import torch
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import gym
from typing import Any, NamedTuple

class OneHotObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.obs_size = env.observation_space.n

    def observation(self, observation):
        return np.eye(self.obs_size)[observation]

class TorchObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def observation(self, observation):
        return torch.from_numpy(np.array(observation))

class Agent(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_action(self, observation):
        pass

@dataclass
class RLTrainStats:
    loss_history: list = field(default_factory=list)
    reward_history: list = field(default_factory=list)
    length_history: list = field(default_factory=list)

class EpisodeBuffer:
    def __init__(self, maxlen, obs_shape, act_shape):
        self.observations = torch.zeros((maxlen,) + obs_shape)
        self.next_observations = torch.zeros((maxlen, ) + obs_shape)
        self.rewards = torch.zeros(maxlen)
        self.actions = torch.zeros((maxlen, ) + act_shape)
        self.size = 0
        self.episodes = []

    def collect_one(self, env: gym.Env, agent: Agent, time_limit: int):
        observation, info = env.reset()
        start = self.size

        for i in range(time_limit):
            cur_obs = observation
            action = agent.get_action(cur_obs)
            observation, reward, terminated, truncated, info = env.step(action)
            self.observations[self.size] = cur_obs
            self.next_observations[self.size] = observation
            self.actions[self.size] = action
            self.rewards[self.size] = reward
            self.size += 1
            if terminated:
                break
        self.episodes.append((start, self.size))

    def cut(self):
        self.observations = self.observations[:self.size]
        self.next_observations = self.next_observations[:self.size]
        self.rewards = self.rewards[:self.size]
        self.actions = self.actions[:self.size]

    def get_episodes(self):
        return self.episodes

#
# Value fitting agent for tabular case
#
# we only try to approximate A(s,a) value in policy graident and set policy as
# pi(a|s) = 1 if a = argmax_a A(s,a)
#
# argmax_a A(s,a) = argmax_a Q(s,a) = r(s,a) + gamma * E[V(s')]
#
# value iteration algorithm:
# Q(s,a) = r(s,a) + gamma * E[V(s')]
# V(s) = max_a Q(s,a)
#
# pro: 
# - it coverges in tabular case gurnateed by fixed point iteration theorem (it's contraction)
#
# con:
# - requires full table which can be huge on memory and need to know state transition beforehand
#
@dataclass
class ValueFittingOptions:
    gamma: float = 0.99
    max_episode_len: int = 1024
    epochs: int = 1000

class ValueFittingAgent(Agent):
    def __init__(self, state_size, action_size):
        inf = 1e9
        self.Q = np.random.rand(state_size, action_size)
        self.V = np.random.rand(state_size)

    def get_action(self, observation):
        return np.argmax(self.Q, axis=1)[observation]

# Assumes determinstic state transition
def value_fitting_train(create_env, agent: ValueFittingAgent, options: ValueFittingOptions):
    env = create_env()
    stats = RLTrainStats()
    state_size = env.observation_space.n
    action_size = env.action_space.n
    transition = [[(-1,-1) for i in range(action_size)] for j in range(state_size)]
    for i in range(options.epochs):
        eb = EpisodeBuffer(options.max_episode_len, (1, ), (1,))
        eb.collect_one(env, agent, options.max_episode_len)

        tot_reward = torch.sum(eb.rewards).item()
        if options.report_reward:
            options.report_reward(tot_reward)
        stats.reward_history.append(tot_reward)

        m = eb.size
        for i in range(m):
            obs = int(eb.observations[i].item())
            act = int(eb.actions[i].item())
            nxt_obs = int(eb.next_observations[i].item())
            reward = eb.rewards[i].item()
            if transition[obs][act][0] == -1:
                transition[obs][act] = (nxt_obs, reward)

        for i in range(state_size):
            for j in range(action_size):
                if transition[i][j][0] != -1:
                    obs, reward = transition[i][j]
                    agent.Q[i][j] = reward + options.gamma*agent.V[obs]
        
        agent.V = np.max(agent.Q, axis=1)

    return stats
            

#
# PPO agent 
#
# We are doing actor critic but try to be conservative about how much we change the
# policy by clipping importance sampling weight
#
# i.e. objective is KL(\theta | \theta_old) \leq delta
# 
# Policy graident is approximated as follows:
#
# ratio = \pi_{\theta}(a|s) / \pi_{\theta_old}(a|s) 
# L = min(ratio * A_{\theta_old}(s,a), clamp(ratio, 1-epsilon,1+epsilon) * A_{\theta_old}(s,a))
# grad = dL/d\theta
#
# pro:
# - it's one of state of art algorithms
# - very flexible on how you feed data (i.e. advantage can be normalized and still works fine)
#
# con:
# - many implementation details
# - very sensitivie on intial conditions as it's still policy graident that data is collected by running its policy
#
#
# implementation details done:
# - samples collected from vectorized environment and batches splitted in this "merged" buffer 
# -- running multiple training with same data randomized is required as PPO assumes we do this
#    and try to be conservative about the change by clipping ratio
# -- ratio must be all 1's in first minibatch'
# -- "inifite tape" that can still run the environment after agent dies so that we always get fixed size of observations
#   not implemented yet
# - layer initialization by orthogonal weight initialization
# -- seems to improve convergence
# -- (TODO) try implemnting this by myself
# - (TODO) implement GAE.
#

class A2CAgent(Agent):
    def __init__(self):
        pass

    @abstractmethod
    def get_policy(self, observation) -> torch.distributions.Distribution:
        pass

    @abstractmethod
    def get_value(self, observation) -> torch.Tensor:
        pass

    def get_action(self, observation):
        return self.get_policy(observation).sample().item()

class A2CMLAgent(A2CAgent):
    def __init__(self, model, device = None):
        if device:
            model = model.cuda()
        self.model = model
        self.device = device

    def get_policy(self, observation) -> torch.distributions.Distribution:
        if self.device:
            observation = observation.to(self.device)
        return self.model.get_policy(observation)

    def get_value(self, observation) -> torch.Tensor:
        if self.device:
            observation = observation.to(self.device)
        return self.model.get_value(observation)

@dataclass
class PPOOptions:
    optimizer: torch.optim.Optimizer
    gamma: float = 0.99
    epsilon: float = 0.1
    batch_nums: int = 4
    max_episode_len: int = 1024
    epochs: int = 1000
    entropy: float = 0.01
    num_envs: int = 4
    train_count: int = 1
    report_train: Any = None
    report_reward: Any = None
    create_env: Any = None
    batch_size: Any = None

def ppo_train(create_env, agent: A2CAgent, device, options: PPOOptions):
    env = create_env()
    stats = RLTrainStats()

    for i in range(options.epochs):
        episode_buffer = EpisodeBuffer(options.num_envs * options.max_episode_len, env.observation_space.shape, (1,),)
        for k in range(options.num_envs):
            episode_buffer.collect_one(env, agent, options.max_episode_len)

        episode_buffer.cut()
        m = episode_buffer.size
        batch_rewards_to_go = torch.zeros(m)
        for (s,e) in episode_buffer.episodes:
            for i in reversed(range(s, e-1)):
                batch_rewards_to_go[i] += episode_buffer.rewards[i] + options.gamma*batch_rewards_to_go[i+1]
            tot_reward = torch.sum(episode_buffer.rewards[s:e]).item()
            if options.report_reward:
                options.report_reward(tot_reward)
            stats.reward_history.append(tot_reward)

        batch_obs = episode_buffer.observations.to(device)
        batch_next_obs = episode_buffer.next_observations.to(device)
        batch_rewards = episode_buffer.rewards.to(device)
        batch_rewards_to_go = batch_rewards_to_go.to(device)
        batch_acts = episode_buffer.actions.to(device)

        A = (batch_rewards_to_go - agent.get_value(batch_obs).reshape(-1)).detach()
        logp_old = agent.get_policy(batch_obs).log_prob(batch_acts.reshape(-1)).detach()

        if options.batch_size:
            batch_size = options.batch_size
        else:
            batch_size = (m + options.batch_nums - 1) // options.batch_nums

        for _ in range(options.train_count):
            perm = np.random.permutation(m)
            batch_obs = batch_obs[perm]
            batch_next_obs = batch_next_obs[perm]
            batch_rewards = batch_rewards[perm]
            batch_rewards_to_go = batch_rewards_to_go[perm]
            batch_acts = batch_acts[perm]

            A = A[perm]
            logp_old = logp_old[perm]
            for j in range((m+batch_size-1)//batch_size):
                start = j*batch_size
                end = min((j+1)*batch_size,m)

                options.optimizer.zero_grad()
                y = options.gamma*agent.get_value(batch_next_obs[start:end]).reshape(-1) + batch_rewards[start:end]

                logp = agent.get_policy(batch_obs[start:end]).log_prob(batch_acts[start:end].reshape(-1))

                log_ratio = (logp - logp_old[start:end])
                ratios = log_ratio.exp()

                # http://joschu.net/blog/kl-approx.html
                approx_kl = ((ratios - 1) - log_ratio).mean()

                surr1 = ratios * A[start:end]
                surr2 = torch.clamp(ratios, 1-options.epsilon, 1+options.epsilon) * A[start:end]

                loss1 = F.mse_loss(y, batch_rewards_to_go[start:end])
                loss2 = -torch.min(surr1,surr2).mean()

                entropy = agent.get_policy(batch_obs[start:end]).entropy()
                ent = entropy.mean()

                total_loss = (0.5*loss1 + loss2 - options.entropy * ent) / options.batch_nums
                total_loss.backward()
                options.optimizer.step()

                if options.report_train:
                    options.report_train({
                        'total_loss': total_loss.item(),
                        'value_loss': loss1.item(),
                        'policy_loss': loss2.item(),
                        'entropy': ent.item(),
                        'ratios': ratios.mean().item(),
                        'approx_kl': approx_kl.item()
                    })
                stats.loss_history.append(total_loss.item())
    return stats

def run(create_env, agent, device, max_episode_len=int(1e5)):
    env = create_env()
    while True:
        observation, info = env.reset()
        env.render()

        for j in range(max_episode_len):
            action = agent.get_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            env.render()
            if terminated:
                break

def run_jupyter(create_env, agent, device, max_episode_len=int(1e5)):
    from IPython.display import clear_output
    import matplotlib.pyplot as plt
    env = create_env('rgb_array')

    while True:
        observation, info = env.reset()

        for j in range(max_episode_len):
            action = agent.get_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            clear_output(wait=True)
            plt.imshow(env.render())
            plt.show()
            if terminated:
                break

