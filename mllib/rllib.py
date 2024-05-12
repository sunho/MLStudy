import torch
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import gym
from typing import Any, NamedTuple

class Agent(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_policy(self, observation) -> torch.distributions.Distribution:
        pass

    @abstractmethod
    def encode(self, observation):
        pass

    def get_action(self, observation):
        return self.get_policy(observation).sample().item()

class A2CAgent(Agent):
    def __init__(self):
        pass

    @abstractmethod
    def get_value(self, observation) -> torch.Tensor:
        pass


class A2CMLAgent(A2CAgent):
    def __init__(self, model, obs_size, onehot = False):
        self.model = model
        self.obs_size = obs_size
        self.onehot = onehot

    def get_policy(self, observation) -> torch.distributions.Distribution:
        return self.model.get_policy(observation)

    def encode(self, observation):
        if self.onehot:
            observation = np.eye(self.obs_size)[observation]
        return observation

    def get_value(self, observation) -> torch.Tensor:
        return self.model.get_value(observation)

@dataclass
class RLTrainStats:
    loss_history: list = field(default_factory=list)
    reward_history: list = field(default_factory=list)
    length_history: list = field(default_factory=list)

class Turn(NamedTuple):
    observation: Any
    next_observation: Any
    action: Any
    reward: Any

class EpisodeBuffer:
    def __init__(self, maxlen, obs_shape, act_shape):
        self.observations = torch.zeros((maxlen,) + obs_shape)
        self.next_observations = torch.zeros((maxlen, ) + obs_shape)
        self.rewards = torch.zeros(maxlen)
        self.actions = torch.zeros((maxlen, ) + act_shape)
        self.last = 0
        self.episodes = []
        

    def collect_one(self, env: gym.Env, agent: Agent, device, time_limit: int):
        observation, info = env.reset()
        start = self.last

        for i in range(time_limit):
            cur_obs = np.array(agent.encode(observation))
            action = agent.get_action(torch.from_numpy(cur_obs).to(device))
            observation, reward, terminated, truncated, info = env.step(action)
            obs = np.array(agent.encode(observation))
            self.observations[self.last] = torch.from_numpy(cur_obs)
            self.next_observations[self.last] = torch.from_numpy(obs)
            self.actions[self.last] = action
            self.rewards[self.last] = reward
            self.last += 1
            if terminated:
                break
        self.episodes.append((start, self.last))

    def get_episodes(self):
        return self.episodes

#
# PPO Implementation
#

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
        sz = 0
        for k in range(options.num_envs):
            episode_buffer.collect_one(env, agent, device, options.max_episode_len)


        m = episode_buffer.last
        batch_rewards_to_go = torch.zeros(m)
        for (s,e) in episode_buffer.episodes:
            for i in reversed(range(s, e-1)):
                batch_rewards_to_go[i] += episode_buffer.rewards[i] + options.gamma*batch_rewards_to_go[i+1]
            if options.report_reward:
                options.report_reward(torch.sum(episode_buffer.rewards[s:e]))

        batch_obs = episode_buffer.observations.to(device)[:episode_buffer.last]
        batch_next_obs = episode_buffer.next_observations.to(device)[:episode_buffer.last]
        batch_rewards = episode_buffer.rewards.to(device)[:episode_buffer.last]
        batch_rewards_to_go = batch_rewards_to_go.to(device)[:episode_buffer.last]
        batch_acts = episode_buffer.actions.to(device)[:episode_buffer.last]


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

                with torch.no_grad():
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


def test():
    N = 48
    env = create_env()
    class NeuralNetwork(nn.Module):
        def __init__(self, obs_size, act_size):
            super().__init__()
            self.affine = nn.Linear(obs_size, N)
            self.affine2 = nn.Linear(obs_size, N)
            self.action_head = nn.Linear(N, act_size)
            self.value_head = nn.Linear(N, 1)

        def forward(self, x):
            x1 = F.relu(self.affine(x))
            x2 = F.relu(self.affine2(x))
            prob = self.action_head(x1)
            value = self.value_head(x2)

            return prob, value

        def get_policy(self, observation):
            logits = self.forward(observation)[0]
            return torch.distributions.Categorical(logits=logits)

        def get_value(self, observation):
            return self.forward(observation)[1]

    model = NeuralNetwork(env.observation_space.n, env.action_space.n)
    agent = A2CMLAgent(model, env.observation_space.n, True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    options = PPOOptions(optimizer)
    options.epochs = 10
    options.report_reward = lambda x: print(x)
    ppo_train('Taxi-v3', agent, options)
    run_jupyter('Taxi-v3', agent)

def run(create_env, agent, device, max_episode_len=int(1e5)):
    env = create_env()
    while True:
        observation, info = env.reset()
        env.render()

        for j in range(max_episode_len):
            action = agent.get_action(torch.as_tensor(agent.encode(observation), dtype=torch.float32).to(device))
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
            xx = torch.as_tensor(agent.encode(observation), dtype=torch.float32).to(device)
            print(agent.get_policy(xx).probs)
            action = agent.get_action(xx)
            observation, reward, terminated, truncated, info = env.step(action)
            clear_output(wait=True)
            plt.imshow(env.render())
            plt.show()
            if terminated:
                break

