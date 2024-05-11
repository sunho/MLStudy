import torch
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
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

@dataclass
class Episode:
    observations: list = field(default_factory=list)
    next_observations: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    rewards: list = field(default_factory=list)

class EpisodeBuffer:
    def __init__(self):
        self.episodes = []
        pass

    def collect_one(self, env: gym.Env, agent: Agent, time_limit: int):
        observation, info = env.reset()
        episode = Episode()

        for i in range(time_limit):
            cur_obs = agent.encode(observation)
            action = agent.get_action(torch.as_tensor(cur_obs, dtype=torch.float32))
            observation, reward, terminated, truncated, info = env.step(action)
            episode.observations.append(cur_obs)
            episode.next_observations.append(agent.encode(observation))
            episode.actions.append(action)
            episode.rewards.append(reward)
            if terminated or truncated:
                break

        self.episodes.append(episode)

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
    batch_size: int = 64
    max_episode_len: int = 1024
    epochs: int = 1000
    entropy: float = 0.01
    num_envs: int = 4
    report_loss: Any = None
    report_reward: Any = None

def ppo_train(envname, agent: A2CAgent, options: PPOOptions):
    env = gym.make(envname)
    stats = RLTrainStats()

    for i in range(options.epochs):
        episode_buffer = EpisodeBuffer()
        for k in range(options.num_envs):
            episode_buffer.collect_one(env, agent, options.max_episode_len)

        batch_obs = []
        batch_next_obs = []
        batch_rewards = []
        batch_rewards_to_go = []
        batch_acts = []

        for episode in episode_buffer.get_episodes():
            batch_obs.extend(episode.observations)
            batch_next_obs.extend(episode.next_observations)
            batch_rewards.extend(episode.rewards)
            rewards_to_go = episode.rewards.copy()
            n = len(episode.observations)
            for i in reversed(range(n-1)):
                rewards_to_go[i] += options.gamma*rewards_to_go[i+1]
            batch_rewards_to_go.extend(rewards_to_go)
            batch_acts.extend(episode.actions)

            if options.report_reward:
                options.report_reward(np.sum(episode.rewards))
            stats.reward_history.append(np.sum(episode.rewards))

        m = len(batch_obs)
        perm = np.random.permutation(m)
        batch_obs = torch.as_tensor(batch_obs, dtype=torch.float32)[perm]
        batch_next_obs = torch.as_tensor(batch_next_obs, dtype=torch.float32)[perm]
        batch_rewards = torch.as_tensor(batch_rewards, dtype=torch.float32)[perm][:, None]
        batch_rewards_to_go = torch.as_tensor(batch_rewards_to_go, dtype=torch.float32)[perm][:, None]
        batch_acts = torch.as_tensor(batch_acts, dtype=torch.float32)[perm]

        A = (batch_rewards_to_go - agent.get_value(batch_obs)).detach()
        logp_old = agent.get_policy(batch_obs).log_prob(batch_acts).detach()[:, None]
        for j in range((m+options.batch_size-1)//options.batch_size):
            start = j*options.batch_size
            end = min((j+1)*options.batch_size,m)

            options.optimizer.zero_grad()
            y = options.gamma*agent.get_value(batch_next_obs[start:end]) + batch_rewards[start:end]

            logp = agent.get_policy(batch_obs[start:end]).log_prob(batch_acts[start:end])[:, None]

            ratios = (logp - logp_old[start:end]).exp()
            surr1 = ratios * A[start:end]
            surr2 = torch.clamp(ratios, 1-options.epsilon, 1+options.epsilon) * A[start:end]

            loss1 = F.mse_loss(y, batch_rewards_to_go[start:end])
            loss2 = -torch.min(surr1,surr2).mean()

            probs = agent.get_policy(batch_obs[start:end]).probs
            ent = -(probs * torch.log(probs)).sum()

            total_loss = (0.5*loss1 + loss2 - options.entropy * ent) / ((m+options.batch_size-1)//options.batch_size)
            total_loss.backward()
            options.optimizer.step()

            if options.report_loss:
                options.report_loss(total_loss.item())
            stats.loss_history.append(total_loss.item())
    return stats


def test():
    N = 48
    env = gym.make('Taxi-v3')
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

def run(envname, agent, max_episode_len=int(1e5)):
    env = gym.make(envname, render_mode="human")
    while True:
        observation, info = env.reset()
        env.render()

        for j in range(max_episode_len):
            action = agent.get_action(torch.as_tensor(agent.encode(observation), dtype=torch.float32))
            observation, reward, terminated, truncated, info = env.step(action)
            env.render()
            if terminated or truncated:
                break

def run_jupyter(envname, agent, max_episode_len=int(1e5)):
    from IPython.display import clear_output
    import matplotlib.pyplot as plt
    env = gym.make(envname, render_mode="rgb_array")

    while True:
        observation, info = env.reset()
        env.render()

        for j in range(max_episode_len):
            action = agent.get_action(torch.as_tensor(agent.encode(observation), dtype=torch.float32))
            observation, reward, terminated, truncated, info = env.step(action)
            clear_output(wait=True)
            plt.imshow(env.render())
            plt.show()
            if terminated or truncated:
                break

