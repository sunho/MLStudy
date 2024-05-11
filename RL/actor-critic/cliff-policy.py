import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self, obs_size, act_size):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_size),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


import gymnasium as gym
env = gym.make('CliffWalking-v0')

model = NeuralNetwork(env.observation_space.n, env.action_space.n)
optimizer = torch.optim.SGD(model.parameters(), lr=0.005)

def run(env):
    observation, info = env.reset()

# make function to compute action distribution
    def get_policy(obs):
        logits = model(obs)
        return torch.distributions.Categorical(logits=logits)

# make action selection function (outputs int actions, sampled from policy)
    def get_action(obs):
        return get_policy(obs).sample().item()

    def compute_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)
        return -(logp * weights).mean()

    for _ in range(100):
        observation, info = env.reset()
        tot_reward = 0
        batch_obs = []
        batch_acts = []
        for _ in range(10000):
            obs = np.eye(env.observation_space.n)[observation]
            batch_obs.append(obs)
            action = get_action(torch.as_tensor(obs, dtype=torch.float32))
            observation, reward, terminated, truncated, info = env.step(action)
            tot_reward += reward
            batch_acts.append(action)
            if terminated or truncated:
                break


        optimizer.zero_grad()
        batch_loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                            act=torch.as_tensor(batch_acts, dtype=torch.int32),
                            weights=torch.as_tensor(tot_reward, dtype=torch.float32)
                            )
        print(batch_loss)
        print(tot_reward)
        batch_loss.backward()
        optimizer.step()

    env.close()

run(gym.make('CliffWalking-v0'))
run(gym.make('CliffWalking-v0', render_mode='human'))

