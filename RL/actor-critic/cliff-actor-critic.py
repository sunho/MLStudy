import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


N = 500

class NeuralNetwork(nn.Module):
    def __init__(self, obs_size, act_size):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(obs_size, N),
            nn.ReLU(),
            nn.Linear(N, N),
            nn.ReLU(),
            nn.Linear(N, act_size),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

class NeuralNetwork2(nn.Module):
    def __init__(self, obs_size):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(obs_size, N),
            nn.ReLU(),
            nn.Linear(N, N),
            nn.ReLU(),
            nn.Linear(N, 1),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

import gymnasium as gym
envname = 'Taxi-v3'
#envname = 'CliffWalking-v0'
env = gym.make(envname, render_mode="rgb_array")

model = NeuralNetwork(env.observation_space.n, env.action_space.n)
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

model2 = NeuralNetwork2(env.observation_space.n)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.002)

env.close()

def run(env):
    def get_policy(obs):
        logits = model(obs)
        return torch.distributions.Categorical(logits=logits)

    def get_action(obs):
        return get_policy(obs).sample().item()

    def compute_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)
        return -(logp * weights).mean()

    
    gs_b = []
    gs = []

    for _ in range(300):
        observation, info = env.reset()
        tot_reward = []
        tot_reward_pf = []
        batch_obs = []
        batch_obs2 = []
        batch_acts = []

        for j in range(1000):
            obs = np.eye(env.observation_space.n)[observation]
            batch_obs.append(obs)
            action = get_action(torch.as_tensor(obs, dtype=torch.float32))
            observation, reward, terminated, truncated, info = env.step(action)
            obs2 = np.eye(env.observation_space.n)[observation]
            batch_obs2.append(obs2)
            tot_reward.append(reward)
            tot_reward_pf.append(reward)

            batch_acts.append(action)
            if terminated or truncated:
                break

        for i in reversed(range(len(tot_reward)-1)):
            tot_reward_pf[i] += tot_reward_pf[i+1] 

        print(tot_reward_pf[0])

        batch_obs = torch.as_tensor(batch_obs, dtype=torch.float32)
        batch_obs2 = torch.as_tensor(batch_obs2, dtype=torch.float32)
        batch_acts = torch.as_tensor(batch_acts, dtype=torch.float32)
        tot_reward = torch.as_tensor(tot_reward, dtype=torch.float32)
        tot_reward_pf = torch.as_tensor(tot_reward_pf, dtype=torch.float32)

        g = 0.99

        y = g*model2(batch_obs2) + tot_reward[:, None]

        optimizer2.zero_grad()
        loss = nn.SmoothL1Loss()(y, tot_reward_pf[:, None])
        loss.backward()
        optimizer2.step()

        weights = tot_reward_pf - model2(batch_obs)

        optimizer.zero_grad()
        batch_loss = compute_loss(batch_obs,
                            act=batch_acts,
                            weights=weights
                            )
        print(loss, batch_loss)
        gs_b.append((tot_reward[0]*torch.norm(batch_loss)**2).item())
        gs.append((torch.norm(batch_loss)**2).item())
        batch_loss.backward()
        optimizer.step()

    env.close()

run(gym.make(envname))
run(gym.make(envname, render_mode='human'))

