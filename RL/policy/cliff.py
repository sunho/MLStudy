import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self, obs_size, act_size):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, act_size),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


import gymnasium as gym
envname = 'Taxi-v3'
#envname = 'CliffWalking-v0'
env = gym.make(envname, render_mode="rgb_array")

model = NeuralNetwork(env.observation_space.n, env.action_space.n)
optimizer = torch.optim.SGD(model.parameters(), lr=0.002)

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

    for _ in range(500):
        observation, info = env.reset()
        tot_reward = []
        batch_obs = []
        batch_acts = []
        for _ in range(1000):
            obs = np.eye(env.observation_space.n)[observation]
            batch_obs.append(obs)
            action = get_action(torch.as_tensor(obs, dtype=torch.float32))
            observation, reward, terminated, truncated, info = env.step(action)
            tot_reward.append(reward)
            batch_acts.append(action)
            if terminated or truncated:
                break

        for i in reversed(range(len(tot_reward)-1)):
            tot_reward[i] += tot_reward[i+1] 
        
        print(tot_reward[0])
        if gs:
            b = np.mean(gs_b) / np.mean(gs)
        else:
            b = 0
        for i in reversed(range(len(tot_reward))):
            tot_reward[i] -= b

        optimizer.zero_grad()
        batch_loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                            act=torch.as_tensor(batch_acts, dtype=torch.int32),
                            weights=torch.as_tensor(tot_reward, dtype=torch.float32)
                            )
        print(batch_loss)
        gs_b.append((tot_reward[0]*torch.norm(batch_loss)**2).item())
        gs.append((torch.norm(batch_loss)**2).item())
        batch_loss.backward()
        optimizer.step()

    env.close()

run(gym.make(envname))
run(gym.make(envname, render_mode='human'))

