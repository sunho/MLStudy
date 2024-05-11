import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import numpy as np

N = 256
class NeuralNetwork(nn.Module):
  def __init__(self, obs_size, act_size):
      super().__init__()
      self.affine = nn.Linear(obs_size, N)
      self.affine2 = nn.Linear(obs_size, N)
  #self.hidden = nn.Linear(N, N)
      self.action_head = nn.Linear(N, act_size)
      self.value_head = nn.Linear(N, 1)

  def forward(self, x):
      x1 = F.relu(self.affine(x))
      x2 = F.relu(self.affine2(x))
      prob = self.action_head(x1)
      value = self.value_head(x2)

      return prob, value

env = gym.make('Taxi-v3', render_mode="rgb_array")

model = NeuralNetwork(env.observation_space.n, env.action_space.n)
model.load_state_dict(torch.load("optimized.model"))

env.close()

def get_policy(model, obs):
    logits = model(obs)[0]
    return torch.distributions.Categorical(logits=logits)
def get_action(model, obs):
    return get_policy(model, obs).sample().item()

def run(envname, model):
  env = gym.make(envname, render_mode="human")

  while True:
    observation, info = env.reset()
    env.render()

    for j in range(int(1e5)):
      obs = np.eye(env.observation_space.n)[observation]
      #action = get_policy(model, torch.as_tensor(obs, dtype=torch.float32)).probs.flatten().argmax().item()
      #print(action)
      action = get_action(model, torch.as_tensor(obs, dtype=torch.float32))
      observation, reward, terminated, truncated, info = env.step(action)
      
      if terminated or truncated:
          break

run('Taxi-v3', model)
