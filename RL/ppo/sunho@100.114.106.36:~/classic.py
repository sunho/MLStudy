import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

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
      action = get_action(model, torch.as_tensor(obs, dtype=torch.float32))
      observation, reward, terminated, truncated, info = env.step(action)
      env.render()
      if terminated or truncated:
          break

def train(envname, N=64, g=0.999, lr=0.001, epsilon=0.1, batch_size=64, buf_size=1024, epoch=1000, entropy=0.01, num_envs=4, loaded=None):
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


#envname = 'CliffWalking-v0'
  env = gym.make(envname, render_mode="rgb_array")

  if loaded:
    model = loaded
  else:
    model = NeuralNetwork(env.observation_space.n, env.action_space.n)
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)

  loss_history = []
  reward_history = []
  length_history = []

  gs_b = []
  gs = []


  for _ in range(epoch):
    tot_reward = []
    tot_reward_pf = []
    batch_obs = []
    batch_obs2 = []
    batch_acts = []
    for k in range(num_envs):
      observation, info = env.reset()
      
      tot_reward_ = []
      tot_reward_pf_ = []
      batch_obs_ = []
      batch_obs2_ = []
      batch_acts_ = []
      for j in range(buf_size):
          batch_obs_.append(observation)
          action = get_action(model, torch.as_tensor(observation, dtype=torch.float32))
          observation, reward, terminated, truncated, info = env.step(action)
          obs2 = np.eye(env.observation_space.n)[observation]
          batch_obs2_.append(obs2)
          tot_reward_.append(reward)
          tot_reward_pf_.append(reward)

          batch_acts_.append(action)
          if terminated or truncated:
              break

      for i in reversed(range(len(tot_reward_)-1)):
          tot_reward_pf_[i] += g*tot_reward_pf_[i+1] 

      print(sum(tot_reward_))
      # if len(tot_reward_) < 200:
      #   for z in range(10):
      #     tot_reward.extend(tot_reward_)
      #     tot_reward_pf.extend(tot_reward_pf_)
      #     batch_obs.extend(batch_obs_)
      #     batch_obs2.extend(batch_obs2_)
      #     batch_acts.extend(batch_acts_)
      reward_history.append(sum(tot_reward_))
      length_history.append(len(tot_reward_))
      tot_reward.extend(tot_reward_)
      tot_reward_pf.extend(tot_reward_pf_)
      batch_obs.extend(batch_obs_)
      batch_obs2.extend(batch_obs2_)
      batch_acts.extend(batch_acts_)

    batch_obs = torch.as_tensor(batch_obs, dtype=torch.float32)
    batch_obs2 = torch.as_tensor(batch_obs2, dtype=torch.float32)
    batch_acts = torch.as_tensor(batch_acts, dtype=torch.float32)
    tot_reward = torch.as_tensor(tot_reward, dtype=torch.float32)
    tot_reward_pf = torch.as_tensor(tot_reward_pf, dtype=torch.float32)

    A = tot_reward_pf[:, None] - model(batch_obs)[1]
    A = A.detach()
    #A= (A- A.mean()) / (A.std() + 1e-8)
    logp_k = get_policy(model, batch_obs).log_prob(batch_acts).detach()
    m = len(batch_obs)
    perm = torch.randperm(m)
    A = A[perm]
    logp_k = logp_k[perm]
    batch_obs2 = batch_obs2[perm]
    batch_obs = batch_obs[perm]
    batch_acts = batch_acts[perm]
    tot_reward = tot_reward[perm]
    tot_reward_pf = tot_reward_pf[perm]
    for i in range((m+batch_size-1)//batch_size):
        start = i*batch_size
        end = min((i+1)*batch_size,m)
        optimizer.zero_grad()
        #print(1-done)
        y = g*model(batch_obs2[start:end])[1] + tot_reward[start:end, None]

        logp = get_policy(model, batch_obs[start:end]).log_prob(batch_acts[start:end])
        ratios = (logp - logp_k[start:end]).exp()[:, None]
        surr1 = ratios * A[start:end]
        surr2 = torch.clamp(ratios, 1-epsilon, 1+epsilon) * A[start:end]
        probs = get_policy(model, batch_obs[start:end]).probs
        #print(y)
        #print(logp)
        loss1 = F.mse_loss(y, tot_reward_pf[start:end, None])
        loss2 = -torch.min(surr1,surr2).mean()
        ent = -(probs * torch.log(probs)).sum()
    #print("ent",ent)
        total_loss = (0.5*loss1 + loss2 - entropy * ent) / ((m+batch_size-1)//batch_size)
        total_loss.backward()
        optimizer.step()
        loss_history.append(total_loss.item())
        # if i == 0:
        #   print(total_loss)

  #torch.save(model, "saved-{}-{}-{}-{}-{}.model".format(N, g, lr, epsilon,batch_size))
  env.close()


  return reward_history, loss_history, length_history, model

#_,_,_,model = train('Taxi-v3', N=1024, g=0.9, lr=0.005, epsilon=0.1, batch_size=64, buf_size=1024, epoch=200, entropy=0.01)
# run('Taxi-v3', model)
