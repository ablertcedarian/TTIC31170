import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class MLP(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)
        logits = self.net(x)
        return torch.softmax(logits, dim=-1)


class Policy:
    def __init__(self, state_dim, action_dim, lr = 0.002, gamma = 0.9):
        self.gamma = gamma
        
        self.policy_net = MLP(state_dim, action_dim)
        self.optim = optim.Adam(self.policy_net.parameters(), lr=lr)

        self.episode_rewards = []
        self.episode_log_probs = []

    def scale_state(self, state):
        # scale state to [0,1] for MountainCar-v0
        pos, vel = state
        pos = (pos + 1.2) / 1.8
        vel = (vel + 0.07) / 0.14
        return np.array([pos, vel], dtype=np.float32)

    def shape_reward(self, state, action, reward, next_state, done):

        if len(state) == 2:
        # TODO: Your code goes here
        # You may choose to use any of the variables above
        # to compute the shaped reward

            pass

        # End of your code
        return reward
    
    def sample_action(self, state):
        if len(state) == 2:
            state = self.scale_state(state)

        # TODO: Your code goes here
        # sample action from policy

        action = 

        # End of your code

        return action
    
        
    def __call__(self, state, action, reward, next_state, done):

        reward = self.shape_reward(state, action, reward, next_state, done)

        self.episode_rewards.append(reward)

        if done:
            # TODO: Your code goes here
            # compute discounted returns for this episode

            returns = 

            # End of your code


            # TODO: Your code goes here
            # perform standard REINFORCE update over this episode

            loss = 

            # End of your code

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            self.episode_rewards = []
            self.episode_log_probs = []




