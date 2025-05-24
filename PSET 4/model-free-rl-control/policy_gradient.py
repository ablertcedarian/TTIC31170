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

            # mechanical energy proxy
            # print(f"Given state {state} with action {action} and pre-reward {reward}")
            # reward += 1 * state[0] + (1/2) * 0 * (state[1]**2)
            # print(f"    to {reward}")
            kp = 15
            kv = 0
            reward += self.gamma * (kp * next_state[0]) - (kp * state[0])
        # End of your code
        return reward

    def sample_action(self, state):
        if len(state) == 2:
            state = self.scale_state(state)
            state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)  # (1,2)

        # TODO: Your code goes here
        # sample action from policy

        probs = self.policy_net(state).squeeze(0)
        dist = Categorical(probs)
        action = dist.sample().item()
        self.episode_log_probs.append(
            dist.log_prob(
                torch.tensor(action, device=probs.device)
                # action
            )
        )

        # End of your code

        return action


    def __call__(self, state, action, reward, next_state, done):

        reward = self.shape_reward(state, action, reward, next_state, done)

        self.episode_rewards.append(reward)

        if done:
            # TODO: Your code goes here
            # compute discounted returns for this episode

            rewards = torch.tensor(
                self.episode_rewards,
            )
            G, returns = 0, []
            for r in reversed(self.episode_rewards):
                G = r + self.gamma * G
                returns.insert(0, G)
            returns = torch.tensor(returns, dtype=torch.float32)

            if not hasattr(self, "baseline"):
                self.baseline = 0.0           # initialise once

            self.baseline = 0.9 * self.baseline + 0.1 * returns.mean().item()
            adv = returns - self.baseline     # advantage
            if adv.std() > 0.1:                        # threshold
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            # End of your code


            # TODO: Your code goes here
            # perform standard REINFORCE update over this episode

            log_probs = torch.stack(self.episode_log_probs)
            loss = -(adv.detach() * log_probs).sum()        # REINFORCE with baseline


            # End of your code

            self.optim.zero_grad()
            loss.backward()
            if done and (len(self.episode_rewards) % 200 == 0):
                print(f"[dbg] return mean {returns.mean():.1f}, grad-norm "
                    f"{torch.norm(torch.stack([p.grad.norm() for p in self.policy_net.parameters()] )):.2f}")

            self.optim.step()

            self.episode_rewards = []
            self.episode_log_probs = []




