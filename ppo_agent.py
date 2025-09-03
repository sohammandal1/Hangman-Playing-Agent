import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class PPOAgent:
    def __init__(self, model, lr=3e-4, gamma=0.95, eps_clip=0.2):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.policy_old = model
        self.loss_fn = nn.CrossEntropyLoss()

    def compute_returns(self, rewards, masks):
        R = 0
        returns = []
        for r, m in zip(reversed(rewards), reversed(masks)):
            R = r + self.gamma * R * m
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32)

    def update(self, log_probs, values, rewards, masks, actions):
        returns = self.compute_returns(rewards, masks)
        values = torch.stack(values)
        log_probs = torch.stack(log_probs)

        advantage = returns - values.detach()

        ratio = torch.exp(log_probs - log_probs.detach())
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - self.eps_clip,
                            1 + self.eps_clip) * advantage

        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = nn.functional.mse_loss(values, returns)

        loss = policy_loss + 0.5 * value_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
