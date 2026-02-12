from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_off_policy import BaseOffPolicy
from .replay_buffer import ReplayBuffer


class DQN(BaseOffPolicy):
    """DQN for discrete action spaces."""

    def __init__(
        self,
        q_network: nn.Module,
        target_network: nn.Module,
        buffer: ReplayBuffer,
        device: torch.device,
        lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        epsilon: float = 0.1,
    ) -> None:
        super().__init__(device, gamma=gamma)
        self.q_network = q_network
        self.target_network = target_network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.buffer = buffer
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.tau = tau
        self.epsilon = epsilon

    def act(self, obs: torch.Tensor, epsilon: float | None = None) -> torch.Tensor:
        eps = self.epsilon if epsilon is None else epsilon
        if torch.rand(1).item() < eps:
            batch = obs.shape[0]
            return torch.randint(
                0, self.q_network(obs).shape[1], (batch,), device=obs.device
            )
        with torch.no_grad():
            q = self.q_network(obs)
            return torch.argmax(q, dim=1)

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = batch["obs"]
        actions = batch["actions"].long()
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]

        q_values = self.q_network(obs).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_q = self.target_network(next_obs).max(dim=1).values
            target = rewards + self.gamma * next_q * (1.0 - dones)
        loss = F.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        for param, target_param in zip(
            self.q_network.parameters(), self.target_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )
        return {"loss": loss.item(), "critic_loss": loss.item(), "q_loss": loss.item()}
