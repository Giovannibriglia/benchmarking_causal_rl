from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import ActionOutput
from .base_off_policy import BaseOffPolicy
from .replay_buffer import ReplayBuffer


class DQN(BaseOffPolicy):
    """DQN for discrete action spaces."""

    action_type = "discrete"

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

    def act(
        self,
        obs: torch.Tensor,
        state=None,
        *,
        deterministic: bool = False,
        epsilon: float | None = None,
    ) -> ActionOutput:
        if deterministic:
            epsilon = 0.0
        eps = self.epsilon if epsilon is None else epsilon
        # NOTE: torch.rand is evaluated unconditionally (even for eps == 0.0)
        # to preserve the exact RNG consumption order of the original code.
        if torch.rand(1).item() < eps:
            batch = obs.shape[0]
            return ActionOutput(
                action=torch.randint(
                    0, self.q_network(obs).shape[1], (batch,), device=obs.device
                ),
                state=state,
            )
        with torch.no_grad():
            q = self.q_network(obs)
            return ActionOutput(action=torch.argmax(q, dim=1), state=state)

    def learn(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
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
