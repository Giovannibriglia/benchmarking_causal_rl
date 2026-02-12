from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_off_policy import BaseOffPolicy
from .replay_buffer import ReplayBuffer


class DDPG(BaseOffPolicy):
    """DDPG for continuous action spaces."""

    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        target_actor: nn.Module,
        target_critic: nn.Module,
        buffer: ReplayBuffer,
        device: torch.device,
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
    ) -> None:
        super().__init__(device, gamma=gamma)
        self.actor = actor
        self.critic = critic
        self.target_actor = target_actor
        self.target_critic = target_critic
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.buffer = buffer
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.tau = tau
        self.noise_std = 0.1

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]

        with torch.no_grad():
            next_actions = self.target_actor(next_obs)
            next_q = self.target_critic(torch.cat([next_obs, next_actions], dim=-1))
            target_q = rewards + self.gamma * next_q * (1.0 - dones)

        # Critic update
        q = self.critic(torch.cat([obs, actions], dim=-1))
        critic_loss = F.mse_loss(q, target_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Actor update (maximize Q)
        actor_actions = self.actor(obs)
        actor_loss = -self.critic(torch.cat([obs, actor_actions], dim=-1)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # soft updates
        for param, target_param in zip(
            self.actor.parameters(), self.target_actor.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )
        for param, target_param in zip(
            self.critic.parameters(), self.target_critic.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "loss": (critic_loss + actor_loss).item(),
        }

    def act(self, obs: torch.Tensor, noise: bool = True) -> torch.Tensor:
        with torch.no_grad():
            action = self.actor(obs)
        if noise:
            action = action + self.noise_std * torch.randn_like(action)
        if hasattr(self, "action_low") and hasattr(self, "action_high"):
            action = torch.max(torch.min(action, self.action_high), self.action_low)
        return action
