from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from .base_actor_critic import BaseActorCritic, RolloutBatch


class VanillaPolicyGradient(BaseActorCritic):
    def __init__(
        self,
        policy: torch.nn.Module,
        device: torch.device,
        lr: float = 3e-4,
        gamma: float = 0.99,
    ):
        super().__init__(policy, device, gamma=gamma, gae_lambda=1.0)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def update(self, batch: RolloutBatch) -> Dict[str, float]:
        advantages, returns = self.compute_gae(
            batch.rewards, batch.dones, batch.values, batch.next_values
        )
        log_probs = batch.log_probs
        loss = -(log_probs * advantages.detach()).mean()
        value_loss = F.mse_loss(batch.values, returns.detach())
        total_loss = loss + 0.5 * value_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return {
            "loss": total_loss.item(),
            "policy_loss": loss.item(),
            "value_loss": value_loss.item(),
            "actor_loss": loss.item(),
            "critic_loss": value_loss.item(),
        }
