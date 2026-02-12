from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from .base_actor_critic import BaseActorCritic, RolloutBatch


class A2C(BaseActorCritic):
    def __init__(
        self,
        policy: torch.nn.Module,
        device: torch.device,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
    ) -> None:
        super().__init__(policy, device, gamma=gamma, gae_lambda=gae_lambda)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

    def update(self, batch: RolloutBatch) -> Dict[str, float]:
        advantages, returns = self.compute_gae(
            batch.rewards, batch.dones, batch.values, batch.next_values
        )
        policy_loss = -(batch.log_probs * advantages.detach()).mean()
        value_loss = F.mse_loss(batch.values, returns.detach())
        entropy = batch.log_probs.exp() * batch.log_probs
        entropy_loss = entropy.mean()
        loss = (
            policy_loss
            + self.value_coef * value_loss
            + self.entropy_coef * entropy_loss
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": (-entropy_loss).item(),
            "actor_loss": policy_loss.item(),
            "critic_loss": value_loss.item(),
        }
