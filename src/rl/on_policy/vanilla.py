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
        distribution = self.policy.distribution(batch.obs)
        logp = self.policy.log_prob(distribution, batch.actions)
        values = self.policy.value(batch.obs)
        advantages = batch.advantages
        returns = batch.returns
        policy_loss = -(logp * advantages).mean()
        value_loss = F.mse_loss(values, returns)
        total_loss = policy_loss + 0.5 * value_loss
        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        self.optimizer.step()
        return {
            "loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "actor_loss": policy_loss.item(),
            "critic_loss": value_loss.item(),
        }
