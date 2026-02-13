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
        # Recompute forward pass so each update has its own graph; stored buffers stay detached.
        distribution = self.policy.distribution(batch.obs)
        logp = self.policy.log_prob(distribution, batch.actions)
        values = self.policy.value(batch.obs)
        advantages = batch.advantages
        returns = batch.returns
        policy_loss = -(logp * advantages).mean()
        value_loss = F.mse_loss(values, returns)
        entropy_term = distribution.entropy()
        if entropy_term.ndim > 1:
            entropy_term = entropy_term.sum(-1)
        entropy = entropy_term.mean()
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "actor_loss": policy_loss.item(),
            "critic_loss": value_loss.item(),
        }
