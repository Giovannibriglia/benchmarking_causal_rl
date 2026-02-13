from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .base_actor_critic import BaseActorCritic, RolloutBatch


class TRPO(BaseActorCritic):
    """Simplified TRPO using KL penalty instead of conjugate-gradient for brevity."""

    def __init__(
        self,
        policy: nn.Module,
        device: torch.device,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        kl_target: float = 0.01,
        value_coef: float = 0.5,
        entropy_coef: float = 0.0,
    ) -> None:
        super().__init__(policy, device, gamma=gamma, gae_lambda=gae_lambda)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.kl_target = kl_target
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    def _kl_divergence(
        self, old_logp: torch.Tensor, new_logp: torch.Tensor
    ) -> torch.Tensor:
        # assumes same actions; KL between old and new categorical/gaussian via log probs
        ratio = torch.exp(new_logp - old_logp)
        return (ratio * (new_logp - old_logp)).mean()

    def update(self, batch: RolloutBatch) -> Dict[str, float]:
        distribution = self.policy.distribution(batch.obs)
        new_logp = self.policy.log_prob(distribution, batch.actions)
        ratio = torch.exp(new_logp - batch.log_probs)
        advantages = batch.advantages
        surrogate = -(ratio * advantages).mean()
        values = self.policy.value(batch.obs)
        value_loss = nn.functional.mse_loss(values, batch.returns)
        entropy_term = distribution.entropy()
        if entropy_term.ndim > 1:
            entropy_term = entropy_term.sum(-1)
        entropy = entropy_term.mean()

        kl = self._kl_divergence(batch.log_probs, new_logp.detach())
        penalty = torch.relu(kl - self.kl_target)
        loss = (
            surrogate
            + self.value_coef * value_loss
            + penalty
            + (-self.entropy_coef * entropy)
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "policy_loss": surrogate.item(),
            "value_loss": value_loss.item(),
            "kl": kl.item(),
            "entropy": entropy.item(),
            "actor_loss": surrogate.item(),
            "critic_loss": value_loss.item(),
        }
