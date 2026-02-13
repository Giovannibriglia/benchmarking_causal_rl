from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from .base_actor_critic import BaseActorCritic, RolloutBatch


class PPO(BaseActorCritic):
    def __init__(
        self,
        policy: torch.nn.Module,
        device: torch.device,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        train_iters: int = 4,
        batch_size: int = 256,
    ) -> None:
        super().__init__(policy, device, gamma=gamma, gae_lambda=gae_lambda)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.train_iters = train_iters
        self.batch_size = batch_size

    def _mini_batches(self, batch: RolloutBatch):
        n = batch.obs.shape[0]
        indices = torch.randperm(n, device=self.device)
        for start in range(0, n, self.batch_size):
            end = start + self.batch_size
            idx = indices[start:end]
            yield RolloutBatch(
                obs=batch.obs[idx],
                actions=batch.actions[idx],
                log_probs=batch.log_probs[idx],
                rewards=batch.rewards[idx],
                dones=batch.dones[idx],
                values=batch.values[idx],
                next_values=batch.next_values[idx],
                advantages=batch.advantages[idx],
                returns=batch.returns[idx],
            )

    def update(self, batch: RolloutBatch) -> Dict[str, float]:
        metrics = {}
        # Safety: old tensors must be constants.
        assert not batch.log_probs.requires_grad
        assert not batch.advantages.requires_grad
        assert not batch.returns.requires_grad
        for _ in range(self.train_iters):
            for mini in self._mini_batches(batch):
                distribution = self.policy.distribution(mini.obs)
                logp = self.policy.log_prob(distribution, mini.actions)
                ratio = torch.exp(logp - mini.log_probs)
                surr1 = ratio * mini.advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                    * mini.advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                values = self.policy.value(mini.obs)
                value_loss = F.mse_loss(values, mini.returns)

                entropy_term = distribution.entropy()
                if entropy_term.ndim > 1:
                    entropy_term = entropy_term.sum(-1)
                entropy = entropy_term.mean()
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

                metrics = {
                    "loss": loss.item(),
                    "policy_loss": policy_loss.item(),
                    "value_loss": value_loss.item(),
                    "entropy": entropy.item(),
                    "actor_loss": policy_loss.item(),
                    "critic_loss": value_loss.item(),
                }
        return metrics
