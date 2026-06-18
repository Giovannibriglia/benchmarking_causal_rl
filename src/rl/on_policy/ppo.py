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
                next_obs=batch.next_obs[idx],
                actions=batch.actions[idx],
                log_probs=batch.log_probs[idx],
                rewards=batch.rewards[idx],
                dones=batch.dones[idx],
                values=batch.values[idx],
                next_values=batch.next_values[idx],
                advantages=batch.advantages[idx],
                returns=batch.returns[idx],
            )

    def learn(self, batch: RolloutBatch) -> Dict[str, float]:
        # Recurrent batches (from rollout_recurrent) carry a (T, N) seq shape and
        # require truncated BPTT over contiguous per-env sequences — the flat
        # shuffled-minibatch path below cannot be used (shuffling breaks the time
        # axis). The MLP path keeps the original flat update UNCHANGED.
        if batch.recurrent_seq_shape is not None:
            return self._learn_recurrent(batch)
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

    def _learn_recurrent(self, batch: RolloutBatch) -> Dict[str, float]:
        """Truncated-BPTT PPO update for recurrent trunks. The whole rollout
        (all N envs × T steps) is one batch per epoch — no flat shuffling, since
        sequences must stay contiguous in time. Forward is a single
        ``evaluate_sequence`` pass with per-env episode-boundary resets;
        ``loss.backward()`` propagates through the recurrent cells (the BPTT)."""
        assert not batch.log_probs.requires_grad
        assert not batch.advantages.requires_grad
        assert not batch.returns.requires_grad
        T, N = batch.recurrent_seq_shape
        obs_seq = batch.obs.view(T, N, *batch.obs.shape[1:])
        if batch.actions.ndim > 1:
            actions_seq = batch.actions.view(T, N, *batch.actions.shape[1:])
        else:
            actions_seq = batch.actions.view(T, N)
        dones = batch.dones.view(T, N)
        adv = batch.advantages.view(T, N)
        ret = batch.returns.view(T, N)
        old_logp = batch.log_probs.view(T, N)
        # episode_starts[t]: step t begins a fresh episode -> reset hidden state.
        # t=0 always (rollout reset the env); t>0 iff the previous step was done.
        episode_starts = torch.zeros(T, N, device=self.device)
        episode_starts[0] = 1.0
        if T > 1:
            episode_starts[1:] = dones[:-1]

        metrics = {}
        for _ in range(self.train_iters):
            logp, values, entropy = self.policy.evaluate_sequence(
                obs_seq, actions_seq, episode_starts, batch.recurrent_states
            )
            ratio = torch.exp(logp - old_logp)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values, ret)
            entropy_mean = entropy.mean()
            loss = (
                policy_loss
                + self.value_coef * value_loss
                - self.entropy_coef * entropy_mean
            )

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

            metrics = {
                "loss": loss.item(),
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
                "entropy": entropy_mean.item(),
                "actor_loss": policy_loss.item(),
                "critic_loss": value_loss.item(),
            }
        return metrics
