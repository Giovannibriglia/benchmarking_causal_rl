"""Confounding-robust DQN with causal identifiability-gap regularization.

Implements a minimal DQN variant that augments TD loss with a sensitivity
penalty based on the causal gap estimator used in the companion paper.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from src.causal_metrics.gap import compute_gap
from src.envs.causal_base import CausalEnv

from .dqn import DQN


class ConfoundingRobustDQN(DQN):
    """DQN + lambda * Delta_phi regularization."""

    def __init__(
        self,
        *args,
        env_oracle: Optional[CausalEnv] = None,
        lam: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.env_oracle = env_oracle
        self.lam = float(lam)

    def update(self, batch: Dict[str, Optional[torch.Tensor]]) -> Dict[str, float]:
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]
        if any(x is None for x in [obs, actions, rewards, next_obs, dones]):
            raise ValueError(
                "ConfoundingRobustDQN requires standard transition tensors."
            )

        obs = obs  # type: ignore[assignment]
        actions = actions.long()  # type: ignore[union-attr]
        rewards = rewards  # type: ignore[assignment]
        next_obs = next_obs  # type: ignore[assignment]
        dones = dones  # type: ignore[assignment]

        q_values = self.q_network(obs).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_q = self.target_network(next_obs).max(dim=1).values
            target = rewards + self.gamma * next_q * (1.0 - dones)
        td_loss = F.mse_loss(q_values, target)

        delta_val = torch.tensor(0.0, device=obs.device)
        if self.env_oracle is not None and actions.shape[0] == int(
            getattr(self.env_oracle, "n_envs", actions.shape[0])
        ):
            gap = compute_gap(
                env=self.env_oracle,
                obs=obs,
                action=actions,
                reward=rewards,
                next_obs=next_obs,
                divergence="tv",
            )
            delta_val = torch.tensor(float(gap.delta), device=obs.device)

        loss = td_loss + self.lam * delta_val
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        for param, target_param in zip(
            self.q_network.parameters(), self.target_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

        return {
            "loss": float(loss.item()),
            "td_loss": float(td_loss.item()),
            "critic_loss": float(td_loss.item()),
            "q_loss": float(td_loss.item()),
            "delta_tv": float(delta_val.item()),
        }
