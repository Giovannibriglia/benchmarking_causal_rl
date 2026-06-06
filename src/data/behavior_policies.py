"""Logging (behavior) policies with EXACT propensities.

Ported from the recovered ``biased_explorer.py`` (Phase-3 gate verdict (d)):
``UniformExplorer`` / ``EpsilonGreedyExplorer`` are the tier-collection
logging policies; every ``select_action`` returns the exact
``log pi_b(a|s)``, which the collection tools write into Minari ``infos``
at collection time — Minari does not store propensities natively, so this
logging is load-bearing (§6.3). The confounded variant arrives in Phase 4.
"""

from __future__ import annotations

import abc
from typing import Optional, Tuple

import torch
import torch.nn as nn


class BehaviorPolicy(abc.ABC):
    """Pluggable logging-policy interface (exact propensities required)."""

    @abc.abstractmethod
    def select_action(
        self,
        obs: torch.Tensor,
        latent: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return sampled action and exact ``log pi_b(a|s[,u])``."""


class UniformExplorer(BehaviorPolicy):
    def __init__(self, n_actions: int, device: torch.device) -> None:
        self.n_actions = int(n_actions)
        self.device = device

    def select_action(
        self, obs: torch.Tensor, latent: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _ = latent
        batch = obs.shape[0]
        probs = torch.full(
            (batch, self.n_actions), 1.0 / self.n_actions, device=self.device
        )
        action = torch.multinomial(probs, num_samples=1).squeeze(-1)
        logp = torch.log(probs.gather(1, action.unsqueeze(-1)).squeeze(-1))
        return action, logp


class EpsilonGreedyExplorer(BehaviorPolicy):
    """ε-greedy over a Q-network with exact mixture propensities."""

    def __init__(self, q_network: nn.Module, epsilon: float = 0.1) -> None:
        self.q_network = q_network
        self.epsilon = float(min(max(epsilon, 0.0), 1.0))

    def select_action(
        self, obs: torch.Tensor, latent: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _ = latent
        with torch.no_grad():
            q_values = self.q_network(obs)
            n_actions = q_values.shape[1]
            greedy = torch.argmax(q_values, dim=-1)
            uniform = torch.randint(
                0, n_actions, (obs.shape[0],), device=obs.device, dtype=torch.long
            )
            choose_uniform = torch.rand(obs.shape[0], device=obs.device) < self.epsilon
            action = torch.where(choose_uniform, uniform, greedy)
            base = torch.full_like(q_values, self.epsilon / n_actions)
            base.scatter_add_(
                1,
                greedy.unsqueeze(-1),
                torch.full((obs.shape[0], 1), 1.0 - self.epsilon, device=obs.device),
            )
            logp = torch.log(
                base.gather(1, action.unsqueeze(-1)).squeeze(-1).clamp_min(1e-8)
            )
        return action, logp

    def log_probs(self, obs: torch.Tensor) -> torch.Tensor:
        """Full [B, A] log-propensity matrix (for OPE target evaluation)."""
        with torch.no_grad():
            q_values = self.q_network(obs)
            n_actions = q_values.shape[1]
            greedy = torch.argmax(q_values, dim=-1)
            base = torch.full_like(q_values, self.epsilon / n_actions)
            base.scatter_add_(
                1,
                greedy.unsqueeze(-1),
                torch.full((obs.shape[0], 1), 1.0 - self.epsilon, device=obs.device),
            )
            return torch.log(base.clamp_min(1e-8))
