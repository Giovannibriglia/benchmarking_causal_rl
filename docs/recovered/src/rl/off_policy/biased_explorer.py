"""Biased exploration policies for confounded offline data collection.

These explorers generate behavior-policy actions and log-probabilities to
support reproducible biased/offline collection under the eight-cell taxonomy.
"""

from __future__ import annotations

import abc
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn


class BiasedExplorer(abc.ABC):
    """Pluggable behavior policy interface."""

    @abc.abstractmethod
    def select_action(
        self,
        obs: torch.Tensor,
        latent: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return sampled action and `log pi_b(a|s[,u])`."""


class UniformExplorer(BiasedExplorer):
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


class EpsilonGreedyExplorer(BiasedExplorer):
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


class ConfoundedExplorer(BiasedExplorer):
    """Pace-style confounded behavior model.

    pi_b(a|s,u) ∝ exp(beta * logit_a(s) + alpha * u * chi_a(s))
    """

    def __init__(
        self,
        logit_fn: Callable[[torch.Tensor], torch.Tensor],
        alpha: float,
        beta: float = 1.0,
        epsilon: float = 0.1,
    ) -> None:
        self.logit_fn = logit_fn
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.epsilon = float(min(max(epsilon, 0.0), 1.0))

    def select_action(
        self, obs: torch.Tensor, latent: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if latent is None:
            raise ValueError(
                "ConfoundedExplorer requires latent input at action selection time."
            )
        base_logits = self.logit_fn(obs)
        if base_logits.ndim != 2:
            raise ValueError("logit_fn(obs) must return shape [B, A].")
        if abs(self.alpha) < 1e-12:
            n_actions = base_logits.shape[1]
            greedy = torch.argmax(base_logits, dim=-1)
            uniform = torch.randint(
                0, n_actions, (obs.shape[0],), device=obs.device, dtype=torch.long
            )
            choose_uniform = torch.rand(obs.shape[0], device=obs.device) < self.epsilon
            action = torch.where(choose_uniform, uniform, greedy)
            probs = torch.full_like(base_logits, self.epsilon / n_actions)
            probs.scatter_add_(
                1,
                greedy.unsqueeze(-1),
                torch.full((obs.shape[0], 1), 1.0 - self.epsilon, device=obs.device),
            )
            logp = torch.log(
                probs.gather(1, action.unsqueeze(-1)).squeeze(-1).clamp_min(1e-8)
            )
            return action, logp
        u = latent[..., -1].reshape(-1, 1).to(base_logits.device)
        gate = torch.tanh(base_logits)
        logits = self.beta * base_logits + self.alpha * u * gate
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        action = torch.multinomial(probs, num_samples=1).squeeze(-1)
        logp = log_probs.gather(1, action.unsqueeze(-1)).squeeze(-1)
        return action, logp
