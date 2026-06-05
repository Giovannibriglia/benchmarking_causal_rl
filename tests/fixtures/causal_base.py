"""Causal environment base abstractions for the eight-cell taxonomy.

This module adds an extension of `BaseEnv` that can expose hidden latent
state and interventional (`do`) reward/transition distributions needed for
online identifiability-gap estimation during training.
"""

from __future__ import annotations

import abc
from typing import Optional

import torch
from src.envs.base import BaseEnv


class CausalEnv(BaseEnv):
    """`BaseEnv` subclass with causal-oracle hooks.

    A `CausalEnv` can expose hidden latent factors (e.g., Z, U) and
    interventional distributions under `do(A=a)`, while still returning the
    standard Gymnasium-compatible step tuple through `reset`/`step`.
    """

    cell: int  # 1..8, set by subclasses
    reward_support: Optional[torch.Tensor] = None

    @abc.abstractmethod
    def latent_state(self) -> torch.Tensor:
        """Return hidden latent state for each vectorized environment."""

    @abc.abstractmethod
    def do_reward(self, action: torch.Tensor) -> torch.Tensor:
        """Return interventional reward distribution at current state."""

    @abc.abstractmethod
    def do_transition(self, action: torch.Tensor) -> torch.Tensor:
        """Return interventional next-state distribution at current state."""

    def supports_oracle(self) -> bool:
        """Whether `do_reward`/`do_transition` are exact analytical oracles."""
        return True

    def observed_reward_distribution(
        self, action: torch.Tensor, reward: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        """Optional observational reward distribution hook."""
        _ = action, reward
        return None
