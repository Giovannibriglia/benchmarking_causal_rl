from __future__ import annotations

import abc
from typing import Tuple

import torch


class BasePolicy(abc.ABC, torch.nn.Module):
    device: torch.device

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.to(device)

    @abc.abstractmethod
    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return action and log probability (if stochastic)."""

    @abc.abstractmethod
    def value(self, obs: torch.Tensor) -> torch.Tensor:
        """Return state-value estimate."""
