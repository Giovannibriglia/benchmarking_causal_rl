from __future__ import annotations

import abc
from typing import Dict

import torch


class BaseOffPolicy(abc.ABC):
    def __init__(self, device: torch.device, gamma: float = 0.99) -> None:
        self.device = device
        self.gamma = gamma

    @abc.abstractmethod
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]: ...
