from __future__ import annotations

import abc
from typing import Dict

import torch

from ..base import Algorithm


class BaseOffPolicy(Algorithm):
    paradigm = "off_policy"

    def __init__(self, device: torch.device, gamma: float = 0.99) -> None:
        super().__init__()
        self.device = device
        self.gamma = gamma

    @abc.abstractmethod
    def learn(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]: ...
