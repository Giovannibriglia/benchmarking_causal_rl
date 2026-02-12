from __future__ import annotations

import abc
from typing import Tuple

import torch


class BaseEnv(abc.ABC):
    device: torch.device

    @abc.abstractmethod
    def reset(self, seed: int | None = None) -> Tuple[torch.Tensor, dict]: ...

    @abc.abstractmethod
    def step(
        self, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]: ...

    @abc.abstractmethod
    def close(self) -> None: ...
