from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Simple feed-forward MLP with configurable hidden layers."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Iterable[int] = (64, 64),
        activation: nn.Module = nn.Tanh,
        output_activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        last = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last, h))
            layers.append(activation())
            last = h
        layers.append(nn.Linear(last, output_dim))
        if output_activation:
            layers.append(output_activation())
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.model(x)
