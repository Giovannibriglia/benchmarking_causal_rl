from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol

import torch
import torch.nn as nn


@dataclass(frozen=True)
class Capabilities:
    has_sample: bool = True
    has_log_prob: bool = True
    is_reparameterized: bool = False
    supports_conditioning: bool = True
    supports_do: bool = True


class BaseNodeCPD(Protocol):
    name: str
    capabilities: Capabilities

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        raise not ImplicitGenerator

    def update(self, X: torch.Tensor, y: torch.Tensor) -> None:
        raise not ImplicitGenerator

    def log_prob(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise not ImplicitGenerator

    @torch.no_grad()
    def sample(self, X: torch.Tensor, n: int = 1) -> torch.Tensor:
        raise not ImplicitGenerator


class ParametricCPD(nn.Module):
    def __init__(self, name: str, cfg: Dict[str, Any]):
        super().__init__()
        self.name = name
        self.cfg = cfg
        self.capabilities = Capabilities(
            has_sample=True,
            has_log_prob=True,
            is_reparameterized=False,
            supports_conditioning=True,
            supports_do=True,
        )


class DifferentiableCPD(nn.Module):
    def __init__(self, name: str, cfg: Dict[str, Any]):
        super().__init__()
        self.name = name
        self.cfg = cfg
        self.capabilities = Capabilities(
            has_sample=True,
            has_log_prob=True,
            is_reparameterized=True,
            supports_conditioning=True,
            supports_do=True,
        )


class ImplicitGenerator(nn.Module):
    def __init__(self, name: str, cfg: Dict[str, Any]):
        super().__init__()
        self.name = name
        self.cfg = cfg
        self.capabilities = Capabilities(
            has_sample=True,
            has_log_prob=False,
            is_reparameterized=True,
            supports_conditioning=True,
            supports_do=True,
        )
