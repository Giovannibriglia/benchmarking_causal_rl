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

    @property
    def is_oracle_u(self) -> bool:
        """True iff this learner carries the OracleU strategy (reads the realized
        U). Drives the runner's value-trace u0-anchor gate. Robust to learners
        that take no strategy (SAC/DDPG): they report False."""
        strat = getattr(self, "_strategy", None)
        return bool(getattr(strat, "requires_confounder_u", False))

    def oracle_anchor_q(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """``Q(s, a_data, u=0)`` for the value-trace u=0 anchor — delegated to the
        OracleU strategy (only meaningful, and only invoked, when ``is_oracle_u``)."""
        return self._strategy.oracle_anchor_q(self.q_network, batch)

    @abc.abstractmethod
    def learn(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]: ...
