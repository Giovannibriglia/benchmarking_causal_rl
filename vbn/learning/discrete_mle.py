from __future__ import annotations

import math
from typing import Dict

import torch

from .. import utils as U
from ..core import BNMeta, DiscreteCPDTable, LearnParams


class DiscreteMLELearner(torch.nn.Module):
    """Maximum-likelihood tabular CPDs with Laplace smoothing."""

    def __init__(self, meta: BNMeta, device=None, dtype=torch.float32, **kwargs):
        super().__init__()
        self.meta = meta

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dtype = dtype

        self.laplace_alpha = kwargs.get("laplace_alpha", 1.0)

    @torch.no_grad()
    def fit(self, data: Dict[str, torch.Tensor]) -> LearnParams:
        tbls: Dict[str, DiscreteCPDTable] = {}

        for child in self.meta.order:
            if self.meta.types[child] != "discrete":
                continue

            pa = self.meta.parents[child]
            child_card = int(self.meta.cards[child])
            parent_cards = [int(self.meta.cards[p]) for p in pa]
            strides = (
                torch.tensor(
                    U.make_strides(parent_cards), device=self.device, dtype=torch.long
                )
                if pa
                else torch.tensor([], device=self.device, dtype=torch.long)
            )

            y = U.to_long(data[child]).to(self.device)
            # N = y.shape[0]

            if not pa:
                counts = U.bincount_fixed(y, minlength=child_card)
                counts = counts + self.laplace_alpha
                probs = (counts / counts.sum()).unsqueeze(0)  # [1, C]
                probs = probs.to(device=self.device, dtype=self.dtype)
                tbls[child] = DiscreteCPDTable(
                    probs=probs,
                    parent_names=[],
                    parent_cards=[],
                    child_card=child_card,
                    strides=strides,
                )
                continue

            Xp = torch.stack(
                [U.to_long(data[p]).to(self.device) for p in pa], dim=1
            )  # [N, n_par]
            pa_idx = (Xp * strides).sum(dim=1)  # [N]
            P = int(math.prod(parent_cards))

            flat_idx = pa_idx * child_card + y
            K = P * child_card
            flat_counts = U.bincount_fixed(flat_idx, minlength=K)  # [K]
            counts = flat_counts.reshape(P, child_card) + self.laplace_alpha
            probs = counts / counts.sum(dim=1, keepdim=True)
            probs = probs.to(device=self.device, dtype=self.dtype)

            tbls[child] = DiscreteCPDTable(
                probs=probs,
                parent_names=pa,
                parent_cards=parent_cards,
                child_card=child_card,
                strides=strides,
            )

        return LearnParams(meta=self.meta, discrete_tables=tbls)
