from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from .. import utils as U
from ..core import BNMeta, LearnParams

from .base import BaseInference


class DiscreteApproxInference(BaseInference):
    def __init__(self, meta: BNMeta, device=None, dtype=torch.float32, **kwargs):
        super().__init__(meta, device, dtype)
        self.meta = meta
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dtype = dtype

    @torch.no_grad()
    def posterior(
        self,
        lp: LearnParams,
        evidence: Dict[str, torch.Tensor],
        query: List[str],
        do: Optional[Dict[str, torch.Tensor]] = None,
        return_samples: bool = False,
        **kwargs,
    ):
        """
        Returns:
            out: dict mapping q -> probs with shape [B?, card_q]

            If return_samples=True:
              (out, samples_dict, weights)
              - samples_dict[q]: [B?, S] (long) particle assignments for each queried q
              - weights:         [B?, S] unnormalized importance weights
        """

        n_samples = kwargs.pop("n_samples", 512)

        do = do or {}
        B = None
        e_vals = {}
        for v, val in evidence.items():
            idx = U.to_long(val).to(self.device)
            if idx.ndim == 1:
                B = idx.shape[0]
            e_vals[v] = idx
        for k, v in do.items():
            if self.meta.types.get(k) == "discrete" and v.ndim == 1:
                B = v.shape[0]

        order = self.meta.order
        samples: Dict[str, torch.Tensor] = {}
        weights = torch.ones(
            (1 if B is None else B, n_samples), device=self.device, dtype=self.dtype
        )

        # helper: probs for a discrete node given sampled parents
        def _disc_probs(node: str) -> torch.Tensor:
            card = int(self.meta.cards[node])
            pa = self.meta.parents[node]
            # tabular path
            if lp.discrete_tables is not None and node in lp.discrete_tables:
                tbl = lp.discrete_tables[node]
                if not tbl.parent_names:
                    return tbl.probs[0].expand((1 if B is None else B), n_samples, card)
                strides = tbl.strides
                pa_idx = sum(
                    samples[p].to(torch.long) * strides[j]
                    for j, p in enumerate(tbl.parent_names)
                )
                return tbl.probs.index_select(0, pa_idx.view(-1)).reshape(
                    (1 if B is None else B), n_samples, card
                )
            # MLP path
            if lp.discrete_mlps is not None and node in lp.discrete_mlps:
                onehots = []
                in_dim = 0
                for p in pa:
                    k = int(self.meta.cards[p])
                    onehots.append(
                        F.one_hot(samples[p].to(torch.long), num_classes=k).to(
                            self.dtype
                        )
                    )
                    in_dim += k
                X = (
                    torch.cat(onehots, dim=-1)
                    if onehots
                    else torch.zeros(
                        (1 if B is None else B, n_samples, 0),
                        device=self.device,
                        dtype=self.dtype,
                    )
                )
                flat = X.reshape(-1, in_dim)
                logits = lp.discrete_mlps[node](flat)
                probs = torch.softmax(logits, dim=-1).reshape(
                    (1 if B is None else B), n_samples, card
                )
                return probs
            raise RuntimeError(f"No CPD for discrete node '{node}'.")

        # ancestral sampling
        for node in order:
            if self.meta.types[node] != "discrete":
                continue

            # do-intervention: clamp without weighting
            if node in do:
                val = U.to_long(do[node]).to(self.device)
                x = (
                    val.view((-1, 1))
                    if (B is not None or val.ndim == 1)
                    else val.view(1, 1)
                )
                samples[node] = x.expand(-1 if B is not None else 1, n_samples)
                continue

            if node in e_vals:
                ev = (
                    e_vals[node].view((-1, 1))
                    if B is not None
                    else e_vals[node].view(1, 1)
                )
                samples[node] = ev.expand(-1 if B is not None else 1, n_samples)
                probs = _disc_probs(node)
                onehot = F.one_hot(
                    samples[node], num_classes=int(self.meta.cards[node])
                )
                w_inc = (probs * onehot).sum(dim=-1).clamp_min(1e-32)
                weights = weights * w_inc
            else:
                probs = _disc_probs(node)
                u = torch.rand((1 if B is None else B), n_samples, device=self.device)
                cdf = probs.cumsum(dim=-1)
                x = (u.unsqueeze(-1) > cdf).sum(dim=-1)
                samples[node] = x

        # weighted marginals for queries
        out: Dict[str, torch.Tensor] = {}
        for q in query:
            card = int(self.meta.cards[q])
            xq = samples[q]
            if B is None:
                flat = xq.view(-1)
                w = weights.view(-1)
                hist = torch.zeros(
                    card, device=self.device, dtype=self.dtype
                ).scatter_add_(0, flat, w)
                out[q] = (hist / hist.sum().clamp_min(1e-32)).unsqueeze(0)
            else:
                hist = torch.zeros((B, card), device=self.device, dtype=self.dtype)
                for c in range(card):
                    hist[:, c] = (weights * (xq == c)).sum(dim=1)
                out[q] = hist / hist.sum(dim=1, keepdim=True).clamp_min(1e-32)

        if not return_samples:
            return out

        # Return particle assignments ONLY for the queried variables (to keep memory low)
        samples_dict: Dict[str, torch.Tensor] = {q: samples[q] for q in query}
        return out, samples_dict, weights
