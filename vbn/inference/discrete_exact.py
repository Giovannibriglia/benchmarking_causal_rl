from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from .. import utils as U
from ..core import BNMeta, LearnParams

from .base import BaseInference


class DiscreteExactVEInference(BaseInference):
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

        num_samples = kwargs.get("num_samples", 512)

        assert lp.discrete_tables is not None, "Need tabular CPDs."
        do = do or {}
        # Build factors
        factors: Dict[str, torch.Tensor] = {}
        var_axes: Dict[str, List[str]] = {}

        # Determine batch size B from batched evidence or batched interventions
        B = None
        for v in evidence.values():
            if v.ndim == 1:
                B = v.shape[0]
        for k, v in do.items():
            if self.meta.types.get(k) == "discrete" and v.ndim == 1:
                B = v.shape[0]

        for child in self.meta.order:
            if self.meta.types[child] != "discrete":
                continue
            tbl = lp.discrete_tables[child]
            C = int(self.meta.cards[child])

            if child in do:
                # do(X=c): replace P(X|pa) with delta(X=c), independent of parents
                val = U.to_long(do[child]).to(self.device)
                if val.ndim == 0:
                    t = F.one_hot(val, num_classes=C).to(self.dtype)  # [C]
                else:
                    # batched delta: [B, C]
                    t = F.one_hot(val, num_classes=C).to(self.dtype)
                factors[child] = t  # [C] or [B, C]
                var_axes[child] = [child]
                continue

            if not tbl.parent_names:
                t = tbl.probs[0]  # [C]
                axes = [child]
            else:
                shape = tbl.parent_cards + [tbl.child_card]  # [..., C]
                t = tbl.probs.reshape(shape)
                axes = tbl.parent_names + [child]
            factors[child] = t.to(device=self.device, dtype=self.dtype)
            var_axes[child] = axes

        # Condition on evidence (skip any var also in do)
        for v, val in evidence.items():
            if v in do:  # intervention overrides evidence
                continue
            idx = U.to_long(val).to(self.device)  # [] or [B]
            for name, tens in list(factors.items()):
                axes = var_axes[name]
                if v not in axes:
                    continue
                ax = axes.index(v)
                if B is not None:
                    t = tens
                    if t.ndim == len(axes):  # no batch
                        t = t.unsqueeze(0).expand(B, *t.shape)
                    t = t.movedim(1 + ax, -1)
                    t = t.gather(
                        -1,
                        idx.view(B, *([1] * (t.ndim - 2)), 1).expand_as(t[..., :1]),
                    ).squeeze(-1)
                    factors[name] = t
                else:
                    factors[name] = tens.index_select(ax, idx.view(()))
                va = axes.copy()
                va.pop(ax)
                var_axes[name] = va

        # Eliminate non-query, non-evidence variables
        qset, eset = set(query), set([k for k in evidence.keys() if k not in do])
        elim = [
            v
            for v in self.meta.order
            if self.meta.types[v] == "discrete" and v not in qset and v not in eset
        ]
        for z in elim:
            to_mul = [k for k, ax in var_axes.items() if z in ax]
            if not to_mul:
                continue
            t, axes = U.product_factors(
                [(factors[k], var_axes[k]) for k in to_mul],
                batch_dim=(0 if B else None),
            )
            ax = axes.index(z)
            t = t.sum(dim=(ax + (1 if B else 0)))
            axes.pop(ax)
            for k in to_mul:
                del factors[k]
                del var_axes[k]
            factors[f"φ_{z}"] = t
            var_axes[f"φ_{z}"] = axes

        # Fuse remaining factors
        t, axes = U.product_factors(
            [(factors[k], var_axes[k]) for k in list(factors.keys())],
            batch_dim=(0 if B else None),
        )

        # Compute each query marginal directly (sum out everything else), no reshape.
        out: Dict[str, torch.Tensor] = {}
        if not query:
            return out if not return_samples else (out, {})

        base = 1 if B else 0
        axes_set = set(axes)

        for q in query:
            if q not in axes_set:
                # If a query got fully summed out, fall back to uniform over its card
                card_q = int(self.meta.cards[q])
                if B is None:
                    out[q] = torch.full(
                        (card_q,), 1.0 / card_q, device=self.device, dtype=self.dtype
                    )
                else:
                    out[q] = torch.full(
                        (B, card_q), 1.0 / card_q, device=self.device, dtype=self.dtype
                    )
                continue

            # Sum over all NON-q variable axes
            sum_dims = [base + i for i, a in enumerate(axes) if a != q]
            marg = t if not sum_dims else t.sum(dim=tuple(sum_dims))
            # Normalize along the last dim
            denom = marg.sum(dim=-1, keepdim=True).clamp_min(1e-32)
            marg = marg / denom  # [B?, card_q]
            out[q] = marg

        if not return_samples:
            return out

        # ---- sampling from marginals (independent across queries) ----
        if num_samples <= 0:
            num_samples = 1
        samples: Dict[str, torch.Tensor] = {}
        for q, probs in out.items():
            if probs.ndim == 1:  # [C]
                samp = torch.multinomial(
                    probs, num_samples=num_samples, replacement=True
                )  # [n_samples]
            else:  # [B, C]
                # torch.multinomial vectorized over rows -> [B, n_samples]
                samp = torch.multinomial(
                    probs, num_samples=num_samples, replacement=True
                )
            samples[q] = samp.to(self.device)

        return out, samples
