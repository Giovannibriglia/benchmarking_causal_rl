from __future__ import annotations

from typing import Dict, List, Optional

import torch

from ..core import BNMeta, LearnParams

from .base import BaseInference


class ContinuousLGInference(BaseInference):
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
        Closed-form Linear-Gaussian posterior (single continuous query).
        Returns:
            pdfs:    [B, S] (or [B, 1] if no sampling)
            samples: [B, S] (or [B, 1] if no sampling)
            extras:  {"mean": [B], "std": scalar or [B], "parents": List[str]}
        Notes:
            - Discrete parents are ignored here (LGParams covers only continuous block).
            - Evidence and do can be [B], [B,1], [1], or scalar.
        """
        import math

        device, dtype = self.device, self.dtype
        num_samples = int(kwargs.get("num_samples", 0) or 0)

        assert len(query) == 1, "LG posterior supports a single query node."
        q_name = query[0]
        assert (
            lp.lg is not None and q_name in lp.lg.name2idx
        ), f"Query '{q_name}' not in LG block."

        # --- helpers -------------------------------------------------------------
        def _to_1d(x: torch.Tensor) -> torch.Tensor:
            x = x.to(device=device, dtype=dtype if x.is_floating_point() else None)
            if x.ndim == 0:  # scalar -> [1]
                return x.view(1)
            if x.ndim == 2 and x.shape[1] == 1:  # [B,1] -> [B]
                return x.view(-1)
            if x.ndim == 1:  # [B]
                return x
            raise ValueError(
                f"Expected evidence shape [B], [B,1], or scalar; got {tuple(x.shape)}"
            )

        def _first_dim(x: torch.Tensor) -> Optional[int]:
            return x.shape[0] if x.ndim >= 1 else None

        ev = {k: v.to(device=device) for k, v in evidence.items()}
        do = {} if do is None else {k: v.to(device=device) for k, v in do.items()}

        # Determine batch size B
        B_candidates: List[int] = []
        for v in list(ev.values()) + list(do.values()):
            b = _first_dim(v)
            if b is not None:
                B_candidates.append(b)
        B = B_candidates[0] if B_candidates else 1
        if B_candidates:
            assert all(
                b == B for b in B_candidates
            ), f"Inconsistent batch sizes: {B_candidates}"

        # --- LG params -----------------------------------------------------------
        lg = lp.lg
        name2idx = lg.name2idx
        W = lg.W.to(device, dtype)  # [n, n]
        b = lg.b.to(device, dtype)  # [n]
        sigma2 = lg.sigma2.to(device, dtype)  # [n]

        q_idx = name2idx[q_name]
        cont_pa_all = [p for p in self.meta.parents.get(q_name, []) if p in name2idx]

        # Build parent values (do overrides evidence)
        parent_vals: List[torch.Tensor] = []
        used_parents: List[str] = []
        for p in cont_pa_all:
            if p in do:
                xp = _to_1d(do[p])
            elif p in ev:
                xp = _to_1d(ev[p])
            else:
                raise KeyError(
                    f"Missing continuous parent '{p}' in evidence/do for '{q_name}'."
                )
            if xp.shape[0] == 1 and B > 1:
                xp = xp.expand(B)
            parent_vals.append(xp)  # [B]
            used_parents.append(p)

        # Mean: b_q + sum_j w_{q<-p_j} x_j
        if len(parent_vals) == 0:
            mean = b[q_idx].expand(B)  # [B]
        else:
            Xp = torch.stack(parent_vals, dim=-1)  # [B, d]
            w_qp = W[q_idx, [name2idx[p] for p in used_parents]]  # [d]
            mean = (Xp * w_qp).sum(dim=-1) + b[q_idx]  # [B]

        # Std
        std = torch.sqrt(torch.clamp(sigma2[q_idx], min=1e-12))  # scalar

        # Prepare for sampling/scoring
        mean_b1 = mean.view(B, 1)  # [B, 1]
        std_b1 = std if std.ndim == 0 else std.view(B, 1)  # scalar or [B,1]

        # Samples
        if return_samples and num_samples > 0:
            eps = torch.randn(B, num_samples, device=device, dtype=dtype)  # [B, S]
            samples = mean_b1 + std_b1 * eps  # [B, S]
        else:
            samples = mean_b1  # [B, 1]

        # Analytic Normal pdf to avoid distribution broadcast guards
        # log N(x | mu, std) = -0.5*((x-mu)/std)^2 - log(std) - 0.5*log(2π)
        z = (samples - mean_b1) / (std_b1 + 1e-12)
        log_p = (
            -0.5 * (z * z) - torch.log(std_b1 + 1e-12) - 0.5 * math.log(2.0 * math.pi)
        )
        pdfs = torch.exp(log_p)  # [B, S] or [B, 1]

        extras = {"mean": mean, "std": std, "parents": used_parents}
        return pdfs, samples, extras
