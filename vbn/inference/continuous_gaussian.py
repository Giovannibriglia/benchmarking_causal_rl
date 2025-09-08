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
        Closed-form Linear-Gaussian posterior for a single continuous query node.
        Returns:
          if return_samples:
              (pdfs_dict, samples_dict, weights)
              - pdfs_dict[q]:   [B, S]  Normal pdf evaluated at returned samples
              - samples_dict[q]:[B, S]  samples from N(mean, std^2)
              - weights:        [B, S]  (all ones here)
          else:
              out_dict where out_dict[q] = {"mean": [B], "var": [B]}
        Notes:
          - Only continuous parents are used (discrete parents ignored here).
          - `evidence`/`do` values can be scalar, [B], or [B,1]; they are broadcast to B.
          - If `do` contains the query `q`, the posterior is a delta at that value.
        """
        import math

        device, dtype = self.device, self.dtype
        num_samples = int(kwargs.get("num_samples", 0) or 0)

        assert len(query) == 1, "LG posterior supports exactly one query node."
        q_name = query[0]
        assert (
            lp.lg is not None and q_name in lp.lg.name2idx
        ), f"Query '{q_name}' not in LG block."

        # ---------- helpers ----------
        def _to_1d(x: torch.Tensor) -> torch.Tensor:
            x = x.to(device=device, dtype=(dtype if x.is_floating_point() else None))
            if x.ndim == 0:  # scalar -> [1]
                return x.view(1)
            if x.ndim == 2 and x.shape[1] == 1:  # [B,1] -> [B]
                return x.view(-1)
            if x.ndim == 1:  # [B]
                return x
            raise ValueError(
                f"Expected tensor scalar / [B] / [B,1]; got shape {tuple(x.shape)}"
            )

        def _first_dim(x: torch.Tensor):
            return x.shape[0] if x.ndim >= 1 else None

        ev = {k: v.to(device=device) for k, v in (evidence or {}).items()}
        do = {} if do is None else {k: v.to(device=device) for k, v in do.items()}

        # Detect batch size B
        B_cands = []
        for v in list(ev.values()) + list(do.values()):
            b = _first_dim(v)
            if b is not None:
                B_cands.append(b)
        B = B_cands[0] if B_cands else 1
        if B_cands:
            assert all(b == B for b in B_cands), f"Inconsistent batch sizes: {B_cands}"

        # ---------- LG params ----------
        lg = lp.lg
        name2idx = lg.name2idx
        W = lg.W.to(device, dtype)  # [n, n]
        b = lg.b.to(device, dtype)  # [n]
        sigma2 = lg.sigma2.to(device, dtype)  # [n]

        q_idx = name2idx[q_name]
        cont_pa_all = [p for p in self.meta.parents.get(q_name, []) if p in name2idx]

        # If intervened on q: degenerate at value
        if q_name in do:
            xq = _to_1d(do[q_name])
            if xq.shape[0] == 1 and B > 1:
                xq = xq.expand(B)
            mean = xq
            var = torch.zeros_like(mean, dtype=dtype, device=device)

            if return_samples and num_samples > 0:
                samples = mean.view(B, 1).expand(B, num_samples)  # [B,S]
            else:
                samples = mean.view(B, 1)  # [B,1]
            # Dirac delta at xq → set pdfs to ones for convenience
            pdfs = torch.ones_like(samples, dtype=dtype, device=device)
            if return_samples:
                return {q_name: pdfs}, {q_name: samples}, torch.ones_like(samples)
            else:
                return {q_name: {"mean": mean, "var": var}}

        # Collect *continuous* parents’ values (do overrides evidence)
        parent_vals = []
        used_parents = []
        for p in cont_pa_all:
            if p in do:
                xp = _to_1d(do[p])
            elif p in ev:
                xp = _to_1d(ev[p])
            else:
                # You can either raise or treat as zero-mean. Here we raise to avoid silent bias.
                raise KeyError(
                    f"Missing continuous parent '{p}' for LG query '{q_name}'."
                )
            if xp.shape[0] == 1 and B > 1:
                xp = xp.expand(B)
            parent_vals.append(xp)  # [B]
            used_parents.append(p)

        # Mean and variance for q | parents
        if len(parent_vals) == 0:
            mean = b[q_idx].expand(B)  # [B]
        else:
            Xp = torch.stack(parent_vals, dim=-1)  # [B, d]
            w_qp = W[q_idx, [name2idx[p] for p in used_parents]]  # [d]
            mean = (Xp * w_qp).sum(dim=-1) + b[q_idx]  # [B]

        var = sigma2[q_idx].expand(B)  # [B]
        std = torch.sqrt(var.clamp_min(1e-12))  # [B]

        if not return_samples or num_samples <= 0:
            return {q_name: {"mean": mean, "var": var}}

        # Sample and compute pdf for each sample (analytic Normal)
        # shapes: mean/std [B] -> [B,1] for broadcasting
        mean_b1 = mean.view(B, 1)
        std_b1 = std.view(B, 1)
        eps = torch.randn(B, num_samples, device=device, dtype=dtype)  # [B,S]
        samples = mean_b1 + std_b1 * eps  # [B,S]

        z = (samples - mean_b1) / (std_b1 + 1e-12)
        # log N(x | mu, std) = -0.5*z^2 - log(std) - 0.5*log(2π)
        log_p = (
            -0.5 * (z * z) - torch.log(std_b1 + 1e-12) - 0.5 * math.log(2.0 * math.pi)
        )
        pdfs = torch.exp(log_p)  # [B,S]

        # weights are uniform here (no importance correction needed)
        weights = torch.ones_like(samples, dtype=dtype, device=device)

        return {q_name: pdfs}, {q_name: samples}, weights
