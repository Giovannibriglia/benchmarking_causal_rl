from __future__ import annotations

from typing import Dict

import torch

from .. import utils as U
from ..core import BNMeta, LearnParams, LGParams


class GaussianLinearLearner(torch.nn.Module):
    """Per-node linear regression y = b + W_par^T x_par + eps, eps~N(0, sigma2)."""

    def __init__(self, meta: BNMeta, device=None, dtype=torch.float32, **kwargs):
        super().__init__()
        self.meta = meta

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dtype = dtype
        self.ridge = kwargs.get("ridge", 1e-6)

    @torch.no_grad()
    def fit(self, data: Dict[str, torch.Tensor]) -> LearnParams:
        order_all = self.meta.order
        cont_order = [n for n in order_all if self.meta.types[n] == "continuous"]
        name2idx = {n: i for i, n in enumerate(cont_order)}
        nc = len(cont_order)

        if nc == 0:
            return LearnParams(meta=self.meta, lg=None)

        Xall = U.stack_float(
            [data[n] for n in cont_order], self.device, self.dtype
        )  # [N, nc]
        N = Xall.shape[0]

        W = torch.zeros((nc, nc), device=self.device, dtype=self.dtype)
        b = torch.zeros(nc, device=self.device, dtype=self.dtype)
        sigma2 = torch.zeros(nc, device=self.device, dtype=self.dtype)

        # Identity = None
        for child in cont_order:
            i = name2idx[child]
            pa = [
                p
                for p in self.meta.parents[child]
                if self.meta.types[p] == "continuous"
            ]
            y = Xall[:, i]
            if not pa:
                b[i] = y.mean()
                sigma2[i] = torch.clamp(y.var(unbiased=False), min=1e-12)
                continue

            P_idx = torch.tensor([name2idx[p] for p in pa], device=self.device)
            Xp = Xall[:, P_idx]
            # Build design with intercept
            X = torch.cat(
                [torch.ones(N, 1, device=self.device, dtype=self.dtype), Xp], dim=1
            )  # [N, d+1]
            d_plus_1 = X.shape[1]

            # Promote to float64 for numerical stability (optional but recommended)
            X64 = X.to(torch.float64)
            y64 = y.to(torch.float64)

            # Tikhonov augmentation: do NOT penalize the intercept
            # Create sqrt(lambda) * I with (0,0) = 0 to avoid penalizing the bias
            lam = float(self.ridge)
            if lam <= 0:
                lam = 1e-8  # ensure some stabilization
            sqrtlam = lam**0.5

            Identity = torch.eye(d_plus_1, device=self.device, dtype=torch.float64)
            Identity[0, 0] = 0.0  # no penalty on intercept
            Xa = torch.cat([X64, sqrtlam * Identity], dim=0)  # [(N + d+1), d+1]
            ya = torch.cat(
                [y64, torch.zeros(d_plus_1, device=self.device, dtype=torch.float64)],
                dim=0,
            )

            # Stable least-squares (SVD-based), handles rank-deficiency gracefully
            theta64 = torch.linalg.lstsq(Xa, ya, rcond=None).solution  # [d+1]

            theta = theta64.to(self.dtype)

            b[i] = theta[0]
            W[i, P_idx] = theta[1:]

            res = y - (X @ theta.to(self.dtype))
            sigma2[i] = torch.clamp((res * res).mean(), min=1e-12)

        lg = LGParams(order=cont_order, name2idx=name2idx, W=W, b=b, sigma2=sigma2)
        return LearnParams(meta=self.meta, lg=lg)
