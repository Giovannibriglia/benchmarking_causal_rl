from __future__ import annotations

from typing import Optional

import torch
from torch import nn, Tensor

from ...learning.base import DifferentiableCPD

# ───────────────── helpers ─────────────────


def _softplus_pos(x: Tensor, eps: float = 1e-6) -> Tensor:
    return torch.nn.functional.softplus(x) + eps


def _gaussian_log_kernel(delta: Tensor, h: Tensor) -> Tensor:
    """
    Product Gaussian kernel log-density:
      log N(0, diag(h^2)) at delta
    delta: [B, N, D], h: [D]
    returns: [B, N]
    """
    inv_var = (1.0 / (h * h)).view(1, 1, -1)
    quad = (delta * delta * inv_var).sum(dim=-1)
    logZ = 0.5 * (
        delta.shape[-1]
        * torch.log(
            torch.tensor(2.0 * torch.pi, device=delta.device, dtype=delta.dtype)
        )
        + 2.0 * torch.log(h).sum()
    )
    return -0.5 * quad - logZ  # [B, N]


def _self_log_kernel_at_zero(D: int, h: Tensor) -> Tensor:
    # log N(0; 0, diag(h^2)) = -0.5 * ( D*log(2π) + 2*Σ log h )
    return -0.5 * (
        D * torch.log(torch.tensor(2.0 * torch.pi, device=h.device, dtype=h.dtype))
        + 2.0 * torch.log(h).sum()
    )


# ───────────────── model ─────────────────


class KDEGaussianLearnBW(DifferentiableCPD):
    """
    Differentiable Gaussian KDE CPD with learnable per-dimension bandwidths.
    Optimizes LOO conditional log-likelihood:
        ℓ = Σ_i log [ Σ_{j≠i} Kx(x_i - x_j) Ky(y_i - y_j) / Σ_{j≠i} Kx(x_i - x_j) ]
    Root nodes (in_dim=0): unconditional KDE on Y.
    """

    def __init__(
        self,
        name: str,
        in_dim: int,
        out_dim: int = 1,
        lr: float = 5e-2,
        epochs: int = 5,
        batch_size: int | None = 1024,
        chunk_size: int = 8192,
        weight_decay: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__(
            name,
            {
                "in_dim": in_dim,
                "out_dim": out_dim,
                "lr": lr,
                "epochs": epochs,
                "batch_size": batch_size,
                "chunk_size": chunk_size,
                "weight_decay": weight_decay,
                "eps": eps,
            },
        )
        self.in_dim = in_dim
        self.out_dim = out_dim
        # data buffers
        self.register_buffer("X_train", torch.empty(0, in_dim))
        self.register_buffer("Y_train", torch.empty(0, out_dim))
        # trainable log-bandwidths (initialized to log(0.5))
        if in_dim > 0:
            self.log_hx = nn.Parameter(torch.full((in_dim,), -0.6931))  # ~ log(0.5)
        else:
            self.log_hx = nn.Parameter(torch.zeros(0), requires_grad=False)
        self.log_hy = nn.Parameter(torch.full((out_dim,), -0.6931))
        self._optim: Optional[torch.optim.Optimizer] = None

    # ───── public API ─────

    def fit(self, X: Tensor, Y: Tensor) -> None:
        X, Y = self._ensure_2d(X), self._ensure_2d(Y)
        self.X_train = X.detach()
        self.Y_train = Y.detach()
        self._train_bandwidths(
            opt_epochs=self.cfg["epochs"], batch_size=self.cfg["batch_size"]
        )

    @torch.no_grad()
    def update(self, X: Tensor, Y: Tensor) -> None:
        X, Y = self._ensure_2d(X), self._ensure_2d(Y)
        if self.X_train.numel() == 0:
            self.X_train = X.detach()
            self.Y_train = Y.detach()
        else:
            self.X_train = torch.cat([self.X_train, X.detach()], dim=0)
            self.Y_train = torch.cat([self.Y_train, Y.detach()], dim=0)
        # Do a short bandwidth refinement on appended data
        with torch.enable_grad():
            self._train_bandwidths(
                opt_epochs=max(1, self.cfg["epochs"] // 2),
                batch_size=self.cfg["batch_size"],
            )

    def log_prob(self, X: Tensor, Y: Tensor) -> Tensor:
        X, Y = self._ensure_2d(X), self._ensure_2d(Y)
        assert self.Y_train.shape[0] > 0, "KDE not fitted"
        hx = _softplus_pos(self.log_hx, self.cfg["eps"]) if self.in_dim > 0 else None
        hy = _softplus_pos(self.log_hy, self.cfg["eps"])
        B = X.shape[0]
        N = self.Y_train.shape[0]
        chunk = self.cfg["chunk_size"]

        # weights in x
        if self.in_dim == 0:
            w_all = torch.full((B, N), 1.0 / max(1, N), device=X.device, dtype=X.dtype)
        else:
            w_chunks = []
            for s in range(0, N, chunk):
                Xi = self.X_train[s : s + chunk]  # [nC, Dx]
                delta_x = X[:, None, :] - Xi[None, :, :]  # [B, nC, Dx]
                logKx = _gaussian_log_kernel(delta_x, hx)  # [B, nC]
                w_chunks.append(logKx.exp())
            w_all = torch.cat(w_chunks, dim=1) + 1e-12  # [B, N]

        # numerator: sum_i w_i(x) * Ky(y - y_i); denominator: sum_i w_i(x)
        num_acc = []
        for s in range(0, N, chunk):
            Yi = self.Y_train[s : s + chunk]
            delta_y = Y[:, None, :] - Yi[None, :, :]
            logKy = _gaussian_log_kernel(delta_y, hy)  # [B, nC]
            num_acc.append(torch.log(w_all[:, s : s + chunk]) + logKy)
        log_num = torch.logsumexp(torch.cat(num_acc, dim=1), dim=1)  # [B]
        log_den = torch.log(w_all.sum(dim=1).clamp_min(1e-12))  # [B]
        return log_num - log_den

    @torch.no_grad()
    def sample(self, X: Tensor, n: int = 1) -> Tensor:
        X = self._ensure_2d(X)
        assert self.Y_train.shape[0] > 0, "KDE not fitted"
        hy = _softplus_pos(self.log_hy, self.cfg["eps"])
        B, N = X.shape[0], self.Y_train.shape[0]
        chunk = self.cfg["chunk_size"]

        # weights over training points
        if self.in_dim == 0:
            w = torch.full((B, N), 1.0 / max(1, N), device=X.device, dtype=X.dtype)
        else:
            hx = _softplus_pos(self.log_hx, self.cfg["eps"])
            w_chunks = []
            for s in range(0, N, chunk):
                Xi = self.X_train[s : s + chunk]
                delta_x = X[:, None, :] - Xi[None, :, :]
                logKx = _gaussian_log_kernel(delta_x, hx)
                w_chunks.append(logKx.exp())
            w = torch.cat(w_chunks, dim=1)
            w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-12)

        cat = torch.distributions.Categorical(probs=w)
        idx = cat.sample((n,))  # [n, B]
        yi = self.Y_train[idx]  # [n, B, Dy]
        eps = torch.randn_like(yi) * hy.view(1, 1, -1)
        return yi + eps

    # ───── internals ─────

    @staticmethod
    def _ensure_2d(T: Tensor) -> Tensor:
        if T.dim() == 1:
            return T.unsqueeze(0)
        return T

    def _train_bandwidths(self, opt_epochs: int, batch_size: Optional[int]) -> None:
        if self.Y_train.shape[0] == 0:
            return
        params = [p for p in [self.log_hx, self.log_hy] if p.requires_grad]
        if not params:
            return

        if self._optim is None:
            self._optim = torch.optim.Adam(
                params, lr=self.cfg["lr"], weight_decay=self.cfg["weight_decay"]
            )

        N = self.Y_train.shape[0]
        device = self.Y_train.device
        idx_all = torch.arange(N, device=device)

        for _ in range(opt_epochs):
            if batch_size is None or batch_size <= 0 or batch_size >= N:
                batches = [idx_all]
            else:
                perm = idx_all[torch.randperm(N, device=device)]
                batches = [perm[s : s + batch_size] for s in range(0, N, batch_size)]

            for ib in batches:
                xb = (
                    self.X_train[ib]
                    if self.in_dim > 0
                    else torch.empty(
                        (ib.shape[0], 0), device=device, dtype=self.Y_train.dtype
                    )
                )
                yb = self.Y_train[ib]
                # compute conditional LOO loglik for this batch
                loss = -self._loo_conditional_loglik(
                    xb, yb, ib
                )  # minimize negative loglik
                self._optim.zero_grad(set_to_none=True)
                loss.backward()
                self._optim.step()

    def _loo_conditional_loglik(self, Xb: Tensor, Yb: Tensor, idx_b: Tensor) -> Tensor:
        """
        Computes sum over i∈batch of log f_{Y|X}^{(-i)}(y_i|x_i)
        Using full dataset, subtracting the self-contribution exactly.
        """
        B = Xb.shape[0]
        N = self.Y_train.shape[0]
        device = Xb.device
        chunk = self.cfg["chunk_size"]
        eps = self.cfg["eps"]

        hx = _softplus_pos(self.log_hx, eps) if self.in_dim > 0 else None
        hy = _softplus_pos(self.log_hy, eps)

        # weights Kx over all j
        if self.in_dim == 0:
            w_all = torch.full(
                (B, N), 1.0 / max(1, N), device=device, dtype=self.Y_train.dtype
            )
        else:
            w_chunks = []
            for s in range(0, N, chunk):
                Xi = self.X_train[s : s + chunk]
                delta_x = Xb[:, None, :] - Xi[None, :, :]
                w_chunks.append(_gaussian_log_kernel(delta_x, hx).exp())
            w_all = torch.cat(w_chunks, dim=1) + 1e-12  # [B, N]

        # Ky terms
        log_num_terms = []
        for s in range(0, N, chunk):
            Yi = self.Y_train[s : s + chunk]
            delta_y = Yb[:, None, :] - Yi[None, :, :]
            logKy = _gaussian_log_kernel(delta_y, hy)
            log_num_terms.append(torch.log(w_all[:, s : s + chunk]) + logKy)
        log_num_all = torch.logsumexp(torch.cat(log_num_terms, dim=1), dim=1)  # [B]
        log_den_all = torch.log(w_all.sum(dim=1).clamp_min(1e-12))  # [B]

        # subtract self contributions (exact LOO)
        # find column indices where j==i (batch positions mapped to global indices)
        # For each b, its global index is idx_b[b]
        # Build a mask to gather self Kx and Ky at zero:
        """if self.in_dim == 0:
            logKx_self = torch.log(
                torch.full(
                    (B,), 1.0 / max(1, N), device=device, dtype=self.Y_train.dtype
                )
            )
        else:
            logKx_self = _self_log_kernel_at_zero(self.in_dim, hx).expand(B)"""

        logKy_self = _self_log_kernel_at_zero(self.out_dim, hy).expand(B)

        # w_all has positive weights; get self weight by indexing
        # Create a helper to gather w_all[b, idx_b[b]]
        w_self = w_all[torch.arange(B, device=device), idx_b]  # [B]
        # Numerator self term: w_self * Ky_self
        log_self_num = torch.log(w_self.clamp_min(1e-12)) + logKy_self  # [B]
        # Denominator self term: w_self
        log_self_den = torch.log(w_self.clamp_min(1e-12))  # [B]

        # LOO: log( num - self_num ) - log( den - self_den )
        # Do this in log-space with small stabilization
        num = (log_num_all.exp() - (log_self_num).exp()).clamp_min(1e-30)
        den = (log_den_all.exp() - (log_self_den).exp()).clamp_min(1e-30)
        loo = torch.log(num) - torch.log(den)  # [B]
        return -loo.mean() * (-1.0)  # return positive loss
