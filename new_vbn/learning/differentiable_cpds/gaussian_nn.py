from __future__ import annotations

import torch
from torch import nn, Tensor

from ...learning.base import DifferentiableCPD


class GaussianNNCPD(DifferentiableCPD):
    """
    y|x ~ N(mu(x), diag(sigma^2(x))) with an MLP head.
    """

    def __init__(
        self,
        name: str,
        in_dim: int,
        out_dim: int = 1,
        hidden: int = 64,
        lr: float = 1e-3,
        epochs: int = 10,
        batch_size: int | None = 1024,
        weight_decay: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__(
            name,
            {
                "in_dim": in_dim,
                "out_dim": out_dim,
                "hidden": hidden,
                "lr": lr,
                "epochs": epochs,
                "batch_size": batch_size,
                "weight_decay": weight_decay,
                "eps": eps,
            },
        )
        p = max(1, in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(p, hidden), nn.ReLU(), nn.Linear(hidden, 2 * out_dim)
        )

    @staticmethod
    def _features(X: Tensor) -> Tensor:
        if X.dim() == 2 and X.shape[1] == 0:
            return torch.ones((X.shape[0], 1), device=X.device, dtype=X.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        return X

    def _stats(self, X: Tensor):
        h = self.mlp(self._features(X))
        mu, logvar = torch.chunk(h, 2, dim=-1)
        logvar = logvar.clamp(-10, 10)
        return mu, logvar

    def _iter_batches(self, X: Tensor, y: Tensor, bs: int | None):
        N = X.shape[0]
        if bs is None or bs <= 0 or bs >= N:
            yield X, y
            return
        perm = torch.randperm(N, device=X.device)
        for s in range(0, N, bs):
            idx = perm[s : s + bs]
            yield X[idx], y[idx]

    def fit(self, X: Tensor, y: Tensor) -> None:
        Xf = self._features(X)
        Y = y
        opt = torch.optim.Adam(
            self.parameters(), lr=self.cfg["lr"], weight_decay=self.cfg["weight_decay"]
        )
        self.train()
        for _ in range(self.cfg["epochs"]):
            for Xb, Yb in self._iter_batches(Xf, Y, self.cfg["batch_size"]):
                opt.zero_grad(set_to_none=True)
                mu, logvar = self._stats(Xb)
                nll = 0.5 * (logvar + (Yb - mu).pow(2) * torch.exp(-logvar))
                loss = nll.sum(-1).mean()
                loss.backward()
                opt.step()

    @torch.no_grad()
    def update(self, X: Tensor, y: Tensor) -> None:
        # small refinement step
        with torch.enable_grad():
            self.fit(X, y)

    def log_prob(self, X: Tensor, y: Tensor) -> Tensor:
        mu, logvar = self._stats(X)
        return -(0.5 * (logvar + (y - mu).pow(2) * torch.exp(-logvar))).sum(-1)

    @torch.no_grad()
    def sample(self, X: Tensor, n: int = 1) -> Tensor:
        mu, logvar = self._stats(X)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(n, *mu.shape, device=mu.device, dtype=mu.dtype)
        return mu.unsqueeze(0) + eps * std.unsqueeze(0)
