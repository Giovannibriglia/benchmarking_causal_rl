from __future__ import annotations

import torch
from torch import nn, Tensor

from ...learning.base import DifferentiableCPD


class MDNCPD(DifferentiableCPD):
    """
    Mixture Density Network: y|x ~ Σ_k π_k(x) N(μ_k(x), diag(σ_k^2(x)))
    """

    def __init__(
        self,
        name: str,
        in_dim: int,
        out_dim: int = 1,
        n_components: int = 8,
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
                "K": n_components,
                "hidden": hidden,
                "lr": lr,
                "epochs": epochs,
                "batch_size": batch_size,
                "weight_decay": weight_decay,
                "eps": eps,
            },
        )
        p = max(1, in_dim)
        K = n_components
        D = out_dim
        H = hidden
        self.backbone = nn.Sequential(nn.Linear(p, H), nn.ReLU())
        self.pi_head = nn.Linear(H, K)
        self.m_head = nn.Linear(H, K * D)
        self.lv_head = nn.Linear(H, K * D)

    @staticmethod
    def _features(X: Tensor) -> Tensor:
        if X.dim() == 2 and X.shape[1] == 0:
            return torch.ones((X.shape[0], 1), device=X.device, dtype=X.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        return X

    def _params(self, X: Tensor):
        h = self.backbone(self._features(X))
        K, D = self.cfg["K"], self.cfg["out_dim"]
        pi = self.pi_head(h).log_softmax(dim=-1)  # [N, K] log-weights
        mu = self.m_head(h).view(-1, K, D)  # [N, K, D]
        lv = self.lv_head(h).view(-1, K, D).clamp(-10, 10)  # [N, K, D]
        return pi, mu, lv

    def _iter(self, X: Tensor, y: Tensor, bs: int | None):
        N = X.shape[0]
        if bs is None or bs <= 0 or bs >= N:
            yield X, y
            return
        perm = torch.randperm(N, device=X.device)
        for s in range(0, N, bs):
            idx = perm[s : s + bs]
            yield X[idx], y[idx]

    def fit(self, X: Tensor, y: Tensor) -> None:
        Xf, Y = self._features(X), y
        opt = torch.optim.Adam(
            self.parameters(), lr=self.cfg["lr"], weight_decay=self.cfg["weight_decay"]
        )
        self.train()
        for _ in range(self.cfg["epochs"]):
            for Xb, Yb in self._iter(Xf, Y, self.cfg["batch_size"]):
                opt.zero_grad(set_to_none=True)
                logpi, mu, lv = self._params(Xb)
                # log N(Y|mu, var): [N,K]
                logN = -0.5 * (lv + (Yb[:, None, :] - mu).pow(2) * torch.exp(-lv)).sum(
                    -1
                )
                # log-sum-exp over components
                loglik = torch.logsumexp(logpi + logN, dim=-1).mean()
                loss = -loglik
                loss.backward()
                opt.step()

    @torch.no_grad()
    def update(self, X: Tensor, y: Tensor) -> None:
        with torch.enable_grad():
            self.fit(X, y)

    def log_prob(self, X: Tensor, y: Tensor) -> Tensor:
        logpi, mu, lv = self._params(X)
        logN = -0.5 * (lv + (y[:, None, :] - mu).pow(2) * torch.exp(-lv)).sum(-1)
        return torch.logsumexp(logpi + logN, dim=-1)

    @torch.no_grad()
    def sample(self, X: Tensor, n: int = 1) -> Tensor:
        logpi, mu, lv = self._params(X)
        pi = logpi.exp()
        cat = torch.distributions.Categorical(pi)
        comp = cat.sample((n,))  # [n, N]
        std = torch.exp(0.5 * lv)  # [N, K, D]
        # gather component parameters
        mu_g = mu[None, torch.arange(mu.shape[0]), comp]  # [n, N, D]
        std_g = std[None, torch.arange(std.shape[0]), comp]  # [n, N, D]
        eps = torch.randn_like(mu_g)
        return mu_g + eps * std_g
