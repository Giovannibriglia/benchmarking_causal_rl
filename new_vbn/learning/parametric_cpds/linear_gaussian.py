from __future__ import annotations

import torch

from ..base import ParametricCPD


# ---------- Linear Gaussian: y|X ~ N(Xβ, σ² I) with sufficient-stats online ----------
class ParametricLinearGaussianCPD(ParametricCPD):
    """
    OLS MLE with intercept:
      β̂ = (X'X)^(-1)X'Y,  σ̂² = RSS / (N - p),  p = features incl. bias
    Maintains sufficient stats for online update:
      S_xx = X'X, S_xy = X'Y, S_yy = sum(Y⊙Y), n = N
    """

    def __init__(self, name: str, in_dim: int, out_dim: int = 1, eps: float = 1e-8):
        super().__init__(name, {"in_dim": in_dim, "out_dim": out_dim, "eps": eps})
        p = in_dim + 1  # +1 bias
        d = out_dim

        # parameters
        self.register_buffer("beta", torch.zeros(p, d))  # [p, d]
        self.register_buffer("sigma2", torch.ones(d))  # [d]

        # sufficient stats
        self.register_buffer("S_xx", torch.zeros(p, p))
        self.register_buffer("S_xy", torch.zeros(p, d))
        self.register_buffer("S_yy", torch.zeros(d))
        self.register_buffer("n", torch.zeros(1))

    @staticmethod
    def _with_bias(X: torch.Tensor) -> torch.Tensor:
        # Handle “no parents” cleanly: X is [N, 0]
        if X.dim() == 2 and X.shape[1] == 0:
            return torch.ones((X.shape[0], 1), device=X.device, dtype=X.dtype)
        # Handle 1D -> make it [N, 1] before adding bias
        if X.dim() == 1:
            X = X.unsqueeze(0)
        ones = torch.ones((X.shape[0], 1), device=X.device, dtype=X.dtype)
        return torch.cat([ones, X], dim=1)  # [N, p]

    def _solve_from_stats(self):
        eps = self.cfg["eps"]
        reg = eps * torch.eye(
            self.S_xx.shape[0], device=self.S_xx.device, dtype=self.S_xx.dtype
        )
        XtX_inv = torch.linalg.pinv(self.S_xx + reg)
        self.beta = XtX_inv @ self.S_xy
        if self.n.item() > self.S_xx.shape[0]:
            rss_vec = self.S_yy - (self.beta * self.S_xy).sum(dim=0)
            dof = max(1.0, self.n.item() - self.S_xx.shape[0])
            self.sigma2 = torch.clamp(rss_vec / dof, min=eps)
        else:
            self.sigma2 = torch.clamp(self.sigma2, min=eps)

    # full refit
    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        Xb = self._with_bias(X)  # [N, p]
        Y = y  # [N, d]
        self.S_xx = Xb.T @ Xb
        self.S_xy = Xb.T @ Y
        self.S_yy = (Y * Y).sum(dim=0)
        self.n[...] = Y.shape[0]
        self._solve_from_stats()

    # online add
    @torch.no_grad()
    def update(self, X: torch.Tensor, y: torch.Tensor) -> None:
        Xb = self._with_bias(X)
        Y = y
        self.S_xx += Xb.T @ Xb
        self.S_xy += Xb.T @ Y
        self.S_yy += (Y * Y).sum(dim=0)
        self.n += Y.shape[0]
        self._solve_from_stats()

    def log_prob(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        Xb = self._with_bias(X)
        mean = Xb @ self.beta
        var = torch.clamp(self.sigma2, min=self.cfg["eps"])
        return -(0.5 * ((y - mean) ** 2 / var + torch.log(var))).sum(dim=-1)

    @torch.no_grad()
    def sample(self, X: torch.Tensor, n: int = 1) -> torch.Tensor:
        Xb = self._with_bias(X)
        mean = Xb @ self.beta
        std = torch.sqrt(torch.clamp(self.sigma2, min=self.cfg["eps"]))
        eps = torch.randn(n, *mean.shape, device=mean.device, dtype=mean.dtype)
        return mean.unsqueeze(0) + eps * std.unsqueeze(0)
