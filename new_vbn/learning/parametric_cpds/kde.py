from __future__ import annotations

import torch
from torch import Tensor

from ...learning.base import ParametricCPD


def _rule_of_thumb_bandwidth(x: Tensor, rule: str = "silverman") -> Tensor:
    """
    Per-dimension bandwidth for Gaussian kernels.
    x: [N, D]
    """
    N, D = x.shape
    std = x.std(dim=0, unbiased=True).clamp_min(1e-8)
    if rule.lower() in ("silverman", "scott"):
        # Scott h = N^{-1/(d+4)}; Silverman adds a constant factor ~ 1.06
        gamma = 1.06 if rule.lower() == "silverman" else 1.0
        h = gamma * std * (N ** (-1.0 / (D + 4.0)))
    else:
        h = std * (N ** (-1.0 / (D + 4.0)))
    return h.clamp_min(1e-6)


def _gaussian_log_kernel(delta: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """
    delta: [B, N, D]  (x_b - x_i) or (y_b - y_i)
    h:     [D]        per-dimension bandwidths
    returns: [B, N]
    """
    D = delta.shape[-1]
    inv_var = (1.0 / (h * h)).view(1, 1, -1)  # [1,1,D]
    quad = (delta * delta * inv_var).sum(dim=-1)  # [B,N]
    # make 2π a Tensor on the right device/dtype
    log2pi = torch.log(
        torch.tensor(2.0 * torch.pi, device=delta.device, dtype=delta.dtype)
    )
    logZ = 0.5 * (D * log2pi + 2.0 * torch.log(h).sum())
    return -0.5 * quad - logZ


class KDEGaussianCPD(ParametricCPD):
    """
    Gaussian KDE CPD with product kernels and conditional density:

        f_{Y|X}(y|x) = [ Σ_i K_X(x - x_i) K_Y(y - y_i) ] / [ Σ_i K_X(x - x_i) ]

    Root nodes: X has dim 0 -> unconditional KDE on Y.

    Capabilities:
      has_log_prob=True, has_sample=True, is_reparameterized=False
    """

    def __init__(
        self,
        name: str,
        in_dim: int,  # dim(X) == |Pa(node)|
        out_dim: int = 1,  # dim(Y)
        rule: str = "silverman",
        chunk_size: int = 8192,
        recompute_bw_on_update: bool = False,
    ):
        super().__init__(
            name,
            {
                "in_dim": in_dim,
                "out_dim": out_dim,
                "rule": rule,
                "chunk_size": chunk_size,
                "recompute_bw_on_update": recompute_bw_on_update,
            },
        )
        self.register_buffer("X_train", torch.empty(0, in_dim))
        self.register_buffer("Y_train", torch.empty(0, out_dim))
        self.register_buffer("h_x", torch.ones(in_dim) if in_dim > 0 else torch.ones(1))
        self.register_buffer("h_y", torch.ones(out_dim))
        self.in_dim = in_dim
        self.out_dim = out_dim

    # ---------- data management ----------
    def fit(self, X: Tensor, Y: Tensor) -> None:
        X = X.detach()
        Y = Y.detach()
        if X.dim() == 1:
            X = X.unsqueeze(0)
        if Y.dim() == 1:
            Y = Y.unsqueeze(0)
        assert X.shape[0] == Y.shape[0], "Mismatched X/Y rows"

        self.X_train = X
        self.Y_train = Y
        if self.in_dim > 0:
            self.h_x = _rule_of_thumb_bandwidth(self.X_train, self.cfg["rule"]).to(
                self.X_train
            )
        else:
            self.h_x = torch.ones(1, device=Y.device, dtype=Y.dtype)
        self.h_y = _rule_of_thumb_bandwidth(self.Y_train, self.cfg["rule"]).to(
            self.Y_train
        )

    @torch.no_grad()
    def update(self, X: Tensor, Y: Tensor) -> None:
        X = X.detach()
        Y = Y.detach()
        if X.dim() == 1:
            X = X.unsqueeze(0)
        if Y.dim() == 1:
            Y = Y.unsqueeze(0)
        assert X.shape[0] == Y.shape[0], "Mismatched X/Y rows"

        if self.X_train.numel() == 0:
            self.X_train = X
            self.Y_train = Y
        else:
            self.X_train = torch.cat([self.X_train, X], dim=0)
            self.Y_train = torch.cat([self.Y_train, Y], dim=0)

        if self.cfg["recompute_bw_on_update"]:
            if self.in_dim > 0:
                self.h_x = _rule_of_thumb_bandwidth(self.X_train, self.cfg["rule"]).to(
                    self.X_train
                )
            self.h_y = _rule_of_thumb_bandwidth(self.Y_train, self.cfg["rule"]).to(
                self.Y_train
            )

    # ---------- densities ----------
    def _weights_x(self, X: Tensor) -> Tensor:
        """
        Return w[b,i] ∝ K_X(x_b - x_i).
        X: [B, Dx]; self.X_train: [N, Dx]
        """
        if self.in_dim == 0:
            # no parents -> uniform weights over training samples
            N = self.Y_train.shape[0]
            return torch.full(
                (X.shape[0], N), 1.0 / max(1, N), device=X.device, dtype=X.dtype
            )

        # Chunked pairwise kernel to avoid OOM
        B, Dx = X.shape
        N = self.X_train.shape[0]
        chunk = self.cfg["chunk_size"]
        out = []
        for s in range(0, N, chunk):
            Xi = self.X_train[s : s + chunk]  # [nC, Dx]
            delta = X[:, None, :] - Xi[None, :, :]  # [B, nC, Dx]
            logK = _gaussian_log_kernel(delta, self.h_x)  # [B, nC]
            out.append(logK)
        logK_all = torch.cat(out, dim=1)  # [B, N]
        # convert to positive weights (not normalized)
        w = logK_all.exp()  # [B, N]
        # avoid all-zero (degenerate h_x): add tiny epsilon
        return w + 1e-12

    def log_prob(self, X: Tensor, Y: Tensor) -> Tensor:
        """
        log f_{Y|X}(y|x) = log Σ_i w_i(x) K_Y(y - y_i)  -  log Σ_i w_i(x)
        with Gaussian Ky normalizer included in K_Y.
        X: [B, Dx], Y: [B, Dy]
        returns: [B]
        """
        if X.dim() == 1:
            X = X.unsqueeze(0)
        if Y.dim() == 1:
            Y = Y.unsqueeze(0)
        # B = X.shape[0]
        N = self.Y_train.shape[0]
        assert N > 0, "KDE not fitted"

        w = self._weights_x(X)  # [B, N]
        # chunk over N to compute Ky and numerator safely
        chunk = self.cfg["chunk_size"]
        num_log_terms = []
        for s in range(0, N, chunk):
            Yi = self.Y_train[s : s + chunk]  # [nC, Dy]
            delta_y = Y[:, None, :] - Yi[None, :, :]  # [B, nC, Dy]
            logKy = _gaussian_log_kernel(delta_y, self.h_y)  # [B, nC]
            # add log(w) + logKy in stable space
            logw = torch.log(w[:, s : s + chunk])  # [B, nC]
            num_log_terms.append(logw + logKy)  # [B, nC]
        num_log = torch.logsumexp(torch.cat(num_log_terms, dim=1), dim=1)  # [B]

        den = w.sum(dim=1).clamp_min(1e-12)  # [B]
        log_den = torch.log(den)
        return num_log - log_den  # [B]

    @torch.no_grad()
    def sample(self, X: Tensor, n: int = 1) -> Tensor:
        """
        Sample y ~ f_{Y|X}(·|x) by:
          1) pick i with prob ∝ K_X(x - x_i)
          2) sample y = y_i + ε, ε ~ N(0, diag(h_y^2))
        Returns: [n, B, Dy]
        """
        if X.dim() == 1:
            X = X.unsqueeze(0)
        # B = X.shape[0]
        N = self.Y_train.shape[0]
        assert N > 0, "KDE not fitted"

        w = self._weights_x(X)  # [B, N]
        w = w / w.sum(dim=1, keepdim=True)  # normalize
        cat = torch.distributions.Categorical(probs=w)  # per-batch categorical
        idx = cat.sample((n,))  # [n, B]
        # gather y_i
        yi = self.Y_train[idx]  # [n, B, Dy]
        eps = torch.randn_like(yi) * self.h_y.view(1, 1, -1)
        return yi + eps
