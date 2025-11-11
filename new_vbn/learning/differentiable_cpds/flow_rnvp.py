from __future__ import annotations

import torch
from torch import nn, Tensor

from ...learning.base import DifferentiableCPD


# ───────────── helper: bias-only features for roots ─────────────
def _feat(X: Tensor) -> Tensor:
    if X.dim() == 2 and X.shape[1] == 0:
        return torch.ones((X.shape[0], 1), device=X.device, dtype=X.dtype)
    if X.dim() == 1:
        X = X.unsqueeze(0)
    return X


# ───────────── RealNVP affine coupling, conditioned on x (FiLM) ─────────────
class _CondAffineCoupling(nn.Module):
    """
    RealNVP-style coupling layer for y in R^D.
    Split y into y_a (pass-through) and y_b (transformed), alternating mask across layers.
    s,t nets are conditioned on concat([y_a, x]) via small MLP.
    """

    def __init__(
        self, y_dim: int, x_dim: int, hidden: int = 128, mask_even: bool = True
    ):
        super().__init__()
        self.y_dim = y_dim
        self.x_dim = max(1, x_dim)
        self.mask_even = mask_even
        # simple half/half split
        self.split = y_dim // 2
        self.split = max(1, self.split)  # guard D=1
        in_s = self.split + self.x_dim
        out_s = y_dim - self.split
        H = hidden
        self.s_net = nn.Sequential(nn.Linear(in_s, H), nn.ReLU(), nn.Linear(H, out_s))
        self.t_net = nn.Sequential(nn.Linear(in_s, H), nn.ReLU(), nn.Linear(H, out_s))

    def forward(
        self, y: Tensor, x: Tensor, reverse: bool = False
    ) -> tuple[Tensor, Tensor]:
        B, D = y.shape
        xa = _feat(x)
        if self.mask_even:
            ya, yb = y[:, : self.split], y[:, self.split :]
        else:
            ya, yb = y[:, self.split :], y[:, : self.split]

        h = torch.cat([ya, xa], dim=-1)
        s = self.s_net(h).tanh()  # bound scale for stability
        t = self.t_net(h)

        if not reverse:
            z_b = (yb * torch.exp(s)) + t
            logdet = s.sum(dim=-1)
            if self.mask_even:
                z = torch.cat([ya, z_b], dim=-1)
            else:
                z = torch.cat([z_b, ya], dim=-1)
        else:
            yb = (yb - t) * torch.exp(-s)
            logdet = (-s).sum(dim=-1)
            if self.mask_even:
                z = torch.cat([ya, yb], dim=-1)
            else:
                z = torch.cat([yb, ya], dim=-1)
        return z, logdet


# --- add this helper near the top (alongside _feat and _CondAffineCoupling) ---


class _CondAffine1D(nn.Module):
    """
    Conditional affine transform for 1D y:
        z = (y - t(x)) * exp(-s(x))
        logdet = -s(x)
    """

    def __init__(self, x_dim: int, hidden: int = 128):
        super().__init__()
        p = max(1, x_dim)
        self.s_net = nn.Sequential(
            nn.Linear(p, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
        self.t_net = nn.Sequential(
            nn.Linear(p, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(
        self, y: Tensor, x: Tensor, reverse: bool = False
    ) -> tuple[Tensor, Tensor]:
        x = _feat(x)  # [B, p]
        s = self.s_net(x).tanh()  # [B, 1] bounded for stability
        t = self.t_net(x)  # [B, 1]
        if not reverse:
            z = (y - t) * torch.exp(-s)
            logdet = (-s).squeeze(-1)  # [B]
            return z, logdet
        else:
            y = y * torch.exp(s) + t
            logdet = s.squeeze(-1)
            return y, logdet


class CondRealNVPFlowCPD(DifferentiableCPD):
    """
    Conditional RealNVP flow CPD:
       z ~ N(0,I),  y = f_theta^{-1}(z; x), exact log_prob via change-of-variables.
    Works for any D (out_dim). Conditioning on x via FiLM/concat in each coupling layer.
    """

    def __init__(
        self,
        name: str,
        in_dim: int,
        out_dim: int,
        n_layers: int = 4,
        hidden: int = 128,
        lr: float = 1e-3,
        epochs: int = 20,
        batch_size: int | None = 1024,
        weight_decay: float = 0.0,
    ):
        super().__init__(
            name,
            {
                "in_dim": in_dim,
                "out_dim": out_dim,
                "n_layers": n_layers,
                "hidden": hidden,
                "lr": lr,
                "epochs": epochs,
                "batch_size": batch_size,
                "weight_decay": weight_decay,
            },
        )
        self.in_dim = in_dim
        self.out_dim = out_dim
        L = []
        if out_dim == 1:
            # use a stack of conditional 1D affine layers
            for _ in range(n_layers):
                L.append(_CondAffine1D(x_dim=in_dim, hidden=hidden))
        else:
            # standard RealNVP coupling with alternating masks
            for i in range(n_layers):
                L.append(
                    _CondAffineCoupling(
                        y_dim=out_dim,
                        x_dim=in_dim,
                        hidden=hidden,
                        mask_even=(i % 2 == 0),
                    )
                )
        self.layers = nn.ModuleList(L)

    # training utilities
    def _iter(self, X: Tensor, y: Tensor, bs: int | None):
        N = y.shape[0]
        if bs is None or bs <= 0 or bs >= N:
            yield _feat(X), y
            return
        perm = torch.randperm(N, device=y.device)
        for s in range(0, N, bs):
            idx = perm[s : s + bs]
            yield _feat(X)[idx], y[idx]

    # log p(y|x) with exact change of variables
    def log_prob(self, X: Tensor, y: Tensor) -> Tensor:
        x = _feat(X)
        z = y
        logdet_sum = y.new_zeros(y.shape[0])
        for layer in self.layers:
            z, logdet = layer(z, x, reverse=False)  # forward: y -> z
            logdet_sum = logdet_sum + logdet
        # base density: standard Normal
        logpz = -0.5 * (
            z.pow(2).sum(dim=-1)
            + self.out_dim
            * torch.log(torch.tensor(2.0 * torch.pi, device=z.device, dtype=z.dtype))
        )
        return logpz + logdet_sum

    def fit(self, X: Tensor, y: Tensor) -> None:
        Xf = _feat(X)
        Y = y
        opt = torch.optim.Adam(
            self.parameters(), lr=self.cfg["lr"], weight_decay=self.cfg["weight_decay"]
        )
        self.train()
        for _ in range(self.cfg["epochs"]):
            for xb, yb in self._iter(Xf, Y, self.cfg["batch_size"]):
                opt.zero_grad(set_to_none=True)
                # maximize log_prob => minimize NLL
                nll = -self.log_prob(xb, yb).mean()
                nll.backward()
                opt.step()

    @torch.no_grad()
    def update(self, X: Tensor, y: Tensor) -> None:
        with torch.enable_grad():
            # few refinement epochs
            old = self.cfg["epochs"]
            self.cfg["epochs"] = max(1, old // 2)
            try:
                self.fit(X, y)
            finally:
                self.cfg["epochs"] = old

    @torch.no_grad()
    def sample(self, X: Tensor, n: int = 1) -> Tensor:
        x = _feat(X)
        B = x.shape[0]
        D = self.out_dim
        z = torch.randn(n, B, D, device=x.device, dtype=x.dtype)
        ys = []
        for i in range(n):
            y_i = z[i]
            # inverse pass: z -> y
            for layer in reversed(self.layers):
                y_i, _ = layer(y_i, x, reverse=True)
            ys.append(y_i.unsqueeze(0))
        return torch.cat(ys, dim=0)  # [n, B, D]
