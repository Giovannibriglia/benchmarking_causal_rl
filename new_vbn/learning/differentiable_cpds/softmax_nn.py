from __future__ import annotations

import torch
from torch import nn, Tensor

from ...learning.base import DifferentiableCPD


class SoftmaxNNCPD(DifferentiableCPD):
    """
    Multiclass NN CPD:
        p(y=k|x) = softmax(g_theta(x))_k
    y must be integer class indices in [0..K-1] with out_dim = K.
    """

    def __init__(
        self,
        name: str,
        in_dim: int,
        out_dim: int,
        hidden: int = 128,
        depth: int = 2,
        lr: float = 1e-3,
        epochs: int = 10,
        batch_size: int | None = 1024,
        weight_decay: float = 0.0,
    ):
        super().__init__(
            name,
            {
                "in_dim": in_dim,
                "out_dim": out_dim,
                "hidden": hidden,
                "depth": depth,
                "lr": lr,
                "epochs": epochs,
                "batch_size": batch_size,
                "weight_decay": weight_decay,
            },
        )
        p = max(1, in_dim)  # bias-only for roots
        layers: list[nn.Module] = [nn.Linear(p, hidden), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)

    @staticmethod
    def _features(X: Tensor) -> Tensor:
        if X.dim() == 2 and X.shape[1] == 0:
            return torch.ones((X.shape[0], 1), device=X.device, dtype=X.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        return X

    def _batches(self, X: Tensor, y: Tensor, bs: int | None):
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
        y_idx = y.squeeze(-1).long()
        K = self.cfg["out_dim"]
        if y_idx.min() < 0 or y_idx.max() >= K:
            raise ValueError(
                f"[{self.name}] y out of range: got [{int(y_idx.min())},{int(y_idx.max())}], K={K}"
            )

        opt = torch.optim.Adam(
            self.parameters(), lr=self.cfg["lr"], weight_decay=self.cfg["weight_decay"]
        )
        loss_fn = nn.CrossEntropyLoss()
        self.train()
        for _ in range(self.cfg["epochs"]):
            for Xb, yb in self._batches(Xf, y_idx, self.cfg["batch_size"]):
                opt.zero_grad(set_to_none=True)
                logits = self.net(Xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()

    @torch.no_grad()
    def update(self, X: Tensor, y: Tensor) -> None:
        # light refinement pass
        with torch.enable_grad():
            self.fit(X, y)

    def log_prob(self, X: Tensor, y: Tensor) -> Tensor:
        Xf = self._features(X)
        y_idx = y.squeeze(-1).long()
        logits = self.net(Xf)
        logp = logits.log_softmax(dim=-1)
        return logp.gather(dim=-1, index=y_idx.unsqueeze(-1)).squeeze(-1)

    @torch.no_grad()
    def sample(self, X: Tensor, n: int = 1) -> Tensor:
        Xf = self._features(X)
        logits = self.net(Xf)
        probs = logits.softmax(dim=-1)
        cat = torch.distributions.Categorical(probs)
        return cat.sample((n,))  # [n, N]
