from __future__ import annotations

import torch
from torch import nn

from ..base import DifferentiableCPD


class MLESoftmaxCPD(DifferentiableCPD):
    """
    Multinomial logistic regression with intercept:
      p(y=k | x) = softmax(W^T x + b)_k
    Expects y as class indices (LongTensor in [0, K-1]).
    - fit: few epochs of plain GD (sequential for-loop)
    - update: 1–few small steps on new data
    Handles root nodes by using bias-only features when in_dim=0.
    """

    def __init__(
        self,
        name: str,
        in_dim: int,
        num_classes: int,
        lr: float = 5e-2,
        epochs: int = 10,
        batch_size: int | None = None,
        weight_decay: float = 0.0,
    ):
        super().__init__(
            name,
            {
                "in_dim": in_dim,
                "num_classes": num_classes,
                "lr": lr,
                "epochs": epochs,
                "batch_size": batch_size,
                "weight_decay": weight_decay,
            },
        )
        p = max(1, in_dim)  # at least bias-only dimension
        self.linear = nn.Linear(p, num_classes, bias=True)

    @staticmethod
    def _features(X: torch.Tensor) -> torch.Tensor:
        # For roots (X: [N, 0]) -> return ones [N, 1]
        if X.dim() == 2 and X.shape[1] == 0:
            return torch.ones((X.shape[0], 1), device=X.device, dtype=X.dtype)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        return X

    def _iter_batches(
        self, X: torch.Tensor, y_idx: torch.Tensor, batch_size: int | None
    ):
        N = X.shape[0]
        if batch_size is None or batch_size <= 0 or batch_size >= N:
            yield X, y_idx
            return
        perm = torch.randperm(N, device=X.device)
        for s in range(0, N, batch_size):
            idx = perm[s : s + batch_size]
            yield X[idx], y_idx[idx]

    # sequential “fit”: a few epochs of GD
    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        Xf = self._features(X)
        y_idx = y.squeeze(-1).long()
        opt = torch.optim.SGD(
            self.parameters(), lr=self.cfg["lr"], weight_decay=self.cfg["weight_decay"]
        )
        loss_fn = nn.CrossEntropyLoss()

        self.train()
        for _ in range(self.cfg["epochs"]):
            for Xb, yb in self._iter_batches(Xf, y_idx, self.cfg["batch_size"]):
                opt.zero_grad(set_to_none=True)
                logits = self.linear(Xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()

    # sequential “update”: 1 epoch small steps
    @torch.no_grad()
    def update(self, X: torch.Tensor, y: torch.Tensor) -> None:
        Xf = self._features(X)
        y_idx = y.squeeze(-1).long()
        # re-enable grads inside
        with torch.enable_grad():
            opt = torch.optim.SGD(
                self.parameters(),
                lr=self.cfg["lr"],
                weight_decay=self.cfg["weight_decay"],
            )
            loss_fn = nn.CrossEntropyLoss()
            self.train()
            for Xb, yb in self._iter_batches(Xf, y_idx, self.cfg["batch_size"]):
                opt.zero_grad(set_to_none=True)
                logits = self.linear(Xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()

    def log_prob(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        Xf = self._features(X)
        y_idx = y.squeeze(-1).long()
        logits = self.linear(Xf)
        logp = logits.log_softmax(dim=-1)
        return logp.gather(dim=-1, index=y_idx.unsqueeze(-1)).squeeze(-1)

    @torch.no_grad()
    def sample(self, X: torch.Tensor, n: int = 1) -> torch.Tensor:
        Xf = self._features(X)
        logits = self.linear(Xf)
        probs = logits.softmax(dim=-1)
        cat = torch.distributions.Categorical(probs)
        return cat.sample((n,))  # [n, N]
