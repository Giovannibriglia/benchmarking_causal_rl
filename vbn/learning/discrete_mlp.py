from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import utils as U
from ..core import BNMeta, DiscreteCPDTable, LearnParams


class _CatMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: int = 64,
        n_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # x: [N, in_dim]
        return self.net(x)  # logits [N, out_dim]


class DiscreteMLPLearner(nn.Module):
    """
    Learns per-node Categorical CPDs via an MLP that maps one-hot(parent) -> logits(child).
    """

    def __init__(
        self,
        meta: BNMeta,
        device=None,
        dtype=torch.float32,
        **kwargs,
    ):
        super().__init__()
        self.meta = meta
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dtype = dtype

        self.hidden = kwargs.get("hidden", 128)
        self.n_layers = kwargs.get("n_layers", 3)
        self.lr = kwargs.get("lr", 1e-3)
        self.epochs = kwargs.get("epochs", 50)
        self.batch_size = kwargs.get("batch_size", 1024)
        self.dropout = kwargs.get("dropout", 0.0)

    def _build_onehot_inputs(
        self, pa: List[str], data: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, int]:
        if not pa:
            N = next(iter(data.values())).shape[0]
            return torch.zeros(N, 0, device=self.device), 0
        onehots = []
        for p in pa:
            x = U.to_long(data[p]).to(self.device)
            k = int(self.meta.cards[p])
            onehots.append(F.one_hot(x, num_classes=k))
        X = torch.cat(onehots, dim=1).to(torch.float32)  # [N, sum(cards)]
        return X, X.shape[1]

    def fit(self, data: Dict[str, torch.Tensor]) -> LearnParams:
        models: Dict[str, nn.Module] = {}
        for child in self.meta.order:
            if self.meta.types[child] != "discrete":
                continue

            pa = self.meta.parents[child]
            X, in_dim = self._build_onehot_inputs(pa, data)
            y = U.to_long(data[child]).to(self.device)
            C = int(self.meta.cards[child])

            model = _CatMLP(
                in_dim,
                C,
                hidden=self.hidden,
                n_layers=self.n_layers,
                dropout=self.dropout,
            ).to(self.device)
            opt = torch.optim.AdamW(model.parameters(), lr=self.lr)
            N = y.shape[0]

            for _ in range(self.epochs):
                for s in range(0, N, self.batch_size):
                    e = min(s + self.batch_size, N)
                    logits = model(X[s:e])  # [B, C]
                    loss = F.cross_entropy(logits, y[s:e])
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()

            models[child] = model

        return LearnParams(meta=self.meta, discrete_mlps=models)

    @torch.no_grad()
    def materialize_tables(self, lp: LearnParams) -> LearnParams:
        """Evaluate trained MLPs on ALL parent configs to create tabular CPDs."""
        assert lp.discrete_mlps is not None, "No neural CPDs to materialize."
        tbls: Dict[str, DiscreteCPDTable] = {}

        for child, model in lp.discrete_mlps.items():
            pa = self.meta.parents[child]
            C = int(self.meta.cards[child])
            if not pa:
                logits = model(torch.zeros(1, 0, device=self.device))
                probs = torch.softmax(logits, dim=-1)  # [1, C]
                tbls[child] = DiscreteCPDTable(
                    probs=probs.to(device=self.device, dtype=self.dtype),
                    parent_names=[],
                    parent_cards=[],
                    child_card=C,
                    strides=torch.tensor([], device=self.device, dtype=torch.long),
                )
                continue

            parent_cards = [int(self.meta.cards[p]) for p in pa]
            grids = torch.meshgrid(
                *[torch.arange(k, device=self.device) for k in parent_cards],
                indexing="ij",
            )
            onehots = [
                torch.nn.functional.one_hot(g.reshape(-1), num_classes=k)
                for g, k in zip(grids, parent_cards)
            ]
            Xall = torch.cat(onehots, dim=1).to(torch.float32)  # [P, sum(cards)]
            logits = model(Xall)  # [P, C]
            probs = torch.softmax(logits, dim=-1)
            strides = torch.tensor(
                U.make_strides(parent_cards), device=self.device, dtype=torch.long
            )

            tbls[child] = DiscreteCPDTable(
                probs=probs.to(device=self.device, dtype=self.dtype),
                parent_names=pa,
                parent_cards=parent_cards,
                child_card=C,
                strides=strides,
            )

        return LearnParams(
            meta=self.meta,
            discrete_tables=tbls,
            discrete_mlps=lp.discrete_mlps,
            lg=lp.lg,
            cont_mlps=lp.cont_mlps,
            cont_mlp_meta=lp.cont_mlp_meta,
        )
