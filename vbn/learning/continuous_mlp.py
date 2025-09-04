from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import utils as U
from ..core import BNMeta, LearnParams, LGParams


class _GaussianMLP(nn.Module):
    def __init__(
        self, in_dim: int, hidden: int = 128, n_layers: int = 3, dropout: float = 0.0
    ):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(d, 2)  # mean, logvar

    def forward(self, x):
        h = self.backbone(x)
        out = self.head(h)
        mean, logvar = out[..., 0], out[..., 1]
        logvar = torch.clamp(logvar, min=-10.0, max=6.0)
        return mean, logvar


class ContinuousMLPLearner(nn.Module):
    """
    Per-node Gaussian MLP CPDs:
      x_i | x_pa ~ N(mean_MLP([onehot(discrete_pa), cont_pa]), exp(logvar_MLP(...)))
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

    def _split_parents(self, child: str):
        pa = self.meta.parents[child]
        disc = [p for p in pa if self.meta.types[p] == "discrete"]
        cont = [p for p in pa if self.meta.types[p] == "continuous"]
        return disc, cont

    def _make_inputs(
        self, disc_pa: List[str], cont_pa: List[str], data: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        xs = []
        N = next(iter(data.values())).shape[0]
        for p in disc_pa:
            k = int(self.meta.cards[p])
            xs.append(
                F.one_hot(U.to_long(data[p]).to(self.device), num_classes=k).to(
                    self.dtype
                )
            )
        for p in cont_pa:
            xs.append(data[p].to(device=self.device, dtype=self.dtype).unsqueeze(-1))
        return (
            torch.cat(xs, dim=1)
            if xs
            else torch.zeros((N, 0), device=self.device, dtype=self.dtype)
        )

    def fit(self, data: Dict[str, torch.Tensor]) -> LearnParams:
        models: Dict[str, nn.Module] = {}
        meta_info: Dict[str, Dict] = {}
        N = next(iter(data.values())).shape[0]

        for child in self.meta.order:
            if self.meta.types[child] != "continuous":
                continue

            disc_pa, cont_pa = self._split_parents(child)
            in_dim = sum(int(self.meta.cards[p]) for p in disc_pa) + len(cont_pa)
            model = _GaussianMLP(
                in_dim, hidden=self.hidden, n_layers=self.n_layers, dropout=self.dropout
            ).to(self.device)
            opt = torch.optim.AdamW(model.parameters(), lr=self.lr)

            X = self._make_inputs(disc_pa, cont_pa, data)  # [N, in_dim]
            y = data[child].to(device=self.device, dtype=self.dtype).flatten()  # [N]

            for _ in range(self.epochs):
                for s in range(0, N, self.batch_size):
                    e = min(s + self.batch_size, N)
                    mean, logvar = model(X[s:e])
                    inv_var = torch.exp(-logvar)
                    # NLL of Gaussian
                    loss = 0.5 * ((y[s:e] - mean) ** 2 * inv_var + logvar).mean()
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()

            models[child] = model
            meta_info[child] = {
                "disc_pa": disc_pa,
                "cont_pa": cont_pa,
                "disc_cards": [int(self.meta.cards[p]) for p in disc_pa],
                "in_dim": in_dim,
            }

        return LearnParams(meta=self.meta, cont_mlps=models, cont_mlp_meta=meta_info)


@torch.no_grad()
def materialize_lg_from_cont_mlp(
    lp: LearnParams,
    pivot: Optional[Dict[str, torch.Tensor]] = None,
    data: Optional[Dict[str, torch.Tensor]] = None,
    eps: float = 1e-12,
) -> LearnParams:
    """
    Linearize each continuous MLP CPD at 'pivot' (or data mean/mode) to produce LGParams.
    """
    assert (
        lp.cont_mlps is not None and lp.cont_mlp_meta is not None
    ), "Need trained continuous MLPs."
    meta = lp.meta
    cont_order = [n for n in meta.order if meta.types[n] == "continuous"]
    if pivot is None:
        assert data is not None, "Provide pivot or data to compute it."
        pivot = U.pivot_from_data(meta.order, meta.types, meta.cards, data)

    name2idx = {n: i for i, n in enumerate(cont_order)}
    nc = len(cont_order)
    dev = (
        next(iter(lp.cont_mlps.values())).backbone[0].weight.device
        if cont_order
        else torch.device("cpu")
    )
    dtype = torch.float32

    W = torch.zeros((nc, nc), device=dev, dtype=dtype)
    b = torch.zeros(nc, device=dev, dtype=dtype)
    sigma2 = torch.zeros(nc, device=dev, dtype=dtype)

    for child in cont_order:
        model = lp.cont_mlps[child]
        info = lp.cont_mlp_meta[child]
        disc_pa, cont_pa = info["disc_pa"], info["cont_pa"]

        # Build discrete one-hot (constant at pivot)
        d_vec = []
        for p, k in zip(disc_pa, info["disc_cards"]):
            idx = int(pivot[p].item())
            oh = torch.nn.functional.one_hot(
                torch.tensor(idx, device=dev), num_classes=k
            ).to(dtype)
            d_vec.append(oh)
        d_vec = (
            torch.cat(d_vec, dim=0)
            if d_vec
            else torch.zeros(0, device=dev, dtype=dtype)
        )

        i = name2idx[child]
        if cont_pa:
            c0 = torch.stack(
                [pivot[p].to(device=dev, dtype=dtype).reshape(()) for p in cont_pa]
            )  # [m]
            c0 = c0.clone().detach().requires_grad_(True)
            x_in = torch.cat([d_vec, c0], dim=0).unsqueeze(0)
            mean0, logvar0 = model(x_in)
            # Jacobian of mean wrt continuous inputs
            J = torch.autograd.grad(
                mean0.sum(), c0, retain_graph=False, create_graph=False
            )[
                0
            ]  # [m]
            for g, p in zip(J, cont_pa):
                W[i, name2idx[p]] = g
            b[i] = (mean0 - (J * c0).sum()).reshape(())
            sigma2[i] = torch.exp(logvar0).reshape(()).clamp_min(eps)
        else:
            x_in = torch.cat(
                [d_vec, torch.zeros(0, device=dev, dtype=dtype)], dim=0
            ).unsqueeze(0)
            mean0, logvar0 = model(x_in)
            b[i] = mean0.reshape(())
            sigma2[i] = torch.exp(logvar0).reshape(()).clamp_min(eps)

    lg = LGParams(order=cont_order, name2idx=name2idx, W=W, b=b, sigma2=sigma2)
    # keep other fields of lp intact
    return LearnParams(
        meta=lp.meta,
        discrete_tables=lp.discrete_tables,
        discrete_mlps=lp.discrete_mlps,
        lg=lg,
        cont_mlps=lp.cont_mlps,
        cont_mlp_meta=lp.cont_mlp_meta,
    )
