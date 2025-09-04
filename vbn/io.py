from __future__ import annotations

from typing import Any, Dict, List, Optional

import networkx as nx
import torch
from tensordict import TensorDict

from .core import BNMeta, DiscreteCPDTable, LearnParams, LGParams

# ───────── helpers ─────────


def _to_cpu(x: torch.Tensor) -> torch.Tensor:
    return x.detach().cpu()


def _extract_catmlp_arch(model: torch.nn.Module) -> Dict[str, Any]:
    """Infer architecture of a Discrete MLP (Sequential of Linear/ReLU[/Dropout] + Linear)."""
    # Collect Linear layer out_features except the last (which is the output dim).
    linear_layers = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
    assert len(linear_layers) >= 1, "Categorical MLP must have at least one Linear."
    in_dim = linear_layers[0].in_features
    out_dim = linear_layers[-1].out_features
    hidden_sizes = [lin.out_features for lin in linear_layers[:-1]]
    # single dropout prob if there are any Dropout modules
    dmods = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
    dropout = dmods[0].p if len(dmods) > 0 else 0.0
    return {
        "in_dim": in_dim,
        "out_dim": out_dim,
        "hidden_sizes": hidden_sizes,
        "dropout": float(dropout),
    }


def _extract_gaussmlp_arch(model: torch.nn.Module) -> Dict[str, Any]:
    """Infer architecture of a Gaussian MLP (backbone + head->2)."""
    # Find *all* Linear layers except the 2-output head
    linear_layers = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
    assert (
        len(linear_layers) >= 1
    ), "Gaussian MLP must have at least one Linear (the head)."
    head = linear_layers[-1]
    assert head.out_features == 2, "Gaussian MLP head must output 2 (mean, logvar)."
    # Backbone in_dim is head.in_features if there are no other Linear layers; else first Linear in_features.
    if len(linear_layers) == 1:
        in_dim = head.in_features
        hidden_sizes: List[int] = []
    else:
        in_dim = linear_layers[0].in_features
        hidden_sizes = [lin.out_features for lin in linear_layers[:-1]]
    dmods = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
    dropout = dmods[0].p if len(dmods) > 0 else 0.0
    return {"in_dim": in_dim, "hidden_sizes": hidden_sizes, "dropout": float(dropout)}


class _CatMLPCompat(torch.nn.Module):
    """Rebuild a categorical MLP from saved arch (works like the original)."""

    def __init__(
        self, in_dim: int, out_dim: int, hidden_sizes: List[int], dropout: float
    ):
        super().__init__()
        layers: List[torch.nn.Module] = []
        d = in_dim
        for h in hidden_sizes:
            layers += [torch.nn.Linear(d, h), torch.nn.ReLU()]
            if dropout > 0:
                layers.append(torch.nn.Dropout(dropout))
            d = h
        layers.append(torch.nn.Linear(d, out_dim))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class _GaussianMLPCompat(torch.nn.Module):
    """Rebuild a Gaussian MLP (mean & logvar) from saved arch."""

    def __init__(self, in_dim: int, hidden_sizes: List[int], dropout: float):
        super().__init__()
        layers: List[torch.nn.Module] = []
        d = in_dim
        for h in hidden_sizes:
            layers += [torch.nn.Linear(d, h), torch.nn.ReLU()]
            if dropout > 0:
                layers.append(torch.nn.Dropout(dropout))
            d = h
        self.backbone = torch.nn.Sequential(*layers)
        self.head = torch.nn.Linear(d, 2)

    def forward(self, x):
        h = self.backbone(x)
        out = self.head(h)
        mean, logvar = out[..., 0], out[..., 1]
        return mean, logvar


# ───────── to/from TensorDict-like payloads ─────────


def learnparams_to_tensordict(lp: LearnParams) -> Any:
    """
    Convert LearnParams to a (nested) TensorDict if available, else a plain dict.
    All tensors are moved to CPU for portability.
    """
    payload: Dict[str, Any] = {}

    # meta: save order, parents, types, cards
    payload["meta"] = {
        "order": lp.meta.order,
        "types": lp.meta.types,
        "cards": lp.meta.cards,
        "parents": lp.meta.parents,
    }

    # discrete tables
    if lp.discrete_tables:
        dt = {}
        for name, tbl in lp.discrete_tables.items():
            dt[name] = {
                "probs": _to_cpu(tbl.probs),
                "parent_names": tbl.parent_names,
                "parent_cards": tbl.parent_cards,
                "child_card": tbl.child_card,
                "strides": _to_cpu(tbl.strides),
            }
        payload["discrete_tables"] = dt

    # discrete MLPs
    if lp.discrete_mlps:
        dm = {}
        for name, model in lp.discrete_mlps.items():
            arch = _extract_catmlp_arch(model)
            state = {k: _to_cpu(v) for k, v in model.state_dict().items()}
            dm[name] = {"arch": arch, "state_dict": state}
        payload["discrete_mlps"] = dm

    # continuous MLPs (+ meta)
    if lp.cont_mlps:
        cm = {}
        for name, model in lp.cont_mlps.items():
            arch = _extract_gaussmlp_arch(model)
            state = {k: _to_cpu(v) for k, v in model.state_dict().items()}
            # parent meta for inputs
            info = lp.cont_mlp_meta[name]
            cm[name] = {
                "arch": arch,
                "state_dict": state,
                "disc_pa": info["disc_pa"],
                "disc_cards": info["disc_cards"],
                "cont_pa": info["cont_pa"],
                "in_dim": info["in_dim"],
            }
        payload["cont_mlps"] = cm

    # linear-Gaussian
    if lp.lg is not None:
        payload["lg"] = {
            "order": lp.lg.order,
            "W": _to_cpu(lp.lg.W),
            "b": _to_cpu(lp.lg.b),
            "sigma2": _to_cpu(lp.lg.sigma2),
        }

    if TensorDict is not None:
        # Convert leaf dicts of tensors to TensorDicts where applicable
        def _maybe_td(x):
            if isinstance(x, dict) and all(
                isinstance(v, torch.Tensor) for v in x.values()
            ):
                return TensorDict(x, batch_size=[])
            return x

        # recursively wrap shallow maps of tensors
        return (
            payload  # returning plain dict is fine; TensorDict can also hold py objects
        )
    return payload


def tensordict_to_learnparams(
    td: Any,
    map_location: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> LearnParams:
    """
    Reconstruct LearnParams (and its BNMeta) from a payload produced by learnparams_to_tensordict.
    """
    device = map_location or torch.device("cpu")

    meta_blob = td["meta"]
    order = list(meta_blob["order"])
    types = dict(meta_blob["types"])
    cards = None if meta_blob["cards"] is None else dict(meta_blob["cards"])
    parents = {k: list(v) for k, v in meta_blob["parents"].items()}

    # rebuild graph quickly from parents
    G = nx.DiGraph()
    G.add_nodes_from(order)
    for ch, pas in parents.items():
        for p in pas:
            G.add_edge(p, ch)
    meta = BNMeta(G=G, types=types, cards=cards, order=order, parents=parents)

    # discrete tables
    disc_tables = None
    if "discrete_tables" in td:
        disc_tables = {}
        for name, blob in td["discrete_tables"].items():
            probs = blob["probs"].to(device=device, dtype=dtype or blob["probs"].dtype)
            strides = blob["strides"].to(device=device)
            disc_tables[name] = DiscreteCPDTable(
                probs=probs,
                parent_names=list(blob["parent_names"]),
                parent_cards=[int(x) for x in blob["parent_cards"]],
                child_card=int(blob["child_card"]),
                strides=strides,
            )

    # discrete mlps
    disc_mlps = None
    if "discrete_mlps" in td:
        disc_mlps = {}
        for name, blob in td["discrete_mlps"].items():
            arch = blob["arch"]
            model = _CatMLPCompat(
                arch["in_dim"],
                arch["out_dim"],
                hidden_sizes=list(arch["hidden_sizes"]),
                dropout=float(arch["dropout"]),
            ).to(device)
            state = {k: v.to(device) for k, v in blob["state_dict"].items()}
            model.load_state_dict(state)
            disc_mlps[name] = model

    # continuous mlps (+ meta)
    cont_mlps = None
    cont_meta = None
    if "cont_mlps" in td:
        cont_mlps, cont_meta = {}, {}
        for name, blob in td["cont_mlps"].items():
            arch = blob["arch"]
            model = _GaussianMLPCompat(
                arch["in_dim"], list(arch["hidden_sizes"]), float(arch["dropout"])
            ).to(device)
            state = {k: v.to(device) for k, v in blob["state_dict"].items()}
            model.load_state_dict(state)
            cont_mlps[name] = model
            cont_meta[name] = {
                "disc_pa": list(blob["disc_pa"]),
                "cont_pa": list(blob["cont_pa"]),
                "disc_cards": [int(x) for x in blob["disc_cards"]],
                "in_dim": int(blob["in_dim"]),
            }

    # linear-Gaussian
    lg = None
    if "lg" in td:
        lg_blob = td["lg"]
        lg_order = list(lg_blob["order"])
        name2idx = {n: i for i, n in enumerate(lg_order)}
        W = lg_blob["W"].to(device=device, dtype=dtype or lg_blob["W"].dtype)
        b = lg_blob["b"].to(device=device, dtype=dtype or lg_blob["b"].dtype)
        sigma2 = lg_blob["sigma2"].to(
            device=device, dtype=dtype or lg_blob["sigma2"].dtype
        )
        lg = LGParams(order=lg_order, name2idx=name2idx, W=W, b=b, sigma2=sigma2)

    return LearnParams(
        meta=meta,
        discrete_tables=disc_tables,
        discrete_mlps=disc_mlps,
        lg=lg,
        cont_mlps=cont_mlps,
        cont_mlp_meta=cont_meta,
    )


# ───────── file-level helpers ─────────


def save_learnparams(path: str, lp: LearnParams) -> None:
    """Save LearnParams to disk (CPU tensors) as a self-contained file."""
    pkg = {
        "format_version": 1,
        "payload": learnparams_to_tensordict(lp),
    }
    torch.save(pkg, path)


def load_learnparams(
    path: str, map_location: Optional[str | torch.device] = None
) -> LearnParams:
    """Load LearnParams from disk (created by save_learnparams)."""
    pkg = torch.load(path, map_location=map_location or "cpu", weights_only=True)
    assert "payload" in pkg, "Invalid file: missing payload."
    return tensordict_to_learnparams(
        pkg["payload"], map_location=map_location or torch.device("cpu")
    )
