# vbn/io.py
from __future__ import annotations

import importlib
import os
from typing import Any, Dict, Optional, Tuple

import networkx as nx
import torch

from . import VBN
from .inference import INFERENCE_METHODS

from .learning import LEARNING_METHODS
from .learning.base import BaseNodeCPD


# ---------------- utils ----------------


def _import_symbol(module_path: str, class_name: str):
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def _nx_to_payload(dag: nx.DiGraph) -> Dict[str, Any]:
    # Pin the "links" key to avoid NetworkX 3.6 warning
    return nx.node_link_data(dag, edges="links")


def _nx_from_payload(payload: Dict[str, Any]) -> nx.DiGraph:
    return nx.node_link_graph(payload, edges="links")


def _maybe_get(val, *attrs, default=None):
    for a in attrs:
        if hasattr(val, a):
            return getattr(val, a)
    return default


def _cpd_dims(cpd: BaseNodeCPD) -> Tuple[Optional[int], Optional[int]]:
    in_dim = _maybe_get(cpd, "in_dim", "input_dim", "x_dim", default=None)
    out_dim = _maybe_get(cpd, "out_dim", "output_dim", "y_dim", default=None)
    return in_dim, out_dim


def _cpd_config(cpd: BaseNodeCPD) -> Dict[str, Any]:
    # Prefer explicit getter
    if hasattr(cpd, "get_config") and callable(cpd.get_config):
        try:
            cfg = cpd.get_config()
            if isinstance(cfg, dict):
                return cfg
        except Exception:
            pass
    # Next best: common attribute used in your base classes
    if hasattr(cpd, "cfg") and isinstance(cpd.cfg, dict):
        return dict(cpd.cfg)
    return {}


def _factory_to_meta(factory) -> Optional[Dict[str, Any]]:
    if factory is None:
        return None
    for name, f in {**LEARNING_METHODS, **INFERENCE_METHODS}.items():
        if f is factory:
            return {"kind": "registry", "name": name}
    mod = getattr(factory, "__module__", None)
    qn = getattr(factory, "__qualname__", None)
    if mod and qn:
        return {"kind": "dotted", "module": mod, "qualname": qn}
    return {"kind": "repr", "repr": repr(factory)}


def _meta_to_factory(meta: Optional[Dict[str, Any]]):
    if not meta:
        return None
    kind = meta.get("kind")
    if kind == "registry":
        name = meta["name"]
        if name in LEARNING_METHODS:
            return LEARNING_METHODS[name]
        if name in INFERENCE_METHODS:
            return INFERENCE_METHODS[name]
        raise RuntimeError(f"Factory '{name}' not found in registries.")
    if kind == "dotted":
        mod = importlib.import_module(meta["module"])
        obj = mod
        for p in meta["qualname"].split("."):
            obj = getattr(obj, p)
        return obj
    if kind == "repr":
        raise RuntimeError(
            "Cannot reload factory saved only as textual repr. "
            "Please register or make it importable."
        )
    raise RuntimeError("Unknown factory serialization kind.")


# ---------------- public API ----------------


def save_vbn(vbn: VBN, path: str) -> None:
    """
    Persist a VBN instance: DAG, seed/device, learning/inference config,
    and per-node CPDs (class, module path, dims, config, state_dict).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    learn_meta = _factory_to_meta(getattr(vbn, "_learning_factory", None))
    infer_meta = _factory_to_meta(getattr(vbn, "_inference_factory", None))

    nodes_payload: Dict[str, Any] = {}
    for n, cpd in vbn._nodes.items():
        cls = cpd.__class__
        in_dim, out_dim = _cpd_dims(cpd)
        cfg = _cpd_config(cpd)
        try:
            sd = cpd.state_dict()
        except Exception as e:
            raise RuntimeError(f"CPD '{n}' must implement state_dict(): {e}")

        nodes_payload[n] = {
            "module": cls.__module__,
            "class": cls.__name__,
            "in_dim": in_dim,
            "out_dim": out_dim,
            "config": cfg,
            "state_dict": sd,
        }

    payload = {
        "version": 2,
        "device": str(vbn.device),
        "seed": vbn.seed,
        "dag": _nx_to_payload(vbn.dag),
        "learning": {
            "factory": learn_meta,
            "kwargs": getattr(vbn, "_learning_kwargs", {}),
        },
        "inference": {
            "factory": infer_meta,
            "kwargs": getattr(vbn, "_inference_kwargs", {}),
        },
        "nodes": nodes_payload,
    }

    torch.save(payload, path)


def load_vbn(path: str, *, map_location: Optional[str | torch.device] = None) -> VBN:
    """
    Reconstruct a VBN saved by save_vbn().
    Tries multiple CPD constructor signatures to support:
      - CPD(name, in_dim, out_dim, **cfg)
      - CPD(name, **cfg)
      - CPD(name, cfg=cfg)
    """
    # Make explicit to silence future PyTorch warning
    payload: Dict[str, Any] = torch.load(
        path, map_location=map_location, weights_only=False
    )

    dag = _nx_from_payload(payload["dag"])

    dev_str = payload.get("device", None)
    device = torch.device(dev_str) if dev_str else None
    if map_location is not None:
        device = torch.device(map_location)

    vbn = VBN(dag=dag, device=device, seed=payload.get("seed", None))

    # Restore learning config
    learn = payload.get("learning", {})
    lf = _meta_to_factory(learn.get("factory"))
    if lf is not None:
        vbn.set_learning_method(lf, **(learn.get("kwargs", {}) or {}))

    # Rebuild nodes
    vbn._nodes = {}
    for name, npld in (payload.get("nodes") or {}).items():
        module_path = npld["module"]
        class_name = npld["class"]
        in_dim = npld.get("in_dim")
        out_dim = npld.get("out_dim")
        cfg = npld.get("config") or {}

        CPDClass = _import_symbol(module_path, class_name)

        # Try multiple constructor styles
        cpd = None
        errs = []

        # 1) name, in_dim, out_dim, **cfg   (only if both dims are known)
        if in_dim is not None and out_dim is not None:
            try:
                cpd = CPDClass(
                    name=name, in_dim=int(in_dim), out_dim=int(out_dim), **cfg
                )
            except Exception as e:
                errs.append(f"(name,in_dim,out_dim,**cfg) -> {e}")

        # 2) name, **cfg
        if cpd is None:
            try:
                cpd = CPDClass(name=name, **cfg)
            except Exception as e:
                errs.append(f"(name,**cfg) -> {e}")

        # 3) name, cfg=cfg
        if cpd is None:
            try:
                cpd = CPDClass(name=name, cfg=cfg)
            except Exception as e:
                errs.append(f"(name,cfg=cfg) -> {e}")

        if cpd is None:
            raise TypeError(
                f"Could not instantiate CPD '{name}' ({module_path}.{class_name}). "
                f"Tried signatures:\n  - " + "\n  - ".join(errs)
            )

        cpd = cpd.to(vbn.device)
        cpd.load_state_dict(npld["state_dict"])
        vbn._nodes[name] = cpd

    # Restore inference (optional; binds inferencer to nodes)
    inf = payload.get("inference", {})
    inf_factory = _meta_to_factory(inf.get("factory"))
    if inf_factory is not None:
        vbn.set_inference_method(inf_factory, **(inf.get("kwargs", {}) or {}))

    return vbn
