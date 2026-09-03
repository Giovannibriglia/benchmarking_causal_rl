"""The GRACE transform cache — fit once per (data, options), reuse everywhere.

**Licence (measured, 2026-09-02/03).** The fit is a pure function of
``(dataset content, fit options)``: 10/10 completed production
(cell, dataset-seed) pairs are bitwise-identical across cql/iql (every float
compared by ``repr``), and two CPU refits with deliberately different global
RNG perturbations produced the same reward-column sha256
(``73c2a173c223…``). The estimator seeds itself, the buffer is runner-owned
and filled from the dataset alone, and deterministic kernels are the default.
Fits are ~1h on d-cells and were recomputed per algorithm per training seed —
half of every grace campaign was redundant. This module removes that term.

**The key is a DICT; the hash only locates.** A candidate entry is a hit only
when its stored key dict equals the query field-by-field — collision is
impossible rather than unlikely. Fields, each with its reason:

* ``data_sha256`` — over the EXACT tensors the fit consumes
  (``EpisodeData`` + next-obs + dones, in order), NOT the dataset id: the
  fingerprint bug means id ↛ content is not guaranteed (a regenerated
  dataset keeps its id), so content addresses the cache and a stale dataset
  is a MISS, never a silent wrong hit.
* ``dataset_id`` — carried for the audit trail (and it catches the reverse
  accident: same content stored under two ids is fine, and visible).
* ``proxy_names``, ``alpha``, ``b``, ``fit_seed``, ``init_seeds``,
  ``fit_kwargs`` (the WHOLE dict — a future kwarg must widen the key by
  construction) — each changes the fit.
* ``code_version`` — sha256 over the SOURCE BYTES of this package + the
  vendored NBN version + torch version. Source bytes, not the git commit:
  an uncommitted edit must invalidate (the S12 lesson — never trust that
  code on disk is the code you remember).
* ``device_kind`` + ``deterministic`` — deterministic kernels are not
  bit-identical across device kinds; a CPU-fitted entry must not serve a
  CUDA run silently.

**Artifacts** under ``<root>/<sha16>/``: ``key.json`` (the full dict),
``serving.json`` (mode/reason/label fields + meta; written with Python's
NaN-literal JSON — this file is read only by this module), and
``rewards.npy`` (float32, buffer order) unless the fit ABSTAINED — abstentions
are cached too (a ~700-950s abstention is still worth not recomputing) and
remain visibly abstentions in every consumer's provenance (S15).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

_CODE_VERSION: Optional[str] = None


def code_version() -> str:
    """sha256 over the fit's ENTIRE code identity: this package's source
    bytes, the vendored NBN's source bytes, and torch's version string.

    NBN is hashed as BYTES, not its version tag — the tag is a second
    construction site for the fact the bytes already carry, and an edited
    vendored tree with an unbumped tag would serve a stale hit as fresh
    (the one failure a cache must not have). torch is not vendored, so its
    version string is the honest identity available. Memoised: the tree walk
    runs once per process, and code cannot change mid-process.
    """
    global _CODE_VERSION
    if _CODE_VERSION is not None:
        return _CODE_VERSION
    h = hashlib.sha256()
    pkg = Path(__file__).resolve().parent
    for f in sorted(pkg.glob("*.py")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    try:
        import nbn

        nbn_root = Path(nbn.__file__).resolve().parent
        for f in sorted(nbn_root.rglob("*.py")):
            h.update(str(f.relative_to(nbn_root)).encode())
            h.update(f.read_bytes())
    except Exception:
        h.update(b"nbn-unavailable")
    h.update(torch.__version__.encode())
    _CODE_VERSION = h.hexdigest()
    return _CODE_VERSION


def _tensor_bytes(h, t: Optional[torch.Tensor], name: str) -> None:
    h.update(name.encode())
    if t is None:
        h.update(b"none")
        return
    a = t.detach().cpu().numpy()
    # Canonical dtypes so a float64-vs-float32 loader difference is a REAL
    # difference (it changes the fit) and an int32/int64 id column is not.
    if a.dtype.kind == "f":
        a = a.astype(np.float32)
    elif a.dtype.kind in "iub":
        a = a.astype(np.int64)
    h.update(str(a.shape).encode())
    h.update(a.tobytes())


def data_fingerprint(data, next_obs, dones) -> str:
    """sha256 over the exact fit inputs, in a fixed order."""
    h = hashlib.sha256()
    _tensor_bytes(h, data.state, "state")
    _tensor_bytes(h, data.action, "action")
    _tensor_bytes(h, data.reward, "reward")
    _tensor_bytes(h, data.episode_ids, "episode_ids")
    for k in sorted(data.proxy):
        _tensor_bytes(h, data.proxy[k], f"proxy_{k}")
    _tensor_bytes(h, next_obs, "next_obs")
    _tensor_bytes(h, dones, "dones")
    return h.hexdigest()


def build_key(
    *,
    dataset_id: str,
    data_sha256: str,
    proxy_names: tuple,
    alpha: float,
    b: int,
    fit_seed: int,
    init_seeds: tuple,
    fit_kwargs: dict,
    device_kind: str,
) -> dict:
    return dict(
        dataset_id=str(dataset_id),
        data_sha256=data_sha256,
        proxy_names=list(proxy_names),
        alpha=float(alpha),
        b=int(b),
        fit_seed=int(fit_seed),
        init_seeds=[int(s) for s in init_seeds],
        fit_kwargs={k: fit_kwargs[k] for k in sorted(fit_kwargs)},
        code_version=code_version(),
        device_kind=str(device_kind),
        deterministic=bool(torch.are_deterministic_algorithms_enabled()),
    )


def _sha_of_key(key: dict) -> str:
    return hashlib.sha256(
        json.dumps(key, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def _entry_dir(root, key: dict) -> Path:
    return Path(root) / _sha_of_key(key)[:16]


def load(root, key: dict):
    """Return the cached ``GraceServing`` or None. Equality on the FULL key
    dict decides; the sha only located the candidate."""
    from src.rl.offline.grace.serving import GraceServing

    d = _entry_dir(root, key)
    kf, sf = d / "key.json", d / "serving.json"
    if not (kf.exists() and sf.exists()):
        return None
    if json.loads(kf.read_text()) != json.loads(json.dumps(key)):
        return None  # located but NOT equal: treat as miss, never serve it
    doc = json.loads(sf.read_text())
    rewards = None
    rf = d / "rewards.npy"
    if rf.exists():
        rewards = torch.from_numpy(np.load(rf))
    meta = dict(doc.get("meta") or {})
    meta["transform_cache_hit"] = True
    meta["transform_cache_entry"] = str(d)
    return GraceServing(
        mode=doc["mode"],
        reason=doc.get("reason", ""),
        fit_label=doc.get("fit_label", ""),
        l4_kind=doc.get("l4_kind", ""),
        lo=doc.get("lo", float("nan")),
        hi=doc.get("hi", float("nan")),
        rewards=rewards,
        meta=meta,
    )


def store(root, key: dict, serving) -> Path:
    """Persist a fresh fit. Returns the entry dir. Verifies the write landed
    (S12: never trust a writer's own success)."""
    d = _entry_dir(root, key)
    d.mkdir(parents=True, exist_ok=True)
    (d / "key.json").write_text(json.dumps(key, indent=1, sort_keys=True))
    doc = dict(
        mode=serving.mode,
        reason=serving.reason,
        fit_label=serving.fit_label,
        l4_kind=serving.l4_kind,
        lo=serving.lo,
        hi=serving.hi,
        meta={
            k: v
            for k, v in (serving.meta or {}).items()
            if isinstance(v, (int, float, str, bool)) or v is None
        },
    )
    (d / "serving.json").write_text(json.dumps(doc, indent=1))
    if serving.rewards is not None:
        np.save(d / "rewards.npy", serving.rewards.detach().cpu().numpy())
    back = load(root, key)
    if back is None:
        raise RuntimeError(f"transform cache write did not land at {d}")
    return d
