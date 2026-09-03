"""Profile ONE representative GRACE fit — where does the time go?

Ruled 2026-09-03: speed work starts with MEASUREMENT. This runs the production
``fit_reward_transform`` on one real dataset and attributes wall time to the
phases the candidates would touch — EM iterations (E-step / per-node M-step
fits), bootstrap replicates, init-seed fits, ``interventional_sweep`` — via
wrappers around the exact call sites, plus a cProfile of everything else. No
optimisation is applied here; the report is the input to that decision.

    uv run python tools/profile_grace_fit.py [--dataset ID] [--n-ep N] [--b B]
        [--out results/profile_grace_fit.json]
"""

from __future__ import annotations

import argparse
import cProfile
import json
import os
import pstats
import sys
import time
from collections import defaultdict

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MINARI_DATASETS_PATH", os.path.expanduser("~/.minari-grace-v2"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.use_deterministic_algorithms(True)

DEFAULT_DID = "grace-v2/d_d_sweep_d100-cartpole/medium-bias_confounded_action-sigma000-seed0-d100-v0"
PROXIES = ("Z", "W", "V")

_t = defaultdict(float)
_n = defaultdict(int)


def _timed(bucket):
    def deco(fn):
        def wrapped(*a, **kw):
            t0 = time.perf_counter()
            try:
                return fn(*a, **kw)
            finally:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                _t[bucket] += time.perf_counter() - t0
                _n[bucket] += 1

        return wrapped

    return deco


def _load(did: str, n_ep: int | None, dev: str):
    import minari

    ds = minari.load_dataset(did)
    obs, act, rew, eps, prox = [], [], [], [], {p: [] for p in PROXIES}
    for i, ep in enumerate(ds.iterate_episodes()):
        if n_ep is not None and i >= n_ep:
            break
        t = len(ep.actions)
        obs.append(np.asarray(ep.observations[:-1], dtype=np.float32))
        act.append(np.asarray(ep.actions))
        rew.append(np.asarray(ep.rewards, dtype=np.float32))
        eps.append(np.full(t, i))
        for p in PROXIES:
            prox[p].append(
                np.asarray(ep.infos[f"proxy_{p.lower()}"], dtype=np.float32)[:t]
            )
    buf = dict(
        obs=torch.tensor(np.concatenate(obs), device=dev),
        actions=torch.tensor(np.concatenate(act), device=dev),
        rewards=torch.tensor(np.concatenate(rew), device=dev),
        episode_ids=torch.tensor(np.concatenate(eps), device=dev),
    )
    for p in PROXIES:
        buf[f"proxy_{p}"] = torch.tensor(np.concatenate(prox[p]), device=dev)
    return buf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT_DID)
    ap.add_argument("--n-ep", type=int, default=None, help="episode cap (None = all)")
    ap.add_argument(
        "--b",
        type=int,
        default=None,
        help="bootstrap replicates (None = production default)",
    )
    ap.add_argument("--out", default="results/profile_grace_fit.json")
    ap.add_argument("--top", type=int, default=25, help="cProfile rows to print")
    ap.add_argument(
        "--device", default=None, help="cuda|cpu (default: cuda if available)"
    )
    args = ap.parse_args()

    from src.rl.offline.grace import bootstrap, estimator, serving

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    buf = _load(args.dataset, args.n_ep, dev)
    n_rows = int(buf["rewards"].shape[0])
    print(f"dataset={args.dataset}\nrows={n_rows} device={dev}", flush=True)

    # ---- wrappers at the exact call sites --------------------------------
    E = estimator.LatentClassEstimator
    E.e_step = _timed("e_step")(E.e_step)
    E.m_step = _timed("m_step")(E.m_step)
    E.interventional_sweep = _timed("interventional_sweep")(E.interventional_sweep)
    E.fit = _timed("fit_total")(E.fit)
    # per-node M-step fits: wrap every mechanism's fit_local by node name
    orig_fit_local = {}

    def _wrap_model(est):
        for node, mech in est.model.mechanisms.items():
            cls = type(mech)
            if cls in orig_fit_local:
                continue
            orig_fit_local[cls] = cls.fit_local
            f = orig_fit_local[cls]

            def fl(self, *a, _f=f, **kw):
                name = next(
                    (
                        n
                        for n, m in _wrap_model.est.model.mechanisms.items()
                        if m is self
                    ),
                    type(self).__name__,
                )
                return _timed(f"m_step/fit_local[{name}]")(_f)(self, *a, **kw)

            cls.fit_local = fl

    _wrap_model.est = None
    orig_init = E.__init__

    def init(self, *a, **kw):
        orig_init(self, *a, **kw)
        _wrap_model.est = self
        _wrap_model(self)

    E.__init__ = init
    # bootstrap replicates: wrap the statistic callable handed to bootstrap_null
    orig_bn = bootstrap.bootstrap_null

    def bn(statistic, *a, **kw):
        return orig_bn(_timed("bootstrap_replicate")(statistic), *a, **kw)

    bootstrap.bootstrap_null = bn
    import src.rl.offline.grace.l4 as l4

    l4.bootstrap_null = bn

    # ---- the fit, under cProfile -----------------------------------------
    opts = dict(proxy_names=PROXIES, device=dev)
    if args.b is not None:
        opts["b"] = int(args.b)
    prof = cProfile.Profile()
    t0 = time.perf_counter()
    prof.enable()
    s = serving.fit_reward_transform(buf, **opts)
    prof.disable()
    total = time.perf_counter() - t0
    print(f"\n[grace] {s.label()}\nTOTAL {total:.0f}s", flush=True)

    # ---- the report --------------------------------------------------------
    rows = sorted(_t.items(), key=lambda kv: -kv[1])
    print(f"\n{'phase':40s} {'calls':>7s} {'total s':>9s} {'mean s':>8s} {'share':>6s}")
    for k, v in rows:
        print(
            f"{k:40s} {_n[k]:7d} {v:9.1f} {v / max(_n[k], 1):8.2f} {100 * v / total:5.1f}%"
        )
    fits = _n["fit_total"]
    print(
        f"\nfits run: {fits} (observed + init seeds + {_n['bootstrap_replicate']} bootstrap replicates)"
    )
    print(
        "\n--- cProfile, cumulative, top rows (everything the wrappers do not name) ---"
    )
    st = pstats.Stats(prof)
    st.sort_stats("cumulative").print_stats(args.top)

    out = dict(
        dataset=args.dataset,
        rows=n_rows,
        device=dev,
        total_s=total,
        label=s.label(),
        phases={
            k: dict(calls=_n[k], total_s=v, mean_s=v / max(_n[k], 1)) for k, v in rows
        },
        options={k: (list(v) if isinstance(v, tuple) else v) for k, v in opts.items()},
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
