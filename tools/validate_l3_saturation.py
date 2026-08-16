"""L3 re-validation on the REAL dataset that failed, not on a synthetic harness.

Rule S5, which this exercise re-earned. A synthetic long-episode fixture built
to reproduce the D-D Acrobot failure **does not reproduce it**: at T = 500 it
recovers 1.000 with no annealing at all, while the real arm sat at 0.53 in 6 of
6 fits. Validating the tempering fix against that fixture would have measured
nothing. So the grid here is real datasets, chosen by their measured episode
length: CartPole s0 (T = 18), Acrobot s0 (T = 150), Acrobot s1 (T = 500).

Each is fitted twice from the SAME random initialisation and the same budget:
``temperature=1.0`` (the pre-fix estimator) against the annealed default. The
comparison is against the logged ``U`` -- ground truth, never a second estimator.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--max-iter", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--fit-seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out", default="results/l3_validation/report.json")
    args = ap.parse_args()

    os.environ.setdefault("MINARI_DATASETS_PATH", str(Path.home() / ".minari-grace-v2"))
    import minari
    import torch
    from src.rl.offline.grace.estimator import EpisodeData, LatentClassEstimator

    from tools.recertify_diagram_arms import rebuild_samples

    device = "cuda" if torch.cuda.is_available() else "cpu"
    grid = [("CartPole-v1", 0), ("Acrobot-v1", 0), ("Acrobot-v1", 1)]
    recert = {
        (r["env"], r["seed"]): r
        for r in json.loads(Path("results/vb_recertification/report.json").read_text())
        if r["cell"] == "d_d" and r["sigma"] == 1.0
    }

    def accuracy(h, t):
        h, t = np.asarray(h), np.asarray(t)
        return float(max((h == t).mean(), (h != t).mean()))

    out = []
    for env, sd in grid:
        ds = minari.load_dataset(recert[(env, sd)]["dataset_id"])
        s, blocks = rebuild_samples(ds, args.episodes)
        state = np.concatenate([b.observations[:-1] for b in blocks], axis=0)
        ep = s["episode"]
        u_ep = np.array([s["u"][ep == e][0] for e in np.unique(ep)], dtype=np.int64)
        T = ep.size / np.unique(ep).size

        def t(x, dtype=torch.float32):
            return torch.tensor(x, dtype=dtype, device=device)

        data = EpisodeData(
            state=t(state),
            action=t(s["a"], torch.long),
            reward=t(s["r"]),
            episode_ids=t(ep, torch.long),
            proxy={"Z": t(s["z"]), "W": t(s["w"])},
        )
        for tag, kw in (("tau=1 (pre-fix)", {"temperature": 1.0}), ("annealed", {})):
            for fs in args.fit_seeds:
                est = LatentClassEstimator(
                    state_dim=state.shape[1],
                    n_actions=int(s["a"].max()) + 1,
                    proxy_names=("Z", "W"),
                    device=device,
                    seed=fs,
                )
                fit = est.fit(
                    data,
                    max_iter=args.max_iter,
                    epochs=args.epochs,
                    init="random",
                    **kw,
                )
                rec = {
                    "env": env,
                    "seed": sd,
                    "T": round(float(T), 1),
                    "arm": tag,
                    "fit_seed": fs,
                    "recovery": accuracy(fit.hard_assignment().cpu().numpy(), u_ep),
                    "separability": float(fit.separability()),
                    "separation_per_step": float(fit.separation_per_step),
                    "initial_saturation": float(fit.initial_saturation),
                    "saturated_at_init": bool(fit.saturated_at_init),
                    "reached_tau_one": bool(fit.reached_tau_one),
                    "n_anneal": int(fit.n_anneal),
                    "converged": bool(fit.converged),
                    "backtrack_exhausted": bool(fit.backtrack_exhausted),
                    "final_ll": float(fit.final_ll),
                }
                out.append(rec)
                print(
                    f"{env:<12} s{sd} T={T:>5.0f} {tag:<16} fs={fs} "
                    f"rec={rec['recovery']:.3f} sat0={rec['initial_saturation']:.3f} "
                    f"sep/step={rec['separation_per_step']:.3f} "
                    f"sep(tel)={rec['separability']:.4f} "
                    f"tau1={rec['reached_tau_one']} conv={rec['converged']}",
                    flush=True,
                )
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                Path(args.out).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
