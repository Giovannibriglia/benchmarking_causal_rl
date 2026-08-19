"""Reproduce the Acrobot optimiser stall on a SUBSAMPLE (S11), then diagnose it.

The stall at production scale: 11 iterations, ~3 of them at tau = 1, then no
improving step at any step size tried -- **while still improving by 8.3x
tolerance**. Not stationary, not converged. Twelve minutes per attempt is far too
slow to debug on.

S11 says cut DATA, not BUDGET. A few hundred real Acrobot episodes preserve
episode length (which drives saturation), the reward support (which drives the
mechanism type), and the objective's curvature (which drives the line search) --
so the pathology should reproduce. The optimisation budget is deliberately
GENEROUS here: a truncated fit would reproduce a stall for the wrong reason and
look like a result.

If it reproduces on a tenth of the data, the blocker is the optimiser. If it does
not, something scales with the data that we have not identified -- and that is
the more interesting answer.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="Acrobot-v1")
    ap.add_argument("--dataset-seed", type=int, default=0)
    ap.add_argument("--episodes", nargs="+", type=int, default=[300])
    ap.add_argument("--max-iter", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--max-backtracks", type=int, default=3)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default="results/l3_validation/acrobot_stall.json")
    args = ap.parse_args()

    os.environ.setdefault("MINARI_DATASETS_PATH", str(Path.home() / ".minari-grace-v2"))
    import minari
    import torch
    from src.rl.offline.grace.estimator import EpisodeData, LatentClassEstimator

    from tools.recertify_diagram_arms import rebuild_samples

    device = "cuda" if torch.cuda.is_available() else "cpu"
    row = next(
        r
        for r in json.loads(Path("results/vb_recertification/report.json").read_text())
        if r["cell"] == "d_d"
        and r["env"] == args.env
        and r["seed"] == args.dataset_seed
        and r["sigma"] == 1.0
    )
    out = []
    for n_ep in args.episodes:
        s, blocks = rebuild_samples(minari.load_dataset(row["dataset_id"]), n_ep)
        state = np.concatenate([b.observations[:-1] for b in blocks], axis=0)
        ep = s["episode"]
        u_ep = np.array([s["u"][ep == e][0] for e in np.unique(ep)], dtype=np.int64)

        def t(x, dtype=torch.float32):
            return torch.tensor(x, dtype=dtype, device=device)

        data = EpisodeData(
            state=t(state),
            action=t(s["a"], torch.long),
            reward=t(s["r"]),
            episode_ids=t(ep, torch.long),
            proxy={"Z": t(s["z"]), "W": t(s["w"])},
        )
        est = LatentClassEstimator(
            state_dim=state.shape[1],
            n_actions=int(s["a"].max()) + 1,
            proxy_names=("Z", "W"),
            device=device,
            seed=0,
        )
        import time

        t0 = time.time()
        fit = est.fit(
            data,
            max_iter=args.max_iter,
            epochs=args.epochs,
            init="proxy",
            max_backtracks=args.max_backtracks,
            verbose=args.verbose,
        )
        dt = time.time() - t0
        h = fit.hard_assignment().cpu().numpy()
        hist = list(fit.log_likelihood)
        rel = [round((b - a) / max(abs(b), 1e-12), 8) for a, b in zip(hist, hist[1:])]
        rec = {
            "episodes": int(np.unique(ep).size),
            "transitions": int(ep.size),
            "mean_T": round(float(ep.size / np.unique(ep).size), 1),
            "seconds": round(dt, 1),
            "n_iter": fit.n_iter,
            "n_anneal": fit.n_anneal,
            "converged": bool(fit.converged),
            "stationary": bool(fit.stationary),
            "finished": bool(fit.finished),
            "backtrack_exhausted": bool(fit.backtrack_exhausted),
            "backtracks": fit.backtracks,
            "backtracks_per_iter": list(fit.backtracks_per_iter),
            "lr_reductions": fit.lr_reductions,
            "final_lr_scale": fit.final_lr_scale,
            # THE discriminator: was the best available step rejected because it
            # made things numerically-negligibly worse (a fixed point) or really
            # worse (an optimiser that cannot ascend)?
            "rejected_step_rel": fit.rejected_step_rel,
            "recovery": float(max((h == u_ep).mean(), (h != u_ep).mean())),
            "rel_deltas": rel,
        }
        out.append(rec)
        print(json.dumps(rec, indent=1), flush=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
