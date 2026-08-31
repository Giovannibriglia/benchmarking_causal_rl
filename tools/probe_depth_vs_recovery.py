"""Did the line-search DEPTH change move the sweep's headline, or the S1c fix?

The post-S1c sweep's WITHOUT arm shifted at the weak end (CartPole s1 d010
0.855 -> 0.682; Acrobot s1 d010 0.990 -> 0.527). That arm declares NO
proxies, so the S1c likelihood correction cannot reach it, and the per-node
M-step is order-for-order identical when there are no proxy nodes -- which
leaves ``max_backtracks`` 6 -> 10 as the only candidate.

Attribution matters: the depth was raised on a d100 measurement where
recovery was 1.0000 at BOTH depths, i.e. where it was demonstrably
innocuous. If it moves recovery on the hard weak-end fits, it is not a
safety limit there but a knob that moves the headline -- which is a ruling
for the author, not a default to keep.

Runs the WITHOUT arm at both depths on the ablation's exact configuration
(400 episodes, max_iter 12, epochs 30, 3 fit seeds, best-LL selection).
CartPole first because it is minutes rather than hours.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MINARI_DATASETS_PATH", os.path.expanduser("~/.minari-grace-v2"))

DEPTHS = (6, 10)
ROWS = (("d010", "CartPole-v1", 1), ("d005", "CartPole-v1", 1))
FIT_SEEDS = (0, 1, 2)
EPISODES, MAX_ITER, EPOCHS = 400, 12, 30


def main() -> int:
    import minari
    import numpy as np
    import torch
    from src.rl.offline.grace.estimator import EpisodeData, LatentClassEstimator

    from tools.recertify_diagram_arms import rebuild_samples

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = {
        (r["cell"], r["env"], r["seed"]): r["dataset_id"]
        for r in json.loads(Path("results/dd_sweep_generation/report.json").read_text())
    }
    out = []
    for tag, env, sd in ROWS:
        did = gen[(f"d_d_sweep_{tag}", env, sd)]
        s, blocks = rebuild_samples(minari.load_dataset(did), EPISODES)
        state = np.concatenate([b.observations[:-1] for b in blocks], axis=0)
        t = lambda x, dt=torch.float32: torch.tensor(x, dtype=dt, device=device)
        data = EpisodeData(
            state=t(state),
            action=t(s["a"], torch.long),
            reward=t(s["r"]),
            episode_ids=t(s["episode"], torch.long),
        )  # WITHOUT arm: no proxies declared
        ep = s["episode"]
        uniq = np.unique(ep)
        u_ep = torch.tensor(
            np.array([s["u"][ep == e][0] for e in uniq]), device=device
        ).long()
        for depth in DEPTHS:
            per_seed = []
            t0 = time.time()
            for fs in FIT_SEEDS:
                est = LatentClassEstimator(
                    state_dim=state.shape[1],
                    n_actions=int(s["a"].max()) + 1,
                    proxy_names=(),
                    device=device,
                    seed=fs,
                )
                fit = est.fit(
                    data,
                    init="random",
                    max_iter=MAX_ITER,
                    epochs=EPOCHS,
                    max_backtracks=depth,
                )
                hard = fit.hard_assignment().reshape(-1)
                acc = max(
                    float((hard == u_ep).float().mean()),
                    float((hard != u_ep).float().mean()),
                )
                per_seed.append(
                    {
                        "fit_seed": fs,
                        "recovery": round(acc, 4),
                        "final_ll": round(float(fit.final_ll), 2),
                        "n_iter": int(fit.n_iter),
                        "backtracks": int(fit.backtracks),
                        "converged": bool(fit.converged),
                        "exhausted": bool(fit.backtrack_exhausted),
                    }
                )
            best = max(per_seed, key=lambda p: p["final_ll"])  # the tool's rule
            rec = {
                "cell": tag,
                "env": env,
                "seed": sd,
                "arm": "without",
                "max_backtracks": depth,
                "recovery_best_ll": best["recovery"],
                "recovery_mean": round(
                    float(np.mean([p["recovery"] for p in per_seed])), 4
                ),
                "per_seed": per_seed,
                "seconds": round(time.time() - t0, 1),
            }
            out.append(rec)
            Path("results/depth_vs_recovery.json").write_text(json.dumps(out, indent=1))
            print(
                f"  {tag} {env} s{sd} without depth={depth:<3} "
                f"recovery(best-LL)={best['recovery']:.4f} "
                f"mean={rec['recovery_mean']:.4f} "
                f"backtracks={[p['backtracks'] for p in per_seed]} "
                f"({rec['seconds']:.0f}s)",
                flush=True,
            )
    print("DEPTH-VS-RECOVERY PROBE COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
