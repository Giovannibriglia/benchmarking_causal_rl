"""Does the line-search depth still never bind under the CORRECTED likelihood?

S10 in force: ``max_backtracks = 6`` was measured never-binding at the
production configuration -- under the PRE-S1c likelihood, where the proxy
channel carried T-fold weight. Removing that term changes the objective's
curvature, and d100 Acrobot s1 (T = 500) now exhausts at 48 backtracks by
iteration 9 while still MONOTONE and recovering 1.0000. Monotone + exhausted
+ correct is the signature of a budget that binds, not of a broken fit -- but
that is a claim to measure, not to assert.

Sweeps the depth on the row that binds, and on a CartPole row that does not
(the control: a deeper budget must not change a fit that already converges).
Reports recovery, end state and final ll per depth. No fix is applied here.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MINARI_DATASETS_PATH", os.path.expanduser("~/.minari-grace-v2"))

DEPTHS = (6, 10, 14)
ROWS = (("Acrobot-v1", 1), ("CartPole-v1", 0))
FK = dict(max_iter=30, m_step_budget=400, batch_size=4096)


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
    for env, sd in ROWS:
        did = gen[("d_d_sweep_d100", env, sd)]
        s, blocks = rebuild_samples(minari.load_dataset(did), 10_000)
        state = np.concatenate([b.observations[:-1] for b in blocks], axis=0)
        t = lambda x, dt=torch.float32: torch.tensor(x, dtype=dt, device=device)
        proxy = {"Z": t(s["z"]), "W": t(s["w"])}
        if s["v"].size:
            proxy["V"] = t(s["v"])
        data = EpisodeData(
            state=t(state),
            action=t(s["a"], torch.long),
            reward=t(s["r"]),
            episode_ids=t(s["episode"], torch.long),
            proxy=proxy,
        )
        ep = s["episode"]
        uniq = np.unique(ep)
        u_ep = torch.tensor(
            np.array([s["u"][ep == e][0] for e in uniq]), device=device
        ).long()
        for depth in DEPTHS:
            est = LatentClassEstimator(
                state_dim=state.shape[1],
                n_actions=int(s["a"].max()) + 1,
                proxy_names=tuple(proxy),
                device=device,
                seed=0,
            )
            t0 = time.time()
            fit = est.fit(data, init="proxy", max_backtracks=depth, **FK)
            hard = fit.hard_assignment().reshape(-1)
            acc = max(
                float((hard == u_ep).float().mean()),
                float((hard != u_ep).float().mean()),
            )
            rec = {
                "env": env,
                "seed": sd,
                "max_backtracks": depth,
                "recovery": round(acc, 4),
                "n_iter": int(fit.n_iter),
                "converged": bool(fit.converged),
                "stationary": bool(fit.stationary),
                "finished": bool(fit.finished),
                "monotone": bool(fit.monotone),
                "backtracks": int(fit.backtracks),
                "backtrack_exhausted": bool(fit.backtrack_exhausted),
                "final_ll": round(float(fit.final_ll), 3),
                "seconds": round(time.time() - t0, 1),
            }
            out.append(rec)
            print(
                f"  {env:<12} s{sd} depth={depth:<3} rec={acc:.4f} "
                f"n_iter={rec['n_iter']:<3} finished={rec['finished']} "
                f"exhausted={rec['backtrack_exhausted']} ll={rec['final_ll']} "
                f"({rec['seconds']:.0f}s)",
                flush=True,
            )
            Path("results/s1c_backtrack_depth.json").write_text(
                json.dumps(out, indent=1)
            )
    print("BACKTRACK DEPTH PROBE COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
