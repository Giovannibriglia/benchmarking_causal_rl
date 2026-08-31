"""T = 16: does annealing cost anything where there is no pathology to fix?

At n = 3 the L3 re-validation read 0.917 -> 0.847, driven entirely by ONE seed
going 0.77 -> 0.54 while the other two went to 1.000. That is neither a
regression nor noise, and T = 16 fits are the cheapest available, so it is
settled by seeds rather than by argument.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--max-iter", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out", default="results/l3_validation/t16_seeds.json")
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
        and r["env"] == "CartPole-v1"
        and r["seed"] == 0
        and r["sigma"] == 1.0
    )
    s, blocks = rebuild_samples(minari.load_dataset(row["dataset_id"]), args.episodes)
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
    out = []
    for arm, kw in (("tau=1", {"temperature": 1.0}), ("annealed", {})):
        for fs in range(args.seeds):
            est = LatentClassEstimator(
                state_dim=state.shape[1],
                n_actions=int(s["a"].max()) + 1,
                proxy_names=("Z", "W"),
                device=device,
                seed=fs,
            )
            fit = est.fit(
                data, max_iter=args.max_iter, epochs=args.epochs, init="random", **kw
            )
            h = fit.hard_assignment().cpu().numpy()
            r = {
                "arm": arm,
                "fit_seed": fs,
                "recovery": float(max((h == u_ep).mean(), (h != u_ep).mean())),
                "separation_per_step": float(fit.separation_per_step),
                "initial_saturation": float(fit.initial_saturation),
                "degenerate_mechanism": bool(fit.degenerate_mechanism),
                "reached_tau_one": bool(fit.reached_tau_one),
                "converged": bool(fit.converged),
                "final_ll": float(fit.final_ll),
            }
            out.append(r)
            print(
                f"{arm:<10} fs={fs} rec={r['recovery']:.3f} "
                f"sep/step={r['separation_per_step']:.3f} "
                f"degen={r['degenerate_mechanism']} ll={r['final_ll']:.0f}",
                flush=True,
            )
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(out, indent=1))

    print("\n=== T=16, 10 seeds ===")
    for arm in ("tau=1", "annealed"):
        rs = [r["recovery"] for r in out if r["arm"] == arm]
        print(
            f"  {arm:<10} mean {np.mean(rs):.3f}  median {np.median(rs):.3f}  "
            f"min {min(rs):.3f}  n_below_0.9 {sum(1 for x in rs if x < 0.9)}/{len(rs)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
