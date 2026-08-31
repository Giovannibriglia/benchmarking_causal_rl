"""T = 500: which fix actually did the work — tempering, or discrete R?

Two changes landed between the 0.563 -> 0.990 result and now, and the
re-validation measured only the first. **Tempering may have been compensating
for a degenerate reward density**, in which case discrete R alone would fix
T = 500 and the anneal is doing less than advertised. Attributing the fix to the
wrong cause is the failure mode here, so the two are separated:

    {tau=1, annealed} x {MDN R, categorical R}, three seeds each.

Read as:
  * categorical alone recovers at tau = 1  -> tempering is NOT what fixed T=500;
  * only the combination works             -> both are load-bearing;
  * tempering alone works under MDN R      -> the original result stands.

Run on the corrected ceil(log2 tau0) schedule throughout, so the annealed arm is
the current one and not the budget-tied schedule the original used.
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
    ap.add_argument("--dataset-seed", type=int, default=1)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--max-iter", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--fit-seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out", default="results/l3_validation/t500_2x2.json")
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
    s, blocks = rebuild_samples(minari.load_dataset(row["dataset_id"]), args.episodes)
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
    print(f"{args.env} s{args.dataset_seed}  T={T:.0f}  n={ep.size}", flush=True)

    out = []
    for rmech in ("mdn", "auto"):
        for tag, kw in (("tau=1", {"temperature": 1.0}), ("annealed", {})):
            for fs in args.fit_seeds:
                est = LatentClassEstimator(
                    state_dim=state.shape[1],
                    n_actions=int(s["a"].max()) + 1,
                    proxy_names=("Z", "W"),
                    device=device,
                    seed=fs,
                    reward_mechanism=rmech,
                )
                fit = est.fit(
                    data,
                    max_iter=args.max_iter,
                    epochs=args.epochs,
                    init="random",
                    **kw,
                )
                h = fit.hard_assignment().cpu().numpy()
                r = {
                    "reward_mechanism": rmech,
                    "resolved": str(est.resolved_reward_mechanism),
                    "temperature_arm": tag,
                    "fit_seed": fs,
                    "recovery": float(max((h == u_ep).mean(), (h != u_ep).mean())),
                    "initial_saturation": float(fit.initial_saturation),
                    "separation_per_step": float(fit.separation_per_step),
                    "degenerate_mechanism": bool(fit.degenerate_mechanism),
                    "n_anneal": int(fit.n_anneal),
                    "reached_tau_one": bool(fit.reached_tau_one),
                    "converged": bool(fit.converged),
                    "final_ll": float(fit.final_ll),
                }
                out.append(r)
                print(
                    f"  R={r['resolved']:<15} {tag:<9} fs={fs} "
                    f"rec={r['recovery']:.3f} sat0={r['initial_saturation']:.3f} "
                    f"rungs={r['n_anneal']} degen={r['degenerate_mechanism']} "
                    f"ll={r['final_ll']:.0f}",
                    flush=True,
                )
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                Path(args.out).write_text(json.dumps(out, indent=1))

    print("\n=== T=500 2x2: which fix did the work ===")
    print(f"  {'reward':<16}{'tau=1':>12}{'annealed':>12}")
    for rmech in ("mdn", "auto"):
        cells = []
        for tag in ("tau=1", "annealed"):
            v = [
                r["recovery"]
                for r in out
                if r["reward_mechanism"] == rmech and r["temperature_arm"] == tag
            ]
            cells.append(f"{np.median(v):.3f}")
        res = next(r["resolved"] for r in out if r["reward_mechanism"] == rmech)
        print(f"  {res:<16}{cells[0]:>12}{cells[1]:>12}   (median of 3)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
