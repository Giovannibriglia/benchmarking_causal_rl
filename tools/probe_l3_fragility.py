"""Does warm-start reduce the fit's SENSITIVITY, not just its noise?

The amplification finding: ulp-level evaluation noise reaches the backtrack /
convergence / stationarity decisions -- each discrete on a continuous input --
and under restart-EM grew into 100+-nat path divergence. Deterministic kernels
removed the run-to-run NOISE; this probe measures the remaining SENSITIVITY:
with everything deterministic, perturb ONE input element by 1e-7 and measure
how far the fit moves, under GEM and under restart-EM on the same data.

If continuation (GEM) does not shrink the response, the optimiser is still
chaotic in its inputs and V-D's thousands of fits inherit that -- worth
knowing before they run.

Also serves as the identical-pair stability check for the new stack: each
condition's baseline runs TWICE and must be bitwise identical (n_iter
included) before the perturbed run is read against it.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="CartPole-v1")
    ap.add_argument("--dataset-seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--max-iter", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--jitter", type=float, default=1e-7)
    ap.add_argument("--out", default="results/cost/l3_fragility.json")
    args = ap.parse_args()

    os.environ.setdefault("MINARI_DATASETS_PATH", str(Path.home() / ".minari-grace-v2"))
    import minari
    import numpy as np
    import torch
    from src.rl.offline.grace.estimator import EpisodeData, LatentClassEstimator

    from tools.measure_consolidate_share import _state_hash
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

    def t(x, dtype=torch.float32):
        return torch.tensor(x, dtype=dtype, device=device)

    def make_data(jitter: float):
        st = state.copy()
        st[0, 0] += jitter  # ONE element, the minimal perturbation
        return EpisodeData(
            state=t(st),
            action=t(s["a"], torch.long),
            reward=t(s["r"]),
            episode_ids=t(ep, torch.long),
            proxy={"Z": t(s["z"]), "W": t(s["w"])},
        )

    def one_fit(data, warm: bool):
        est = LatentClassEstimator(
            state_dim=state.shape[1],
            n_actions=int(s["a"].max()) + 1,
            proxy_names=("Z", "W"),
            device=device,
            seed=0,
        )
        fit = est.fit(
            data,
            max_iter=args.max_iter,
            epochs=args.epochs,
            init="proxy",
            consolidate=False,
            warm_start=warm,
        )
        h = fit.hard_assignment().cpu().numpy()
        return {
            "n_iter": fit.n_iter,
            "final_ll": float(fit.final_ll),
            "recovery": float(max((h == u_ep).mean(), (h != u_ep).mean())),
            "backtracks": fit.backtracks,
            "state_sha256_ex_ewc": _state_hash(est.model.state_dict()),
        }

    base_data, jit_data = make_data(0.0), make_data(args.jitter)
    out = {
        "env": args.env,
        "episodes": args.episodes,
        "jitter": args.jitter,
        "modes": {},
    }
    for name, warm in (("gem", True), ("restart-EM", False)):
        a = one_fit(base_data, warm)
        b = one_fit(base_data, warm)
        stable = a == b
        print(
            f"  {name:>10} baseline x2: {'BITWISE STABLE' if stable else 'UNSTABLE'}"
            f"  n_iter={a['n_iter']}  ll={a['final_ll']:.4f}  "
            f"rec={a['recovery']:.3f}  backtracks={a['backtracks']}",
            flush=True,
        )
        j = one_fit(jit_data, warm)
        dll = abs(j["final_ll"] - a["final_ll"])
        print(
            f"  {name:>10} jitter {args.jitter:g}: d_ll={dll:.6f} nats  "
            f"d_n_iter={j['n_iter'] - a['n_iter']}  "
            f"params {'SAME' if j['state_sha256_ex_ewc'] == a['state_sha256_ex_ewc'] else 'MOVED'}",
            flush=True,
        )
        out["modes"][name] = {
            "baseline": a,
            "baseline_repeat_bitwise": stable,
            "jittered": j,
            "delta_ll": dll,
        }
    g, r = out["modes"]["gem"]["delta_ll"], out["modes"]["restart-EM"]["delta_ll"]
    print(
        f"\n  SENSITIVITY to a 1e-7 single-element input perturbation: "
        f"GEM {g:.4f} nats vs restart-EM {r:.4f} nats"
        + (
            "  -> continuation damps it"
            if g < r / 3
            else "  -> continuation does NOT damp it; the optimiser path is chaotic in its inputs either way"
        )
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
