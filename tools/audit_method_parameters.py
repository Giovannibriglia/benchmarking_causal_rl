"""The binding audit: does any GRACE cap actually bind? Measured, not asserted.

See docs/grace_v2_method_parameters.md for the rule and the classification.
Two parts:

1. AGGREGATE existing fit artifacts (cost runs, L3 re-validation, probes) for
   the per-run binding flags: tau1_budget_bound is new, so older artifacts are
   read through their converged/finished/backtrack fields where present.
2. The m_step_budget PLATEAU PROBE: identical fits at budgets
   100/200/400/800/1600 on one dataset per environment (subsampled real data,
   S11). The curve of final_ll against budget is the evidence; a budget on the
   flat part is a safety guard, one on the rising part is a tuning knob.
   Deterministic kernels make each point exact for its configuration.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--max-iter", type=int, default=30)
    ap.add_argument(
        "--budgets", nargs="+", type=int, default=[100, 200, 400, 800, 1600]
    )
    ap.add_argument("--out", default="results/cost/method_parameter_audit.json")
    args = ap.parse_args()

    os.environ.setdefault("MINARI_DATASETS_PATH", str(Path.home() / ".minari-grace-v2"))
    import minari
    import numpy as np
    import torch
    from src.rl.offline.grace.estimator import EpisodeData, LatentClassEstimator

    from tools.recertify_diagram_arms import rebuild_samples

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out: dict = {"aggregate": {}, "budget_probe": []}

    # ---- Part 1: aggregate existing artifacts -------------------------------
    def rows_from(path, keys):
        f = Path(path)
        if not f.exists():
            return []
        data = json.loads(f.read_text())
        rows = data if isinstance(data, list) else [data]
        got = []
        for r in rows:
            if isinstance(r, dict) and any(k in r for k in keys):
                got.append(r)
        return got

    sources = {
        "l3_validation_gem": "results/l3_validation/report_gem.json",
        "grace_fit_cost_gem": "results/cost/grace_fit_cost_gem.json",
    }
    agg = {"n_fits": 0, "converged": 0, "backtrack_exhausted": 0, "unfinished": 0}
    for name, path in sources.items():
        for r in rows_from(path, ("converged",)):
            agg["n_fits"] += 1
            agg["converged"] += bool(r.get("converged"))
            agg["backtrack_exhausted"] += bool(r.get("backtrack_exhausted"))
            fin = r.get("finished")
            if fin is None:
                fin = bool(r.get("converged")) or bool(r.get("stationary"))
            agg["unfinished"] += not fin
    out["aggregate"] = agg
    print(
        f"  existing artifacts: {agg['n_fits']} fits, {agg['converged']} converged, "
        f"{agg['backtrack_exhausted']} backtrack-exhausted, "
        f"{agg['unfinished']} unfinished (cap candidates)",
        flush=True,
    )

    # ---- Part 2: the budget plateau probe -----------------------------------
    recert = {
        (r["env"], r["seed"]): r
        for r in json.loads(Path("results/vb_recertification/report.json").read_text())
        if r["cell"] == "d_d" and r["sigma"] == 1.0
    }
    for env in ("CartPole-v1", "Acrobot-v1"):
        s, blocks = rebuild_samples(
            minari.load_dataset(recert[(env, 0)]["dataset_id"]), args.episodes
        )
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
        for budget in args.budgets:
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
                init="proxy",
                consolidate=False,
                m_step_budget=budget,
                batch_size=4096,
            )
            h = fit.hard_assignment().cpu().numpy()
            rec = {
                "env": env,
                "m_step_budget": budget,
                "final_ll": float(fit.final_ll),
                "n_iter": fit.n_iter,
                "converged": bool(fit.converged),
                "finished": bool(fit.finished),
                "tau1_budget_bound": bool(fit.tau1_budget_bound),
                "backtracks": fit.backtracks,
                "backtrack_exhausted": bool(fit.backtrack_exhausted),
                "recovery": float(max((h == u_ep).mean(), (h != u_ep).mean())),
            }
            out["budget_probe"].append(rec)
            print(
                f"  {env:<12} budget={budget:>5}  ll={rec['final_ll']:.2f}  "
                f"n_iter={rec['n_iter']}  conv={rec['converged']}  "
                f"budget_bound={rec['tau1_budget_bound']}  "
                f"backtracks={rec['backtracks']}  rec={rec['recovery']:.3f}",
                flush=True,
            )
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(out, indent=1))

    print("\n  READ: the ll-vs-budget curve per env. 400 on the flat part ->")
    print("  safety guard; on the rising part -> tuning knob (grow or derive).")
    print("  Any tau1_budget_bound=True or backtrack_exhausted=True above is a")
    print("  cap binding and goes in table 4 of the audit doc.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
