"""RE ladder — dataset-size sweep for the G3 sample-efficiency gate (D5).

The regime sweep's axis is (beta, sigma); dataset size (``rollout_episodes``,
"RE") is a fixed budget. This driver reuses ``run_cell`` VERBATIM per rung
with ``rollout_episodes`` overridden, writing each rung under its own results
root so every existing reader (regime_report, render_regime_report) works
unchanged per rung:

    <results-root>/re_{0300|1000|3000}/{regime}/beta_*_sigma_*/{env}/{algo}/{critic}/{seed}/

G3 (approved thresholds): tau = the observational arm's value-MSE at the
largest RE; the adaptive arm must reach <= tau at <= half the data (log-RE
interpolation). The scorer lives in the reporting step; this file only runs
the ladder and stamps a manifest with full provenance.

Usage (the Night-2 E4 block):
    uv run python tools/re_ladder.py \
        reproducibility/rl_regimes/offline_mdp/critic_ablation.yaml \
        --re 300 1000 3000 \
        --points beta_000_sigma_000 beta_000_sigma_100 \
        --envs CartPole-v1 --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.benchmarking.regime_sweep import load_sweep_spec, run_cell


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cell_yaml", help="cell critic_ablation.yaml to ladder")
    ap.add_argument(
        "--re",
        type=int,
        nargs="+",
        default=[300, 1000, 3000],
        help="rollout_episodes rungs (default: 300 1000 3000)",
    )
    ap.add_argument(
        "--points",
        nargs="+",
        default=["beta_000_sigma_000", "beta_000_sigma_100"],
        help="param dirnames to run per rung (default: basic + sigma=1.0)",
    )
    ap.add_argument("--envs", nargs="+", default=None)
    ap.add_argument("--algos", nargs="+", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--results-root", default="results_re_ladder")
    ap.add_argument("--dataset-prefix", default="re_ladder")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    spec = load_sweep_spec(args.cell_yaml)
    root = Path(args.results_root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "cell_yaml": str(args.cell_yaml),
        "regime": spec.regime,
        "re_rungs": list(args.re),
        "points": list(args.points),
        "envs": args.envs or spec.envs,
        "algos": args.algos or spec.algos,
        "seeds": args.seeds if args.seeds is not None else spec.seeds,
        "rungs": {},
    }
    for re_val in args.re:
        rung_root = root / f"re_{re_val:04d}"
        print(f"[re_ladder] rung RE={re_val} -> {rung_root}", flush=True)
        written = run_cell(
            args.cell_yaml,
            results_root=rung_root,
            dataset_prefix=f"{args.dataset_prefix}_re{re_val}",
            envs=args.envs,
            algos=args.algos,
            seeds=args.seeds,
            budget_overrides={"rollout_episodes": int(re_val)},
            device=args.device,
            points=list(args.points),
        )
        manifest["rungs"][str(re_val)] = {
            "results_root": str(rung_root),
            "n_leaves": len(written),
        }
        (root / "ladder_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[re_ladder] done — manifest at {root / 'ladder_manifest.json'}")


if __name__ == "__main__":
    main()
