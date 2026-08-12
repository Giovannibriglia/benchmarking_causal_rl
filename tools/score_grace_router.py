"""G4 — score the GRACE router against the ground-truth arm labels.

Walks a regime's grace ablation leaves, reads each leaf's LOGGED router
components (router_delta_a / router_delta_r / router_coverage /
ensemble_width from the last checkpoint row), applies the CALIBRATED per-env
thresholds (grace_router_reference.yaml), and scores the verdicts against the
arm label DERIVED from the leaf's (beta, sigma) path segments — the G4 gate:
macro-F1 >= 0.8 over {basic, biased, confounded}.

Scoring from logged components (not the run-time verdict string) makes the
verdict reproducible under re-calibration and covers runs that executed
before the reference existed; the run-time verdict is printed alongside for
the audit trail. Ground truth here is the config-derived arm label — the
generation-time U-derived gate metadata is never consumed (R5).

Usage:
    uv run python tools/score_grace_router.py offline_mdp [--results-root results]
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def _last_row(csv_path: Path) -> dict | None:
    try:
        rows = list(csv.DictReader(open(csv_path)))
    except OSError:
        return None
    return rows[-1] if rows else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("regime")
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--critic", default="grace", help="router-carrying arm")
    ap.add_argument("--out", default=None, help="optional CSV of per-leaf verdicts")
    args = ap.parse_args()

    from src.benchmarking.regime_report import iter_leaves
    from src.rl.offline.grace.router import RegimeRouter

    per_label: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    leaves_out = []
    n_scored = 0
    routers: dict[str, RegimeRouter] = {}
    for leaf in iter_leaves(args.results_root, args.regime):
        if leaf["critic"] != args.critic:
            continue
        row = _last_row(Path(leaf["path"]) / "critic_ablation_metrics.csv")
        if row is None:
            continue
        env = leaf["env"]
        if env not in routers:
            routers[env] = RegimeRouter.from_reference(env)
        stats = {}
        for key, col in (
            ("delta_a", "router_delta_a"),
            ("delta_r", "router_delta_r"),
            ("coverage", "router_coverage"),
            ("width", "ensemble_width"),
        ):
            val = row.get(col, "")
            if val not in ("", None):
                try:
                    stats[key] = float(val)
                except ValueError:
                    pass
        verdict = routers[env].verdict(stats)
        truth = leaf["arm"]
        per_label[truth][verdict.label] += 1
        n_scored += 1
        leaves_out.append(
            {
                "path": str(leaf["path"]),
                "env": env,
                "algo": leaf["algo"],
                "seed": leaf["seed"],
                "truth": truth,
                "verdict": verdict.label,
                "serve": verdict.serve,
                "runtime_verdict": row.get("router_verdict", ""),
                **{f"stat_{k}": v for k, v in stats.items()},
            }
        )

    labels = ("basic", "biased", "confounded")
    print(f"[g4] {args.regime}: {n_scored} grace leaves scored")
    print("[g4] confusion (rows = truth, cols = verdict):")
    all_verdicts = sorted({v for d in per_label.values() for v in d})
    header = "truth".ljust(12) + "".join(v.ljust(14) for v in all_verdicts)
    print("  " + header)
    for t in labels:
        line = t.ljust(12) + "".join(
            str(per_label[t].get(v, 0)).ljust(14) for v in all_verdicts
        )
        print("  " + line)
    f1s = []
    for lab in labels:
        tp = per_label[lab].get(lab, 0)
        fn = sum(per_label[lab].values()) - tp
        fp = sum(per_label[t].get(lab, 0) for t in per_label if t != lab)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
        print(f"[g4] {lab}: precision={prec:.3f} recall={rec:.3f} f1={f1:.3f}")
    macro = sum(f1s) / len(f1s) if f1s else 0.0
    print(
        f"[g4] macro-F1 = {macro:.3f}  (gate G4: >= 0.8 -> {'PASS' if macro >= 0.8 else 'FAIL'})"
    )

    if args.out and leaves_out:
        with open(args.out, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(leaves_out[0].keys()))
            writer.writeheader()
            writer.writerows(leaves_out)
        print(f"[g4] per-leaf verdicts -> {args.out}")


if __name__ == "__main__":
    main()
