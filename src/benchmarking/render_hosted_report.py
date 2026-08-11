"""Cross-arm report for a hosted-dataset behavior-policy sweep.

Reads the leaves a ``hosted_sweep`` run wrote under
``results/{regime}/{simulation}/{arm}/{env}/{algo}/{seed}/`` and renders the
family comparison — the hosted analog of the classical report's arm axis:

  * ``report/aggregate.csv``  — one row per (arm, env, algo, seed): final and
    best ``eval_return_mean`` from the leaf's eval curve;
  * ``report/summary.csv``    — mean/sd over seeds per (arm, env, algo);
  * ``report/{regime}_{simulation}_{env}.png`` — grouped bars per env: x = arm
    (manifest order, i.e. the behavior-quality axis), one bar per algo, height
    = final return (mean over seeds), error bar = sd.

Run:
    uv run python -m src.benchmarking.render_hosted_report <regime> \
        --simulation <simulation> [--results-root results] [--metric final|best]
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

# Fixed categorical order (validated palette — see docs/minari_adoption_report.md
# tooling notes); color follows the algo slot, never the arm.
_PALETTE = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
]


def _leaf_stats(leaf: Path) -> tuple[float, float] | None:
    f = leaf / "eval_metrics.csv"
    if not f.exists():
        return None
    vals = [float(r["eval_return_mean"]) for r in csv.DictReader(f.open())]
    if not vals:
        return None
    return vals[-1], max(vals)


def render(
    regime: str,
    simulation: str,
    results_root: str = "results",
    metric: str = "final",
) -> Path:
    family = Path(results_root) / regime / simulation
    manifest = json.loads((family / "manifest.json").read_text())
    arms, envs, algos, seeds = (
        manifest["arms"],
        manifest["envs"],
        manifest["algos"],
        manifest["seeds"],
    )
    report = family / "report"
    report.mkdir(exist_ok=True)

    rows = []  # (arm, env, algo, seed, final, best)
    for arm in arms:
        for env in envs:
            env_tag = env.replace("/", "-")
            for algo in algos:
                for seed in seeds:
                    stats = _leaf_stats(family / arm / env_tag / algo / f"seed{seed}")
                    if stats is None:
                        print(
                            f"[render_hosted_report] MISSING leaf: {arm}/{env_tag}/{algo}/seed{seed}"
                        )
                        continue
                    rows.append((arm, env, algo, seed, *stats))
    if not rows:
        raise SystemExit(f"No complete leaves under {family}/ — run the sweep first.")

    with (report / "aggregate.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "env", "algo", "seed", "final_return", "best_return"])
        w.writerows(rows)

    idx = 4 if metric == "final" else 5
    summary = {}  # (arm, env, algo) -> (mean, sd, n)
    with (report / "summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "arm",
                "env",
                "algo",
                "final_mean",
                "final_sd",
                "best_mean",
                "best_sd",
                "n_seeds",
            ]
        )
        for arm in arms:
            for env in envs:
                for algo in algos:
                    sel = [
                        r for r in rows if r[0] == arm and r[1] == env and r[2] == algo
                    ]
                    if not sel:
                        continue
                    fin = [r[4] for r in sel]
                    bst = [r[5] for r in sel]
                    sd = statistics.pstdev
                    w.writerow(
                        [
                            arm,
                            env,
                            algo,
                            f"{statistics.mean(fin):.4f}",
                            f"{sd(fin):.4f}",
                            f"{statistics.mean(bst):.4f}",
                            f"{sd(bst):.4f}",
                            len(sel),
                        ]
                    )
                    m = [r[idx] for r in sel]
                    summary[(arm, env, algo)] = (statistics.mean(m), sd(m), len(sel))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    for env in envs:
        env_tag = env.replace("/", "-")
        fig, ax = plt.subplots(figsize=(max(6.8, 1.4 + 2.2 * len(arms)), 4.4), dpi=150)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        n_a = len(algos)
        width = min(0.18, 0.8 / n_a)
        xs = np.arange(len(arms))
        for j, algo in enumerate(algos):
            c = _PALETTE[j % len(_PALETTE)]
            offs = (j - (n_a - 1) / 2) * (width + 0.02)
            means = [
                summary.get((arm, env, algo), (float("nan"), 0, 0))[0] for arm in arms
            ]
            sds = [summary.get((arm, env, algo), (0, 0, 0))[1] for arm in arms]
            ax.bar(
                xs + offs,
                means,
                width,
                yerr=sds,
                capsize=3,
                color=c,
                label=algo,
                error_kw={"ecolor": "#3a3a38", "elinewidth": 1},
            )
        ax.set_xticks(xs, arms)
        ax.set_xlabel("behavior-policy arm (dataset)", color="#3a3a38")
        ax.set_ylabel(
            f"{metric} eval return (mean ± sd over {len(seeds)} seeds)",
            color="#3a3a38",
            fontsize=9,
        )
        ax.set_title(f"{regime}/{simulation} — {env}", color="#1a1a19", fontsize=11)
        ax.grid(True, axis="y", color="#e8e7e0", linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#c3c2b7")
        ax.tick_params(colors="#3a3a38", labelsize=9)
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(report / f"{regime}_{simulation}_{env_tag}.png")
        plt.close(fig)
    print(
        f"[render_hosted_report] wrote {report}/ (aggregate.csv, summary.csv, {len(envs)} figure(s))"
    )
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("regime")
    p.add_argument("--simulation", required=True)
    p.add_argument("--results-root", default="results")
    p.add_argument("--metric", choices=["final", "best"], default="final")
    args = p.parse_args()
    render(args.regime, args.simulation, args.results_root, args.metric)


if __name__ == "__main__":
    main()
