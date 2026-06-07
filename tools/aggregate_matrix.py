#!/usr/bin/env python3
"""Validate + aggregate the discrete cells matrix sweep.

Validation (sweep-harness hardening, post-hoc part): each spec's
``causal_cells_metrics.csv`` must contain one reference row per seed, one
random row, and ``len(roles) x len(seeds) x len(tiers)`` result rows.

Aggregation: per (cell, tier, role): IQM of normalized regret over seeds with
a 95% bootstrap CI, plus probability of improvement (variant vs basic, lower
regret = better). Writes ``outputs/matrix_summary.csv``.

Usage: python -m tools.aggregate_matrix <run_dir> [<run_dir> ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from src.eval.regret import iqm, probability_of_improvement, stratified_bootstrap_iqm_ci


def validate(df: pd.DataFrame, path: str) -> None:
    res = df[df.role.isin(["basic", "variant"])]
    seeds = sorted(res.seed.dropna().unique())
    tiers = sorted(res.tier.dropna().unique())
    roles = sorted(res.role.unique())
    expected = len(roles) * len(seeds) * len(tiers)
    assert len(res) == expected, (
        f"{path}: expected {expected} result rows "
        f"({roles} x {len(seeds)} seeds x {len(tiers)} tiers), got {len(res)}"
    )
    n_ref = len(df[df.role == "reference"])
    assert n_ref == len(seeds), f"{path}: {n_ref} reference rows != {len(seeds)} seeds"
    assert len(df[df.role == "random"]) == 1, f"{path}: missing random row"
    assert res.J.notna().all(), f"{path}: NaN J values"


def main() -> None:
    frames = []
    for run_dir in sys.argv[1:]:
        csv = Path(run_dir) / "causal_cells_metrics.csv"
        df = pd.read_csv(csv)
        validate(df, str(csv))
        frames.append(df)
        print(f"validated {csv}")
    allrows = pd.concat(frames, ignore_index=True)
    res = allrows[allrows.role.isin(["basic", "variant"])]

    out = []
    for (cell, tier, role), grp in res.groupby(["cell", "tier", "role"]):
        vals = grp.normalized_regret.to_numpy(dtype=float)
        lo, hi = stratified_bootstrap_iqm_ci(vals.reshape(1, -1))
        out.append(
            {
                "cell": cell,
                "tier": tier,
                "role": role,
                "algo": grp.algo.iloc[0],
                "n_seeds": len(vals),
                "iqm_norm_regret": round(iqm(vals), 4),
                "ci_low": round(lo, 4),
                "ci_high": round(hi, 4),
                "iqm_J": round(iqm(grp.J.to_numpy(dtype=float)), 1),
            }
        )
    summary = pd.DataFrame(out).sort_values(["cell", "tier", "role"])

    # probability of improvement: variant beats basic (LOWER regret wins)
    poi = []
    for (cell, tier), grp in res.groupby(["cell", "tier"]):
        b = grp[grp.role == "basic"].normalized_regret.to_numpy(dtype=float)
        v = grp[grp.role == "variant"].normalized_regret.to_numpy(dtype=float)
        if len(b) and len(v):
            poi.append(
                {
                    "cell": cell,
                    "tier": tier,
                    "p_variant_improves": round(probability_of_improvement(-v, -b), 3),
                }
            )
    poi_df = pd.DataFrame(poi)

    outdir = Path("outputs")
    outdir.mkdir(exist_ok=True)
    summary.to_csv(outdir / "matrix_summary.csv", index=False)
    poi_df.to_csv(outdir / "matrix_prob_improvement.csv", index=False)
    print(summary.to_string(index=False))
    print()
    print(poi_df.to_string(index=False))


if __name__ == "__main__":
    main()
