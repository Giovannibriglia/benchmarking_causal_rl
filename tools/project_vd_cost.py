"""V-D cost projection, on TOTALS.

The earlier reading -- "B = 99 is ~5.5 h serial, well under an hour six-way, so
option A may be back in reach" -- was **per constraint on one dataset**, and the
grid was never multiplied in. The grid size was always the dominant term:

    total = sum over datasets of ( constraints(diagram) * B * fit_cost(env) )

Two inputs that were previously estimated and are now measured:

* **constraints per diagram** -- read from the catalogue
  (``testable_implications`` plus the assumptions that have a testable shadow),
  per diagram rather than as a flat average, because it varies 1..5;
* **fit cost** -- the measured per-env converged/stationary fit.

**The M-step lever is reported as a RANGE, and projections use the CONSERVATIVE
end.** Measured ratios: x2.01, x3.05, x3.62, x3.92 on CartPole across
configurations, and x12.68 on Acrobot. Numerator and denominator both move with
configuration, so the latest is not "the" factor; and Acrobot's x12.68 mostly
reflects the epoch-based BASELINE paying O(n*epochs) on 8.6x the rows, which is a
property of the comparison rather than a speedup GRACE realises.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Measured M-step lever, across configurations. Range, not a point.
LEVER_MEASURED = {"cartpole": [2.01, 3.05, 3.62, 3.92], "acrobot": [12.68]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, default=99)
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--recert", default="results/vb_recertification/report.json")
    ap.add_argument(
        "--fit-minutes",
        nargs="+",
        default=["CartPole-v1=8.6"],
        help="measured per-env fit cost; envs without a measurement are reported "
        "as UNMEASURED rather than filled with a guess",
    )
    args = ap.parse_args()

    from src.rl.offline.grace.cell_graph import CATALOGUE

    fit_min = {}
    for spec in args.fit_minutes:
        k, v = spec.split("=")
        fit_min[k] = float(v)

    rows = json.loads(Path(args.recert).read_text())
    constraints = {}
    for name, g in CATALOGUE.items():
        constraints[name] = len(g.testable_implications) + sum(
            1 for a in g.assumptions if not a.untestable
        )

    print("constraints per diagram (MEASURED, not the estimated 4):")
    for d in sorted({r["diagram"] for r in rows}):
        print(f"  {d:<14} {constraints.get(d, '?')}")
    print()

    total_fits = 0
    total_min = 0.0
    unmeasured = set()
    per_cell = {}
    for r in rows:
        c = constraints.get(r["diagram"], 0)
        fits = c * args.b
        total_fits += fits
        env = r["env"]
        if env not in fit_min:
            unmeasured.add(env)
            continue
        m = fits * fit_min[env]
        total_min += m
        key = (r["cell"], env)
        per_cell[key] = per_cell.get(key, 0.0) + m

    print(f"{len(rows)} datasets, B = {args.b}")
    print(f"  total EM fits            {total_fits:,}")
    if unmeasured:
        print(f"  !! UNMEASURED envs, excluded from the total: {sorted(unmeasured)}")
        print("     the total below is a LOWER BOUND, not the projection")
    print(
        f"  serial                   {total_min / 60:,.0f} h  "
        f"({total_min / 60 / 24:,.1f} days)"
    )
    print(
        f"  at {args.parallel}-way parallel        {total_min / 60 / args.parallel:,.0f} h  "
        f"({total_min / 60 / args.parallel / 24:,.1f} days)"
    )
    print()
    print("  by cell x env (six-way hours):")
    for k, v in sorted(per_cell.items(), key=lambda kv: -kv[1]):
        print(f"    {k[0]:<11}{k[1]:<13}{v / 60 / args.parallel:8.1f}")
    print()
    print("  M-step lever, measured range (projections use the CONSERVATIVE end):")
    for env, vals in LEVER_MEASURED.items():
        print(f"    {env:<10} x{min(vals):.2f} .. x{max(vals):.2f}  from {vals}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
