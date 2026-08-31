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
    ap.add_argument("--b", type=int, default=99, help="replicates per null")
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--recert", default="results/vb_recertification/report.json")
    ap.add_argument(
        "--declarations",
        type=int,
        default=1,
        help="declarations tested per dataset. V-D's misspecification arms are "
        "wrong DECLARATIONS, not different datasets -- the same dataset is "
        "tested under several, each with its own null under its own declared "
        "model. Default 1 UNDER-counts for exactly the experiment V-D is; L5 is "
        "unstarted so the arm structure is modelled parametrically rather than "
        "invented.",
    )
    ap.add_argument(
        "--pool-seeds",
        action="store_true",
        help="LEVER A: one null per (cell, env, sigma) CONFIGURATION rather than "
        "per dataset. Seeds within a configuration are i.i.d. draws from the "
        "same generator, so under H0 their nulls are the same distribution and "
        "computing five is estimating one object five times. Licensed by a "
        "tested exchangeability check, not by the argument (see bootstrap.py).",
    )
    ap.add_argument(
        "--declaration-matrix",
        action="store_true",
        help="use the per-cell declaration counts from docs/grace_v2_vd_design.md "
        "instead of a scalar. A misspecification applies only where the thing it "
        "misspecifies EXISTS, so the count is cell-dependent: declaring an "
        "omission of an edge never declared is not a misspecification.",
    )
    ap.add_argument("--fit-minutes", nargs="+", default=["CartPole-v1=8.6"])
    ap.add_argument("--scenarios", action="store_true", help="run all three")
    args = ap.parse_args()

    from src.rl.offline.grace.cell_graph import CATALOGUE

    fit_min = {}
    for spec in args.fit_minutes:
        k, v = spec.split("=")
        fit_min[k] = float(v)

    rows = json.loads(Path(args.recert).read_text())
    constraints = {
        name: len(g.testable_implications)
        + sum(1 for a in g.assumptions if not a.untestable)
        for name, g in CATALOGUE.items()
    }

    # From docs/grace_v2_vd_design.md. Value = list of DECLARED diagrams tested
    # on that cell; the constraint count of each is the DECLARED diagram's, not
    # the data's. M2 is excluded here and scoped to a single demonstration
    # configuration, because it is undetectable in principle and running it
    # everywhere buys a repeated null result at full price.
    DECLARATION_MATRIX = {
        "d_a_null": ["D-A-null"],
        "d_b_prime": ["D-B-prime", "D-A"],
        "d_e": ["D-E", "D-A"],
        "d_d": ["D-D", "D-A", "D-D"],
    }

    def project(b, pool, declarations):
        """(total_fits, total_minutes, unmeasured_envs, per_cell_minutes)."""
        # Pooling makes the NULL a property of the configuration, so the unit of
        # work is the configuration; without it, the dataset.
        units = {}
        for r in rows:
            key = (
                (r["cell"], r["env"], r["sigma"])
                if pool
                else (r["cell"], r["env"], r["sigma"], r["seed"])
            )
            units[key] = r
        total_fits, total_min, unmeasured, per_cell = 0, 0.0, set(), {}
        for r in units.values():
            if args.declaration_matrix:
                decls = DECLARATION_MATRIX.get(r["cell"], [r["diagram"]])
                c = sum(constraints.get(d, 0) for d in decls)
                fits = c * b
            else:
                c = constraints.get(r["diagram"], 0)
                fits = c * b * declarations
            total_fits += fits
            if r["env"] not in fit_min:
                unmeasured.add(r["env"])
                continue
            m = fits * fit_min[r["env"]]
            total_min += m
            per_cell[(r["cell"], r["env"])] = (
                per_cell.get((r["cell"], r["env"]), 0.0) + m
            )
        return total_fits, total_min, unmeasured, per_cell

    print("constraints per diagram (MEASURED, not the estimated 4):")
    for d in sorted({r["diagram"] for r in rows}):
        print(f"  {d:<14} {constraints.get(d, '?')}")
    n_cfg = len({(r["cell"], r["env"], r["sigma"]) for r in rows})
    print(
        f"\n{len(rows)} datasets in {n_cfg} configurations "
        f"({len(rows) // n_cfg} seeds each)"
    )

    if not args.scenarios:
        scenarios = [("as configured", args.b, args.pool_seeds, args.declarations)]
    else:
        # LEVER A pools 5 seeds, so B per dataset drops 5x for the same
        # configuration-level precision. LEVER B then trades p-value resolution
        # for cost, with the MC error reported alongside.
        # ``b`` is the replicate count OF THE NULL. Unpooled that is per
        # dataset; pooled it is per CONFIGURATION, drawn evenly across the
        # configuration's seeds -- so LEVER A holds the null's precision fixed
        # (100 ~ 99) and divides the fits by the seed count, rather than cutting
        # precision. Conflating the two units understates A's cost by 5x, which
        # is the seed count exactly.
        scenarios = [
            ("current: B=99 per DATASET", 99, False, args.declarations),
            (
                "+A: B=100 per CONFIG (20/seed) — same precision, 5x fewer fits",
                100,
                True,
                args.declarations,
            ),
            (
                "+A+B: B=39 per CONFIG — precision traded, MC error quoted",
                39,
                True,
                args.declarations,
            ),
        ]

    for label, b, pool, decl in scenarios:
        fits, mins, unmeasured, per_cell = project(b, pool, decl)
        six = mins / 60 / args.parallel
        # When the declaration matrix is active the scalar is UNUSED -- echoing
        # it would claim a multiplier the math ignored.
        tag = (
            "declaration-matrix (1/2/2/3 per cell)"
            if args.declaration_matrix
            else f"declarations={decl}"
        )
        print(f"\n=== {label}   [{tag}] ===")
        print(f"  total EM fits          {fits:,}")
        if unmeasured:
            print(
                f"  !! UNMEASURED envs excluded: {sorted(unmeasured)} "
                "-- the figures below are a LOWER BOUND"
            )
        print(f"  serial                 {mins / 60:,.0f} h ({mins / 60 / 24:,.1f} d)")
        print(f"  {args.parallel}-way parallel        {six:,.0f} h ({six / 24:,.1f} d)")
        if b < 99:
            # MC error of a quantile from b replicates, reported rather than
            # hidden: cutting B is honest degradation only if the degradation
            # is quoted.
            print(
                f"  p-value resolution     ~1/{b + 1} = {1 / (b + 1):.3f} "
                f"(against 1/100 = 0.010 at B=99)"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
