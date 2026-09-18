"""C1 cost projection — every input measured, reported BEFORE launch (§7.4).

    uv run python tools/project_c1_cost.py --fit-hours F [--speedup S] [--contended]

Inputs (defaults are the measured values, cited):
  ratios       k=1 augmented fit / unaugmented = 1.71, k=2 = 1.84 (cost probe,
               d100 sigma=0 s0, 49k rows, GPU; 2026-09-03)
  training     per-run medians from the pilot's leaves (results/e1 provenance):
               cql base 391-512 s, iql base 625-865 s (quiet vs contended);
               base cells run the critic axis (4 critics tmdp / 3 tpomdp) in
               ONE run; multiplier for the extra critic heads is a guess
               (1.3x) and is flagged as such.
  fits         per dataset seed, under the content-addressed cache:
               tmdp: dmdp k0 (1.0) + k1 sufficient? (1.71) = 2.71; dpomdp: k0,k1 hits = 0
               tpomdp: dmdp 2.71; dpomdp: k0,k1 hits + k2 (1.84) = 1.84
               -> 3 x (2.71 + 0) + 3 x (2.71 + 1.84) = 21.8 fit-units
  L5 records   one per (dataset, k), content-cached; ~250 s each at n_ep=500
               on a quiet CPU (measured) -> CPU, overlaps GPU work; not GPU-h.
"""

from __future__ import annotations

import argparse


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fit-hours",
        type=float,
        required=True,
        help="unaugmented fit wall time, hours (the profile's TOTAL)",
    )
    ap.add_argument(
        "--speedup",
        type=float,
        default=1.0,
        help="Phase-2 factor applied to fits (measured; 1.0 = none)",
    )
    ap.add_argument(
        "--contended", action="store_true", help="use the contended training medians"
    )
    ap.add_argument(
        "--critic-mult",
        type=float,
        default=1.3,
        help="base-cell multiplier for the extra critic heads (GUESS)",
    )
    a = ap.parse_args()
    r1, r2 = 1.71, 1.84
    fit_units = 3 * (1.0 + r1) + 3 * (1.0 + r1 + r2)
    fit_h = fit_units * a.fit_hours / a.speedup
    cql, iql = (512, 865) if a.contended else (391, 625)
    # runs: per cell 2 algos x 3 ds x 3 ts = 18; 8 cells = 144
    base_cells, grace_cells = (
        4,
        4,
    )  # tmdp_base, tpomdp_base, both s0 companions | 4 grace cells
    per_cell_train = 9 * (cql + iql)  # seconds, single critic
    train_s = base_cells * per_cell_train * a.critic_mult + grace_cells * per_cell_train
    train_h = train_s / 3600
    total = fit_h + train_h
    print(
        f"fit-units {fit_units:.1f} x {a.fit_hours:.2f} h / speedup {a.speedup:.2f} = {fit_h:.1f} GPU-h (fits)"
    )
    print(
        f"training: 144 runs; base cells x{a.critic_mult} for the critic axis (guess) = {train_h:.1f} GPU-h ({'contended' if a.contended else 'quiet'} medians)"
    )
    print(
        f"TOTAL = {total:.1f} GPU-h  -> {'STOP (>60, §7.4)' if total > 60 else 'GO (<60)'}"
    )
    print(
        "not counted: L5 records (CPU, content-cached, ~4 min per (dataset,k)); the cache is assumed SHARED across cells (results/grace_cache)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
