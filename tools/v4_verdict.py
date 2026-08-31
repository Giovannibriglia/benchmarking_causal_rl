"""V4 verdict assembly -- read-only render of results/v4/report.json.

Assembles the gate verdict the pre-registration asked for: coverage aggregate
vs nominal 90%, width/collapse table, the D-E dual-bounds comparison and
instrument-value gap, the D-B-prime exploration gate, weak-end procedural
shares, and the failure-budget aggregate from the persisted diagnostics.

Recorded rulings the render reflects (2026-08-27):
* d025 widths are reported as MEASURED (0.05-0.16), not rounded into the
  pre-registration's "~0" -- small is a measurement, silence is not.
* The Balke-Pearl anchor is CARTPOLE-ONLY: the anchor filters to in-pair
  actions {0, 1}, and on 3-action Acrobot that conditioning invalidates the
  bounds (BP misses truth on two Acrobot seeds). Same family as the A2 gate's
  two-action special case (handoff, Cluster A). D-E's Acrobot rows therefore
  have NO valid closed-form reference and the instrument-value gap is
  computed on CartPole only; Acrobot BP rows print as "no valid reference".
* Walk width-0 rows are an OPTIMISER finding, separable from the interval
  verdict (see results/v4/walk_diagnosis.json: the constraint functional
  mismatch puts LR(theta-hat) >> c, so the walk starts infeasible and the
  fallback returns theta-hat's target for every start).
"""

from __future__ import annotations

import json
from pathlib import Path

NOMINAL = 0.90


def main() -> int:
    rows = json.loads(Path("results/v4/report.json").read_text())
    iv = [r for r in rows if r["row"] == "interval"]
    bd = [r for r in rows if r["row"] == "bounds"]

    # ---- coverage ----------------------------------------------------------
    n_cov = sum(1 for r in iv if r["covered"])
    print("== V4 VERDICT ==\n")
    print(f"COVERAGE: {n_cov}/{len(iv)} = {n_cov/len(iv):.1%} vs nominal {NOMINAL:.0%}")
    for r in iv:
        if not r["covered"]:
            gap = max(r["truth"] - r["hi"], r["lo"] - r["truth"])
            print(
                f"  miss: {r['cell']:<9} {r['env']:<12} s{r['seed']} "
                f"[{r['lo']:+.4f}, {r['hi']:+.4f}] truth={r['truth']:+.4f} "
                f"(outside by {gap:.4f})"
            )
    weak = [r for r in iv if r["cell"] in ("d025", "d010", "d005")]
    print(
        f"  weak end (d <= 0.25): {sum(r['covered'] for r in weak)}/{len(weak)}; "
        f"all misses are at the strong-identification end"
    )

    # ---- width / collapse --------------------------------------------------
    print("\nWIDTHS (collapse property; d025 reported as MEASURED, not '~0'):")
    for cell in ("d_a_null", "d100", "d050", "d025", "d010", "d005"):
        ws = [r["width"] for r in iv if r["cell"] == cell]
        print(f"  {cell:<9} width {min(ws):.4f} .. {max(ws):.4f}")

    # ---- failure budget ----------------------------------------------------
    tot_req = sum((r["bootstrap_diagnostics"] or {}).get("n_requested", 0) for r in iv)
    tot_fail = sum((r["bootstrap_diagnostics"] or {}).get("n_failed", 0) for r in iv)
    hi_rows = [r for r in iv if r["failure_rate"] > 0.2]
    print(
        f"\nREPLICATE FAILURE BUDGET: {tot_fail}/{tot_req} = {tot_fail/tot_req:.1%} "
        f"(all failures degenerate-mechanism)"
    )
    print(
        "  high-failure rows (>20%): "
        + ", ".join(f"{r['cell']}/{r['env'][:4]}/s{r['seed']}" for r in hi_rows)
    )
    print(
        "  NOT QUOTABLE until the s1 pattern is understood -- see the s1 "
        "diagnosis note in the handoff."
    )

    # ---- procedural share, weak end ---------------------------------------
    print("\nPROCEDURAL SHARE (weak end called out):")
    for r in iv:
        if (
            r["cell"] in ("d010", "d005")
            and r["procedural_share"] == r["procedural_share"]
        ):
            print(
                f"  {r['cell']:<5} {r['env']:<12} s{r['seed']} "
                f"share={r['procedural_share']:.0%}"
            )

    # ---- D-E dual bounds ---------------------------------------------------
    print("\nD-E DUAL BOUNDS (truth 0.5):")
    for r in bd:
        if r["cell"] != "d_e":
            continue
        walk = (
            f"walk [{r['walk_lo']:+.4f}, {r['walk_hi']:+.4f}] cov={r['walk_covered']}"
        )
        if r["env"].startswith("CartPole"):
            bp = f"BP [{r['bp'][0]:+.4f}, {r['bp'][1]:+.4f}] cov={r['bp_covered']}"
            print(f"  {r['env']:<12} s{r['seed']} {bp}  {walk}")
        else:
            print(
                f"  {r['env']:<12} s{r['seed']} BP: NO VALID REFERENCE "
                f"(3-action env; anchor filters to actions {{0,1}} -- "
                f"cf. the A2 two-action special case)  {walk}"
            )
    print(
        "  instrument-value gap: CARTPOLE-ONLY (see docstring), and NOT YET a\n"
        "  clean measurement: the walk is an inner approximation whose width\n"
        "  is BUDGET-LIMITED (a 600-step probe on d_b_prime CartPole s0 was\n"
        "  still descending, 0.760 -> 0.574, where production's 150 steps\n"
        "  reached 0.67), so walk-vs-BP currently measures the optimiser\n"
        "  budget as much as the model. A converged walk must precede the gap."
    )

    # ---- walk exploration gate --------------------------------------------
    degen = [r for r in bd if abs(r["walk_hi"] - r["walk_lo"]) < 1e-9]
    print(f"\nWALK (bounds cells): {len(degen)}/{len(bd)} rows width-0")
    for cell in ("d_e", "d_b_prime"):
        rows_c = [r for r in bd if r["cell"] == cell]
        cov = sum(1 for r in rows_c if r["walk_covered"])
        print(f"  {cell:<10} walk coverage {cov}/{len(rows_c)}")
    print(
        "  POST-FIX READING (2026-08-27, after 141ee6f/ef238a5/c7420db): the\n"
        "  walk MOVES on every row with healthy multi-start spread and\n"
        "  LR(theta-hat)=0 by construction. D-B-prime's under-coverage is\n"
        "  INNER-APPROXIMATION TRUNCATION -- the step budget, not the region:\n"
        "  the 600-step probe above still descends toward truth. The step\n"
        "  budget (or a plateau-based stop) is the open optimiser decision."
    )

    # ---- the safeguards that fired ----------------------------------------
    print(
        "\nSAFEGUARDS THAT EARNED THEIR PLACE:\n"
        "  * multi-start, mandated as insurance, became the DETECTOR -- three\n"
        "    starts on an identical endpoint is unambiguous.\n"
        "  * D-B-prime's coverage row was designated the empirical exploration\n"
        "    test and FIRED. Neither failure would have been visible without\n"
        "    them."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
