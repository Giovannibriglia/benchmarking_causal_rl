"""The corrected V-B certification table, read against the recorded predictions.

Reports what CHANGED and why, not just the new numbers: a re-certification whose
output is a fresh table invites the reader to accept it, while one that shows
every flip against the original invites them to check it.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recert", default="results/vb_recertification/report.json")
    args = ap.parse_args()
    rows = json.loads(Path(args.recert).read_text())

    print(f"=== corrected certification, {len(rows)} datasets ===\n")
    print(
        f"{'cell':<11}{'env':<13}{'n':>4}{'was ok':>8}{'now ok':>8}"
        f"{'gate ok':>9}{'usable':>8}"
    )
    grid = defaultdict(list)
    for r in rows:
        grid[(r["cell"], r["env"])].append(r)
    for (cell, env), rs in sorted(grid.items()):
        was = sum(1 for r in rs if r["was_preflight_passed"] is not False)
        now = sum(1 for r in rs if r["preflight_passed"])
        gate = sum(1 for r in rs if r["gate_passed"])
        both = sum(1 for r in rs if r["preflight_passed"] and r["gate_passed"])
        print(f"{cell:<11}{env:<13}{len(rs):>4}{was:>8}{now:>8}{gate:>9}{both:>8}")

    flips = [
        r
        for r in rows
        if (r["was_preflight_passed"] is not False) != bool(r["preflight_passed"])
    ]
    print(f"\n--- {len(flips)} preflight verdict flips ---")
    for r in flips:
        d = "FAIL->PASS" if r["preflight_passed"] else "PASS->FAIL"
        was = "; ".join(r["was_reasons"] or []) or "-"
        print(f"  {d} {r['cell']:<10} {r['env']:<12} s{r['seed']} sig={r['sigma']:<5}")
        print(f"     was: {was[:150]}")
        if not r["preflight_passed"]:
            print(f"     now: {'; '.join(r['preflight_reasons'])[:150]}")

    still = [r for r in rows if not r["preflight_passed"]]
    print(f"\n--- {len(still)} datasets still FAIL preflight ---")
    c = Counter()
    for r in still:
        for s in r["preflight_reasons"]:
            c[s.split("(")[0].split("--")[0].strip()[:90]] += 1
    for k, v in c.most_common():
        print(f"  {v:>3}  {k}")

    gate_only = [r for r in rows if r["preflight_passed"] and not r["gate_passed"]]
    print(
        f"\n--- {len(gate_only)} pass preflight but FAIL the confounding gate "
        "(a separate claim: 'is the declared confounding present at strength', "
        "not 'is the arm valid') ---"
    )
    gc = Counter((r["cell"], r["env"], r["sigma"]) for r in gate_only)
    for k, v in sorted(gc.items()):
        print(f"  {v:>3}  {k[0]:<10} {k[1]:<12} sigma={k[2]}")

    # P1: the k-rank / margin question the third-proxy decision rests on.
    dd = [r for r in rows if r["cell"] == "d_d"]
    print("\n=== P1: D-D k-ranks and margins, transition-level -> episode-level ===")
    kr_was = Counter(json.dumps(r["was_k_ranks"], sort_keys=True) for r in dd)
    kr_now = Counter(
        json.dumps(r["preflight_proxy_k_ranks"], sort_keys=True) for r in dd
    )
    print("  was:", dict(kr_was))
    print("  now:", dict(kr_now))
    for env in sorted({r["env"] for r in dd}):
        sub = [r for r in dd if r["env"] == env]
        for view in ("Z", "W", "R"):
            was = [
                r["was_margins"][view]
                for r in sub
                if r["was_margins"]
                and r["was_margins"].get(view) not in (None,)
                and r["was_margins"][view] != float("inf")
            ]
            now = [r["preflight_proxy_margins"][view] for r in sub]
            wa = sum(was) / len(was) if was else float("nan")
            na = sum(now) / len(now) if now else float("nan")
            print(
                f"  {env:<12} {view}: mean margin {wa:6.2f} -> {na:6.2f}"
                f"   (n_was={len(was)}, n_now={len(now)})"
            )
    degen = [r for r in dd if any(r["preflight_proxy_binning_degenerate"].values())]
    print(f"\n=== P2: views whose quantile grid still collapses: {len(degen)} ===")
    for r in degen:
        print(
            "  ",
            r["env"],
            r["seed"],
            r["sigma"],
            r["preflight_proxy_binning_degenerate"],
            r["preflight_proxy_k_ranks"],
        )

    dbp = [r for r in rows if r["cell"] == "d_b_prime"]
    if dbp:
        print("\n=== D-B' : the S1b exemption, MEASURED ===")
        gaps = [r["preflight_drift_length_weighting_gap"] for r in dbp]
        inert = sum(1 for r in dbp if r["preflight_drift_length_weighting_inert"])
        print(
            f"  length_weighting_gap: min {min(gaps):.4f} median "
            f"{sorted(gaps)[len(gaps)//2]:.4f} max {max(gaps):.4f}"
        )
        print(f"  inert (gap < 0.05): {inert}/{len(dbp)}")
        for r in dbp:
            if not r["preflight_drift_length_weighting_inert"]:
                print(
                    f"    NOT INERT {r['env']} s{r['seed']} sig={r['sigma']} "
                    f"short={r['preflight_drift_autocorr_short_episodes']:+.3f} "
                    f"long={r['preflight_drift_autocorr_long_episodes']:+.3f}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
