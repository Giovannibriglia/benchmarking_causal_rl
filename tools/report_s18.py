"""The S18 report — the calibration rows' ONE remaining purpose.

Ruled 2026-09-03: dr2_cut is stripped (A2), the calibration sweep is stopped,
and its rows are kept solely as evidence for the S18 finding:

    A point null of exact Markovianity, tested against a flexible model on a
    DETERMINISTIC system, rejects at floor effect sizes — the null is false
    by construction, at any capacity.

This tool READS ``results/l5_calibration/rows.jsonl`` (the frozen pre-S19
record; family OAR, stamped) and renders the finding's three exhibits:

  1. the null rows' p distribution (statistical-tier rejection rate at floor
     effect sizes on true-MDP data);
  2. the two Delta-R^2 magnitude distributions (true-MDP nulls vs constructed
     POMDP masks) and their separation;
  3. the capacity-shrink ratios (approximation error is capacity-dependent;
     information is not).

No import of ``l5`` — this is a reader of evidence already on disk, immune to
the post-ruling API. Nothing here gates anything.

Run:  uv run python tools/report_s18.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROWS = Path("results/l5_calibration/rows.jsonl")
OUT = Path("results/l5_calibration/s18_report.json")


def _dist(vals):
    a = np.array([v for v in vals if v is not None], dtype=float)
    if not a.size:
        return None
    return dict(
        n=int(a.size),
        min=float(a.min()),
        median=float(np.median(a)),
        max=float(a.max()),
    )


def main() -> int:
    rows = [json.loads(x) for x in ROWS.read_text().splitlines() if x]
    rows = [r for r in rows if "error" not in r]
    nulls = [r for r in rows if r["kind"] == "null"]
    masked = [r for r in rows if r["kind"].startswith("masked")]

    ps = np.array([r["p"] for r in nulls], dtype=float)
    shrink = lambda rr: [  # noqa: E731
        (r.get("capacity") or {}).get("shrink")
        for r in rr
        if (r.get("capacity") or {}).get("shrink") is not None
    ]
    null_dr2 = _dist([r["stat"] for r in nulls])
    masked_dr2 = _dist([r["stat"] for r in masked])
    report = dict(
        provenance=dict(
            source=str(ROWS),
            n_rows=len(rows),
            family_history="OAR (pre-S19 sweep; valid for S18 — the finding "
            "concerns the point null's behaviour, not the family)",
            envs=sorted({r["env"] for r in rows}),
            datasets=len({r["dataset"] for r in rows}),
        ),
        exhibit_1_null_p=dict(
            n=int(ps.size),
            frac_rejected_at_005=float(np.mean(ps <= 0.05)) if ps.size else None,
            p_min=float(ps.min()) if ps.size else None,
            p_median=float(np.median(ps)) if ps.size else None,
            reading="the statistical tier rejects true-MDP data at floor "
            "effect sizes — the point null is false by construction on "
            "deterministic systems (S18)",
        ),
        exhibit_2_dr2=dict(
            null=null_dr2,
            masked=masked_dr2,
            separation=(
                float(masked_dr2["min"] / null_dr2["max"])
                if null_dr2 and masked_dr2 and null_dr2["max"] > 0
                else None
            ),
            reading="the rejected effects are orders of magnitude below the "
            "true-POMDP effects — statistically detectable, practically "
            "nothing; the S18 rule's equivalence-region form exists because "
            "of exactly this gap",
        ),
        exhibit_3_capacity_shrink=dict(
            null=_dist(shrink(nulls)),
            masked=_dist(shrink(masked)),
            reading="null effects shrink with base capacity (approximation "
            "error); masked effects do not (information)",
        ),
    )
    OUT.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
