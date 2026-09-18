"""ARCHIVED (2026-09-03) — the L5 calibration sweep, superseded by ruling.

This tool ran the dr2-cut calibration campaign: p-uniformity, the two
Delta-R^2 distributions, power across effect size/horizon/sample size, and
the selector's as-deployed error rates (including the --report-only /
per-stage readings added in 4d539d8). The ruling that ended it: ``dr2_cut``
is a per-environment constant (A2 forbids it); falsification is REPORT-ONLY
(`l5.MarkovVerdict.record`) and the window is chosen by MATERIALITY-BY-REFIT
against L4's own interval (`pomdp_branch.transform_offline_rewards_declared`)
— no constant to calibrate, so no calibration campaign.

What survives:

* the 61 collected rows (``results/l5_calibration/rows.jsonl``, pre-S19
  (O,A,R)-family, stamped) as EVIDENCE for the S18 finding — rendered by
  ``tools/report_s18.py`` into ``s18_report.json`` (97% of true-MDP nulls
  rejected at floor effect sizes; 925x Delta-R^2 separation; capacity-shrink
  6.9 vs 0.0007);
* the materiality criterion's power question, which moved to SYNTHETIC
  fixtures with dialable truth (the l5/pomdp_branch test files), not
  environments.

The sweep implementation is preserved in git history (last working version:
commit 4d539d8). It imports APIs removed by the ruling (`select_window`,
`declaration_falsified`) and will not run against current code — that is the
point, not a defect.
"""

raise SystemExit(
    "calibrate_l5 is archived (dr2_cut stripped by ruling, 2026-09-03). "
    "The S18 evidence report is tools/report_s18.py; the sweep's last "
    "working version is in git history at 4d539d8."
)
