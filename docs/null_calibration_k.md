# Pinning `k` for the null-calibration gate — fixed-denominator gate, k = 2.4

> The initial `gap/(cell-noise)` attempt STOPPED (the finding below); it was RESOLVED by
> switching to a fixed reference denominator — see "Resolution" at the end.

`null_calibrated = gap < k * noise` (see `regime_report.compute_null_calibration`),
`k = 2.0` provisional. This measures both endpoints so `k` is set from the
**separation** between the correct pipeline and the exact bug the gate catches, not
from the passing number alone. Reproduce with
`uv run python tools/measure_null_calibration_k.py`.

## Measurement (offline_mdp, CQL, 5 seeds, NE=30, rollout_len=16, batch 128, seq 8; 2026-07-15)

Both configs measured in one harness per seed — `oracle_u`/`proximal`/`observational`
all CQL, plus an injected **bare-DQN** observational — all trained on the SAME
episode-grouped stream and scored against the SAME CQL oracle, so A and B differ ONLY
in the observational's base class (J5's bare-DQN-vs-CQL confound, reusing the existing
`_build_strategy_critic("observational", "offline_dqn", ...)` build).

| config | obs MSE→oracle (mean ± seed-sd) | prox MSE (mean ± sd) | gap | noise | **ratio = gap/noise** |
|---|---|---|---|---|---|
| **A. correct** (obs=CQL) | 0.0692 ± 0.0493 | 0.0258 ± 0.0345 | 0.0434 | 0.0426 | **1.02** |
| **B. broken** (obs=bare-DQN) | 0.2707 ± 0.2168 | 0.0258 ± 0.0345 | 0.2449 | 0.1552 | **1.58** |

`noise_B / noise_A = 3.6`.

## Result: STOP — do not pin k at this budget

**`ratio_B = 1.58 < 2`** — the ratio statistic does NOT cleanly separate correct from
broken here. Per the follow-up's decision rule, we do not split a gap this small.

**Why (probe C fired):** the bug inflates the SEED VARIANCE, not just the mean. The bare-DQN
observational's per-seed MSE swings 0.065–0.608 (sd 0.217, ≈ **6×** the CQL floor's 0.049).
So while the **gap widens 5.6×** (0.043 → 0.245), the **noise also widens 3.6×** (0.043 →
0.155), and the ratio only rises 1.5× (1.02 → 1.58). `gap/noise` is the wrong statistic
here because the denominator moves with the defect.

**The gap itself separates cleanly.** In noise-units of the CORRECT cell (a fixed
denominator that the bug can't inflate):

| config | `gap / noise_A` |
|---|---|
| A. correct | **1.02** |
| B. broken | **5.75** |

That is a clean 5.6× separation; log-halfway `k = sqrt(1.02 × 5.75) ≈ 2.4`.

## Resolution (feat/null-cal-fixed-denominator)

Adopted option 1 — **the gate now divides by a FIXED reference denominator**, the
CORRECT pipeline's basic-point pooled seed-sd (`noise_ref`), stored per (env, algo) in
`reproducibility/rl_regimes/_base/null_cal_reference.yaml` and read by
`regime_report.compute_null_calibration`. It is NOT recomputed from the judged cell (the
denominator the defect moves). A missing (env, algo) key → **uncalibrated** (blank),
never a silent pass.

    null_calibrated = gap < k * noise_ref        k = NULL_CALIBRATION_K = 2.4

**`k` pinned = 2.4** = log-halfway `sqrt(1.02 × 5.75)`. `noise_ref`'s own ~25% (n=5)
uncertainty puts the effective threshold band at ~1.9–3.2; the correct endpoint (1.02)
sits clear below and the broken endpoint (5.75) clear above.

**Second correct endpoint (generality check).** A correct IQL basic run (obs and oracle
both IQL), same budget, 5 seeds: `gap=0.0656`, `noise_ref=0.1149`, `gap/noise_ref=0.57` —
calibrated, well below k. IQL's `noise_ref` is ~2.7× CQL's, so `noise_ref` is legitimately
**per-(env, algo)** (a single global denominator would be wrong); no second finding.

Stored references (measured 2026-07-15, CartPole-v1, 5 seeds, NE=30 / rollout_len=16 /
batch 128 / seq_len 8):

| (env, algo) | noise_ref | correct gap/noise_ref |
|---|---|---|
| CartPole-v1, cql | 0.04257 | 1.02 |
| CartPole-v1, iql | 0.11485 | 0.57 |

Broken (CartPole-v1, cql, obs forced to bare DQN vs the CQL oracle): gap/noise_ref = 5.75
→ NOT calibrated at k=2.4. Re-measure and update the references when the budget or the
estimator changes.
