# Offline-budget calibration — findings

**Measurement only.** Branch `feat/offline-budget-probe`; tool
`tools/probe_offline_budget.py`. No changes to the runner, gates, critics, or
`budgets.yaml`. This diagnoses the budget; the fix is a later PR. **This reports; it
does not decide.**

## What was run

The shipped `BenchmarkRunner` at the **σ=0 basic point** (`bias_confounded_action`,
`c_r=1.0`, so `oracle_u` is defined), base `cql`, CartPole-v1. Grid:
`rollout_episodes ∈ {30, 300} × seeds {0,1,2}` = 6 runs, each trained **once** to
256 000 grad steps (`n_episodes=250 × rollout_len=1024`), `n_checkpoints=25`. The
step-curve is read off the 25 checkpoints (grad steps 1024 → 256 000). Critics:
observational, proximal, oracle_u. All 6 runs completed; the σ=0 gate passed for all.

- **`value_mse_to_oracle`** and **`apparent_q_mean`** per (RE, seed, critic) per checkpoint.
- **`eval_return`**: base-actor greedy return on the runner's **clean** eval env
  (`build_env`, no confounder). It reads ~940–1020 — i.e. eval episodes are **not**
  500-capped here (the policy balances ~1000 steps to pole-fall). Read it as a
  saturation signal, not on the nominal-500 scale.
- **Q bound**: the σ=0 basic confounder shifts reward (base 1 + `c_r·U·1[a=a_bad]`),
  so dataset **r_max = 2.0 → r_max/(1−γ) = 200** (γ=0.99). The naive r=1 → 100 line is
  drawn too. Because eval episodes run long (no early termination), the **true**
  discounted value is itself near this bound, so a mean-Q in the low hundreds is *not*
  by itself proof of divergence — the MSE and the tail are.

Figures: `runs/fig_mse_vs_steps.png`, `runs/fig_q_vs_steps.png`. Table:
`runs/table.md` / `table.csv` (values below).

## The table (mean over 3 seeds; nearest checkpoint to each target)

| target | steps | RE | obs_mse | prox_mse | obs_Q | prox_Q | oracle_Q | eval_ret |
|--|--|--|--|--|--|--|--|--|
| ~1k | 1024 | 30 | 0.27±0.19 | 0.24±0.39 | 6.70 | 7.01 | 6.69 | 942±60 |
| ~1k | 1024 | 300 | 0.07±0.03 | 0.05±0.07 | 6.62 | 6.90 | 6.76 | 986±18 |
| 20k | 22528 | 30 | 302±201 | 328±401 | 55.5 | 53.7 | 46.6 | 976±77 |
| 20k | 22528 | 300 | 75±51 | 39±19 | 59.0 | 66.3 | 63.4 | 1020±0 |
| 50k | 54272 | 30 | 591±527 | 271±242 | 68.9 | 64.0 | 58.3 | 994±44 |
| 50k | 54272 | 300 | 394±347 | 286±461 | 80.1 | 100.8 | 93.5 | 1015±8 |
| 100k | 96256 | 30 | 3768±4667 | 3484±5198 | 71.6 | 70.1 | 85.0 | 988±55 |
| 100k | 96256 | 300 | 1063±1390 | 2304±2862 | 88.5 | 118.8 | 111.3 | 1013±11 |
| 256k | 256000 | 30 | 11841±10789 | 4569±5240 | 141.1 | 96.8 | 65.8 | 981±62 |
| 256k | 256000 | 300 | 1215±1957 | 21935±37523 | 89.5 | 186.2 | 105.5 | 1016±5 |

(The "~5k" target's nearest checkpoint is 1024; the probe's first checkpoint is 1 epoch
= 1024 steps, so there is no readout strictly between 1k and 11k.)

## Answers to the questions

**1. Knee, or monotone?** **Monotone worsening — no knee.** `value_mse_to_oracle` is
*lowest at the first checkpoint* (~1k steps: 0.07–0.27) and rises ~monotonically
(near power-law on log–log) to 10³–10⁴ by 256k. There is no improving-then-worsening
knee inside the probed range; the best value fidelity is the **earliest** budget.
obs and prox track each other on the way up.

**2. Does Q exceed the bound, and when?** **The mean Q does *not* clearly exceed the
r_max=2 bound (200).** CQL's conservatism keeps mean `apparent_q` bounded: it climbs
from ~7 (1k) to ~65–140 (RE30) / ~90–186 (RE300) at 256k, under 200. Only
**proximal at RE=300 reaches the bound (~186 → touches 200) by 256k**. This is *not*
the bare-DQN mean-blowup from the brief (that Q=221 885 was the excluded bare-DQN
arm). **But** two things do diverge: (a) `value_mse_to_oracle` reaches ~10⁴ →
RMSE ~100 on a value scale of ~100, i.e. **~100 % relative per-state error** —
divergence lives in the tail/per-state Q, invisible in the mean; (b) the **seed-sd on
Q explodes late** (the RE=30 band spans roughly [−1, +100] around 10⁵ steps). So:
mean Q respects the bound, but value *estimation* collapses and becomes
seed-unstable. Onset of the blow-up is early — MSE is already ~50–300 by 11k–22k steps.

**3. Does 300 episodes move the knee / cut the seed-sd, and by how much?** **It lowers
the whole curve ~4–10×, but does not move the onset.** obs_mse at 1k: 0.07 (RE300) vs
0.27 (RE30), ~4×; seed-sd 0.03 vs 0.19, ~6×. At 256k: 1215 (RE300) vs 11841 (RE30),
~10×. So more data **reduces the magnitude** of the value divergence and stabilises
early seeds — but both sizes still worsen monotonically from ~1k steps, and RE=300
still reaches MSE ~10³ by 256k (with one proximal seed spiking to 2×10⁴). **Dataset
size is a partial lever, not a cure**: if the divergence is driven by both, more data
alone won't fix it — you'd also need fewer steps.

**4. Do obs and prox agree at σ=0 anywhere?** **Yes — only at the earliest
checkpoints (~1k steps).** There obs≈prox≈oracle (MSE 0.05–0.27, all Q ≈ 6.7): the
null-calibration property holds. They **diverge** as steps grow (256k, RE=300: obs_mse
1215 vs prox_mse 21935; Q obs 90 vs prox 186). **The σ=0 null-calibration property is a
low-step property.** Crucially, `eval_return` is *already saturated* (~942–986) at that
same 1k-step budget — so the policy is already good where the value estimates are still
trustworthy.

**5. Recommended (grad_steps, rollout_episodes), with the tradeoff.**
- The evidence points to **drastically fewer grad steps**. Value estimates are best
  (agree with the oracle, MSE minimal, obs≈prox) at the **earliest** probed budget
  (~1k steps), where the **policy is already saturated** (eval ~940–986). Everything
  past ~1k–5k steps only inflates the value-MSE and seed variance **without improving
  the policy**.
- **Recommendation to consider:** cut the offline budget by **1–2 orders of magnitude**
  — on the order of **1k–5k grad steps** instead of 256k — and read the value metric at
  an **early checkpoint** (its minimum) rather than the final one. Pair with
  **rollout_episodes ≈ 300** if the null-calibration *value* metric matters (it cuts the
  residual noise ~4–10× and stabilises seeds); if only the *policy* matters, RE=30 is
  already saturated.
- **Tradeoffs / caveats (why this is a report, not a decision):**
  - The 1k-step sweet spot is **CartPole + CQL specific**. Harder envs (Acrobot) and
    other bases (IQL, BCQ, offline_dqn) converge at different rates and must be
    re-probed per (env, algo) before any budget is changed.
  - The probe's floor is 1024 steps (first checkpoint); the true optimum may be lower.
    A follow-up should probe **100–5000 steps densely** to locate it.
  - The current budget numbers (`n_train_envs=16`, `rollout_len=1024`) are
    **on-policy vectorized-rollout parameters reused verbatim as offline gradient
    steps** — that reuse is the likely root cause; 256k SGD steps against ~750
    transitions (RE=30) is ~44 000 epochs of overfitting.

## Connection to the null-calibration reference

This explains the `noise_ref` explosion measured on `feat/…@44b1edc`
(cql 0.043→574, iql 0.115→214 from the NE=30 probe to the 256k production budget):
those production numbers are **budget-driven value-divergence artifacts**, not signal.
The seed-sd that inflates `noise_ref` is the same late-step instability seen here. If
the offline budget is cut per this probe, `noise_ref` should be **re-measured at the
chosen budget** (and would likely return toward the small probe-scale values). Either
way the k=2.4 separation still holds — but the denominator is budget-dependent.

## Reproduce

```
python -m tools.probe_offline_budget --orchestrate \
    --out-root docs/offline_budget_probe/runs --max-workers 2 --device cuda
# figures/table only, from existing runs:
python -m tools.probe_offline_budget --plot --out-root docs/offline_budget_probe/runs
```
Per-run raw curves are under `runs/ds{RE}_seed{S}/` (`critic_ablation_metrics.csv`,
`eval_metrics.csv`, `probe_meta.json`); ~2–3 h/seed on cuda.
