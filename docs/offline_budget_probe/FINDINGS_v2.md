# Offline-budget calibration v2 — findings

**Measurement only.** Branch `feat/offline-budget-probe-v2`; tool
`tools/probe_offline_budget_v2.py`. No changes to runner/gates/critics/`budgets.yaml`.
**This reports; it does not decide.** All 9 runs completed (RE {300,1000,3000} × seeds
{0,1,2}, base cql, CartPole-v1 σ=0 basic, 256k grad steps, 25 checkpoints).

## Why v1 was wrong, and the fix

v1 minimised `value_mse_to_oracle` and picked 1024 steps — but there every critic's
`Q ≈ 6.7` against a true discounted value of ~100: the critics were **untrained**, and
three near-initialisation heads trivially agree, so MSE was tiny. The MSE also
references the **oracle, which moves with the budget**, so it cannot tell "both
accurate" from "both untrained." v2 adds an **absolute, budget-independent anchor**.

**What `apparent_q_mean` averages (read from `critic_ablation.py`, not assumed):**
`set_sequence_buffer` caches a fixed eval set = **every dataset transition `(obs,
a_data)`** from `iter_episodes()`, deterministically subsampled to 4000 via
`linspace(0,n-1,4000)`. Then `q_c = predict_q_adj(obs_e).gather(1, act_e)` — **Q at the
dataset action** — and `apparent_q_mean = q_c.mean()`; `value_mse_to_oracle` is over
the same set. **The anchor matches this exactly:** MC return-to-go
`G_t = Σ γ^k r_{t+k}` over the same reconstructed buffer and the same 4000 subsample.
(Verified: seed0/RE300 anchor = 8.693 reproduces a standalone pre-flight byte-for-byte.)

**The anchor is the BEHAVIOUR value, and that is the load-bearing caveat.** `pi_basic`
is random-tier (mean episode ~13.7 steps), so the MC anchor ≈ **18.4, flat across RE**
(18.27 / 18.57 / 18.54 — the sanity check passes) but is the *behaviour-policy* value,
**not** the improved-policy value the conservative critic targets. The learned policy
survives ~1020 steps (eval), whose discounted value is **~5.4× the anchor**
(eval-derived). So:
- `Q/anchor ≈ 1` = behaviour scale = **UNDER-trained** (not "healthy," as a naive
  reading of the [0.5,2] band would suggest).
- The **target is ~5.4× the anchor** (improved scale).
- The anchor is **per-seed bimodal** (seed0 ≈ 8.5 vs seeds 1,2 ≈ 23) because it tracks
  each seed's `pi_basic` quality — so ratios are computed per-run against that run's
  own anchor. RE-independence (the requested sanity check) holds; seed-dependence does not.

Figures: `runs_v2/fig_ratio_vs_steps.png`, `fig_relerr_vs_steps.png`,
`fig_anchor_eval.png`. Tables: `runs_v2/table.md`, `proximal_per_seed.md`.

## The curve (mean over 3 seeds)

`ratio = Q/anchor` (target ≈ 5.4×); `rel_err = √mse/|Q|`:

| steps | RE | obs_ratio | prox_ratio | oracle_ratio | obs_rel | prox_rel | eval |
|--|--|--|--|--|--|--|--|
| 1k | 300/1k/3k | 0.42 / 0.41 / 0.42 | 0.45 / 0.45 / 0.45 | 0.43 / 0.42 / 0.43 | .04/.03/.03 | .03/.04/.04 | ~960–986 |
| 20k | 300 | 3.34 | 3.90 | 3.72 | 0.184 | 0.107 | 1020 |
| 20k | 1000 | 3.55 | 3.70 | 3.67 | 0.082 | 0.052 | 1009 |
| 20k | 3000 | 3.76 | 3.94 | 3.91 | **0.061** | **0.030** | 1009 |
| 50k | 300 | 4.38 | 6.20 | 5.36 | 0.297 | 0.155 | 1015 |
| 50k | 1000 | 4.95 | 5.53 | 5.93 | 0.217 | 0.130 | 1001 |
| 50k | 3000 | 5.44 | 6.09 | 5.90 | **0.093** | **0.061** | 1013 |
| 100k | 300 | 4.84 | 8.00 | 6.19 | 0.353 | 0.333 | 1013 |
| 100k | 1000 | 5.17 | 6.39 | 6.83 | 0.362 | 0.094 | 1009 |
| 100k | 3000 | 6.03 | 6.85 | 6.51 | **0.091** | **0.062** | 1009 |
| 256k | 300 | 4.83 | **15.60** | 6.72 | 0.558 | 0.356 | 1016 |
| 256k | 1000 | 5.10 | 6.38 | 6.71 | 0.374 | 0.086 | 1012 |
| 256k | 3000 | 5.99 | 6.61 | 7.11 | 0.205 | 0.130 | 1009 |

## Answers to the deliverable

**Where is Q/anchor closest to improved-scale AND rel_err lowest — do they trade off?**
The ratio reaches improved scale (~5.4×) around **30–50k steps** for all RE and then
**plateaus** (no further climb → not diverging in the mean; the oracle lands at
5.4–7.1× = improved scale, so **the oracle reference is healthy, not broken** — it is
"far from 1" only because of the behaviour→improved gap). rel_err, however, is lowest
at ~1k (the *untrained* regime) and rises with steps. **So yes, they trade off at
RE=300 and RE=1000** — you cannot have both reached-scale and minimal rel_err there.
**At RE=3000 the tradeoff nearly dissolves:** by 20–50k the ratio is 3.8–5.4 (≈improved)
*and* rel_err is 0.03–0.09 (both critics). Per-RE best window:
- **RE=300:** no clean window — ratio reaches ~5 by 50k but rel_err is already 0.15–0.30
  and climbing, and proximal diverges (below).
- **RE=1000:** ~50–100k, ratio 5–7, prox_rel 0.09–0.13, obs_rel 0.22–0.36 (prox good,
  obs mediocre).
- **RE=3000:** ~30–50k, ratio 5.4–6.1, rel_err **0.06–0.09** for both — the clean window.

**Does more data break the tradeoff (does rel_err at the right-scale budget keep
falling)?** **Yes, decisively.** At ~100k (ratio already ≈ improved):
`obs_rel 0.353 → 0.362 → 0.091` and `prox_rel 0.333 → 0.094 → 0.062` for RE
300→1000→3000. More data **flattens the rel_err-vs-steps curve**, so the value estimate
can reach improved scale *and* stay accurate. v1's "monotone MSE explosion" was largely
a **data-starvation artifact**, not an intrinsic property of long training.

**Is there an RE beyond which returns diminish?** The proximal **instability is cured by
RE=1000** (below) and does not improve further at 3000 — so for *stability*, returns
diminish past ~1000. But **obs accuracy keeps improving 1000→3000** (obs_rel@100k
0.362→0.091), so for *value fidelity* there is no clear diminishing point by 3000 — obs
is still getting better. Net: the practical knee is **RE≈1000** (instability gone,
proximal accurate); **RE=3000** buys observational accuracy and high-step margin.

**Does the proximal instability reproduce?** **Yes at RE=300, and only there.**
Per-seed at 256k:

| RE | seed0 | seed1 | seed2 |
|--|--|--|--|
| 300 | Q=302.6, ratio **35.8**, rel_err **0.84** | ratio 5.0, 0.07 | ratio 6.0, 0.16 |
| 1000 | ratio 8.8, 0.15 | ratio 4.9, 0.05 | ratio 5.5, 0.06 |
| 3000 | ratio 9.3, 0.16 | ratio 5.4, 0.08 | ratio 5.2, 0.15 |

The seed-0 blow-up (Q=302, mse 65262) at RE=300 is **gone at RE≥1000** (Q≈77–81). It is a
small-data under-identification pathology, and seed-0 is the *worst behaviour policy*
(anchor 8.5 vs 23) — the shortest-episode dataset is where proximal's latent-class
E-step under-identifies. ≥1000 episodes cures it.

**Recommended (offline_grad_steps, rollout_episodes), with the tradeoff.**
**≈ (30k–50k grad steps, rollout_episodes ≈ 3000).** At RE=3000/50k: ratio 5.4–6.1
(improved scale reached), rel_err 0.06–0.09, no instability, eval saturated. This is
both **cheaper in training** than 256k (~5–8× fewer steps) and only costlier in a
one-time ~10× generation (training cost is O(1) in dataset size). If generation-bound,
**RE=1000 at ~50–100k** is the fallback: proximal is still cured and correctly scaled;
observational is noisier (rel_err ~0.2–0.36) but not diverging. **Avoid** both v1
endpoints: 1k (under-trained, ratio 0.4) and 256k (obs rel_err climbs, and proximal
explodes at RE=300).

*Tradeoffs / caveats (why this is a report, not a decision):* CartPole+CQL specific —
**re-probe per (env, algo)** (Acrobot/IQL/BCQ converge at different rates). The anchor
is the behaviour value; the ~5.4× improved-scale target is eval-derived on the CLEAN env
while Q is trained on confounded rewards, so treat the improved-scale line as
approximate. The anchor's per-seed bimodality means ratios must be read per-seed, not
just in the mean.

## Connection to the null-calibration reference
The stored `noise_ref` (`@44b1edc`: cql 574, iql 214) was measured at RE=40 / 256k —
**both data-starved *and* over-trained** by this probe's lights, i.e. the worst corner.
Those numbers are budget-driven artifacts; re-measured at ~50k / RE≈3000 the value MSE
(and its seed-sd) would be far smaller and more stable. The k=2.4 *separation* is
unaffected; the *denominator* should be re-pinned once a budget is chosen.

## Reproduce
```
python -m tools.probe_offline_budget_v2 --orchestrate \
    --out-root docs/offline_budget_probe/runs_v2 --max-workers 3 --device cuda
python -m tools.probe_offline_budget_v2 --plot --out-root docs/offline_budget_probe/runs_v2
```
Per-run curves under `runs_v2/ds{RE}_seed{S}/`. Note: RE=3000 runs took ~12 h each under
3-way GPU sharing (heavier generation), not the ~1 h solo estimate.
