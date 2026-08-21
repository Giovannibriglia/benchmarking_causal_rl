# GRACE method parameters — the binding audit

**The rule (2026-08-21, binding):** GRACE's own parameters must be
environment-independent — one configuration across every cell, environment and
sweep point, or the method is fitted to CartPole and Acrobot rather than
tested on them. For every numeric parameter, **report whether it binds**: a
limit never reached across the full grid is a safety guard and not a
parameter; a limit that is reached is a tuning knob and must be either derived
from the data or disclosed as an assumption. Cell parameters (σ, `d`, `M`,
`c_r`, `n_proxies`, episode budget) are the experiment's independent
variables, declared in the catalogue, and are NOT in scope.

Binding evidence: `tools/audit_method_parameters.py`, writing
`results/cost/method_parameter_audit.json`; per-run evidence travels on the
C3 labels (`tau1_budget_bound`, `backtrack_exhausted`, `degenerate_mechanism`,
`converged`).

## 1. Derived from data (fine by construction)

| parameter | derivation |
|---|---|
| `temperature` (τ₀) | mean episode length — at τ = T the tempered episode log-lik IS the per-step average |
| anneal rungs | ⌈log₂ τ₀⌉ — halving until 1, no free parameter |
| `epochs` under the fixed-step budget | `m_step_budget / steps_per_epoch` — derived so the two cannot disagree |
| R's type (categorical vs MDN) | support saturation: uniques(first half) vs uniques(full) — a derived criterion, no magnitude threshold |
| proxy-init split | median of the first proxy's episode mean — a data quantile |
| degeneracy ceiling | `−log(min_scale·√2π)` per dim, derived from the mechanism's OWN declared floor |
| `u_card` | cell-declared here (`\|U\| = 2` is part of the scenario); where unknown it is selected by held-out likelihood, which is the method's business and stays environment-independent |

## 2. Disclosed reporting choices (reported like a confidence level)

| parameter | value | note |
|---|---|---|
| `alpha` (L4/L5) | reported | never a constant inside the estimator |
| `B` | reported with its own MC error | a precision parameter, not calibration |
| `max_failure_rate` | 0.0 default | the strictest reading; relaxing is an explicit caller decision |
| `tol` = 1e-4 + window = 2 | disclosed convergence CRITERION | defines the `converged` label; a fit near it reports its tail deltas (`ll_tail_rel_deltas`), so whether it decided anything is visible |

## 3. Caps and guards — must be shown never to bind (measured, not asserted)

| parameter | value | binds when… | evidence |
|---|---|---|---|
| `max_iter` | 30 (default) | a fit ends non-finished with the budget spent | **`tau1_budget_bound`** on every fit. Structural note: since the anneal became a PREFIX of extra iterations, `max_iter` counts τ=1 iterations ON TOP of the derived rungs — the total already has the `rungs + cap` shape and carries no hidden τ₀ dependence. The audit's job is the cap itself. |
| `max_backtracks` | 3 | `backtrack_exhausted` fires at τ=1 | per-fit flag, aggregated by the audit |
| `m_step_budget` | 400 | the M-step's improvement has NOT plateaued at the budget | the budget-sweep probe: fits at 100/200/400/800/1600 steps, ll curve reported — O(steps) makes the budget environment-independent by construction; whether 400 SUFFICES is what the probe measures |
| `lr` (base) | 1e-3 | fits routinely need halvings | `backtracks` distribution; under GEM the line search adapts DOWNWARD from it per iteration, so lr is a starting point with a self-correcting mechanism, but persistent nonzero backtracks would mean the start is wrong |
| numerical guards | 1e-9 (monotone compare), 1e-12 denominators | never by design | magnitude ≪ any measured quantity |
| saturation diagnostics | `_SATURATION_EPS` = 1e-3, `_SATURATION_FLAG` = 0.5 | never — TELEMETRY ONLY | nothing in the control path reads the detector (a gate on it was tried and withdrawn as an A2 violation) |

## Measured (2026-08-21, `results/cost/method_parameter_audit*.{json,log}`)

* **`max_iter`**: never bound at defaults — `tau1_budget_bound` False in all
  10 probe fits, including one at 34 iterations. The 17/20 unfinished fits in
  archived artifacts trace to the L3 re-validation's *deliberately reduced*
  diagnostic cap (max_iter = 15, random init) — a diagnostic-run choice, not
  the production default binding.
* **`m_step_budget` = 400**: recovery invariant (0.987–0.993) across a 16×
  budget range in both environments; final-ll differences are sub-percent and
  NON-monotone on CartPole (budget changes the optimisation path — the
  input-chaos finding, not unclaimed improvement). Not binding on the
  estimand.
* **`max_backtracks`**: the audit's genuine catch, in two parts. At depth 3
  the production CartPole config exhausted MID-ASCENT (best rejected step
  worsening 3.3e-2 with accepted improvements still 0.3–4%; a deeper search
  recovered ~98 further nats) — a knob. And a depth-6 fit reached a genuine
  fixed point (rejected step flat to 3e-6) yet classified STUCK, because the
  stationarity test read only the ACCEPTED tail — the abrupt-convergence gap.
  **Resolved by two changes, neither adding a constant:** default depth
  3 → 6 (64× lr span), and stationary granted when the best rejected step's
  worsening is itself below `tol` (flat at the optimiser's resolution in both
  directions). Re-measured: the production configuration finishes; remaining
  exhaustions occur only at sub-production budgets (100/200) and carry their
  labels.
* **base `lr`**: backtracks 1–155 across configs — the line search does real
  work downward from 1e-3; a disclosed starting point with a measured
  self-correction mechanism, spanning 64× under the new depth.

## 4. Hand-set and binding — MUST BE EMPTY, or each entry justified

| parameter | status |
|---|---|
| *(EMPTY as of the 2026-08-21 audit. `max_backtracks` was here for one afternoon — measured binding at depth 3 — and left by measurement: the depth raise plus the fixed-point stationarity grant, both derived or tol-reusing, re-measured non-binding at the production configuration.)* | |

**Disclosed model-class choices (fixed everywhere, adequacy tested by the
benchmark itself):** MDN `num_components = 3`, `hidden = (64, 64)`,
`batch_size = 4096`, mechanism `min_scale = 1e-3` (the library's declared
floor; when it binds, `degenerate_mechanism` fires — that is the detector
working, and it is how the mis-specified continuous R was caught). These are
the method's model class, not tuning knobs: they never vary by environment,
and the degeneracy/saturation detectors are what make a bad choice loud
rather than silently absorbed.

## Longer term (queued after V-C, not now)

Two environments cannot fully answer the two-environment worry however well
the parameters behave. Queued: the parametric generalisation sweep (synthetic
arms varying episode length, reward support size, state dimension and `|U|`
independently) plus one real environment with a genuinely continuous reward —
the highest-risk untested path, since every current arm resolves R to
categorical.
