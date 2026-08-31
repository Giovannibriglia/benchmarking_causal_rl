# Q2 — the sequential value, scoped and pre-registered (2026-08-24)

Giovanni's route decision: the full causal critic, not the de-confounded
reward-model shortcut. Scoped BEFORE any q2 run; the git timestamp is the
registration. The split follows the estimator-before-seam discipline:

* **Q2-A — sequential value estimation**: V^pi / Q^pi under do, FIXED target
  policy (the generator's greedy policy, declared), against MC return-to-go.
  The scientific claim; failures attributable. THIS BLOCK.
* **Q2-B — the critic seam**: GRACE's Q inside CQL/IQL training, learned-
  policy return. Deployment demonstration. NOT this block.

## Prerequisite, before any q2 number: validate the transition model

q1 never needed P(S' | S, A, U); q2 does, and that mechanism has NEVER been
validated (R was validated twice; the transition model's profile only ever
measured scoring cost). Measure held-out predictive accuracy — one-step and
multi-step rollout error against the true environment — and report it as its
own result. If it is poor, q2 fails for non-causal reasons and the
attribution would otherwise burn days in the wrong layer.

## Estimand, anchor, policy

* Q^pi_do(s,a) = E[R | do(a), s] + gamma * E_{s'|do(a),s}[ V^pi(s') ],
  sampling-based fitted iteration on the do-channel (the original L3 design).
* Anchor: DISCOUNTED MC return-to-go from the true environment under pi,
  with gamma MATCHED TO THE BENCHMARK'S OWN — consistency with the offline
  algorithms outranks the particular value. (The standing MC rule applies
  here for the first time: q1 had analytic truth, q2 does not.)
* Target policy: the generator's greedy policy, fixed and declared. Never a
  learned one (that is Q2-B).

## Cells and q2 verdicts (deliberately different from q1 — that is the point)

| cell | q2 verdict |
|---|---|
| d_a_null | point-ID (no latent; pure machinery check) |
| d_d | point-ID under finite-K (the headline; u_card declared — LABEL it) |
| d_e | bounds-only |
| d_b_prime | bounds-only |

**Gap noted, not built:** D-F — point-ID for q1, NON-ID for q2, the
taxonomy's most interesting off-diagonal case; its arms were never built.

## REGISTERED PREDICTIONS (before any run)

1. **Error compounds ~ 1/(1 - gamma).** Per-step error is 0.03–0.11 M; the
   return-level error should be roughly that divided by (1 - gamma). Much
   worse means the fitted iteration AMPLIFIES rather than accumulates — a
   finding about the iteration, not the estimator.
2. **The weak end (d <= 0.10) should be WORSE than q1 suggested** — the
   path-chaos enters every backup. Registered because the opposite result
   (backups averaging the instability away) would be good news worth
   understanding rather than assuming.

## Gates (V-C1's shape, at return level)

No harm on d_a_null and d_d at d = 1.0; accuracy vs MC-RTG against the
observational OPE floor, per seed; coverage and width where L4 supplies
intervals; refusal where L2 returns non-ID or the fit's conditions fail.

## Sequencing

Transition validation → fitted iteration on d_a_null (machinery; MEASURE THE
COST and report before scaling — many sample(do=) calls per backup; if
prohibitive, that changes the design and is better known at cell one) →
d_d at d = 1.0 → d_d weak end → bounds cells via L4. Reports after the
transition validation and after the d_a_null machinery check, before the
substantive cells.

Out of scope: the seam, learned policies, new cells, D-F.
