# L5's equivalence tolerance — the derivation (2026-09-02)

Ruled: L5's hypothesis is an equivalence region, not a point null (S18).
Falsified ⇔ `p ≤ α` **and** the violation exceeds a tolerance `τ` derived
from L4's own interval — the uncertainty GRACE already reports on the served
value. This document is the derivation, written down before the calibration
sweep, as directed. Where the propagation is clean it is given exactly; where
it is not derivable from L4 alone, that is said plainly and a constant-free
closure is proposed instead of a plausible number.

## What the tolerance must protect

The MDP branch serves two things: the substituted reward column
`r̂(O, a)` and the contrast with its L4 interval `[lo, hi]`, half-width
`w = (hi − lo)/2`, in per-step reward units, measured per fit. The ruling's
anchor: a Markov violation that cannot perturb the served quantity by more
than `w` cannot change a decision made on it.

A Markov violation (hidden state `ξ` beyond the observed `O`) reaches the
served object through exactly two channels.

## Channel 1 — the reward channel: derivable, and derived

If `ξ` carries **aligned, fast** (non-episode-constant) predictive
information about `R` beyond `(O, A)`, the fitted `E[r | O, a, U]`
mis-estimates `E[r | S, a, U]` and the contrast can shift.

The verdict already measures this channel: `reward_channel.improvement` is
the held-out ΔR² of `R` from history, and the **shift placebo nets out the
episode-constant part by construction** (a shifted lagged action carries the
same episode-constant `U`), so the excess of the observed improvement over
the placebo draws isolates the fast component:

    ΔR²_fast = max(0, improvement − Q_placebo)        (Q_placebo: the draws' upper quantile at the stated α)

The per-step reward-prediction perturbation attributable to `ξ` is bounded in
RMS by the predictive mass it carries:

    δ_r = sd(R) · sqrt(ΔR²_fast)

The contrast is a difference of two conditional means of `R` over the visited
distribution; a mean perturbation is bounded by the RMS perturbation
(Cauchy–Schwarz), each side contributing at most `δ_r`:

    |Δ contrast| ≤ 2 · sd(R) · sqrt(ΔR²_fast)

Suppression condition (`|Δ contrast| < w`), solved for the tolerance:

    τ_R = ( w / (2 · sd(R)) )²          — every term measured per fit.

**Order of magnitude on the pilot's own numbers**: d100 s0 has
`w ≈ 0.0123`, `sd(R) ≈ 0.5` ⇒ `τ_R ≈ 1.5e-4`. The measured fast reward
components on both H0 and masked CartPole are ≤ 0 (negative improvements);
a genuinely reward-relevant hidden state must carry ΔR² > 1.5e-4 to matter —
and anything that small genuinely cannot move the contrast outside the
interval GRACE already serves it in. The bound is loose (factor 2, RMS→mean)
in the direction that *widens* falsification, never narrows it.

## Channel 2 — the observation channel: NOT derivable from L4 alone, said plainly

The obs-dim ΔR² is denominated in next-state variance units. Its route to
the **served transform** at one step ends in the reward fit — which is
channel 1, already bounded. Its route beyond one step (hidden state → future
states → termination and future reward) does not pass through the transform
at all: it passes through the base learner's Q, which the MDP branch
deliberately does not model (no transition model — that is the reduction),
and which is *common to both arms*. Converting obs-ΔR² into served-value
units would require an effective-horizon × value-Lipschitz factor that GRACE
neither measures nor bounds. **A τ for the obs channel is therefore not
cleanly derivable from L4's interval, and per the ruling this is stated
rather than papered over with a plausible constant.**

### The constant-free closure: materiality by refit

When the obs-family test rejects statistically (`p ≤ α`), run the POMDP
branch's own machinery — the k-augmented fit — and compare its contrast to
the un-augmented one against `w`:

    |contrast(k) − contrast(0)| ≤ w   ⇒  IMMATERIAL to the served value
    otherwise                          ⇒  MATERIAL ⇒ full falsification

This is decision-anchored (the tolerance is again L4's own interval), uses
no new constants, and costs one extra fit paid only on statistical
rejection. It is literally "would acting on the violation change the answer
beyond the answer's own uncertainty".

### The tiered verdict, and what each contract cell reports

| tier | quantity | meaning |
|---|---|---|
| statistical | `p`, ΔR² (family max) | is the declared MDP exactly true? (it never is, at resolution — S18) |
| mechanism | capacity-shrink ratio (`capacity.shrink`) | approximation error shrinks with capacity (measured 56× on H0); information does not (~1× on H1) — corroboration, never a gate |
| channel 1 | `ΔR²_fast` vs `τ_R` | can the violation reach the transform through the reward fit? |
| channel 2 | materiality refit vs `w` | did acting on the violation change the served value? |

**SUPERSEDED ON ONE POINT (ruled 2026-09-03): falsification never abstains
and never switches.** The declaration is an input, not a hypothesis GRACE may
overrule — on falsification GRACE serves AS DECLARED, warns, and records the
evidence as a C3 condition on the served value; `serving_material` GRADES the
warning ("falsified and the correction moves" vs "falsified, correction
unaffected"), never gates serving. Abstention remains L4's, for fit health
only. The deterministic (MDP, MDP) cell then reads: statistically rejected at
ΔR² ≈ 1.6e-7, capacity-shrink 56×, both channels clear ⇒ warning suppressed
by the stated cut, serving unchanged, evidence attached.

### An honest consequence, flagged before it is measured

On masked CartPole the reward is one-step-independent of the hidden
velocities, so channel 1 may read ~0 and channel 2 may find the transform
genuinely unmoved — in which case the honest statement is that **GRACE's own
served object is unaffected by this violation while the learner's values are
belief-limited in both arms equally**. The user-contract warning (row 3) is
still issued from the statistical + mechanism tiers (ΔR² five orders above
the H0 range, capacity-stable); what the tolerance changes is only whether
GRACE *abstains from serving*. Whether the masked-CartPole materiality refit
actually moves the contrast is an empirical question — it is measured in the
calibration sweep, not assumed either way. If the (MDP, POMDP) grid cell's
headline "detection rate" should count statistical-tier detections rather
than materiality-tier falsifications, that is a reporting choice for the
grid design, and both rates will be in the calibration report.

## Interaction with the window selector

The selector consumes the same equivalence semantics **at every stage**:
stage k passes when `declaration_falsified(α, dr2_cut)` is False. Measured
without the cut (calibration, as-deployed): null-row `k_selected` read
1/2/None — the statistical tier rejects floor effects at every lag, so the
cut-less selector chases them to `k_max` on exactly-Markov data, and contract
row 2 ("over-assumption is cheap") fails for a reason unrelated to
over-assumption. With any cut in the measured gap the selector returns k = 0
on true MDPs, and the selected-k distribution on true-MDP arms is row 2's
headline metric. Selector exhaustion (`k = None`) is a BUDGET-BOUND
fit-mechanism condition in the L4-abstention family — not a declaration
override.

## What is deliberately NOT here

No noise-injection scale, no "negligible ΔR²" constant, no z-scores, no
per-dimension cutoffs. The only scales in the decision are `w` (measured per
fit by L4), `sd(R)` (measured from the data), and the placebo quantile
(measured from the draws).
