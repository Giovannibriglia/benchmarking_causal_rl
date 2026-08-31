# V-C1 pre-registration — V1 and V5 at estimator level

**Committed BEFORE any V-C1 fit is run** (2026-08-22); the git timestamp is
the pre-registration. Scope per review: V1 + V5 only — no L4, no bounds, no
q2, no critic-seam wiring, no new cells; a result appearing to need one of
those stops and reports rather than widening.

## The query, stated in every artifact

**q1 — the per-step interventional value** `E[R | do(a), s]`, read through the
gate channel as the action contrast `E[R | do(a_bad), s] − E[R | do(other), s]`.
Nothing here is a sequential-value claim: D-D's q2 is point-ID only under
finite-K and is out of scope.

## Truth: analytic, and why the MC rule does not apply

The generator defines the gate exactly: contrast truth = `c_r · q̄` with
`q̄ = 0.5` (symmetric pairs; the d = 1.0 deterministic gate has q̄ = 0.5 in
expectation over U as well), and `0` on `d_a_null` (c_r = 0). The standing
MC-return-to-go rule was written for value-MSE at the RETURN level (q2); a
per-step gate contrast has a closed form and using MC for it would add
variance to a known number.

## Estimators and metrics

* **GRACE**, production configuration (audited defaults; `init="proxy"` where
  the arm declares proxies, `"random"` on `d_a_null`; 3 fit seeds, best-LL
  selection — likelihood-valid under categorical R), full 3000 episodes
  (reported verdicts run at production scale; S11's subsampling is for
  diagnostics). C3 binding flags recorded per fit.
* **Floor: the naive estimator** — transition-pooled contrast as the
  headline, episode-level as the collider-isolating companion. Its error is
  also computable analytically per dataset (`naive − truth` from logged
  data), which is what makes the mechanical component checkable.
* All errors **|error| / M**, per dataset seed, never seed-averaged signed
  (the measured sign instability makes signed means read ≈ 0 while wrong).

## Gates (criteria fixed in advance)

* **V1 — no harm.** On `d_a_null` (no latent) and D-D d = 1.0 (confounded,
  identified, proxies decorative): GRACE's per-seed |error| must sit within
  the per-seed spread of the floor's |error| on the same datasets. Determinism
  is on; the spread is measured, not assumed.
* **V5 — point-ID accuracy.** On D-D CartPole d ≤ 0.10: GRACE's |error| must
  beat the floor's by more than the per-seed spread, per seed.

## The pre-registered curve, DECOMPOSED

The GRACE-vs-floor advantage alone cannot carry the proximal claim, because
the floor's normalised error is `c_r · |tilt| / M` — the same σ-tilt
multiplied by a c_r growing 20× along the axis. Growth of the raw advantage
with shrinking d is therefore partially MECHANICAL and predicted everywhere
GRACE identifies, Acrobot included. Three components are registered instead:

1. **Floor error grows ∝ c_r·|tilt|** on both environments — mechanical,
   verified against the analytic per-seed tilt.
2. **GRACE's own |error|/M stays within its measured ±0.15 band across the
   whole axis on BOTH environments** — identification maintained: via the
   proxies on CartPole at weak d, via the non-proxy channels on Acrobot.
3. **The PROXIMAL content is the with-vs-without value gap**: predicted to
   track the recovery transition on CartPole (≈ 0 at d = 1.0, largest at
   d ≤ 0.10) and to stay small on Acrobot throughout (the boundary-case
   prediction). The without-arm values come from the sweep ablation already
   run; no new fits are needed for this component.

**Falsifiers, stated in advance:** (2) failing on CartPole at weak d means
GRACE loses identification despite the proxies — V5 cannot pass honestly
even if the raw advantage looks good. (2) holding while (3) is flat on
CartPole means GRACE's accuracy is coming from something other than the
proximal channel — the false-positive case the review named, and V5 would be
reported as NOT demonstrating proximal identification regardless of the
gate arithmetic. (3) large on Acrobot would contradict the boundary-case
finding and reopen it.

## Deliverables

V1 and V5 verdicts under the criteria above; the decomposed curve on both
environments against these predictions; the binding-flag aggregate; one
paragraph on whether the predictions held, including the Acrobot half.


---

## POST-HOC AMENDMENTS (2026-08-22, marked as such — written AFTER the run)

1. **Component (1) as registered was WRONG, by algebra, and the data shows
   the correct form.** Registered: floor error grows ∝ c_r·|tilt|. Correct:
   the naive transition-pooled bias is `c_r · d · tilt = M · tilt` — the
   compensation pins the naive bias exactly as it pins the estimand, so the
   floor is FLAT along the axis. Measured: CartPole 0.086–0.116, matching
   the naive-bias gate's independent tilt measurements (0.090–0.112) almost
   digit-for-digit; Acrobot 0.033–0.118 vs gate 0.030–0.084. The original
   review intuition — that raw advantage growth would be meaningful — is
   thereby partially restored: the floor does not grow, so any advantage
   shape comes from GRACE's side alone.
2. **Component (iii) is evaluated WITHIN the sweep ablation only** (matched
   configuration, matched random init, 400 episodes) and never combined
   numerically with V-C1's production fits: the two runs differ in episodes
   (400 vs 3000) and with-arm init (random vs proxy), so a cross-run gap
   would mix a configuration difference with the proxy difference.
3. **A finding the registration did not anticipate: the symmetric gate makes
   the VALUE contrast forgiving of latent collapse.** With q̄ = 0.5, a
   class-collapsed fit still estimates E[bonus | a_bad] ≈ q̄ · c_r ≈ truth —
   so the value-level with/without gap is intrinsically weaker than the
   latent-level one (measured: positive in 11/15 CartPole seeds but weak
   against noise, largest +0.24 at mid-d, vs the recovery gap of +0.5). A
   value-level proximal demonstration would need per-class contrasts or an
   asymmetric gate — out of V-C1 scope, reported rather than widened.


## ADJUDICATION (2026-08-23) and prospective amendments

* **`d_a_null` CartPole V1: INCONCLUSIVE — criterion undefined on a
  zero-variance reference.** Not a pass (unearned under the registered
  wording), not a fail (it reflects the criterion, not the method). Measured
  value on record: GRACE |error| ≤ 0.001 (0.1% of M).
* **Prospective zero-spread rule, stated before any block it applies to:**
  when the reference's per-seed spread is zero, GRACE's |error| is compared
  against **GRACE's own per-seed spread** — the question becomes "consistent
  with zero at the precision available", with the method's own noise as the
  only yardstick present. Derived, no constant. Applies to future blocks
  only, never retroactively.
* **V5 stands exactly as registered**: joint d ≤ 0.10 FAILS; d = 0.10 passes
  3/3 at ~3× spread.

## REGISTERED BEFORE RUNNING — the V5-failure scaling prediction (2026-08-23)

Mechanism conjectured for the d = 0.05 failure, in the same family as the
corrected floor algebra: a probability-estimation error δ in the categorical
reward becomes a value error c_r · δ, i.e. **δ/d after M-normalisation** —
bias stays zero, variance grows as 1/d: a bias–variance trade along the
informativeness axis with a predicted optimum where one was observed.
Since δ ∝ 1/√n, the prediction is

    normalised |error| ∝ 1 / (d · √n)

checkable by subsampling episodes: at fixed d, err(n)/err(3000) ≈ √(3000/n)
(2.0 at n = 750, 1.41 at n = 1500); across d at fixed n,
err(d = 0.05)/err(d = 0.10) ≈ 2. If it holds, the d = 0.05 gate failure is a
**quantified sample-size requirement (~4× the data)**, not a structural
limit. Falsifier: no √n improvement at fixed d would mean the error is not
estimation noise and the mechanical story is wrong again.


## SCALING CHECK RESULT (2026-08-23) — the registered prediction is FALSIFIED

Error vs n on CartPole (per seed, M-normalised): NON-monotone in 5 of 6
(cell, seed) series — n = 1500 beats n = 3000 in five of them (e.g. d = 0.05
s1: 0.035 → 0.006 → 0.041 across 750/1500/3000). There is no 1/√n decay: the
weak-end value error is NOT estimation variance. It is the
**chaotic-optimiser-path regime** (the measured input-fragility family: a
1e-7 perturbation moves final_ll by 60 nats; a subsample change is a vastly
larger input change) arriving at the value level. Consequences:
* the "quantified sample-size requirement (~4× data)" story is dead — more
  data does not reliably move the demonstration point outward;
* the ANTI-CORRELATION finding takes its sharper form: the proximal channel
  becomes necessary exactly where value estimation enters the path-chaotic
  regime, and the tension is structural at this configuration, not
  sample-limited;
* the second registered component falsified by its own check this round —
  the registration discipline is doing its job in both directions.

## NEGATIVE-SHARE CROSS-TAB (2026-08-23) — the SERIOUS branch

The four negative-share (harmful) best-LL fits carry labels
INDISTINGUISHABLE from the high-share fits at the same points: all converged,
finished, un-exhausted, un-bound, sep/step 13.6–15.1, SATURATED-AT-INIT on
every weak-end fit (a risk flag with zero discrimination there). **GRACE at
the weak end can be confidently wrong — value error of order the floor's own
bias — with no C3 label separating harm from help.** The C3 system detects
PROCESS pathologies; this error is not a process failure but estimation in a
chaotic landscape, invisible to process labels by construction. Serving
consequence for V-C2, stated for that conversation: a point-serving critic is
NOT safe at the weak end on fit labels alone — refusal there needs an
UNCERTAINTY statement, which is L4's job; at d ≥ 0.25 the shares are
+0.80…+0.97 with small stable errors and point-serving on labels is
defensible.


## V-C1 CLOSE-OUT (2026-08-23)

**Matched pair (full scale, both arms random-init, CartPole):** the d = 1.0
control's differential share is ≈ 0 (+0.02/−0.17/+0.08) — the registered
false-positive check PASSES. At d = 0.10 the differential share is
+0.12/+0.22/−0.09 (mean ≈ +0.08), INSIDE the control's noise band, while the
same fits carry a latent-level recovery gap of +0.09…+0.35 — and the
collapsed without-arm (recovery 0.65) still removes 65–85% of the floor's
bias. The symmetric gate's contrast is intrinsically robust to latent
collapse, measured cleanly at matched full scale.

**The four deliverables:**
1. Cross-tab: harmful fits label-indistinguishable from helpful ones — the
   serious branch; serving safety at the weak end needs L4-style uncertainty,
   labels suffice at d ≥ 0.25.
2. Scaling: the registered 1/(d√n) prediction FALSIFIED — weak-end value
   error is path-chaotic, not estimation variance; more data does not
   reliably help.
3. Differential share: control ≈ 0 ✓; at the claim point genuinely low
   (≈ +0.08, within control noise) against a latent gap of +0.35.
4. Anti-correlation, in the reading the scaling check supports: STRUCTURAL —
   the proximal channel becomes necessary exactly where value estimation
   enters the path-chaotic regime; the value-level proximal proposition on
   the symmetric contrast is small because that estimand forgives latent
   collapse, so the proximal value case lives in latent-reading quantities
   (per-class contrasts, L5 statistics).

**Verdicts:** V1 pass (3 blocks) + inconclusive (zero-variance reference,
prospective rule recorded). V5 as registered: fail on the joint d ≤ 0.10
criterion; pass 3/3 at d = 0.10 at ~3× spread. V-C1 CLOSED; V-C2 scoping
takes §1's serving-safety input and §3's number as its premises.


## THE ASYMMETRIC TEST — registered 2026-08-23, BEFORE generation

**Correction of framing first: the symmetric gate is not "forgiving" — it
makes the marginal correct BY CONSTRUCTION.** With `q0 + q1 = 1`, `q̄ = 0.5`
exactly at every sweep point, so `E[R | do(a_bad)] = 1 + c_r · q̄` is
independent of the latent structure under balanced `P(U)`: a fully collapsed
fit estimates the pooled rate, which sits at the right marginal by
arithmetic. Symmetry was chosen to hold the marginal bonus rate constant
along the sweep — a reasonable control that turns out to trade away the
thing the cell exists to demonstrate at the value level. Arithmetic, not a
property of the estimator — and therefore fixable.

**The test:** ONE asymmetric variant at the demonstration point —
`(q0, q1) = (0.05, 0.15)`: `d = 0.10`, `c_r = M/d = 10`, `q̄ = 0.10`.
CartPole, 3 dataset seeds, matched pair (both arms random-init, full scale,
production configuration), 18 fits. No sweep.

**Registered prediction, with its arithmetic:** a collapsed fit now estimates
the observationally tilted pooled rate, whose error is `c_r·d·(P(U|a_bad) −
P(U)) = M·tilt` — the FULL confounding bias, the same as the naive floor —
while a latent-recovering fit corrects it. The differential share at the
asymmetric point should therefore be **large, approaching the correctable
bias** (≈ +1 up to the with-arm's own noise), where the symmetric point gave
+0.08. Control expectation: the with/without recovery gap should look like
the symmetric d = 0.10 point's (proxies load-bearing either way — asymmetry
changes the VALUE arithmetic, not the latent problem).

**Decision rule, fixed in advance:** large differential → D-D carries the
value-level proximal claim and V-C2 may serve marginal contrasts. Not large →
the value-level case is genuinely unavailable on this cell; D-D's claim is
scoped to latent-reading quantities; no further cycle.

Truth at this point: `contrast = c_r · q̄ = 1.0`; floor bias `M · tilt` as
everywhere.


## ASYMMETRIC TEST RESULT (2026-08-23) — the decision rule fires the scoping branch

Measured (CartPole, matched full scale, truth = c_r·q̄ = 1.0): differential
share **+0.17/+0.22/+0.20** — tripled versus the symmetric point and
sign-consistent, but NOT "large, approaching the correctable bias". The
registered prediction fails quantitatively, and the premise that missed is
identified: "a collapsed fit carries the full M·tilt bias" assumed TOTAL
collapse, while weak-end without-arms are partially informed (recovery
0.66–0.94 via the action channel and R), carrying only ~0.45 of the floor's
bias; with the with-arm's own weak-end noise at ~0.28 of floor, the
ACHIEVABLE differential between the two arms is ~0.2 — exactly what was
measured. Co-reading on record: the proxies deliver essentially all the
differential available between the arms; the HEADROOM is what is small.

**Per the pre-committed rule: D-D's value-level proximal claim is SCOPED TO
LATENT-READING QUANTITIES** (per-class contrasts, heterogeneous effects,
L5's statistics); V-C2 does not serve the marginal contrast as the proximal
demonstration. Recovery control held (with/without gap matches the symmetric
point). No further cycle. Third registered prediction falsified by its own
check in this exercise — each time with the failing premise named, which is
the discipline's product.
