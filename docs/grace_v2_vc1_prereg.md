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
