# GRACE v2 — architecture and design constraints

**Status: the estimator is not yet implemented.** L1 (the declared-diagram
layer) is in `src/rl/offline/grace/cell_graph.py`, documented in
`docs/diagram_catalogue.md`. L2–L5 are specified but unwritten. This document
records the design constraints that bind that implementation; it grows into the
as-built description as the layers land.

Companion documents: `docs/diagram_catalogue.md` (the assumption surface),
`docs/grace_v2_conditions.md` (identification derivations, C1/C2/C5),
`docs/nbn_requirements.md` (what the library owes us, and what it now
delivers).

---

## The assumption surface

v2 asserts exactly one thing per scenario: **the declared causal diagram**.
Everything else must be derived from it, learned from data, selected by a
held-out criterion, or named as an assumption in the catalogue. See
`diagram_catalogue.md` — every verdict carries the assumptions it rests on,
and `Verdict.label()` makes that travel with each number produced.

---

## Standing statistical rules

Two rules, each promoted from repeated independent rediscovery. **A new
statistic must argue against these, not rediscover them.**

### S1 — Nulls are built at EPISODE granularity, never transition granularity

Whenever a statistic is computed over data carrying an **episode-static latent**,
its null must be constructed by resampling or permuting **whole episodes**. `U`
is drawn once per episode; so are the D-D proxies and the D-E instrument. Their
effective sample size is the number of *episodes*, and step-level resampling
destroys exactly the block dependence being measured — producing a null far
tighter than the statistic's own sampling law, which then reports associations
that are not there.

Four independent instances before it was written down:

| site | step-level null said | truth |
|---|---|---|
| C1's splitter | leakage across the split | blocks straddled the boundary |
| L5's bootstrap | intervals far too narrow | resamples broke episode blocks |
| k-rank permutation | a zero-signal view is rank 2 | rank 1 (obs 0.031 vs null max 0.035 step-level, 0.105 episode-level) |
| every preflight independence check | `corr(I,U) = +0.086` → "not exogenous" | fixed tolerance justified at transition *n*; true SE ≈ 0.04, sign flipped across strengths |

The practical consequence: **a fixed magnitude tolerance is almost always the
wrong instrument.** The preflight now compares each statistic to a null
re-estimated from the data at hand by permuting whole episodes — optionally only
*within* strata of `U`, so the null preserves `P(· | U)` and destroys only the
association under test. Nothing needs calibrating per environment or per sample
size, which is also what keeps it inside v2's no-calibration-constants rule.

### S2 — Test the CONDITIONAL claim, never its marginal shadow

Nearly every assumption in the catalogue is a conditional independence, and the
marginal version is routinely nonzero *by design*. Testing the marginal produces
confident, wrong verdicts about correct generators:

| claim | marginal | why nonzero by design | conditional |
|---|---|---|---|
| proxy ⟂ A | +0.50 | proxy measures `U`, `A` is driven by `U` | +0.003 given `U` |
| proxy ⟂ S | +0.226 | `U → A → S` | 0.9 null SDs given `U` |
| I ⟂ R | −0.048 given `A` | `A` is a **collider** on `I → A ← U`, so conditioning on it *opens* the path | +0.005 given `(A, U)` |

The collider row is the sharp one: conditioning on *more* made the test wrong.
Read the conditioning set off the graph, and check whether each member is a
collider or a descendant of one before adding it.

### The corollary both rules share

Three of these were caught only against the **real** generator, after passing a
synthetic harness — the harness's states carried no `U → A → S` path, so its
marginals happened to equal its conditionals. Validate against real generated
data, and validate the generator against **ground truth** (logged `U`, declared
parameters), never against the estimator that will later consume it.

## Constraints imposed by NBN v0.14.0

These are properties of the library, verified on the vendored copy. Each has a
consequence GRACE must respect; none is a defect.

### N1 — Differentiability contract: route targets through `sample(do=)`

| path | differentiable through caller tensors? |
|---|---|
| `mechanism.log_prob`, `model.log_prob`, `model.sample(±do)` | **yes** |
| `query` / `query_batch` | **no — by design** (`inference_mode`) |
| `intervene()` | **no** — returns a deep copy, severing the caller's graph |

**Consequence.** L3's interventional target computation and L4's bound
optimiser must route through **`model.sample(n, do=…)`**, never `query` or
`intervene()`. Verified: `sample(do=)` gradient **1.0000** against an analytic
1.0. This is why the target computation sits behind one function boundary —
that boundary is where the differentiable path is guaranteed.

`query`/`query_batch` remain the right tool for *diagnostics* that need no
gradient (ESS, PSIS-k̂, batched dose-response readouts).

### N1a — Which interventional API: the discriminator is GRADIENTS

| need | API | why |
|---|---|---|
| a target, or anything feeding a loss | **`sample(do=)`**, looped over intervention values | the only differentiable interventional path |
| L4's bound evaluation, any read-only quantity | **`query_batch(do=)`**, one batched call | takes a per-row intervention vector; ~0.3 ms batched against ~37.7 ms looped for 256 interventions |

`query`/`query_batch` are non-differentiable **by design**. Routing a target
through `query_batch` for the speed would not raise — it would return a value
with no gradient, presenting downstream as a model that will not train. Decide
per call site and **say which you chose in the code comment**; both call sites in
`estimator.py` do, and both are pinned by a contract test.

The loop cost is negligible where it applies (~1.5 ms per intervention value),
so there is no reason to trade the gradient away for it.

**Shape contract for `sample(do=)`**, which cost real time to find: `do` values
are `[1, D]` — *not* 0-d scalars and *not* `(n,)` vectors — because the
do-dispatch builds a deterministic mechanism that indexes a batch axis, so a 0-d
value fails inside the sampler rather than at the call. Evidence is expanded to
`n` rows, since the sampler reshapes each parent to `(n, -1)`. Ancestral sampling
genuinely has no batch axis for a *varying* intervention and says so explicitly —
but that was not the failure here, and assuming it would have sent the fix in the
wrong direction.

**`query_batch`'s return is not always a tensor.** For a discrete target it is
`[B, K]`; for a continuous one (which `R` is) the engine returns the
likelihood-weighting particle representation `(weights [B, N], samples [B, N, D])`
and the posterior mean is the weighted average over particles.

Cross-checked on the L3 fixture: `do(a=0)` gives 0.998 (sample) against 0.999
(query_batch), `do(a=1)` gives 1.812 against 1.808. They answer the same question
by different machinery, so agreement is evidence and a drift would mean one is
wrong.

### N2 — `update_local` refuses weights: the online refresh must **refit**

`update_local(..., weights=…)` raises `NotImplementedError` (verified). There
is no incremental *weighted* update, and the EM M-step is inherently weighted
by per-episode responsibilities.

**Consequence for the online blocks.** The rolling-buffer EM refresh must
**refit** on the current episode view rather than incrementally update. This is
a design constraint recorded *before* that work starts, not a bug to discover
mid-block. Cost is bounded and already measured: a full EM fit is ≈ 22 s at
production scale, so a periodic refit is affordable at the cadences the online
cells use.

### N3 — KDE's bandwidth rule is unweighted: prefer parametric mechanisms in EM

The weighted Nadaraya–Watson mixture is exact, but the Scott/Silverman
bandwidth rule still uses the **unweighted** spread and row count. Under EM a
stratum's bandwidth would therefore reflect the pooled data rather than that
stratum's, biasing the component densities toward each other — precisely the
direction that would make strata look *less* separable than they are.

**Direction of the bias, and what it costs.** Pooling makes latent classes look
**less** separable than they are. That matters for L5's accounting: it produces
**false negatives, not false alarms** — it degrades V3's *detection power* and
its detection curve, while leaving the false-positive rate (measured on the
D-A-null arm, where there are no classes to blur) intact. A KDE-induced failure
would therefore look like "the diagram was not refuted", the quiet failure mode,
which is exactly why the mechanism choice is constrained rather than left to
taste. This belongs in L5's limitations section alongside M2's in-principle
undetectability.

**Consequence.** In the weighted EM path, use **MDN, normalizing flow, or
LinearGaussian** for continuous nodes. `supports_weights` is `True` for all
three (verified), and `False` for KNN — so a mistaken choice fails fast in
`fit()` rather than silently biasing. If KDE ever becomes load-bearing, making
it exact requires a weighted quantile, which would be a new NBN request.

---

## What the library now provides (v0.14.0), and how v2 uses it

| capability | v2 use |
|---|---|
| `model.log_prob(data, per_node=False)` | L3's complete-data likelihood, and — with `per_node=True` — **L5's channel-split diagnostic** (action channel vs reward channel) directly, instead of re-deriving the decomposition. Raises on a missing node, so a silently-marginalised variable cannot masquerade as a fitted one. |
| `fit(..., weights=)` with `supports_weights` | The EM M-step's per-episode responsibilities, applied to whole episodes (the latent is episode-static, so one responsibility broadcasts across that episode's rows). Verified exact against replication: weighted == replicated to 4.8e-07. |
| Parent-gradient transparency (pinned upstream) | Lets **all recurrence live GRACE-side** (Hypothesis B) — a GRACE encoder's output is passed as an ordinary parent and still receives gradient. No recurrent mechanism class is needed in NBN. |

### The do-semantics obligation this leaves with GRACE

Because recurrence is GRACE-side, GRACE owns a correctness trap that no library
API can enforce: **under `do(A_t = a)`, a recurrent hidden state must advance on
the *realized* trajectory including the intervened action**, not on what the
un-mutilated mechanism would have emitted. An implementation that advances the
state from the observational action produces plausible, wrong interventional
values. Keeping recurrence GRACE-side is what makes this tractable — the
component performing the intervention is the one that owns the state update —
but it must be written deliberately and tested.

---

## Superseded workarounds

Recorded so the implementation is not written against the old plan:

* **A GRACE-side weighted M-step loop is no longer needed.** The earlier plan
  had GRACE bypass `fit_local` and run its own Adam loop over `log_prob`
  (measured 1.08 s/M-step) because sample weights were absent. Use
  `fit(..., weights=)` instead wherever `supports_weights` is `True`, and keep
  a GRACE-side loop only for a mechanism that refuses.
* **A manual per-node log-likelihood loop is no longer needed.** Use
  `model.log_prob(..., per_node=True)`.

Neither workaround was ever implemented — the estimator did not exist when the
capabilities landed — so this is a change of plan, not a refactor.
