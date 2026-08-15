# L4 — uncertainty: design decisions taken before implementation

Recorded ahead of the code because both are expensive to retrofit and one of
them is where v2's central discipline would quietly fail.

---

## 1. "Compatible with the observational distribution" — the threshold must be calibrated, not chosen

For a non-identified cell, L4's bound is a min/max of the target functional over
the set of models **compatible with the observational distribution**. That word
carries the whole layer, and every obvious operationalisation smuggles in a free
parameter:

* "likelihood within δ of the MLE" — δ is a calibration constant;
* "moments matched to tolerance ε" — ε is a calibration constant;
* "any model in a hand-specified family" — the family is the constant.

**If a δ appears in L4, the layer has reintroduced exactly what dropping `k` was
meant to remove.** v1 died of `k` and `noise_ref`; a bound whose width is set by
an unjustified threshold is the same failure wearing different notation, and
worse, it is *invisible* — the bound still looks like a bound.

### The decision

The compatible set is a **likelihood-ratio confidence region**:

> `C(α) = { θ : 2·(ℓ(θ̂) − ℓ(θ)) ≤ c(α) }`

with the critical value `c(α)` **calibrated by the within-dataset parametric
bootstrap** — resampling **whole episodes** (S1), refitting, and reading the
LR statistic's realised distribution. The bound is then the min/max of the
target over `C(α)`.

**Why the bootstrap is not a convenience here.** The usual χ² asymptotics for
the LR statistic are *unavailable*, for the same reason they are unavailable for
L5's falsification test: a finite mixture with an unidentified latent puts the
null on a **boundary of the parameter space**, where the regularity conditions
behind `χ²_df` fail. The reference distribution is not χ² and is not known in
closed form, so estimating it from the data at hand is the only sound route —
not a fallback.

**This is the same mechanism as L5's null, deliberately.** L4 and L5 share one
calibration device rather than maintaining two: episode-level parametric
bootstrap, within dataset, refit each replicate. Stating it explicitly is part
of the design, because two separately-tuned calibrations would be two places for
a constant to hide, and any drift between them would be invisible in the
outputs.

### What is owed at delivery

1. **Where the threshold comes from** — the bootstrap procedure, its
   granularity, and its replicate count.
2. **Why it is not a constant** — it is re-estimated per dataset from that
   dataset's own refits; nothing carries across environments or sample sizes.
3. **What it reduces to when the model IS identified** — the interval must
   **collapse** toward a point as the identifying information sharpens. This is
   a *checkable property*, not a claim: run the bound engine on **D-D** (point-ID
   by the proximal criterion) and on **D-A** (point-ID by back-door) and assert
   the width goes to ~0; run it on **D-E** (bounds-only by construction) and
   assert it does not. A bound engine that returns a wide interval on an
   identified cell is broken in a way no single-cell test would reveal.

`α` itself is a **reported coverage level**, not a calibration constant: it is
chosen and disclosed like any confidence level, and the *threshold* realising it
is estimated from data. That distinction is the whole point and should be stated
wherever a bound is reported.

---

## 2. Two validations, planned together: q1 exact, q2 by coverage

### q1 — Balke–Pearl, exactness

D-E gives an **exact closed-form reference**, which is why keeping `R` binary
was worth the redesign of its reward gate (see `diagram_catalogue.md`, R2 route
(a)): the Balke–Pearl LP is a per-step, binary-instrument, binary-treatment,
binary-outcome result. It anchors **q1 only**.

Validation: the engine's q1 bounds must match the closed form to numerical
tolerance on D-E. This tests the *optimiser* — that the min/max over the
compatible set is actually found.

### q2 — Monte-Carlo ground truth, coverage

There is **no closed form for the sequential value**, so the anchor cannot
validate the query that actually matters for a critic. But this is a benchmark
and **the true SCM is known**, which the general setting never affords: `V^π`
under the target policy is computable by **Monte-Carlo rollout in the true
environment**.

Validation: **coverage**, not exactness — does the interval contain the true
`V^π` at the nominal rate across seeds and cells? And **width must be reported
alongside**, because an interval that covers by being vacuous is not a result.
A coverage table without widths is uninterpretable, and the failure it hides
(bounds that are correct and useless) is the likely one.

### Why both, and why planned from the start

The two tests localise faults to different components:

| q1 (Balke–Pearl) | q2 (MC coverage) | diagnosis |
|---|---|---|
| exact | covers | engine sound |
| **fails** | — | the **optimiser** is wrong; fix before reading q2 at all |
| exact | **under-covers** | the **sequential extension** is wrong, not the optimiser |
| exact | covers but vacuous | bounds sound, **compatible set too loose** |

The third row is the valuable one and is only readable if both exist. Building
q2's harness later would mean discovering a sequential-extension fault with no
way to exonerate the optimiser.

---

## Constraints inherited

* Bound optimisation runs through **`sample(do=)`** — the only differentiable
  interventional path (N1). `query_batch(do=)` is for *evaluating* a fixed
  bound, never for the optimisation target.
* Bootstrap resampling is **episode-level** (S1), like every other null here.
* Any assumption the bound rests on rides on the estimate object (C3), and any
  with no observable shadow goes in the consolidated section of
  `diagram_catalogue.md`.
* q2 stays **non-ID under confounded dynamics** regardless of q1 — L2 already
  enforces this, and L4 must not present an interval that implies otherwise.
