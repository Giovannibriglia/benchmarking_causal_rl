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


---

## 3. Measured cost of a bootstrap replicate under the monotone guard

Pre-measured because it changes the V-D block plan, and is better known now than
discovered mid-run. **Caveat on every number here: taken on 2 threads while V-B
was generating**, so these are upper bounds on a contended machine, not clean
timings. Re-measure on an idle machine before the block plan is fixed.

Fixture: 300 episodes x 12 steps (3600 transitions), `epochs=30`, `max_iter=8`.

| base lr | mean fit | backtracks | iters | per-iter | monotone |
|---|---|---|---|---|---|
| 1e-2 | 117–146 s | 6–7 | 5 | 23.5 s | yes |
| **3e-3** | **62.6 s** | **3.0** | 3.5 | **17.9 s** | yes |
| 1e-3 | 100.3 s | 3.5 | 3.0 | 33.4 s | yes |

**The guard's cost is the BACKTRACKS, not the extra likelihood evaluation.** The
claim that the happy path is free stands — the checking E-step *is* the next
iteration's E-step — but the happy path is not where we were: at lr = 1e-2 the
M-step overshoots on roughly every iteration, and each rejected step is a full
M-step redone. A run with retries disabled finished in 37 s, but only by
stopping early on a decrease it could not repair, so that is not a like-for-like
baseline and should not be quoted as "the unguarded cost".

**Consequence for planning.** The earlier estimate (refit ≈ 8.65 s, B = 99 ≈ 7.7
min per constraint) did not include backtracking and is now optimistic by more
than an order of magnitude: at this fixture scale and thread budget a guarded
fit is ~63–146 s, so B = 99 is roughly **1.7–4 hours per constraint**. That is a
block-plan-changing difference.

**Two levers before accepting it**, in order:

1. **Base step size.** lr = 3e-3 halves the cost of lr = 1e-2 by overshooting
   less. Note the non-monotonicity in the *table*: 1e-3 is slower than 3e-3
   despite fewer backtracks, because a smaller step needs more epochs to make
   the same progress. There is an optimum and it is measurable — but it is a
   *performance* knob, not a calibration constant, and nothing downstream reads
   it. It must be chosen by wall-clock on an idle machine and reported.
2. **Fit budget per replicate.** A bootstrap replicate does not need the
   precision of the point fit; `epochs` and `max_iter` can be lower, provided
   the reduction is applied to the *observed* fit as well so the null and the
   statistic come from the same procedure. Applying it to only one would bias
   the threshold — a subtler version of the dropped-replicate bias.

Do NOT reach for a third lever of dropping slow or non-converging replicates:
that is exactly the bias the module refuses (see `bootstrap.py`).


### 3a. Is `lr` only a wall-clock knob? **NOT ESTABLISHED — treat it as disclosed**

Checked directly, because mixture EM has multiple local optima and a step size
that lands in a different one changes the *estimand*, not merely the time to
reach it. 200 episodes x 10, two seeds per setting:

| lr | mean final LL | recovery (seed 0 / 1) | prior₀ |
|---|---|---|---|
| 1e-2 | −3633.74 | 0.960 / 0.935 | 0.453 / 0.499 |
| 3e-3 | −3658.63 | 0.960 / 0.945 | 0.454 / 0.494 |
| 1e-3 | −3674.70 | **0.980 / 0.990** | 0.440 / 0.455 |

Spread across `lr` was 40.96 against a within-`lr` seed spread of 43.37, so the
formal comparison says "within noise" — **but that verdict is not usable**, for
two reasons:

1. **The trend is monotone in `lr`** (−3633.7, −3658.6, −3674.7). Systematic
   ordering is not what sampling noise looks like, and with n = 2 seeds the
   comparison has no power to separate the two.
2. **Log-likelihood and recovery accuracy disagree in direction.** The *worst*
   LL (lr = 1e-3) gives the *best* latent recovery (0.980 / 0.990). If `lr` were
   pure wall clock, both would be flat. Instead this looks like different optima
   with different qualities — exactly the failure mode the check was for.

**Ruling for now: `lr` is disclosed like `α`** — reported with every result, not
treated as a free knob — until a properly powered comparison (≥ 10 seeds, idle
machine) either establishes agreement or characterises the dependence. The
optimiser setting must not be chosen by wall clock while it may be choosing the
answer.

### 3b. Lever priority, fixed

1. **Parallelism across replicates** — statistically neutral, implemented
   (`n_jobs`, thread pool, deterministic by index). Take this first.
2. **Fewer SGD epochs per M-step** — GEM permits a partial M-step, so this is
   legitimate. **Symmetrically only.**
3. **Reducing B** — honest degradation, with the MC error reported alongside.
   After the free levers, never instead of them.

**Excluded: warm-starting replicates from the null-generating parameters.** The
replicate is generated *from* those parameters, so warm-starting hands it a head
start the observed fit never got, and the replicate statistics come out
systematically better-optimised. Same asymmetry as an uneven fit budget. The
general rule now lives in `bootstrap.py`'s docstring: **the procedure that
produces the observed statistic must produce the replicate statistics.**


## Path-chaos vs identification width — a check to build in (added 2026-08-23, from V-C1)

V-C1's falsified scaling check showed the weak-end value error is
OPTIMISER-PATH-CHAOTIC, not sample-limited (error non-monotone in n; the
1e-7-perturbation fragility family). Consequence for L4: bootstrap replicates
perturb the data, and on a chaotic likelihood surface the replicate fits will
vary widely — the interval stays conservative (sound) but may be VACUOUS at
the weak end, wide for the wrong reason.

**Build the diagnostic in from the start:** compare the bootstrap replicate
spread against the spread of repeated fits on IDENTICAL data with perturbed
initialisation only. Comparable spreads mean the interval is measuring the
optimiser, not the identification uncertainty, and the interval must be
labelled accordingly (C3) rather than served as an identification statement.
Much cheaper built in now than discovered as a vacuous interval during V-C3.
