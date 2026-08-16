# GRACE v2 — handoff

Branch `feat/grace-v2` (off `feat/grace-critic`), worktree
`~/PycharmProjects/bcrl-grace-v2`, own venv with torch 2.10.0+cu130.

**Read this before touching anything.** The rules below were each learned by
getting them wrong, usually more than once. Rediscovering them costs a day
apiece.

---

## The standing rules

Each has a one-line reason. **A new statistic or component must argue against
these, not rediscover them.**

| # | rule | why |
|---|---|---|
| **A1** | The declared **diagram is the only assumption**. Everything else is derived from it, learned, or named in the catalogue. | If a config can independently switch a channel on, the diagram stops being the assumption surface. `diagram_arms.py` derives *which* channels exist from the catalogue entry; YAML supplies only strengths, and a mismatch either way is refused. |
| **A2** | **No calibration constants.** No `k`, no `noise_ref`, no tuned tolerance, no discretisation in the estimator. | v1 died of them. Where a threshold seems needed, build a null from the data instead (see S1). |
| **S1** | **Nulls at EPISODE granularity**, never transition. | `U`, the proxies and the instrument are all episode-constant, so effective *n* is the episode count. Step-level resampling shatters the blocks and gives a null far too tight. Five instances: C1's splitter, L5's bootstrap, the k-rank permutation, and two preflight checks. |
| **S1b** | **RULE (operational, deliberately broad): match the statistic's unit of observation to the unit at which the tested quantity varies.** A quantity constant within an episode (`U`, `Z`, `W`, `I`) enters statistics as **one row per episode**; a per-step companion is reduced to an episode statistic (the **mean**, never the sum — a sum is proportional to length by construction). Quantities that genuinely vary within an episode stay at transition level, with the null still clustered by episode. | **In RL, episode length is an OUTCOME.** Pooling an episode-constant quantity at transition level weights each episode by its own length. **MECHANISM (see below) — the rule is broader than the mechanism, on purpose.** Measured: `corr(I,U)` = **−0.590** at transition level against **−0.034** at episode level, on an instrument drawn from its own Bernoulli that never reads `U`. The permutation null does not rescue it — permuting whole episodes destroys the length-weighting in the null while the observed statistic keeps it, so it surfaces as a huge z rather than as noise. |
| **S2** | Test the **conditional** claim, never its marginal shadow. | Marginals are nonzero *by design*: proxy⟂A reads +0.50 marginally vs +0.003 given `U`; proxy⟂S reads 0.226 marginally because `U→A→S`. And conditioning on *more* can be wrong — `A` is a **collider** on `I→A←U`, so conditioning on it alone opens `I→U→R`. |
| **S3** | A statistic must be compared against **a null that describes the statistic actually computed**. | A *maximum* over 8 tests judged by a per-test 3-SD cutoff runs ~8× the intended false-alarm rate — it fired on provably covariate-free proxies. Build the null of the max. Likewise a *binning failure* (Acrobot's reward is −1 almost everywhere) is not an uninformative view. |
| **S4** | Validate the **generator against ground truth**, and the estimator against the generator. Never mutually. | If each validated the other, a shared misconception passes in silence. Preflight reads logged `U` and declared parameters only — never L5, never the estimator. |
| **S5** | Validate on **real generated data**, not only a synthetic harness. | Three checks passed a synthetic harness and failed on the real generator: the harness's states carried no `U→A→S` path, so its marginals happened to equal its conditionals. |
| **S6** | **Identity has exactly one construction site**, and injectivity is unit-tested over the full grid. | Three V-B bugs in one session were all identity/collision. `dataset_name` omits the seed, so five seeds collided on one id and each run deleted the last — 27 report rows, 8 surviving datasets, silently. A helper-level test is not enough: an inline construction in `main()` collided while the test passed. |
| **S7** | Every long run asserts a **completion invariant**: surviving artifacts == report rows == expected grid. | "27 rows against 8 datasets" was a contradiction the driver could have noticed and did not; it ran 1h39m first. |
| **S8** | Report **"not testable"** as distinct from "passed". | D-E's exclusion had zero residual variance under the deterministic gate — a measurement of nothing that read as a clean pass. |
| **S9** | A check is validated by the **calibration of its p-values under a true null**, not by the absence of failures. A statistic that cannot vary returns the **most confident possible pass on no evidence** — report it as *untestable*, never as passed. | A check that has gone blind and a check that works are indistinguishable by failure count; they differ only in the distribution. Two instances, both found this way and both running in the **flattering** direction. (a) **D-E's exclusion**: `R` deterministic in `(A, U)` under the gate ⇒ zero residual variance ⇒ the statistic is identically zero for the data and every permutation. (b) **D-A-null's `U ⫫ R`**: on CartPole with `c_r = 0` the reward is exactly 1.0 every step ⇒ `var(R) = 0` ⇒ **p = 1.000, the cleanest pass in the table, on nothing** — in the very arm L5's false-positive rate is read from. The suite's p-values must be checked for uniformity: 4 of 7 families came out uniform (KS p 0.69–0.95), and the two that deviated did so *conservatively*, which is what led to (b). Same discipline retires a false alarm: 40 uniform D-D p-values (KS p 0.686) make a single p = 0.005 the expected minimum, not a defect. |
| **C3** | Estimates carry their conditions **on the object**. | `fit.estimate()` is the only way a number is produced, so none escapes without `monotone`/`converged`/`separability`. |

### Library rules (NBN v0.14.0, vendored)

| # | rule | why |
|---|---|---|
| **N1** | Targets and anything feeding a loss → **`sample(do=)`**, looped. Read-only quantities → **`query_batch(do=)`**. | The discriminator is **gradients**, not samples-vs-posterior. `query`/`query_batch` are non-differentiable by design; routing a target through them returns a value with no gradient and presents as a model that will not train. Say which you chose in the comment at each call site. |
| **N1a** | `sample(do=)` shapes: `do` values are **`[1, D]`** (not 0-d, not `(n,)`); evidence expanded to `n` rows. | A 0-d value fails *inside* the sampler. The batched-do axis is a *different* restriction and was **not** the cause of our failure — assuming it would have sent the fix the wrong way. |
| **N1b** | `query_batch` returns **`(weights [B,N], samples [B,N,D])`** for continuous targets, contradicting its annotation. | Posterior mean is the weighted average over particles. Reported upstream. |
| **N2** | `update_local` refuses weights → a refresh **refits**. | The EM M-step is inherently weighted. |
| **N3** | Weighted EM uses **MDN / flow / LinearGaussian**, never KDE. | KDE's bandwidth rule is unweighted, biasing strata toward each other — false negatives, the quiet direction. `supports_weights` is asserted per node so a mistake raises. |
| **N4** | Snapshot parameters with **`copy.deepcopy(state_dict())`**. | `state_dict()` aliases live parameters: `_bias` −0.012977 → **−0.017710 through the snapshot** after an in-place step. A backtrack written the natural way accepts every step it meant to reject, silently. |

---

## State by layer

| layer | state |
|---|---|
| **L1** declared diagram | **Done.** `cell_graph.py`, 9 entries. The entire assumption surface. |
| **L2** identification | **Done.** `identify.py`, derived from graph structure only (never reads `Verdict`, so gate V2 is a real test). Precedence: backdoor → frontdoor → proximal → instrument → lagged (gated) → **unknown ⇒ bounds-only, never point-ID by default**. |
| **arms** | **Done.** D-A-null / D-D / D-E / D-B′ generate, certify, and are byte-frozen against the historical arms. |
| **L3** estimation | **Core done, guarded.** `estimator.py`: EM over an episode-static latent, both channels, monotone-guarded GEM. |
| **L4** uncertainty | **Design decided, not implemented** — see `grace_v2_l4_design.md`. Compatible set = LR confidence region with the threshold **bootstrap-calibrated** (shared mechanism with L5, since chi-2 asymptotics fail on the mixture boundary); validation is q1 against Balke-Pearl for exactness and q2 against MC ground truth for **coverage plus width**. `interventional_sweep` is the evaluation seam; the optimiser must use `sample(do=)`. |
| **L5** falsification | **Not started.** The headline capability. |

### ⚠ S1b — the rule, the mechanism, and what it invalidates

S1 (episode-level *nulls*) was necessary and not sufficient. An episode-level
null over a length-weighted *statistic* is still wrong, and that is what broke
the D-E arm: mean episode length by `(U, I)` cell was 19.5 / 59.0 / 67.4 / 15.4,
so the long "disagreeing" episodes dominated the pooled correlation.

#### The mechanism: episode length is a COLLIDER

The earlier statement of this — "length correlates with anything influencing
behaviour, so it manufactures dependence" — is too loose, and the looseness is
detectable: a regression fixture built to that description **passed the buggy
code**. If a quantity is independent of length, length-weighting is unbiased in
expectation, and no amount of length variation changes that.

What the bias actually requires is that length be a **common descendant** of
both quantities under test:

> `I → A → L ← A ← U`. Everything that drives the action drives survival, so
> `L` is a collider. Weighting transitions by `L` is a form of **conditioning on
> that collider**, and conditioning on a collider manufactures dependence
> between its causes.

This is the version with predictive content — it says *which* checks are hit and
*how hard*:

* a **passive** quantity that drives nothing (the D-D proxies: `parents(Z) =
  {U}`, and `Z` enters no policy) is **barely touched**;
* an **instrument**, whose entire purpose is to move the action, is hit
  **hardest**;
* the bias **peaks when the two interact in the action law**, because then the
  surviving episodes are systematically the ones where they disagree — which is
  precisely the 19.5 / 59.0 / 67.4 / 15.4 pattern above.

Reproduced in `tests/test_arm_preflight.py::_length_coupled_arm`, whose per-step
hazard is a function of the realised action mix: pooled `corr(I,U) = −0.53`
against `−0.003` per episode, on an instrument drawn from its own Bernoulli. The
pre-fix module rejects that instrument at **z = 9.01**.

#### ⚠ THE CAVEAT — do not use the mechanism to narrow the rule

The collider condition governs **dependence** tests, and only those. For a
**marginal** quantity — a proportion, or the mean of an episode-constant
variable — length-weighting biases the estimate whenever `L` depends on **that
one variable alone**. No collider, no second variable, no interaction required:
`E_w[U] = E[L·U]/E[L] ≠ E[U]` the moment `L` and `U` are related at all.

So the operational rule stays **broad**. It is cheap, it is always safe, and it
needs no case analysis at the point of use. The mechanism explains and predicts;
it does not license an exemption. A future reader reaching for "the collider
condition doesn't hold here, so I can pool over transitions" is making the
marginal mistake, and `P(a = a_bad)` — a proportion of an episode-constant-`U`
stratum, and the headline number in the D-D coupling note — is exactly the
quantity it would be made on.

**Do not apply this as a blanket rule.** D-B's lagged-proxy construction is
*genuinely* per-step — repeated within-episode measurement is the whole
construction — so its views stay at transition level. But check that pooling
triples across episodes does not reintroduce length-weighting there rather than
assuming it does not.

**Conclusions currently resting on a transition-level statistic over an
episode-constant quantity — all must be recomputed before they stand:**

1. **The D-D k-rank/informativeness numbers that retired the third-proxy
   proposal** — P(a=a_bad) 0.389 vs 0.485, all three views at k-rank 2, `R`
   strongest at margin 21.82. `Z` and `W` are episode-constant, so these were
   measured the wrong way. The finding may survive; if length-weighting inflated
   `R`'s apparent informativeness, D-D's coupling to P(a_bad) is *stronger* than
   the catalogue records and **the reserve third-proxy remedy becomes live
   again**. Recompute, then confirm or revise the D-D entry's coupling note.
2. **Every preflight certification stamped by V-B** (130 datasets) — hence the
   re-certification pass. The 35 failures are not trustworthy as diagnoses.
3. **The proxy-strength sweep** (margins 1.59 / 2.74 / 4.12 / 3.99 / 3.97) that
   pinned D-D's production strength at 1.5 — same defect, so the pin is
   provisional.
4. **The D-B′ drift autocorrelations** are safe — but understand the exemption
   rather than inheriting it, because the reason is narrower than it looks.
   *Not* simply "`U` is per-step there". Pooling within-episode lag-1 pairs
   across episodes **does** weight longer episodes more. What makes it sound is
   that the autocorrelation is **homogeneous across episodes by construction**
   (a single declared flip probability ρ, identical in every episode), so
   length-weighting merely reweights an already-unbiased quantity.
   **This safety would evaporate silently** if a future variant made the drift
   rate depend on state or policy — ρ would then vary across episodes, longer
   episodes would carry more weight, and the pooled autocorrelation would drift
   toward whatever ρ prevails in long episodes with nothing raising an error.
   Any state- or policy-dependent drift variant must revisit this.

**Worth a line in the paper.** In RL, episode length is an outcome, so any
transition-level statistic over an episode-constant quantity is implicitly
weighted by a variable that policy quality controls — coupling statistics to
behaviour-policy quality through an undeclared channel. A close cousin of the R4
finding, and a trap specific to sequential settings that the proximal literature
never has to confront.

### PREDICTIONS, recorded before the measurement that resolves them

Written down *before* step 3 was run, so the re-measurement is a **test** of the
mechanism rather than a confirmation of it. Resolve each explicitly; a miss is
the more interesting outcome, because it means the mechanism above is
incomplete.

| # | prediction | why | resolution |
|---|---|---|---|
| **P1** | **D-D's k-rank and informativeness numbers largely SURVIVE** the move to episode granularity. | The proxies are passive so the collider mechanism barely touches them; `R` is per-step and downstream of the action, so it is the one view that could move. | **HIT on the conclusion, MISS on the reasoning — and the measure had to be replaced to see it.** *k-ranks:* all 40 read `{Z:2, W:2, R:2}`, margins ≈ 5× on every view, and `R` is the min-margin view in only 13/40 — **`R` does not bind.** *Informativeness:* `s2/s1` is not a valid cross-view scale at all. It is bin-dependent (one Acrobot seed: 0.98 → 0.74 over 4 → 32 bins) because for disjoint conditional supports the histogram rows are orthogonal, so `s2/s1 = ‖p₁‖/‖p₀‖` — **relative concentration, not separation**. On the binning-free AUC, **`R` is the strongest view in 40/40** (1.0000 / 0.9999 vs Z 0.982, W 0.984). Granularity **raised** `R` (transition AUC 0.68–0.74), so length-weighting had *deflated* it — the opposite of the recorded worry, and in the safe direction. |
| **P1′** | If P1 holds the third proxy stays retired; if the numbers move materially the mechanism is incomplete. | — | **Neither branch — the third branch flagged in advance is the one that happened.** The numbers moved *exactly as the mechanism predicts* (`R` is not passive), so the mechanism is intact, and the movement runs **toward** `R`: it retires the third proxy on stronger evidence rather than reviving it. The coupling to `P(a_bad)` (`corr = +0.876` per step) is an artefact of the per-step view and vanishes per episode. The governing quantity is **gated steps per episode, `P(a_bad) × E[T]`** — measured 5.5–239, AUC ≥ 0.9974 throughout — which degrades only near ~1. |
| **P2** | **The three Acrobot Kruskal sum-5 cases reclassify as a FAILED MEASUREMENT** (collapsed quantile grid), not an uninformative view. | Acrobot's reward takes two values (−1, and −1 + c_r under the gate); with 8 quantile bins the grid collapses unless >12.5% of steps are gated, which is exactly the seed/σ-dependent boundary those three sit on. | **MISS, informatively.** The *cause* was diagnosed right — a two-valued view against quantile bins — but the predicted *observable* never appears: **0 of 40** D-D datasets show a collapsed grid, because at episode granularity the view is the episode-MEAN reward, which is continuous. The fix removed the low cardinality rather than labelling it. Consequence: the `binning_degenerate` flag added for this is **dead code on these arms**. Kept (cheap, and the condition returns with other envs or a per-step view) but recorded as never having fired here, so nobody later cites it as evidence the guard is working. |
| **P2′** | If they instead survive as genuine k-rank-1 views, that is a **real result about Acrobot and D-D**. | — | **Did not occur.** All three now read `R:2` and pass. They were a check artifact — of a different kind than predicted, but an artifact. |

### A RULE RECORDED IS NOT A RULE APPLIED

S3 was written down, justified, and applied to the *action* family — and never
reached the *covariate-free* family sitting six lines above it in the same
function, a max over `dims × proxies = 8` statistics read against a per-test
cutoff. That is almost certainly the source of the 3.1–3.7 SD cluster in V-B's
D-D failures.

**Standing task, not a reaction to the next failure:** whenever the statistics
layer is touched, sweep it for **maxima or minima over families still read
against per-test cutoffs**. The tell is a `max(...)` or `min(...)` over a list of
independently-computed test statistics.

**Design constraint on L5 (the headline capability, not yet started).** L5's
conditional-independence tests will face **both** defects — episode granularity
*and* family-wise multiplicity — and it is far cheaper to build them right than
to debug them out of certification failures later. Concretely, L5 must:

1. compute every statistic at the granularity of the quantity tested (S1b),
   reusing `arm_preflight`'s `_episode_constant` / `_episode_mean` rather than
   re-deriving them;
2. judge any family of tests against **the null of the family statistic**
   (S3) — L5 tests many independences at once, so this is its default case, not
   an edge case;
3. read tails as **quantiles of the permutation draws**, never as z-scores, since
   family maxima are right-skewed.

L5 is where a false-alarm rate becomes a headline number, so a multiplicity bug
there is not a nuisance — it is the result.

### NEXT SESSION — start here, in this order

1. ~~Write S1b into this document.~~ **Done — you are reading it.**
2. **Apply the granularity fix** to the correlation families and the k-rank
   views in `src/envs/offline/arm_preflight.py`. Collapse `U`/`Z`/`W`/`I` to one
   row per episode; summarise per-step companions (state, action, reward) by an
   episode statistic. **Keep the exclusion check conditioning on `(A, U)`** —
   `A` is a collider on `I → A ← U` and `U` blocks the opened path — and add the
   docstring note explaining why exogeneity must *not* condition on `A` while
   exclusion *must*. The two sit side by side and the correct treatment differs.
   *A previous attempt failed because an `old` string no longer matched after
   the formatter had touched the file, and an `n_classes` replacement landed in
   `_k_rank_permutation` instead of `check_proxies`. Read the current file
   before editing; do not trust remembered text.*
3. **Re-certify from stored samples** (`~/.minari-grace-v2`, 130 datasets).
   Minutes, not another 8 hours — certification is a stamp, not generation.
4. **Re-measure the D-D numbers** at episode granularity; update the entry.
   **This gates step 6 — see the note below.**
5. **Fix the skewed-tail cutoff** — the null of the maximum is built correctly
   then read with a 3-SD z-score, but a maximum's distribution is right-skewed,
   so use a *quantile* of the permutation draws. Likely secondary to the
   granularity bug, but wrong independently of it.
6. **Then**: fixed-step-budget M-step measurement (O(steps) not O(n·epochs),
   legitimate under GEM, symmetric-safe), constraints-per-diagram count
   (measured, not the estimated 4), and the V-D re-projection — on an idle
   machine, since every timing so far is a contended upper bound.

**Why 4 must precede 6, and is not arbitrary sequencing.** The two open
strategic items — the third-proxy question and the V-D cost projection — look
independent and are not. They are coupled through the **constraint count per
diagram**, which is the multiplier in the V-D projection that is already an
order of magnitude over budget:

> D-D re-measurement → possible third-proxy reinstatement → **a changed
> declared diagram** (A1: the diagram is the only assumption, so a new proxy
> channel is a new catalogue entry, not a config knob) → more views, hence more
> moment constraints per diagram → a larger per-cell cost in the V-D
> projection.

Re-projecting first would price a diagram set that step 4 may invalidate, and
the error runs in the expensive direction: a third proxy adds constraints, so
the projection would come out *low* and a V-D scope chosen against it would be
under-budgeted. Any reordering has to answer this, not just note the
dependency.

### GATE FAILURES — diagnosed (2026-08-16). Two clusters, both CHECK defects.

The gate asks "is the declared confounding present at the declared strength",
which is a different claim from the preflight's "is the arm valid" — so a gate
failure is not automatically a defect. Diagnosed rather than assumed; **neither
cluster turned out to be an arm property.**

#### Cluster B — D-E, 15 failures (14 CartPole + 1 Acrobot): a STALE STAMP

`_instrument_signature` derives `gate_test_passed` from `check_instrument` — the
same function the collider bug lived in. The re-certification pass only
re-stamped `preflight_*` keys, so the gate carried the pre-fix verdict.
Recomputed from stored samples with the fixed check: **all 15 flip to pass, D-E
is 40/40**, and `corr(I, U)` on the flipped rows is −0.070 … +0.054. Re-stamped.

**D-E CartPole goes from 6/20 usable to 20/20**, which was the priority: D-E is
L4's only exact anchor via Balke–Pearl, and six datasets was a thin basis for the
reference that validates the bound engine.

*Generalisable lesson:* a re-certification must re-stamp **every** derived key,
not the ones named after the layer being fixed. Two metadata blocks were computed
by one function and only one of them was refreshed.

#### Cluster A — Acrobot D-D and D-B′, 14 failures: the A2 identity is the TWO-ACTION SPECIAL CASE

Derived from the policy's own swap rule (redraw w.p. σ when `a0 ∈ {a_good,
a_bad}`; within-pair `P(a_bad) = pbar(2−pbar)` if `U` else `pbar²`, `pbar =
p/(p+g)`):

> `E[(1{a=a_bad} − p_s)(2U−1)] = σ · mean( p_s·g_s / (p_s+g_s) )`

The gate predicts `σ · mean( p_s(1−p_s) )`. **These agree iff `p + g = 1`, i.e.
iff the action space is binary.** On a 3-action env the gate over-predicts by
`(p+g)(1−p)/g`. Simulated against ground truth: ratio 1.00 at 2 actions, **0.69–
0.72 at 3** — and V-B measured **0.60–0.78 across all 14 Acrobot rows, stable in
σ**. CartPole (2 actions) passes at every σ, as the derivation requires.

**Compounded by an absolute tolerance.** `corr_tolerance = 0.03` is both a tuned
constant (an **A2-rule violation** that survived the v1 purge) and absolute, so a
*constant relative* error passes at σ = 0.25 and fails at σ = 1.0. That is the
only reason the failures looked σ-dependent and read as "the mechanism does not
bite at high σ on Acrobot". It does; the check is wrong at every σ.

**Consequence, in the direction that matters.** The 26 Acrobot rows that *passed*
passed for lack of absolute magnitude, not because the check was right — so the
A2 gate has been **uninformative on Acrobot throughout**, in both directions. The
arms themselves are fine: the preflight certifies them and the derivation above
plus its simulation establish the generator does what it declares.

**Fix, NOT yet applied — it needs a decision.** The correct target needs
`g_s = π_basic(a_good|s)` logged alongside `p_s`; a one-line generator change,
but an exact re-stamp then needs the 40 Acrobot datasets re-rolled (`g_s` is not
recoverable from `p_s` alone — at 3 actions the same `p_s` is consistent with two
different `g_s`). Options: (a) log `g_s` and regenerate Acrobot only; (b) log it
and leave existing stamps marked "A2 not evaluated on >2 actions"; (c) replace
the absolute tolerance with a relative one, which would make the existing rows
pass but on a formula still known to be wrong. **(c) is not recommended** — it
converts a visible failure into an invisible one.

### L3 — EM SATURATION ON LONG EPISODES, and the four bugs found fixing it

**The failure.** The E-step sums per-row log-likelihoods over the episode, so a
between-class difference of `d` nats/step becomes `T·d` per episode. At `T=500`
the class softmax is a step function: responsibilities are 0/1 after the FIRST
E-step and EM is frozen in whatever basin the initialisation picked. Measured on
D-D Acrobot: **6/6 fits at chance recovery, every one reporting
`separability = 1.0000`.** At `T = 18–38` the same code recovers 0.997–1.000.

**It is OPTIMISATION, not identification, and the objective says so.** Same
model, same data: the proxy warm start reaches `ll = +24k…+42k` at recovery
0.99, random starts sit at `−15k…−35k` at 0.53. The likelihood ranks the correct
solution far higher and EM cannot reach it. That is also what makes the proxies
"load-bearing" in D-D **through initialisation only** — an optimiser-dependent
property of this implementation, not of the identification problem. A better
optimiser could make them decorative again.

**Fixed by deterministic annealing (τ₀ = mean episode length, a derivation:
at τ = T the tempered episode log-lik IS the per-step average). Re-validated on
the real data that exhibited it:**

| env | T | τ=1 (pre-fix) | annealed |
|---|---|---|---|
| CartPole s0 | 16 | 0.917 | 0.847 *(one seed 0.77→0.54; see the seed sweep)* |
| Acrobot s0 | 150 | 0.783 | **0.923** |
| Acrobot s1 | 500 | **0.563** | **0.990** |

#### Four bugs found *inside* the fix — three of them silent

1. **The guard compared objectives ACROSS a temperature change.** τ=500 against
   τ=63 is not a comparison; it read as a catastrophic decrease, exhausted the
   backtracks on iteration 1, and stopped the fit before it ever reached τ=1.
   **Annealing present in the code and absent in effect, reporting an ordinary
   fit.** The reference is now re-evaluated at the new τ.
2. **Mid-anneal exhaustion terminated the run**, leaving the fit maximising a
   surrogate it was only meant to pass through — a do-effect of 1.22 against a
   true 0.75 with every other diagnostic ordinary. Exhaustion during the anneal
   now advances to the next temperature; only τ=1 exhaustion stops the run.
   `reached_tau_one` is what caught it and is now propagated under C3.
3. **A plausible constant that would have been wrong on real data.** The first
   version gated annealing on `initial_saturation ≥ 0.99`, justified as "the
   statistic is bimodal, so any cut decides identically". Asserted from two
   observations and **false** — measured 0.41 / 0.93 / 0.99 across T = 16 / 150 /
   500, a graded statistic. **A 0.99 cut would have declared T=150 unaffected,
   and a third of the pre-fix fits failed there.** Worth reading next to A2: the
   constant was not merely unjustified, it was wrong. The anneal is now extra
   iterations rather than a slice of `max_iter`, which removes the conflict that
   made a gate look necessary; nothing in the control path reads the detector.
4. **`_canonicalise` was silently resetting fields, and had been all along.** It
   rebuilt the fit member-by-member and defaulted everything it did not list, so
   **any fit whose classes happened to need swapping reported `backtracks = 0`
   and `backtrack_exhausted = False` however badly it had struggled — the C3
   labels were lying on roughly half of all runs, at random, for as long as that
   function has existed.** Now `dataclasses.replace`: it copies by construction,
   so the bug class is unreachable rather than fixed, and every field added since
   (and hence) is carried automatically.

#### Diagnostics, and why `separability` had to go

`separability` is a function of the responsibilities ALONE, so it measures the
posterior's confidence and nothing about whether it is right — **1.0000 at 0.53
recovery**. It is retained as telemetry and labelled as such. The correctness
diagnostic is now `separation_per_step` (nats/step between the best and
second-best class), which is length-normalised.

`separation_per_step` is **unbounded above, so a very large value means
DEGENERACY, not confidence** — one fit reported 287,155 nats/step. The two are
distinguished by a **floor detector, never by magnitude**: a continuous
mechanism declares `min_scale`, which bounds its log-density at
`−log(min_scale·√(2π))` per dimension (5.99 nats at the 1e-3 default). That
ceiling is *derived* from the library's own parameter, so it introduces no
constant — a magnitude cut-off on `separation_per_step` would have been an
A2 constant wearing a diagnostic's clothes.

#### Blocking consequence for L4 and L5

Every bootstrap replicate is an EM fit. A saturated replicate's statistic is a
draw from the **initialiser**, not from the sampling distribution, so a null
built from them measures initialisation variance — while looking impeccable:
narrow, smooth, and about the wrong thing. Since L4's compatible set and L5's
thresholds are both read off these nulls, that would propagate into every
calibrated number without ever presenting as an error. Saturated and
stopped-while-tempered replicates are therefore **failed** replicates, in the
same category as an exhausted backtrack budget, counted in `diagnostics()`.

### ⚠ SINGLE-SEED RESULTS AT SHORT T ARE NOT TRUSTWORTHY

The T = 16 anneal question was settled by 10 paired seeds: mean paired difference
**−0.042 at 0.85 SE**, medians 0.995 vs 0.993, **5 of 10 seeds bit-identical**,
and the non-zero differences going both ways (−0.457, −0.100, +0.030, +0.113).
Annealing is a wash at short T and the uniform anneal stands.

**The finding underneath is the more useful one.** Both arms fail on the same
seeds at the same rate — 2/10 and 3/10 below 0.9, with seed 7 failing
*identically* in both (0.517, ll = −8144). So this configuration has a
**multi-modal likelihood at T = 16 regardless of temperature**: two to three
seeds in ten land in a bad basin whatever you do.

Consequences, both of which apply retroactively:

* a **single-seed** result at short T carries a ~20–30% chance of being a
  bad-basin draw, and several earlier readings in this project were single-seed;
* an apparent effect of size ~0.1 between two arms at n = 3 is **within the
  basin lottery** and means nothing. The n = 3 reading that started this
  (0.917 → 0.847) was exactly that.

Any L3 comparison from here reports **paired seeds and the median**, not a
single fit and not a mean over three.

### Open threads

* **V-B** running (`results/vb_generation/`, relaunched after the id fix). Its first run's 4 failures are **discarded** — computed on data later overwritten by the collision. Re-certification happens as part of generation, so no separate pass.
* **D-D's reward-view coupling** — documented in the catalogue, to be *quantified by R4*, deliberately not engineered away. Third-proxy remedy held in reserve, evidence-driven only.
* **L4's threshold must never become a constant** — this is the single most
  likely place for v1's `k` to reappear, since a bound with an unjustified width
  still looks like a bound. The identified-cell collapse check (D-D and D-A to
  ~0 width, D-E not) is what would catch it.
* **C1's splitter assertion** attaches to V-D.
* **D-B's q2 stays gated** — degrades to bounds, does not serve point values.
* The **untestable-assumption section** in `diagram_catalogue.md` is the paper's limitations section in draft. A test enforces that a new untestable assumption cannot be added without appearing there.

### Gotchas that cost real time

* **pre-commit reformats and aborts the commit.** Always `git log` after committing; re-`git add -A` and re-commit.
* **One agent per worktree.** Kill by **PID**, never a bare pattern (`pkill -f regime_sweep` matches `test_regime_sweep.py`). Note also that `pgrep -f <own pattern>` **matches its own shell**, so "still running" can be entirely self-reference — check `ps -eo args | grep -v grep` before concluding a job survived a kill.
* **`cd X && ... &` BACKGROUNDS THE `cd` TOO — a silent write into a sibling checkout.** The `&` applies to the whole `&&` chain, so any *following* line in the same tool call runs in the ORIGINAL cwd. A heredoc written that way landed a script in the frozen `benchmarking_causal_rl` checkout instead of this worktree, with no error anywhere. Guards: parenthesise, `(cd X && ...) &`, or use absolute paths in anything backgrounded. **The sharper risk is the one that did not happen this time:** it created an *untracked* file, which was harmless and visible in `git status`. The same slip onto an *existing* path would have been a silent modification to the frozen v1 branch with nothing to signal it. Check the sibling checkout is clean after any backgrounded write.
* The full test suite exceeds a 10-minute tool timeout; run in chunks with `-k`.
* Sampling from an **unfitted** model raises `assert self._bias is not None` — that is "sample before fit", unrelated to any do-semantics issue.
