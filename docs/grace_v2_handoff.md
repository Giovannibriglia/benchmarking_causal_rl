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
| **P1** | **D-D's k-rank and informativeness numbers largely SURVIVE** the move to episode granularity. | The proxies are passive — `parents(Z) = parents(W) = {U}` and neither enters the policy — so by the collider mechanism they are barely touched. The reward view is the one that could move, since `R` is per-step and genuinely downstream of the action. | **SPLIT.** *k-ranks:* **HIT** — all 40 D-D datasets now read `{Z:2, W:2, R:2}`; no view lost rank, and the 3 that read `R:1` gained it. *Informativeness:* **NOT YET SETTLED, and the early sign is against P1.** The recorded margins are not comparable across the change (the denominator moved from `max` of 40 draws to the `q99` of 200), so the old→new table conflates two edits. On the comparable raw statistic `s2/s1`, three spot datasets show **R at 0.426 on CartPole against Z's 0.923** — i.e. R is markedly the *weakest* view — where the old transition-level margins made it the **strongest** (8.66 vs 4.09). On Acrobot R is 0.991, the strongest. If that reversal holds under step 4's matched measurement, **P1 fails on exactly the quantity the third-proxy decision rests on**, and the caveat above applies: `R` is per-step and downstream of the action, so it is the one view the collider mechanism predicts *should* move. |
| **P1′** | If P1 holds, **the third proxy stays retired** on sound evidence. If the numbers move materially, **the mechanism is incomplete** and that finding outranks the catalogue question. | — | **Open — step 4.** Note the third branch neither P1 nor P1′ anticipated: the numbers may move *as the mechanism predicts* (R is not passive), which would leave the mechanism intact and still revive the third-proxy question. Do not read "the mechanism held" as "the proxy stays retired". |
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
* **One agent per worktree.** Kill by **PID**, never a bare pattern (`pkill -f regime_sweep` matches `test_regime_sweep.py`).
* The full test suite exceeds a 10-minute tool timeout; run in chunks with `-k`.
* Sampling from an **unfitted** model raises `assert self._bias is not None` — that is "sample before fit", unrelated to any do-semantics issue.
