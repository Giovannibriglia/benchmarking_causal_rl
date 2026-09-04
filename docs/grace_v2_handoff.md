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
| **S1c** | **A quantity that is constant within an episode enters the likelihood — and every statistic — ONCE PER EPISODE, never once per row.** | Per-row entry silently multiplies its weight by the episode length, and episode length is an OUTCOME. Four instances in four modules, one of them documented as a trap and recurring anyway: (1) the length-weighted exogeneity statistic (S1b); (2) L4's walk constraint carrying `T·log p_U(k)` per episode (the V4 bounds failure); (3) the per-row-prior trap `_episode_log_liks`'s own docstring warned about; (4) **the proxies, measured 2026-08-27**: episode-constant Z/W/V enter the likelihood per row, weighting each proxy channel by exactly T (d100 CartPole s0: between-class gaps 78–90 nats as coded vs ~5 counted once — the *dominant* channels, vs A 0.78 / R 39.7). Belongs in the paper's methods discussion, not only here. |
| **S2** | Test the **conditional** claim, never its marginal shadow. | Marginals are nonzero *by design*: proxy⟂A reads +0.50 marginally vs +0.003 given `U`; proxy⟂S reads 0.226 marginally because `U→A→S`. And conditioning on *more* can be wrong — `A` is a **collider** on `I→A←U`, so conditioning on it alone opens `I→U→R`. |
| **S3** | A statistic must be compared against **a null that describes the statistic actually computed**. | A *maximum* over 8 tests judged by a per-test 3-SD cutoff runs ~8× the intended false-alarm rate — it fired on provably covariate-free proxies. Build the null of the max. Likewise a *binning failure* (Acrobot's reward is −1 almost everywhere) is not an uninformative view. |
| **S4** | Validate the **generator against ground truth**, and the estimator against the generator. Never mutually. | If each validated the other, a shared misconception passes in silence. Preflight reads logged `U` and declared parameters only — never L5, never the estimator. |
| **S5** | Validate on **real generated data**, not only a synthetic harness. | Three checks passed a synthetic harness and failed on the real generator: the harness's states carried no `U→A→S` path, so its marginals happened to equal its conditionals. |
| **S6** | **Identity has exactly one construction site**, and injectivity is unit-tested over the full grid. | Three V-B bugs in one session were all identity/collision. `dataset_name` omits the seed, so five seeds collided on one id and each run deleted the last — 27 report rows, 8 surviving datasets, silently. A helper-level test is not enough: an inline construction in `main()` collided while the test passed. |
| **S7** | Every long run asserts a **completion invariant**: surviving artifacts == report rows == expected grid. | "27 rows against 8 datasets" was a contradiction the driver could have noticed and did not; it ran 1h39m first. |
| **S8** | Report **"not testable"** as distinct from "passed". | D-E's exclusion had zero residual variance under the deterministic gate — a measurement of nothing that read as a clean pass. |
| **S10** | **A failure condition is a claim about how often something goes wrong, and that claim EXPIRES.** Whenever the estimator's end-state semantics change, re-examine every condition that consumes them. | Twice now a condition was specified when it was rare and became **typical** as the estimator matured, and in both cases it would have failed the **modal** production replicate while looking principled. (a) *Saturation* — written as a failure, then measured at 0.95–1.00 on **every** T = 500 fit including those recovering at 0.99, so it would have rejected every null on the long-episode environments L4 and L5 most need. (b) *`backtrack_exhausted`* — written as a failure, then the **stationary** end state was introduced and every stationary fit sets it; the production-scale CartPole fit finishes by stationarity, so the rule would have failed the typical replicate. Both surfaced only because something else forced a re-read. A third instance is the dead test found the same way: a fixture asserting `converged=True` **and** `backtrack_exhausted=True` — impossible for the estimator to produce — silently stopped exercising anything once the test became `finished`. |
| **S11** | **Subsample the REAL data. Cut data, never cut the optimisation budget.** Default to subsampled real datasets for anything diagnostic, iterative or exploratory; reserve production scale for runs meant to be reported. | We over-corrected from S5. That lesson was real — a hand-built T = 500 fixture recovered 1.000 *without* annealing, so validating the tempering fix against it would have measured nothing — but the conclusion drawn was "use production scale", which conflates two things. **The fixture failed because it was hand-built and lacked the structure that caused the pathology, not because it was small.** A few hundred episodes drawn from an actual arm preserves episode lengths, reward support, the `U` mechanism, the proxies, the covariate distribution and the discrete/continuous structure — everything that produced every pathology found so far — at a tenth of the cost. Saturation is driven by episode length (preserved); reward degeneracy by the support (preserved); line-search behaviour by the objective's curvature (largely preserved). **The distinction that must be held:** the D-D ablation cut episodes *and* `max_iter`, and the budget cut is why every fit came back unconverged and the value-level half was unanswerable. **Fewer episodes with a full fit is a smaller experiment; the same episodes with a truncated fit is a broken one — and it fails in the direction that looks like a result.** |
| **S9** | A check is validated by the **calibration of its p-values under a true null**, not by the absence of failures. A statistic that cannot vary returns the **most confident possible pass on no evidence** — report it as *untestable*, never as passed. | A check that has gone blind and a check that works are indistinguishable by failure count; they differ only in the distribution. Two instances, both found this way and both running in the **flattering** direction. (a) **D-E's exclusion**: `R` deterministic in `(A, U)` under the gate ⇒ zero residual variance ⇒ the statistic is identically zero for the data and every permutation. (b) **D-A-null's `U ⫫ R`**: on CartPole with `c_r = 0` the reward is exactly 1.0 every step ⇒ `var(R) = 0` ⇒ **p = 1.000, the cleanest pass in the table, on nothing** — in the very arm L5's false-positive rate is read from. The suite's p-values must be checked for uniformity: 4 of 7 families came out uniform (KS p 0.69–0.95), and the two that deviated did so *conservatively*, which is what led to (b). Same discipline retires a false alarm: 40 uniform D-D p-values (KS p 0.686) make a single p = 0.005 the expected minimum, not a defect. |
| **S12** | **Bit-identical output from a procedure that should vary means THE CODE DID NOT CHANGE.** Check the file on disk before believing any relaunch — never trust a patch script's own success message. | Two runs of a stochastic procedure cannot agree to three decimals. Caught the Q2-A "exact anchor" run reproducing the sampled-U run bit-for-bit: the heredoc printed success and `ast.parse` passed, but the write never persisted, so the same code ran twice and a verdict computed from a DUPLICATE row was about to be reported as a third measurement. This project has now tripped on unpersisted or misdirected edits four times (the `old`-string that no longer matched after the formatter; the `n_classes` replacement landing in the wrong function; the backgrounded `cd` writing into a sibling checkout; this one). The tell is cheap and universal, and it generalises past patching: any "changed" configuration whose output is bitwise identical to the unchanged one changed nothing. |
| **S13** | **Where a nuisance variable perturbs only the REWARD — never the dynamics, and unseen by the policy — take its expectation in CLOSED FORM, never by sampling.** | The trajectory is then identical under every draw, so `E_U[·]` is exact and carries NO Monte-Carlo noise. Sampling instead makes the metric a one-draw estimate of the same quantity, and the variance lands wherever the comparison's noise band is. Measured twice: the Q2-A anchor's RMSE fell **3-5x** (s1 16.35 → 5.85, s2 5.73 → 1.16) purely from switching sampled-U to analytic-U, because a sampled anchor leaves each state's return `U`-CONDITIONAL while the estimate is `U`-MARGINAL; the same substitution at `eval_env` buys the experiment statistical power for free. **Always ask whether the nuisance touches dynamics: if it does, this does not apply.** |
| **S14** | **For a component whose failures are SILENT, the smoke must include a POSITIVE CONTROL: a quantity predicted to move, and a check that it moved in the predicted direction.** "No error" is not evidence that such a component works. | The GRACE seam produced three silent failures in one build, each of which would have yielded plausible numbers and a clean-looking experiment: (a) an additive offset that DOUBLE-COUNTED the base critic's own contrast — and because doubling preserves sign, argmax policies mostly would not move, so the no-harm prediction would have PASSED while the seam was broken; (b) an extractor reading buffer attributes that `ReplayBuffer` does not expose, which would have made every run abstain, and abstention is *designed* to look like a scope decision; (c) declared proxy channels never reaching the buffer, so a proximal cell would have quietly fitted the ablation's "without" arm. None raises. The control that separates them is one number: the SERVED contrast against the BASE critic's contrast — predicted strictly lower, since the base carries the upward `M · tilt` bias that GRACE removes. Equal ⇒ passthrough in disguise; higher ⇒ sign inverted. Same family as S12 (bit-identical output means nothing changed): both are cheap tells for changes that fail without complaining. |
| **S15** | **Require every run to record WHAT IT ACTUALLY DID, not only what it produced.** For a component that can no-op silently, the artifact must carry evidence of the action — which branch ran, what changed, how long it took — because output that looks right is compatible with nothing having happened. | The E1 smoke's leaf said `"grace": null` and `seconds: 400` where a served run takes ~5000; the CSVs, the dataset id and the returns were all correct and plausible. `_needs_episode_grouping_run()` is True whenever critic ablation is configured, so every run took the GROUPED offline path while the transform hook sat on the FLAT one — every grace arm would have been byte-identical to its baseline, for 36 hours, with **P1 passing perfectly, P3 absent and P5 reading "the bias did not cross a decision boundary"**, and the conclusion that GRACE does nothing. No test caught it and no error was raised; the provenance record did. Companion to S14 (positive controls) and to the input-check rule: of six silent failures in this seam, FIVE were inputs or wiring arriving wrong rather than outputs computed wrong, and that class is invisible to output inspection **by construction**. The fix form matters too — ONE construction site called from both paths, never a second hook, which removes the class rather than the instance (the same move as `_episode_log_liks`' deduplication). |
| **S16** | **A component that fails silently needs a check PER INPUT, not a check per output.** | Same tally: the additive offset (double-counted the base's own contrast), the unreadable `ReplayBuffer` (attributes vs `gather()`), the declared proxies never reaching the buffer, the inverted pessimism sign, the dataset-id collision across cells, and the hook on the unused branch. Only the first is an output defect; the rest are inputs arriving wrong. Checking outputs would have caught one of six. |
| **S17** | **A positive control belongs on the REPORTED ENDPOINT, not only on the components that feed it.** Before committing a long campaign, run one cell and verify the headline metric *behaves like the quantity it is named after* — in a plausible range, and MOVING as the thing it measures changes. A flat or few-valued series fails, however healthy every component is. | The seventh silent failure, and the only one invisible to every component check. `rollout_len` carries three meanings — on-policy collection length, the legacy offline budget, and the EVAL HORIZON. On an offline run the first two are inert (`offline_grad_steps` sizes the learner), so E1 set it to 2 for them and thereby evaluated every policy over **two environment steps**. Deployment return was then `2.0 + bonus_rate × (a_bad steps among those 2)`: exactly three reachable values, and the observed set across all 17 completed cells was `{2.0, 2.5, 3.0}` — the predicted set, nothing else. A pole cannot fall in two steps, so trained, untrained and broken policies all scored the same, and **every return-based prediction (P1b included) was untestable while looking like a clean pass**. Nothing errored, the CSVs were well-formed, the seam was correct, the critic ablation was correct, S14's positive control on the served contrast PASSED — the components all worked and the metric they fed measured nothing. Six of seven failures were caught by checks on components; this one required checking the endpoint itself. Fix form per S15: separate the meanings (`eval_rollout_len`, defaulting to `None` so existing runs are byte-identical) rather than correcting the one call site. |
| **S18** | **A point null of exact equality, tested against a flexible model, rejects given enough resolution — the null is false by construction. When the test feeds a decision, the hypothesis must be an EQUIVALENCE REGION whose tolerance is derived from the decision's own scale.** | Ruled 2026-09-02 on L5's Markov falsifier: on exactly-Markov deterministic CartPole the history block always improves the structured approximation residual by a positive sliver (measured Delta-R^2 1.6e-7, shrinking 56x with base capacity), so "history adds exactly zero" rejects at ANY capacity while the true-POMDP effect sits 5 orders higher (1e-2, capacity-stable). Same move as the walk's derived stop and the bootstrap's MC-error criterion: the tolerance comes from a quantity the pipeline already measures (here L4's own interval), never from a chosen number. Derivation: `docs/l5_equivalence_tolerance.md`. |
| **S19** | **A feature set used to certify a model's sufficiency must be the feature set the model receives.** Certifying `k` on richer features than the estimator gets is the same defect as validating one estimand and serving another. | Ruled 2026-09-03. The fifth instance of the validated-≠-served family — the additive offset, the max operator, the learned termination head, and now the POMDP branch's augmentation: L5 selected `k` with lagged `(O, A, R)` history blocks while `pomdp_branch._augmented_cols` served lagged `(A, S)` only, so a `k` certified because of past rewards was a `k` the estimator could not honour. Stated as a rule about FEATURE SETS rather than quantities because that is the generalisation the four earlier instances were pointing at. Resolution: the family's history blocks are `(O, A)` only (the served columns); the R-inclusive block survives as the reward-channel DIAGNOSTIC, where a reward-only-visible hidden state belongs by construction. |
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

**ELEVATED TO A PATTERN (2026-08-21) — a stated property of sequential
benchmarks for the paper, not a lab note.** Four independent instances, each
found the hard way: (1) the length-weighted exogeneity statistic
(corr(I,U) −0.590 pooled vs −0.034 per episode); (2) the k-rank permutation
null; (3) the D-D proxy-vs-state test; (4) the `a_bad` identity inference in
the naive-bias gate (row-level tilt flipped the inferred action on low-σ
CartPole seeds). And a fifth manifestation INSIDE a demonstrand rather than a
check: the transition-pooled naive bias on CartPole flips SIGN across policy
seeds while the episode-level version is uniformly positive — the collider
term can dominate and invert a naive estimate, not merely blur it
(`results/cost/naive_bias_gate.log`). The general statement: **in sequential
data, any transition-pooled statistic involving an episode-constant quantity
is implicitly reweighted by a collider that policy quality controls; at low
coupling strength the collider term can exceed and re-sign the effect being
measured.**

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
   observations and **false**. ⚠ **The conclusion holds; the reason first
   recorded for it does not — corrected here rather than left resting on a
   superseded measurement.** The original evidence was a gradation
   0.41 / 0.93 / 0.99 across T = 16 / 150 / 500, read as sat0 tracking episode
   length. Those were **MDN-R** fits. Under categorical-R the *same environment
   at the same T* reads **0.983**, not 0.41 — so sat0 is **not primarily a
   function of T**. It tracks **per-step separation × episode length**, and the
   categorical mechanism raises per-step separation sharply; the original
   gradation confounded episode length with mechanism mis-specification. The A2
   conclusion survives and is stronger: sat0 is near-saturated almost everywhere,
   including on a **converged, correct, recovery-0.991 CartPole fit at T = 18**,
   so *any* cut in the upper range misclassifies healthy fits. **A 0.99 cut would
   have declared T=150 unaffected, and a third of the pre-fix fits failed
   there.** The constant was not merely unjustified, it was wrong. The anneal is now extra
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

**And bitwise comparisons are the wrong instrument entirely (2026-08-19).**
The L3 fit is nondeterministic run-to-run on CUDA at a fixed seed: two
bitwise-identical `consolidate=False` runs (same data, same config, RNG
verified identical by construction) returned the same `n_iter` and recovery
but final_ll −18330.6 vs −18342.8 and different parameter hashes
(`consolidate_share_isolated.log`). So a fixed-seed A/B cannot establish that
two configurations produce "the same fit" — a ~12-nat ll spread is the noise
floor of a single run pair, and any smaller effect is unreadable. Equivalence
claims need the paired-seed distributional test above; fixed-seed pairs only
measure rates.

**RESOLVED (same day): the nondeterminism was removable, and determinism is
now the default.** `tools/probe_l3_determinism.py`: torch's
`use_deterministic_algorithms(True)` raises on nothing in the fit path and
makes repeats bitwise identical — re-confirmed at the scale that failed
(`results/cost/l3_noise_floor.log`: two 300-episode fits bit-identical, ll
equal to the last bit) at ~5% cost. The measured pre-determinism floor,
k = 5 identical fits at 300 episodes: **ll spread 131.7 nats, n_iter
8/8/8/10/11, five distinct parameter states, recovery constant at 0.99** — so
every ll gap the consolidate A/B produced (12, 143, 210 nats) was at or inside
the floor and none was readable. Decision (2026-08-19):
`fit(deterministic=True)` is the DEFAULT for every reported run — commit
`9625b85` is the switchover point, and pre- and post-switchover numbers are
not naively comparable, since deterministic kernels are not bit-identical to
the default ones. The mode travels on the fit and its estimates under C3;
`NONDETERMINISTIC-KERNELS` flags the unusual case. **FAIL-LOUD:** torch raises
if an op with no deterministic implementation ever enters the fit path (none
does today). If a future change trips it, that is the flag working — add a
deterministic implementation or consciously label the run; never silently
disable. One consequence for the consolidate question: with deterministic
kernels plus `--isolate-consolidate-rng`, the A/B becomes bitwise-decidable in
a single run, so re-measurement-queue item 4 can be that instead of a
paired-seed campaign.

**The transferable lesson — amplification through discrete decisions.** The
mechanism behind the floor: parameters can return bitwise identical while the
evaluation differs by 5e-4 nats (measured at 100 episodes), and **a guard that
makes a DISCRETE decision on a continuous quantity converts arbitrarily small
numerical noise into macroscopic path divergence.** Three such sites, all in
the EM loop: the backtrack accept/reject, the convergence window, and the
stationary rule — whence 131 nats and three different n_iter values from one
configuration. Two consequences: (a) fit fragility is a property of the
OPTIMISER, not only of the measurement — re-check after warm-start, since
continuation from θ_old should reduce the sensitivity, and if it does not,
that is worth knowing before V-D runs thousands of fits; (b) the
pre-determinism bootstrap nulls contained this fit noise on top of resampling
variance — conservative, so nothing is invalidated, but see the note in
`bootstrap.py`'s docstring.

**The floor reaches the COST numbers too — n_iter is 40% of the story.**
`n_iter` came back 8/8/8/10/11 across identical fits, and total fit time is
`n_iter × per-iteration cost`, so every total-time figure quoted so far is a
single draw from a distribution with ~40% spread in its dominant factor: the
fork verdicts (CartPole 8.6 min, Acrobot 11.9 min — the qualitative verdict
"minutes, not hours" survives easily; the figures do not support their quoted
precision), the M-step lever range (×2.01 / ×3.92 / ×3.62 / ×12.68 — some of
its cross-run drift, previously read as configuration-dependent, is this, and
using the conservative end in projections was right for a reason unidentified
at the time), and the consolidate share (66.2 / 56.2 / 59.2%). None of these
is wrong and none changes a decision already made, but until re-measured they
are RANGES, not figures.

**The re-measurement queue** (post-merge, post-sync, everything with
deterministic kernels ON, diagnostics on subsampled real data per S11):

1. **L3 re-validation** at T = 16/150/500, warm-started. Nothing at
   production scale before this confirms the estimator behaves warm-started.
   Includes the fragility re-check above.
2. **Cost re-measurement**, both environments, fork verdict per environment —
   explicitly including every figure in the paragraph above, quoted as a
   figure only from here on. The consolidate SHARE will shrink as warm-start
   cheapens the baseline while the absolute saving persists; label
   accordingly.
3. **V-D re-projection** against the declaration matrix, three scenarios
   (current, +A, +A+B).
4. **Consolidate equivalence** — one bitwise run (deterministic kernels +
   `--isolate-consolidate-rng`), upgraded from the paired-seed campaign.
5. **D-D proxy ablation** — a FIRST measurement, not a repeat; it was void
   for three separate reasons.

**What the floor does NOT touch: recovery held at 0.99 in all five runs.**
The noise lives in the log-likelihood and the parameters, never in the
latent — so every conclusion resting on RECOVERY (tempering fixing T = 500,
discrete-R rescuing CartPole, the 2×2 attribution) is unaffected by the
floor. Do not over-generalise the invalidation: it reaches numbers denominated
in nats or seconds, not the latent-recovery results.

**The boundary.** The first post-sync measurement will be the first in this
project made with a stable instrument — warm-started (GEM, not restart-EM)
AND deterministic. Everything before it was measured through one or both
instabilities; comparisons across the boundary go through the C3 labels
(`algorithm`, `deterministic`), never through memory.

### THE BOUNDARY WAS CROSSED 2026-08-19 — first stable-instrument results

NBN v0.15.0 synced (`c007b7a`), warm-start adopted (`a75b274`: GEM by
default, lr-halving retry, per-iteration reset, the tau=1 retry loop dead
under GEM, `algorithm="gem"` on the label).

**L3 re-validation (GEM + deterministic, random init, 3 fit seeds,
`results/l3_validation/report_gem.json`; pre-warm-start record kept at
`report.json`):**

| dataset | tau=1 | annealed |
|---|---|---|
| CartPole T=16 | 0.980/0.980/0.960 | 0.980/0.980/0.980 |
| Acrobot T=150 | 0.630/0.940/0.920 | 0.940/0.980/0.980 |
| Acrobot T=500 | 0.990/0.990/**0.690** | 0.990/0.990/0.980 |

**Tempering SURVIVES, re-measured** — and the attribution sharpened:
warm-start alone rescues 2/3 untempered seeds at T=500 (was 0/3), the anneal
turns that into 3/3. GEM shrinks the damage saturation does; the anneal
removes the trigger. Anneal still a wash at T=16. The bad-basin tail shrank
everywhere (worst fit 0.630 vs 0.530 pre-warm-start).

**Fragility (`results/cost/l3_fragility.*`):** identical-fit pairs are
BITWISE STABLE under the new stack, n_iter included — so identical
configurations now have quotable costs. But a 1e-7 perturbation of ONE input
element moves final_ll by **60 nats under GEM** (146 under restart-EM):
continuation damps the response ~2.4× and does NOT remove it — **the
optimiser path is chaotic in its inputs in both regimes.** Recovery was
bit-for-bit unmoved in every case; the chaos lives in ll/parameters only.
Consequences: (a) a single fit's ll carries no meaningful precision below
tens of nats with respect to data representation — two fits on
NEARLY-identical data are incomparable at the ll level; only identical
inputs compare, and then bitwise; (b) bootstrap replicates legitimately
absorb this response as part of sampling variability (the symmetry rule
covers it: observed and replicates share the procedure); (c) V-D must never
read a raw ll difference across arms without its null — which its design
already forbids, but this is now a measured reason rather than a policy.

### OPEN ARCHITECTURAL ITEM FOR THE CRITIC-SEAM BLOCK (recorded 2026-08-23)

**L2's verdict does not select the estimator — OR ITS CHANNELS (one item,
two faces; merged 2026-08-24).** L2 returns point-ID for
`d_a_null` by back-door adjustment on S with NO latent required — and nothing
in the pipeline acts on that: L3 is always the latent-class model regardless
of the verdict. The architecture's promise is that L2 delivers the estimand
FORM and L3 estimates that form; in practice the verdict is not consulted.
Tolerable now (the value estimate is right either way — V-C1's V1 measured
no harm at ≤ 0.001), but it bites at the critic seam: a critic applying
latent machinery to unconfounded cells will abstain or waste effort where the
observational estimate is exactly correct. Why it never surfaced: every cell
exercised in anger has a latent. L4's smoke surfaced it through the
`d_a_null` abstention (resolved at the mechanism level by the Dirac routing;
the estimator-selection gap remains). The SAME item's second face, surfaced
by L4's D-E work: GRACE's model has no INSTRUMENT channel, so an LR region
over the I-blind latent-class model is validly wider than Balke–Pearl.
Ruled 2026-08-24: **serving the closed form on D-E is the seam WORKING, not
a workaround** — L2's verdict there is "bounds-only via Balke–Pearl" and L4
implementing exactly that estimand form is verdict-directed estimation,
hand-wired for one cell; the general mechanism (verdict selects estimator
AND channels) is the seam block's job, and both faces must be fixed
together. The LR-region optimiser's production home is **D-B-prime**
(bounds-only with no instrument, 40 certified datasets): exact method where
the diagram licenses one, LR region where it does not. On D-E both are
reported — the closed form as what the DECLARED DIAGRAM licenses, the
I-blind LR region as what the latent-class model alone licenses; the gap
between them is the measured value of declaring the instrument.

### THE D-D REVISION — OUTCOME (2026-08-22, the one-place record)

What a fresh session needs to know about why the cell looks the way it does;
close to paper-ready. The chain: ablation found the proxies decorative → the
diagnosis was R too strong, not proxies too weak → V added as the ENABLER
({Z, W, V} triple decoupled from R) → R's informativeness swept via
compensated gate separation (d swept, c_r = M/d derived, estimand invariant)
→ σ = 0.25 uniform (gate: naive bias clears its σ = 0 null at 4.5× / 18.4×)
→ regenerated, certified, swept.

**The four results:** (1) the decorative→load-bearing transition exists and
is LOCATED on CartPole (details below); (2) **the demonstration is
CartPole-scoped by measurement** — Acrobot is the boundary case locating the
limit, by the structural claim now in the catalogue entry: action-channel
informativeness and confounding both scale with σ while the reward channel's
scales with d × E[T], so at long horizon there may exist NO regime where the
proxies are load-bearing and confounding is present; (3) estimand invariance
of the compensated instrument is measured (M-normalised do-contrast errors
within ±0.15 M across c_r 1 → 20, no drift); (4) the generalisation test
passes — ONE configuration, zero per-environment values, 176/228 converged,
0 non-monotone, across the full decorative→load-bearing range and an order
of magnitude in episode length. Non-convergence CONCENTRATES where expected:
44/52 in the without-arm, 23 of them at CartPole d ≤ 0.10 — the fits whose
information was removed; the with-arm converged 93%.

### The sweep, measured (2026-08-22)

One GRACE configuration (the audited defaults, nothing per-environment)
across d ∈ {1.0, 0.5, 0.25, 0.10, 0.05} at M = 1.0, σ = 0.25, both
environments, 228 converged-budget fits
(`results/dd_sweep_ablation/summary.txt`; per-seed, M-normalised, per the
entry's binding reporting constraints):

* **The transition exists and is LOCATED on CartPole**: proxies decorative at
  d = 1.0 (gap ≈ 0, replicating the original finding with V present), the
  gap opens at d = 0.5 (+0.15 on s0), is decisive by d = 0.25 (+0.42), and
  by d ≤ 0.10 the without-arm sits at chance on 2/3 dataset seeds while the
  with-arm holds ≥ 0.995 (gaps +0.43…+0.50).
* **Acrobot's transition is beyond the grid's weak end for 2/3 seeds** —
  gaps reach only +0.18 at d ≤ 0.10 and its without-arm holds 0.81–0.98 even
  at d = 0.05. The separation × E[T] law, quantified: at T ≈ 150 the
  episode-mean R keeps AUC ≥ 0.86 at d = 0.05 and the action channel adds
  the rest, so the reward-side lever alone cannot make the proxies fully
  load-bearing there. That asymmetry is a RESULT (the same law, fourth
  appearance), not a failure of the sweep.
* **The compensation works at the value level**: M-normalised do-contrast
  errors stay within ±0.15 M across the whole sweep with no drift in c_r
  (1 → 20), so the estimand-invariance claim is measured, not argued.
* **Per-seed reporting earned its keep**: CartPole's s1 without-arm resists
  collapse (0.86–0.99) where s0/s2 fall to chance — the policy-seed
  dependence the reporting constraint anticipated.
* **Generalisation reading**: 176/228 fits converged, 0 non-monotone, one
  configuration throughout — the direct answer to "are you overfitting to
  CartPole and Acrobot", pending the caveat that the ablation records
  predate the C3 binding flags (the tool must record
  `tau1_budget_bound`/`backtrack_exhausted` before the paper's binding table
  can cite this sweep; the audit's own probes carried them).

The σ = 1.0 contrast stands on the frozen datasets (2026-08-19 ablation:
without ≥ with in all six blocks) — reported together with the curve as the
declared contrast per the catalogue entry.

### THE RE-MEASUREMENT QUEUE — COMPLETED 2026-08-19/20, first quotable figures

**2. Cost + fork (`results/cost/grace_fit_cost_gem.*`, 3000 episodes,
fixed-step M-step, converged, clean C3):** CartPole **65.1 s**, Acrobot
**42.7 s** — Acrobot FASTER despite 8.6× the transitions (the fixed-step
budget decouples per-iteration cost from n; it needed 9 iterations to
CartPole's 16). **THE FORK DOES NOT SPLIT: minutes everywhere, cadence refit
viable in both environments, the per-cell architecture split is unnecessary,
and the online Acrobot cells are unblocked.** M-step lever ×13.98 (CartPole)
/ ×147.95 (Acrobot) — the lever is a function of n, which is what the old
cross-run drift was.

**3. V-D re-projection (`results/cost/vd_projection_gem.log`, declaration
matrix, measured costs):** full grid B=99/dataset = 57,420 fits = 6.0 d at
6-way; **+A (pooled nulls) = 1.2 d**; +A+B (B=39) = 0.5 d at p-resolution
0.025. Option A is in reach.

**4. Consolidate equivalence — CLOSED, BITWISE
(`results/cost/consolidate_share_gem.*`):** determinism control identical;
True vs False identical (n_iter, final_ll to the bit, ex-EWC params). Pure
overhead ESTABLISHED; the divergence that opened the question was the Fisher
pass consuming global RNG. Direction note: the SHARE rose to 83% under GEM
(warm-start cheapens the denominator); the invariant is the absolute ~49
s/iter. The `consolidate=False` default now rests on measurement.

**5. D-D proxy ablation — FIRST measurement
(`results/dd_ablation_gem.log`, matched random inits, 3×3×3×2):**

**The proxies are DECORATIVE for latent recovery, everywhere.** The
without-proxies arm recovers 0.995–1.000 (best-LL) in all six
(env, dataset-seed) blocks and is ≥ the with-proxies arm in every one
(with: 0.980–0.998). Under the fixed stack (GEM + anneal + discrete-R +
deterministic), U is recoverable from (S, A, R) alone from a RANDOM init —
so even the "load-bearing through initialisation only" role the proxies
held under the old optimiser is gone. **The docstring's feared conclusion is
the measured one: D-D currently functions as a back-door cell wearing
proximal clothing — an estimator can succeed on it without exercising the
proximal channel, and V-C would report success on machinery it never used.**
This threatens the CELL'S PURPOSE and is a design decision for the author,
not a parameter choice: candidate remedies are the reserve third proxy /
weaker R-coupling (the catalogue's coupling note quantifies why R is doing
all the work: episode-mean R separates U at AUC 1.0), or re-scoping D-D's
claim. A1 applies — whatever is chosen is a new catalogue entry.

Caveats carried on the numbers: the ablation ran the EPOCH-BASED M-step
(the tool has no m_step_budget passthrough — worth adding before any re-run;
a fixed-step re-run would be ~50× cheaper), and most verdict-arm fits are
`conv=False` at max_iter=12 — recovery conclusions are plateaued and
seed-stable, but the VALUE-LEVEL do-contrasts (CartPole spread 0.32–0.74
across arms and seeds; Acrobot 0.48–0.55) are unconverged-fit numbers and
carry their C3 labels; do not quote them as estimates.

### ⚠ WHAT WAS MEASURED THROUGH THE MIS-SPECIFIED REWARD — an audit

Recovery **0.543 against 0.983** on CartPole means the MDN reward was not a
precision issue: on that arm the estimator was **failing outright**. Every
measurement taken with it on the real arms is therefore suspect until re-checked,
and the ones below are **void rather than imprecise** — an unconverged number can
be tightened, a number from a broken estimator cannot.

| measurement | status |
|---|---|
| **D-D proxy ablation** | **VOID, for two independent reasons.** (1) Its value-level numbers (the 14× do-contrast error on CartPole s0, the −0.099 on Acrobot s1) were computed through a degenerate reward density, and its *latent-level* conclusion is equally suspect because the estimator that measured it was failing at chance on that arm. (2) **Its SELECTION criterion is also void.** It picked the reported fit by `max(per_seed, key=final_ll)` — deliberately, so the answer would not choose the method — but under a floor-dominated reward density **a higher likelihood may indicate a MORE degenerate fit, not a better one**. The criterion may have been selecting *for* degeneracy. **The re-run is a FIRST measurement, not a repeat.** |
| **L3 re-validation** (T = 16/150/500) | Ran under MDN-R. The 0.563 → 0.990 headline is confounded — see the 2×2 below, which is the more interesting consequence. |
| **T = 16 seed sweep** | Ran under MDN-R. Lower stakes (its conclusion is "no effect"), but it carries the caveat until re-checked. |
| **Cost numbers** | Being redone with both fixes. |
| **D-D informativeness / third-proxy retirement** | **UNAFFECTED — confirmed, not assumed.** `tools/measure_dd_granularity.py` imports only `numpy`/`json`/`os`/`pathlib`; the AUC, the k-ranks and the margins are pure functions of the logged data and never construct a model. The retirement stands. |

**Likelihood-based SELECTION is only valid when the likelihood is.** This is the
sharper form of the point above and it reaches further than the ablation. The
proxy-init probe's central evidence — proxy-init at `ll ≈ +24k…+42k` against
random-init's `−15k…−35k`, taken as clean proof that the estimand is identified
and the optimiser is the defect — was also measured under MDN-R. Under
categorical-R log-probabilities are bounded above by zero and comparable across
fits, so **re-check that comparison**. A 40k-nat gap is unlikely to be
manufactured by a scale floor and the conclusion will probably hold, but the
load-bearing evidence for "optimisation, not identification" currently rests on a
criterion we now distrust, and that should not be left standing on trust.

**The 2×2 that the re-validation cannot substitute for.** Tempering and discrete
R both landed between the 0.563 → 0.990 result and now, and only the first was
measured. **Tempering may have been compensating for a degenerate reward
density** — in which case discrete R alone fixes T = 500 and the anneal is doing
less than advertised. `tools/validate_t500_2x2.py` separates them:
`{τ=1, annealed} × {MDN-R, categorical-R}`, three seeds each, on the corrected
`ceil(log2 τ₀)` schedule. Any of the three outcomes is a result; **attributing
the fix to the wrong cause is not.**

### T = 500 2×2 — RESULT: both fixes are load-bearing, partially redundant

| reward | τ=1 | annealed |
|---|---|---|
| **MDN-R** | **0.550** (0/3 good) | **0.990** (3/3) |
| **categorical-R** | **0.980** (2/3) | **0.990** (3/3) |

**Tempering alone rescues T = 500 under MDN-R**, so the 0.563 → 0.990 headline
stands as measured. Discrete-R alone also gets there in the median but not
reliably (one seed in three still at 0.56). Only the combination is 3/3. Neither
fix is doing the other's work, and neither owns the result.

Two things the 2×2 showed that it was not designed to look for:

1. **SATURATION IS NECESSARY BUT NOT SUFFICIENT.** `initial_saturation` is
   0.95–1.00 in **all twelve** fits, including the categorical cells that recover
   perfectly at τ=1. So the E-step saturates regardless; what separates recovery
   from chance is the basin structure the mechanism creates. Saturation is the
   precondition, the mis-specified likelihood was the trigger. **This corrected
   the bootstrap contract — see below.**
2. **RECOVERY AND LIKELIHOOD VALIDITY COME APART.** `degenerate_mechanism` fires
   on 2 of the 3 MDN *annealed* fits — the ones recovering at 0.99. A fit can get
   the latent right while its reward density sits on the floor. Since L4 and L5
   read the **likelihood**, not the labels, **discrete-R is required for them
   even where tempering already fixes recovery.** An independent argument for the
   change, separate from the CartPole recovery result.

#### ⚠ CORRECTED: saturation is a RISK flag, not a FAILURE flag

The first version of the bootstrap contract failed any replicate whose E-step
saturated. **That was wrong, and finding 1 above is what showed it**: on the
T = 500 arm every fit saturates, so the rule would have failed *every* replicate
on exactly the long-episode environments L4 and L5 most need — and with
`max_failure_rate` defaulting to zero, rejected every null there **while looking
principled**. The wrong version is preserved because the general lesson is:

> **A diagnostic that fires on healthy fits is a RISK flag, not a FAILURE flag.**
> Failing on it conflates "this was hard" with "this went wrong".

Revised: saturation fails a replicate only when **nothing was done about it** —
saturated *and* no annealing, i.e. init-determined. Saturated with the anneal
active is the diagnostic doing its job. The genuine failure conditions are an
exhausted backtrack budget, a fit that stopped mid-anneal, a **degenerate
mechanism**, and non-convergence.

### 🔍 THE PATTERN: a diagnostic added for one failure keeps catching a larger one

Four times now, and it is an argument for the C3 discipline itself rather than an
anecdote — **it belongs in the paper's methods discussion, not only here**:

| diagnostic added for… | …immediately caught |
|---|---|
| **saturation detector** (frozen E-step) | the anneal terminating mid-schedule and returning a tempered surrogate |
| **`reached_tau_one`** (incomplete anneal) | a smoothed surrogate being returned as an estimate, reading like a converged fit |
| **scale-floor detector** (one 287,155 outlier) | **R mis-specified as continuous on every arm in the benchmark** |
| **C3 labels** (conditions travelling with values) | `_canonicalise` resetting them, lying on ~half of all runs at random |

The common structure: each pathology was **already occurring and already
invisible**, and what made it visible was attaching a condition to the number
rather than inspecting the number. None of the four was found by looking at a
result and doubting it; all four were found because a value arrived carrying a
label that contradicted it.

### THE FORK — CartPole answered; the architecture SPLITS rather than collapsing

**CartPole: 198.7 s (3.3 min), CONVERGED, 6 iterations, every C3 condition
clean** — against 3913 s unconverged before the discrete-R and anneal-rung
fixes. `MINUTES → cadence refit viable`. Measured lever: fixed-step M-step
33.1 s/iter against epoch-based 129.9 s/iter, **×3.92** (up from ×2.01, because
the categorical R made the fixed-step path cheaper while the baseline still pays
O(n·epochs)).

**If Acrobot lands on hours the architecture does not collapse, it SPLITS**:
GRACE is a cadence-refit critic where fits are cheap and a fit-once-then-serve
critic where they are not. That is defensible **only if the split is DECLARED
PER CELL rather than discovered at runtime** — a critic that silently changes
its refresh semantics with the environment is exactly the kind of undeclared
channel A1 exists to forbid.

The concrete consequence, which is narrower than a general limitation: it would
block **the online Acrobot cells specifically**, because those need refresh and
refresh means refitting (N2: `update_local` refuses weights). Offline Acrobot is
unaffected — it fits once by construction.

**V-D re-projection, ready to run but NOT yet run.** At 3.3 min per converged
fit, `B = 99` is ~5.5 h serial per constraint and well under an hour at six-way
parallelism, against the earlier estimate of 1.7–4 h *per replicate*. That is a
different problem, and **option A (the full grid) may be back in reach on
CartPole**. Do not project until Acrobot reports: Acrobot decides whether the
grid is uniform or must be split by environment, and projecting on CartPole
alone would answer the easy half.

### 🛑 THE M-STEP IS NOT A STEP — NBN R3 is OPEN and BLOCKING

> **STATUS 2026-08-19: implemented and verified upstream** (branch
> `feat/warm-start-fit-local`, full suite green, PR #260 in review). Nothing
> below is retired until it is **merged, tagged and synced into `nbn/`** —
> GRACE is still running restart-EM against the vendored snapshot, the
> `RESTART-EM-PARAMS-PROVISIONAL` label stays on, and the re-measurement
> freeze stays in force.

`MDNMechanism.fit_local` executes `self.net = _build_mlp(...)` with a fresh Adam
**on every call**, so GRACE's M-step is an **independent refit from random
init**, not a partial maximisation of `Q(θ | θ_old)`. GEM's guarantee
presupposes continuity from `θ_old`. **Without warm-start, GRACE's EM is not
EM.**

Four things chased as separate problems are one cause:

| symptom | explanation |
|---|---|
| the line search is **inverted** — measured Δ objective at `lr ×1 / ×0.5 / ×0.25 / ×0.0625 / ×2.4e-4` = **−26 / −110 / −1224 / −3148 / −4573** | a smaller `lr` is not a gentler step, it is a **worse fresh fit**, so every retry is worse than the last and exhaustion is guaranteed |
| the guard compares against a **moving baseline** — even `lr ×1` returns worse by 26 | the refit is stochastic |
| **non-monotone likelihood tails**, twice read as ill-conditioning | refit stochasticity |
| per-iteration cost never falls as the fit approaches its optimum | every iteration relearns from scratch |

**Interim: the algorithm is `restart-EM`, not GEM, and every fit says so.** The
workaround retries with **more epochs** rather than a smaller `lr` — with a
fresh-fit M-step, more optimisation is what gets closer to the maximiser. It is
directionally right and it changes the algorithm: even *accepted* steps are
stochastic refits, so the parameter sequence may never settle after the objective
plateaus. `converged` can therefore fire on ΔLL while the parameters still jump,
and **anything reading the PARAMETERS rather than the objective — interventional
values, L4 bounds — is provisional**. `Estimate.label()` carries
`RESTART-EM-PARAMS-PROVISIONAL` until warm-start lands.

#### Retroactive scope — what survives, what must be re-measured

| result | status |
|---|---|
| **discrete-R** | **SURVIVES UNTOUCHED** — a modelling-correctness result, independent of the optimiser. |
| **tempering / T=500** | **Probably survives, re-check rather than assume.** Saturation is an **E-step** property: responsibilities saturate, the partition freezes, and refitting mechanisms from a frozen partition reinforces it whether the refit is warm or cold. Tempering attacks the E-step. It is load-bearing, so re-check it after warm-start. |
| **all cost numbers, both fork verdicts, the V-D projection's absolutes** | **MUST BE RE-MEASURED.** They rest on per-iteration costs that change materially once iterations stop restarting from scratch. Expect improvement. The *lever ratios* (5× pooling, the declaration matrix) are unaffected. |
| **CartPole "finished by stationarity", 8.6 min** | **WITHDRAWN** — `stationary` meant "no fresh refit at a degraded `lr` beat the incumbent", not stationarity of the objective. |
| **D-D proxy ablation** | **VOID**, now for a third reason. |

**Do not re-run any of it until warm-start is available.** Measuring twice is the
one thing worse than measuring late. The productive work meanwhile is the V-D
design document, which needs no optimiser.

### V4 — THE L4 GATE: verdict accepted (2026-08-27)

Run complete (48 rows, `results/v4/report.json`); verdict assembled by
`tools/v4_verdict.py` (read-only render — run it for the tables).

**Intervals PASS.** Coverage 32/36 = 88.9% against nominal 90%; all four
misses are hair's-breadth (outside by 0.0001–0.0059) and at the
strong-identification end; the weak end (d ≤ 0.25) is 18/18. Collapse behaves
as designed. **d025's widths are 0.05–0.16, recorded as MEASURED** — the
pre-registration wrote "~0", and small is a measurement where silence is not.

**Bounds FAIL on the optimiser, not on the bounds** — 8/12 walk rows return
width 0 with all three starts on an identical endpoint. The separable reading
the pre-registration set up: an optimiser finding, with the interval result
standing independently. Two safeguards earned their place: **multi-start**,
mandated as insurance, became the *detector* (three starts on one endpoint is
unambiguous), and **D-B-prime's coverage row** was designated the empirical
exploration test and fired. Neither failure would have been visible without
them.

**The Balke–Pearl anchor is CARTPOLE-ONLY, now measured** (flagged as a
possibility at D8, confirmed here): the anchor filters to in-pair actions
{0, 1}, and on 3-action Acrobot BP misses truth on two seeds. D-E's Acrobot
rows have no valid closed-form reference and the instrument-value gap cannot
be computed there. Same family as the A2 gate's two-action special case
(Cluster A above); on CartPole the gap awaits the walk fix.

#### The walk diagnosis (2026-08-27) — the constraint measures the wrong functional

`tools/diagnose_l4_walk.py` on one frozen row (d_e CartPole s0) and one
moving row (d_b_prime CartPole s0); log at `results/v4/walk_diagnosis.log`,
data `results/v4/walk_diagnosis.json`. The four prescribed checks all come
back CLEAN — starts distinct (param distances 0.29/0.59), target gradient
non-zero (|g_t| ≈ 1.06), the feasibility gate rejecting *correctly*, one
tangent step moving LR by only ~0.15 — because the defect is upstream of all
four:

> **`_observed_ll_differentiable` and `fit.final_ll` compute DIFFERENT
> functionals.** `final_ll` mirrors `e_step`: per-row log-liks of the
> channels (A, R, proxies) summed per episode, prior added ONCE per episode.
> The walk's functional calls full `model.log_prob`, which also includes the
> S marginal AND `U`'s mechanism log-prob **per row** — each class column
> gets `T·log p_U(k)` per episode instead of `log p_U(k)` once. That is the
> documented "classic bug that makes the posterior scale with episode
> length" (`_episode_log_liks` docstring), sitting in the L4 walk.

Measured on the frozen row: `LR(θ̂) = 70,686` against `c = 821` — the walk
starts 86× outside its own region at the UNPERTURBED θ̂ clone, 150 steps of
restoration cannot close it, no iterate is ever feasible, and the fallback
returns θ̂'s target for every start: width 0, identical endpoints. Verified
exactly: recomputing with the e_step-mirroring functional gives
**LR(θ̂) = 0.0 to the bit** (per-row U sums −42,340/−30,763 per class are the
injected tilt; the S marginal adds −36.7).

Two consequences beyond the 8 frozen rows. (1) **All twelve bounds rows are
invalid, ruled 2026-08-27**: d_b_prime CartPole s0 has LR(θ̂) = 2,498 ≈ 3c —
small enough for restoration to bridge in two steps, so the walk ran, but
inside an offset region. The frozen/moved split is just the mismatch's
magnitude (rows × prior skew) against c, which is also why the frozen rows
are the LARGE-c rows — the inversion that flagged this. (2) The exactness
test's RF parametrisation moving is consistent: what differs on production
is the per-row U tilt's size, not the walk.

**FIXED by deduplication, ruled and landed 2026-08-27 (`141ee6f`).** Not by
re-synchronising the two implementations — two implementations of one
functional that must agree by discipline is what produced the bug, and
re-synchronising re-arms it. `_episode_log_liks` is now **the one
construction site** (gains `model=`/`differentiable=` for L4's clone walk);
the L4 wrapper delegates, so agreement is structural. The invariant
`LR(θ̂) == 0` is pinned by `test_lr_at_theta_hat_is_zero` — the one-line
test that would have caught this — which also asserts the constraint still
carries gradients. The autograd leak at the threshold check landed
separately (`f880840`). The V4 bounds block re-runs in full after the fix;
the interval block is untouched and stands.

##### The bounds block RE-RUN (2026-08-27, post 141ee6f/ef238a5/c7420db)

All 12 rows re-ran with the fixed functional, replicate pinning and the
stratified criterion (`results/v4/report.json`, bounds rows replaced; the
pre-fix record is in git history at `2f5411d`). Three findings:

1. **The walk mechanics are fixed.** Every row moves; multi-start spreads
   are third-decimal (healthy variation, no fallback signatures); widths
   0.05–0.17.
2. **D-E: 6/6 walk rows cover truth**, Acrobot included. But the measured
   relation to Balke–Pearl INVERTS the D8 expectation ("the I-blind LR
   region is validly wider than BP"): the walk is much NARROWER (CartPole:
   walk 0.08–0.14 vs BP 0.78–0.84). Two readings, both live: the
   latent-class model's parametric assumptions (finite-K + mechanism
   classes) are far stronger than BP's nonparametric ones, so its
   confidence region is legitimately tight — and the walk is an inner
   approximation whose width is BUDGET-LIMITED (finding 3), so part of the
   narrowness is truncation. **The instrument-value gap is therefore NOT
   yet a clean measurement anywhere**: it needs a walk run to plateau, and
   it remains CartPole-only (BP invalid on 3-action Acrobot).
3. **D-B-prime: 0/6 cover — and a 600-step probe attributes it to the STEP
   BUDGET, not the region.** The designated exploration gate fired
   genuinely this time: the min-walk on CartPole s0 descends 0.760 → 0.574
   over 600 steps and is still descending where production's 150 steps
   stopped at 0.67. So the under-coverage is inner-approximation
   truncation. The disclosed inner-approximation semantics ("every bound is
   achieved by some compatible model") remain TRUE — the bounds are valid
   inner bounds — but as a COVERAGE gate the walk needs either a much
   larger budget or a plateau-based stop (a relative-improvement criterion,
   reported not thresholded, to stay inside A2). That is the open optimiser
   decision; note D-B′ is also the drift arm, so once the budget question
   is settled, any residual gap between the converged region and truth
   reads as the episode-static model's bias under within-episode drift —
   worth keeping separable.

#### The s1 pattern — diagnosed: a CHECK artifact of reward-type resolution (S10)

All seven high-failure rows (>20% replicate failures) are dataset-seed s1,
both environments, every failure `degenerate_mechanism`. Diagnosed at the
data level, no fits needed (`results/v4/report.json` diagnostics + exact
replicate resampling):

* **s1 is the long-episode seed**: Acrobot s1 has **exactly 1 episode in
  3000 that terminates** (all others truncate at T=500; CartPole s1 means
  131/86 steps vs 15–46 on s0/s2). The terminal step's reward creates a
  ONE-EPISODE rare level (`0.0`, or gate+terminal `4.0/10.0/20.0`).
* On an episode resample that draws the rare-level episode but lands it
  only in the second half of the data order, `_resolve_reward_type`'s
  half-vs-full support-growth criterion **mis-resolves R as continuous →
  MDN-R on two-atom data → scale floor → `degenerate_mechanism` → failed
  replicate.** Reproducing the exact resamples (seeds `fit_seed+1+i`)
  predicts the recorded failure counts on every Acrobot s1 row: 6/6, 6/7,
  6/6, 6/6, 6/8.
* This is an **S10 instance recorded in advance by its own docstring**:
  "errs toward CONTINUOUS … merely restores the previous behaviour" was
  written when MDN-R *was* the previous behaviour; discrete-R then made
  MDN-R a flagged failure, and the harmless error direction became a
  20–40% replicate-failure budget. The failed replicates are not broken
  fits of the estimand — they are replicates fitted with a DIFFERENT
  mechanism class than the observed fit (a symmetry-rule violation induced
  by re-resolving per replicate).
* Consequence for the verdict: the 10.1% failure budget is **part artifact**.
  **Both fixes ruled and landed 2026-08-27, as separate commits**: replicate
  pinning (`ef238a5` — the symmetry rule applied to the model CLASS: the
  observed fit's resolved mechanism travels to its replicates via
  `pin_reward_resolution`, wired into both L4 statistic closures) and
  episode-stratified half-sampling (`c7420db` — the old half was the first
  n//2 rows IN DATA ORDER, a fragility of the observed fit too; Acrobot s1
  resolved correctly only by luck of where its one terminating episode sat).
  Pinning fixes the symmetry violation; stratifying fixes the criterion.
  Until the affected blocks re-run, quote the budget with this note.
* The remaining ~39 background failures (1–4 per row, s0/s2 included) are
  NOT this path — attributed by exact replicate refit
  (`tools/attribute_replicate_failures.py` →
  `results/v4/replicate_attribution.json`, counts reproduce the recorded
  4/19 and 3/19): **a PROXY MDN (V, W or Z — one channel at a time) on the
  scale floor, R categorical throughout.** Mechanism: proxies are
  episode-constant, so every episode contributes T identical copies of one
  value and a resample-duplicated episode contributes 2T; a mixture
  component can buy likelihood by spiking onto such a point mass. The
  propensity scales with episode LENGTH, which unifies the s1 pattern:
  both failure modes are episode-length-driven, which is why s1 — the
  long-episode seed in both environments — is elevated everywhere. The
  detector excluding these replicates is correct behaviour (their
  likelihood is floor-measuring); the quotable decomposition of the 10.1%
  budget is ~30 support-growth mis-resolutions (Acrobot s1) + ~39 proxy
  point-mass collapses (background, length-elevated).

#### ✅ THE PROXY PSEUDO-REPLICATION BUG — RULED A BUG AND FIXED (2026-08-27)

**Ruled (not an estimator-semantics choice): a misspecification, and the most
serious violation available under A1.** The declared diagram draws Z/W/V once
per episode from `p(·|U)`; entering them per row implements `p(Z|U)^T` — a
model the diagram does not describe — while every other decision in this
project has been defended on the grounds that the diagram drives the
estimator. Textbook pseudo-replication: one observation counted `T` times as
`T` independent draws.

**Fixed in two commits, because the E-step alone is incoherent.**

* `ee2c0f6` — **E-step**: per-step channels (A, R) summed per row,
  episode-constant channels added ONCE, prior once. Same construction site as
  the walk dedup, so L4 inherits it. `EpisodeData` now REFUSES a proxy that
  varies within an episode (exact equality — the rows are copies), so D-B's
  genuinely per-step lagged construction cannot be routed through this
  channel silently. Tests pin the semantics: doubling every episode's rows
  doubles A/R and leaves the proxy term untouched (`2·ll(T) − ll(2T) == P`,
  which fails by a factor of T pre-fix).
* `68908cc` — **M-step**, part of the same bug: `model.fit` has no node
  subset, so one stacked call necessarily fits the proxies on per-ROW
  duplicates. EM would then maximise `p(Z|U)^T` while the E-step scores
  `p(Z|U)`, and the monotonicity guard would watch a quantity the fit is not
  optimising. Fitting them again afterwards does NOT fix it — the two pull
  against each other every M-step and the corrected objective DECREASES
  (measured: backtrack exhaustion at 24 backtracks / 5 iterations). Each
  channel is now fitted once at its own granularity, with the fixed step
  budget derived per node from that node's rows, and `set_mechanism`
  (public) re-registering each fit so the factor cache cannot go stale.

**Measured after, on real data (d100):** proxy gaps **0.58–0.60 nats**
against R 4.03 (CartPole s0) and R **134.7** (Acrobot s1, T = 500) — where
pre-fix the proxies read 78–90 and dominated everything. Recovery **1.0000**
on both, CartPole converged and monotone with 7 backtracks. So the corrected
estimator is materially different exactly as predicted, R now dominates by a
wide margin, and **the D-D decorative-proxies finding should be expected to
STRENGTHEN** — but every result measured through the old likelihood is
re-run, not re-labelled.

**The line-search depth expired with it — RESOLVED BY MEASUREMENT (`d60cb23`).**
Acrobot s1 (T = 500) recovered 1.0000 and stayed MONOTONE but exhausted its
backtrack budget at iteration 9, so `finished=False` — an L4 abstention
condition, i.e. Acrobot rows would have abstained rather than reported.
Monotone + exhausted + correct is the signature of a line search whose DEPTH
binds, and S10 applies exactly: `max_backtracks = 6` was measured
*never-binding* **under the pre-S1c likelihood**, and removing the T-fold
proxy term changes the objective's curvature. Measured rather than argued
(`tools/probe_backtrack_depth_s1c.py`, `results/s1c_backtrack_depth.json`):

| row | depth 6 | depth 10 | depth 14 |
|---|---|---|---|
| Acrobot s1 (binds) | exhausted, `finished=False`, ll −1,556,324 | **finished**, ll −1,312,563 | identical to 10 |
| CartPole s0 (control) | finished, ll −42,654.398, n_iter 22 | identical | identical |

Depth 10 recovers **243,761 further nats** and finishes; 14 is identical, so
10 is non-binding rather than merely deeper. **The control is the half that
makes it a measurement**: an already-converging fit is bit-identical at every
depth, so a deeper budget cannot perturb a fit that already converges.
Default raised 6 → 10. `backtrack_exhausted` still reports if it binds again
— that is the flag working, not licence to raise again without measuring.

#### The walk's stop is now DERIVED, not a step count (`02b275f`, ruling 2)

A fixed step count is a constant and a relative-improvement threshold is a
constant wearing a different hat; neither survives A2. The bound already
carries a Monte-Carlo error, because `c(α)` is a QUANTILE of B replicates and
`bootstrap.mc_error` is its resampling SE — in LR units. Converting it to
TARGET units is free: the walk records every feasible iterate, so the bound
under a perturbed threshold is the best target among iterates satisfying it,
and the tolerance is the bound's spread across `c(α)`'s own MC interval.
Below that, further movement is unmeasurable — smaller than the noise in the
region the bound is taken over.

`steps` becomes a SAFETY LIMIT (4000 in V4, since the 600-step probe was
still descending where the old fixed 150 stopped at 0.67). Per the binding
audit's own rule — a budget that never binds is a safety limit, one that
binds is a knob and must be disclosed — a truncated walk keeps its valid
INNER-APPROX semantics but gains a **`BUDGET-TRUNCATED`** sub-condition on
the label, and `meta` records which of plateau/budget ended each start. So a
truncated bound can never be mistaken for a converged one.

#### The measurement that established it — confirmed 2026-08-27, the S1c fourth instance

Checked before any fix was applied, on Giovanni's direction, because it
outranks the fixes: under the true generative model an episode's likelihood
carries `p(Z|U) p(W|U) p(V|U)` **once**; as coded, `_episode_log_liks` sums
the proxy channels per row, so each proxy channel is weighted by the episode
length. Verified structurally (proxies are transition-aligned in
`EpisodeData`, within-episode variance ~1e-10 — exactly episode-constant)
and quantitatively on fitted d100 CartPole s0: per-episode between-class
gaps **Z 90.5 / W 86.4 / V 78.5 nats as coded** against **~5 nats counted
once** — per-row/once ratio median 15.0 vs median episode length (mean T
16.6). **The proxies are the dominant likelihood channels by an artifact of
weighting** (A contributes 0.78, R 39.7). The M-step has the same shape: the
proxy MDN is fitted on T duplicates per episode (2T when a resample
duplicates an episode), which is the point-mass-collapse mechanism above.

Honest scope of what it explains: the saturation link is **partial** — on
this short-T dataset removing the over-weight does NOT de-saturate (R's
39-nat gap saturates alone; saturation is over-determined at short T); at
T = 500 the proxy term is ~2,500 nats as coded vs ~5 once, so long-T
behaviour is where the amplification lives. And it means every
latent-recovery number was produced under a misspecified likelihood — not
necessarily wrong (over-weighting an informative channel can still recover
the latent, and recovery held 0.98–1.0), but **known before anything is
re-run**. Changing the weighting is an estimator-semantics decision
(S10: every downstream condition re-examines; it touches D-D's
decorative-proxies result, the proxy warm start, and both calibration
layers) — **not yet ruled**. The refactor that closed the walk bug carries
the current semantics unchanged, with the misspecification recorded in
`_episode_log_liks`'s docstring. Bounds cells are untouched by it (their
fits declare no proxies), so the V4 bounds re-run does not wait on the
ruling.

#### Q2-A step 1 — transition model VALIDATED, with a split verdict (2026-08-27)

`tools/validate_transition_model.py` → `results/q2a_transition/report.json`.
Held-out (episode-split) one-step and open-loop multi-step error along
logged action sequences (the logged trajectory IS ground truth —
deterministic dynamics), both candidate mechanisms, per catalogue fact 3 the
parents are (S, A). Note the d-sweep shares trajectories across d, so the
distinct data distributions are d_a_null + one per (env, seed).

* **CartPole: LinearGaussian is essentially exact** — one-step 0.0008–0.0014
  normalised RMSE, open-loop error ≤ 0.05 at horizon 50. MDN is 30–100×
  worse one-step (0.03–0.13). LG's residual scale sits on the `min_scale`
  floor on 3 of 4 dims (predicted: deterministic dynamics, the discrete-R
  property) — fine for sampling/backups, but its log-densities are
  ceiling-pinned and must not become likelihood-bearing without the same
  treatment R got.
* **Acrobot: the verdict flips** — LG one-step 0.20–0.41 (genuinely
  nonlinear dynamics), MDN 0.08–0.32. But MDN's open-loop rollouts DIVERGE
  catastrophically on s1 (normalised error 242–489 from horizon 10) while
  LG stays 0.4–1.2 to horizon 499.
* **For q2's fitted iteration the load-bearing metric is ONE-STEP accuracy**
  (backups sample s′ at logged states; the model is never rolled far), so:
  LG on CartPole is unambiguous; on Acrobot neither mechanism is clean and
  the choice (or a better mechanism) is a design decision the d_a_null
  machinery check should inform. Long-horizon mean-rollouts leave the data
  support and explode for BOTH mechanisms on CartPole s1 (h=200: ~4,500) —
  expected extrapolation, recorded so nobody reads it as a fit defect.

#### Q2-A step 2 — the d_a_null machinery check (2026-08-27): CartPole works, Acrobot does not, and the cost question is closed

`tools/run_q2a_danull.py` → `results/q2a_danull/report.json` (K = 500;
the K = 60 first pass is preserved at `report_k60_cartpole.json` as the
horizon-truncation measurement — each sweep extends the backup horizon ~one
step, so K must exceed the discount horizon; at K = 60, CartPole s1 came
back biased −13.3 on RTG ≈ 57 and the bias vanished at K = 500).

* **Cost: NOT prohibitive — the design stands.** ~0.02 s per sweep; 80M
  transition samples per row-pair; VI 10–12 s per mechanism per row;
  estimator fit 5–7 s; target-policy rebuild ~60 s. The prereg's "many
  sample(do=) calls per backup" worry is answered at cell one.
* **CartPole, LinearGaussian: the machinery WORKS.** V̂ vs exact on-policy
  RTG: RMSE 0.36 / 2.86 / 0.97 on mean |RTG| 6.5 / 56.6 / 17.3 (≈ 5–6%
  relative), biases −0.02 / −0.69 / −0.38. With per-step reward error ~0
  (Dirac R) this error is pure transition + termination-boundary — no
  amplification. MDN is 3–10× worse throughout, matching step 1.
* **Acrobot: NEITHER mechanism is usable as-is** — the check did its job
  before any substantive cell. LG: RMSE 26–31 on |RTG| 36–80 with large
  biases of BOTH signs (its 0.2–0.4 one-step error compounds). MDN: s0 is
  its best row (RMSE 17.5, bias −0.6) but s1 is biased −42 and **s2
  DIVERGES outright — RMSE 34,188, sup-change 18,377**: step 1's MDN
  rollout divergence materialising as fitted-iteration divergence.
  **Registered prediction 1's amplification branch FIRED here** ("much
  worse means the fitted iteration AMPLIFIES rather than accumulates — a
  finding about the iteration"): function approximation + bootstrapping +
  off-support model samples. The Acrobot transition mechanism is now a
  REQUIRED design decision before any Acrobot q2 cell (a mechanism class
  fitted for deterministic multi-dim dynamics — neither an LG nor a 3-MDN).
* Two pipeline notes for the assembled q2 block: (1) a diverging fitted
  iteration must FAIL a cell, never report a number — wire sup-change into
  the C3 conditions (the s2 row shows the failure shape); (2) the final
  sup-change can be large at buffer states outside the anchor's support
  while anchor-region V is stable (CartPole s1: supΔ 31 with RMSE 2.86) —
  report it, don't gate on it blindly.

### KNOWN GAP — a diagram cell has never trained through `run_cell` (2026-09-01)

`SweepSpec.arm_generator_kwargs` splats `proxy_strength`, `instrument_strength`,
`u_drift`, `n_proxies` into `EnvConfig`, which has none of them, so
`_run_point` raises `TypeError` on any cell with a `diagram:` key. The diagram
cells were only ever GENERATED (`tools/generate_diagram_arms.py`) and consumed
by estimator-level tools; none was ever trained through the sweep driver.

**Worse, and the reason E1 does not use it:** `_dataset_id` is built from
`(prefix, regime, env, beta, sigma, seed)` — **the cell name is not in it**.
Every D-D sweep cell is `offline_mdp` at beta=0, sigma=0.25, so they all
collide onto ONE id. Training them through the sweep would have generated
fresh, uncertified data and trained every cell on the SAME dataset, with every
comparison between identical arms, completing without error.

**Not fixed** — deliberately, and not to be fixed mid-experiment. `tools/run_e1.py`
bypasses it and PINS the certified id read from the generation reports, with two
assertions before any training: ids distinct across cells, and every id present
in the store and carrying its certification stamp.

### E1 PILOT STOPPED BY DECISION (2026-09-03) — d025/d010asym incomplete, deliberately

The pilot was stopped after `d025/cql/grace/s1` promoted. `danull`, `d100s0`
and `d100` are COMPLETE (12 leaves each, every cross-algorithm grace pair
bitwise-identical — 10/10) and are harvested as the observability contract's
(declared MDP, true MDP) row. **`d025` and `d010asym` are instrument-design
archive under the 2026-09-02 reframe** — they were built to measure P3's
correction-share ordering and the asymmetric point's return gains, neither of
which survives as a claim — and finishing them cost ~1 GPU-day plus the CPU
contention that was doubling the L5 calibration sweep's runtime. The 6
completed d025 leaves are kept as-is; the cell is marked incomplete-by-
decision with this paragraph as the reason. Do not resume them without a new
ruling; `tools/run_e1.py` would happily continue if re-invoked.

### RECORDS OF 2026-09-03 (afternoon) — selector readings, parity pass, probe capture, two near-misses

> **Superseded the same evening (see "RULING 2026-09-03 (evening)" below):
> the selector readings and the under-cut table are a measurement of a
> WITHDRAWN design — `dr2_cut` was stripped as a per-environment constant
> (A2) and selection moved to materiality-by-refit. The parity pass, the
> probe capture and the two near-misses stand.

**The L5 selector's two readings (calibration, CartPole, 55 rows at the time
of writing).** `k_selected = None` was ambiguous on the page between "every
stage rejected" and "budget-bound"; they are the SAME event (a run of
`k_max + 1` tests with no pass), and `tools/calibrate_l5.py` now states both
readings per env (`summary.<env>.selector`): the as-deployed statistical
selector (`dr2_cut=None`, what the sweep ran) and the same rows re-read
under the stated cut (`--dr2-cut`, default 1e-4), with the cut CHECKED
against the measured gap (`cut_in_measured_gap`). Null rows (n = 27):

| reading | k = 0 | k = 1 | k = 2 | budget-bound |
|---|---|---|---|---|
| as deployed (statistical only) | 1 | 6 | 4 | 16 |
| under the cut 1e-4 | 27 | 0 | 0 | 0 |

Gap: null max 5.8e-7, masked min 5.4e-4 (separation 925x), so the cut is
inside it. This pair IS contract row 2's headline ("over-assumption is
cheap" holds only if k = 0 is selected on true-MDP data): the cut-less
selector chases floor rejections to `k_max` on 16/27 nulls; under the cut
k = 0 on 27/27. Rows now store every stage's (lag, p, stat) so any cut can
be re-read post hoc; rows written before that field existed decide k = 0
from stage 0 alone and are otherwise reported "undetermined" (so the masked
rows' under-cut k is undetermined for the early rows, never guessed).
**Rule:** the in-flight sweep process holds the OLD code and writes its own
`report.json` at exit — run `uv run python tools/calibrate_l5.py
--report-only` afterwards to rebuild the two-reading report from
`rows.jsonl`.

**Parity re-run RECORDED.** The chain `pilot-exit -> GPU preflight -> parity
tests -> cost probe` ran as a background task whose output file survived:
`tests/test_proximal_vectorized_parity.py` — 4 passed, 60 s, 2026-09-03
11:09, GPU 12 MiB used at preflight. That is the clean-tree check for
everything built during the CUDA-OOM window, on record. The POMDP cost probe
(d100 sigma=0 seed 0, GPU) is the tail of the same chain, so its output
lands in that task file at exit; a watcher copies it to
`results/pomdp_cost_probe.log` the moment the probe exits (it was NOT
dead-piped, and was not killed — 5 h in, progressing). Augmented state dim
is analytic, not measured: obs + k(obs + 1) — CartPole 9 at k = 1, 14 at
k = 2.

**Two near-misses pre-commit surfaced on the first commit of the L5 build
(recorded because each resolved to the intended value by luck, not design).**
(1) `build_generator_agent` passed `mask_indices=behavior_mask_indices` to
`_train_generator` without DECLARING the parameter (flake8 F821): every
caller so far took the default path, so the masked-behavior generation
through that entry point had never executed — it would have raised
`NameError` on first use. Declared with default `None`. (2)
`e1_d100s0_grace.yaml` carried TWO identical `critics:` blocks (a copy-edit
artefact); PyYAML keeps the LAST mapping for a duplicate key and both read
`basic: [observational]`, so the finished `d100s0` runs are unaffected — a
duplicate that resolved to the intended value is a near-miss, not a
non-event. The earlier block is removed; `check-yaml` (ruamel) rejects
duplicates, which is how it surfaced. Result artefacts are committed with
`--no-verify` so the end-of-file hook does not rewrite them.

### RULING 2026-09-03 (evening) — `dr2_cut` STRIPPED; falsification report-only; selection by materiality; the parameter taxonomy

**The objection, owned:** the measured Delta-R^2 gap was a PER-ENVIRONMENT
CONSTANT and A2 forbids those. It entered only as `dr2_cut`, in two places,
and neither needs it.

**1. Falsification is report-only — no threshold at all.** By the standing
ruling L5 warns and never overrides; a verdict that changes no behaviour
needs no binary. `MarkovVerdict.declaration_falsified` is REMOVED;
`MarkovVerdict.record(alpha)` travels on the served value (C3) with: the
effect size Delta-R^2 and its p; the CAPACITY-SHRINK ratio (the mechanistic,
dimensionless separator — approximation error shrinks with base capacity,
measured 56x; information does not, ~1x); base R^2 per dimension with
`scale_invalid` where negative; the reward-channel diagnostic. The user
reads the magnitude and its evidence. Nothing branches on it.

**2. Selection is materiality-by-refit against L4's own interval.** The
predictive test asks "is the process exactly Markov?", whose answer is
always no (S18). The estimator's question is whether another lag changes
WHAT GRACE SERVES by more than the uncertainty GRACE already reports:
`k* = min { k : |contrast(k+1) - contrast(k)| <= w_k }`, contrast = the
served action contrast on the lag-k augmented state, w_k = L4's half-width
there — every term measured per fit, no constant, no environment dependence
(same family as the walk's derived stop, the bootstrap MC-error criterion
and tau_R). `l5.select_window` is REMOVED; the selector lives in
`pomdp_branch.transform_offline_rewards_declared`. Cost: fits at k = 0 and
k = 1 on a true MDP (~1.7x, measured) replacing the 912 s selection pass;
under the cache the k = 0 fit is a hit whenever the MDP-declared arm ran on
the same data. **First empirical point** (peer session, max-null era): the
k = 0 and k = 1 transforms on d100s0 s0 produced IDENTICAL intervals to four
decimals ([+0.4989, +0.5235]) — the criterion stops at k = 0 immediately.

**3. A user-supplied k is an INPUT, never a hypothesis.** The declaration
surface is `(observability, optionally k)` and declared-MDP IS k = 0 — ONE
code path (the runner calls `transform_offline_rewards_declared` for every
GRACE arm): MDP -> k = 0; POMDP with k -> that k; POMDP without k -> §2.
When k is supplied GRACE uses it and reports two diagnostics, both
report-only, neither overriding: **sufficient?** (does lag k+1 move the
served contrast by more than w_k — if so the window is too short: warn,
serve anyway; label `WINDOW-TOO-SHORT(warn)`) and **necessary?** (does k-1
already suffice — if so the window is longer than needed: compute and
estimator variance, no correctness harm; `WINDOW-LONGER-THAN-NEEDED(info)`).
The second is contract row 2 in its exact form. `k_max` applies only when
selection is delegated; `k_diagnostics` is a BUDGET switch (the extra fits),
disclosed when off. Config: `declared_observability`, `grace_window_k`,
`grace_k_max`, `grace_k_diagnostics`, `grace_l5_alpha`, `grace_l5_b`;
`grace_dr2_cut` is GONE. Tests: `tests/test_pomdp_branch.py` (12) pin the
path; `tests/test_l5_markov.py` pins the record.

**4. Calibration re-scoped; the sweep DISCARDED.** The running sweep was
stopped (peer session, 2026-09-03 ~18:45; 61 rows preserved). Its purpose
was choosing a number we no longer use. What calibration is still for, and
it is not per-environment: (i) the POWER of the materiality criterion — how
large a violation must be before it is caught — a property of the METHOD,
measured on synthetic fixtures where the truth is dialable; (ii) the S18
result — a point null of exact Markovianity rejects at floor effect sizes on
deterministic systems — reported ONCE as a finding from the rows on disk.
The "two selector readings" table and the under-cut numbers recorded this
afternoon are now a measurement of a WITHDRAWN design, kept as such.

**5. The parameter taxonomy — the environment-independence claim the paper
asserts.**

| class | members | status |
|---|---|---|
| **declarations** | the diagram, observability and k, `u_card` | user inputs; honoured; contradictions REPORTED |
| **budgets** | B, folds, RFF count, `k_max`, iteration caps, `k_diagnostics` | compute limits; DISCLOSED when they bind |
| **derived** | tau_R from w; the walk's stop from bootstrap MC error; `min_scale` and the sqrt-eps truncation from the float representation; the materiality criterion from w_k | measured per fit |
| **calibration constants** | — | **NONE** |

Nothing measured on one environment transfers to another, because nothing
is measured to be transferred.

**Fixture finding, recorded (S8 doing its job, and a fragility to keep in
view):** the unit fixture "hidden AR state drives the reward" at obs dim 2
gave a memoryless base fit with held-out base R² of −1.3 / −0.2 on the obs
dims — `scale_invalid` on every dim, IDENTICAL under the pre- and post-S19
code (same conditioning block). The same dynamics at obs dim 3 fit at R²
0.92. So the random-feature basis on a 4-wide standardised block can
generalise badly on a small synthetic fixture; the flag caught it, the
fixture was moved to d = 3, and the positive-control test now asserts the
obs dims are NOT scale-invalid so it cannot pass vacuously. On real data
(CartPole full view) base R² is 0.99999+ at every capacity probed; the
materiality-power calibration on synthetic fixtures must report base R²
alongside its power numbers for exactly this reason.

**The reference priming MISSED (found by the peer on the first grid leaf):**
my `phase2_speed.py` built its buffer straight from the Minari episodes
(49,125 rows, no `next_obs`/`dones`) while the runner's fill writes
49,762 rows with both — different content, a different `data_sha256`,
correctly a MISS (the cache did its job; the priming used the wrong
construction site). One-time cost 1.15 h; the grid's own first k = 0 fit
(`d2961b84`) and k = 1 fit (`59883afe`) are the entries the later cells
reuse. Lesson: prime a cache only through the consumer's own fill.

**Also fixed on the way (recorded):** the k >= 1 augmented view handed to
the fit carried no `next_obs`/`dones`, so the extractor rolled next-obs
ACROSS episode boundaries and saw no terminations; the view now carries the
exact next augmented state (lag blocks shifted by one) and the dones. And
`serving.transform_offline_rewards(apply=False)` lets the branch compare
fits before writing the real buffer exactly once.

**Next, in order (ruled):** (a) the critic end to end — done in this commit,
pending the grid's first run; (b) repo adoption — the YAML driver consuming
the e1 configs (`CELLS` gone), the contract cells as configs with
`declared_observability` and optional k, the deployed plotting extensions
(per-seed curves, paired base-vs-grace, the critic-accuracy table, the
return decomposition, abstentions separate), the README section — split:
driver/plotting/README-driver = peer session, config keys + README-critic =
this session; (c) SPEED, measurement first — profile one representative
grace fit (EM iterations, M-step steps, bootstrap replicates,
`interventional_sweep`) and report where the time goes before optimising;
candidates in likely payoff order: bootstrap replicates in parallel
(`n_jobs`, GPU-memory-gated), the transform cache actually used by the grid,
the per-node M-step budget on 1-D proxy channels, fit-once-per-(cell,
dataset seed, k) across training seeds — none applied on argument alone.

### THE S18 EVIDENCE REPORT + THE WINDOW-COST TABLE (2026-09-03, sweep-holder session)

**S18, measured and filed:** `results/l5_calibration/s18_report.json`
(rendered by `tools/report_s18.py` from the 61 preserved sweep rows — the
rows' ONE remaining purpose under the dr2-cut ruling). Headlines: 29 true-MDP
null replicates, **97% rejected at α = 0.05, median p = 0.005** — the point
null of exact Markovianity is false by construction on deterministic systems;
ΔR² separation **925×** between the largest null effect (5.8e-7) and the
smallest constructed-POMDP effect (5.4e-4); capacity-shrink median **6.9 on
nulls vs 0.0007 on masked** (approximation error dies with capacity;
information does not). Cite the JSON, not this paragraph.

**The window branch's measured costs** (d100s0 s0, 49k rows, GPU,
production budgets; `fit_unaug = 7709 s` is the denominator):

| fit | state dims | seconds | ratio |
|---|---|---|---|
| k = 0 (unaugmented) | 4 | 7,709 | 1.00 |
| k = 1 | 9 | 13,184 | **1.71** |
| k = 2 | 14 | 14,159 | **1.84** |

Window cost grows GENTLY with k (the fixed-step M-step decouples
per-iteration cost from dimension). With the transform cache, the materiality
selector's k=0 fit is a HIT on any campaign where the base arm ran first, so
its marginal cost is the k+1 fit alone. Third stability datum for the
materiality criterion on true-MDP data: intervals at k = 0/1/2 are
[+0.4989,+0.5235] / [+0.4989,+0.5235] / [+0.4989,+0.5237] —
|Δcontrast| ≈ 2e-4 against w ≈ 1.2e-2, a ~60× margin, so the selector stops
at k = 0 immediately. (The probe's earlier SELECTION timing, 912 s, was the
pre-S19 statistical selector and is VOID — recorded, not quotable.)

**danull leaf paths moved (2026-09-03):** the pilot's 12 `offline_mdp_danull`
leaves now live under `beta_000_sigma_000` (was `sigma_025` — the driver's
campaign default leaking into a cell where σ is meaningless; the diagram-arm
validator correctly rejects σ > 0 on a no-latent cell, and the tree now
agrees with the declaration `e1_danull*.yaml: basic {β=0, σ=0}`). Anything
hardcoding the old path must update.

### PHASE 3 PRE-REGISTRATION (2026-09-03, before generation) — the true-POMDP datasets

**What runs:** `tools/generate_diagram_arms.py --cells d_d_sweep_d100_om13
--envs CartPole-v1 --device cpu` — the FIRST execution of the
masked-behaviour path (the generator is trained on the masked view under its
own generator dir `<out>/generator/CartPole-v1_s<seed>_om13`; the rollout
acts through `_MaskedViewPolicy`; the dataset stores the FULL observation).
Six datasets: seeds 0–2 × {σ = 0 basic, σ = 0.25 confounded}, ids carrying
`-om13` (S6). Generator training on CPU because the card is held by the
k = 2 timing (footprint 7.6 GB of 8.2, measured).

**Predictions, written before the result:**
1. Every dataset's stamp reads `behavior_information_set: masked:1,3`; the
   full-view d100 datasets read `full` (checked side by side).
2. Gate test passes and preflight passes on all six; the one licensed
   regeneration is the covariate-free preflight at p ≈ α (the known ~1%
   rate). Any other failure is stop condition §7.1.
3. **L5 positive control (certification check 1):** the masked view fails
   the Markov test at lag 0, α = 0.05, with ΔR² in the 1e-3..1e-1 range
   (the calibration's masked rows on load-time-masked d100: 5e-4..36), while
   the full-view d100 s0 sits at ~1e-7 (measured today). Capacity-shrink
   ratio < 1 (information, not approximation error).
4. **Certification check 3 (recoverable from history):** the masked
   velocities regress from one lag of positions with R² > 0.9 (a finite
   difference of positions reconstructs them), so k = 1 is the EXPECTED
   selection on this data (contract plan scope statement).
5. The behaviour policy is genuinely less competent on the masked view:
   the generator's tier-selection return and the rollout's return
   distribution sit BELOW the full-view d100 generator's (a POMDP has a
   worse achievable policy) — recorded, and the reason cross-column readings
   are forbidden.
Checks 2 and 4 of the plan's certification list (memory pays; per-context
return spread) are grid-side measurements and are run with the contract
cells, not here.

**OUTCOME (2026-09-03 21:25) — PHASE 3 COMPLETE: 6/6 certified on the first
pass, no regeneration.** `results/dd_sweep_om13_generation/report.json`
(committed): every row gate True, preflight True, reasons empty, proxy
margins ≈ 5 (as the full-view cell's), 57–136 s each after the generator.
Against the predictions:
1. ✓ every stamp reads `behavior_information_set: masked:1,3`; the full-view
   d100 datasets read `None` — they predate the stamp (the field did not
   exist), not `full`; the reader must treat None as full-view-historical.
2. ✓ gate + preflight on all six; the licensed regeneration was never needed.
3. ✓ **L5 positive control** on s0 σ = 0, masked view: p = 0.010 (the
   floor at b = 99), ΔR² = 3.0e-2, capacity-shrink 0.001 (information);
   the full view of the same data: ΔR² = 1.3e-7, shrink 577 (approximation
   error). Base R² on the masked view 0.99/0.97 (clean fit). The
   reward-channel diagnostic reads 0.22 / 0.24 on both views — lagged R
   carries the episode-constant U (U → R survives at σ = 0), as the
   catalogue says. **σ = 0.25 s0 (the grid's operating point):** masked view
   p = 0.010, ΔR² = 3.4e-2, shrink 0.000; full view ΔR² = 1.3e-7, shrink
   3702 — same picture. The reward-channel diagnostic reads −0.02 on the
   masked view vs +0.20 on the full view there: under the masked view the
   memoryless reward base is weaker (base R² 0.43 vs 0.47) and the lagged
   block adds nothing beyond the placebo — reported, not decided on. Cost
   datum: a 500-episode L5 record takes ~250 s on a quiet CPU (1930 s under
   the earlier load), so the `l5_n_ep = 500` budget prices the record at
   minutes per (dataset, k), cached by content across training seeds.
4. ✓ velocities from ONE lag of positions (+ action): R² 0.93 / 0.90 on the
   masked data → k = 1 is the expected selection.
5. ✗ **REVERSED, with a mechanism:** the masked behaviour policy is MORE
   competent — rollout return 41.9 (mean; episode length 33.7) vs 19.2
   (16.4) on the full-view d100 s0. Mechanism, measured in the generator
   curves: `select_tier_episode("medium")` takes the FIRST checkpoint
   reaching a third of each generator's OWN return range; the masked DQN's
   first checkpoint (61.5, range 52–62.5, target 55.5) already qualifies,
   while the full-view generator's medium checkpoint is a genuine 56 of
   52–64. A tier-selection artefact of the D4RL-style rule, not a defect of
   the masked path; every generation knob is identical between the two
   datasets (metadata diffed field by field: only `p_hat` and the U-edge
   statistics differ, which the behaviour difference explains). It is
   exactly the cross-column competence confound the plan forbids reading,
   and touches no within-column claim; recorded so nobody reads "the POMDP
   column scores higher" as a GRACE effect.

**First execution, first surprise (19:20, loud, fixed, relaunched):** the
masked generator trained fine on the 2-dim view, then
`build_generator_agent` rebuilt the agent at the CANONICAL 4-dim obs and
the checkpoint refused to load (state-dict size mismatch [64, 2] vs
[64, 4]). The same reduction `generate_offline_dataset` already applies is
now applied at the build site; pinned by
`test_build_generator_agent_is_masked_dim_when_behaviour_is_masked` (no
training: the 'random' tier path). Second latent defect on this never-run
path in one day (the first was the undeclared parameter) — the reason the
plan makes "generate one and certify it" a gate rather than a formality.

**Comparison record (S19, closing the afternoon's measurement):** on real
data the (O, A)-only family is invariant to the reward channel by
construction — d_a_null s0 and d_b_prime s0 share their (O, A) trajectories
and now give IDENTICAL statistics (lag 0: 1.669e-7; lag 1: 4.264e-9), where
the (O, A, R) family had differed (1.100e-7 / 2.004e-8): lagged R had been
leaking into the observation-channel statistic. Reward-channel diagnostics
unchanged (0.0204 / 0.0185).

### PHASE 4 PRE-REGISTRATION (2026-09-03, before any grid result) — the CartPole contract grid C1

**Cells** (`reproducibility/rl_regimes/diagrams/c1_*.yaml`, campaign prefix
`c1`, results under `results/c1`): truth ∈ {tmdp = d_d_sweep_d100, tpomdp =
d_d_sweep_d100_om13 with the learner masked [1, 3]} × arm ∈ {base, grace
declared MDP, grace declared POMDP (k delegated)}; σ = 0.25; 3 dataset
seeds × 3 training seeds; cql + iql. The base arm carries the Phase-6 critic
axis (observational, proximal, oracle_u, sensitivity on tmdp; proximal
EXCLUDED on tpomdp — L2: D-G q1 bounds-only, q2 non-ID). Analysis is
WITHIN-COLUMN only.

**Predictions (pre-authorised readings, written as predictions):**
1. **Row 2 (declared POMDP, true MDP):** the materiality selector returns
   k = 0 on ≥ 8 of 9 (ds, ts) leaves per algorithm (first empirical point:
   identical k = 0 / k = 1 intervals on d100s0 s0); served rewards are
   bitwise those of the declared-MDP arm (cache hit), so the paired return
   delta vs row 1 is 0 up to training-seed noise — "over-assumption is
   cheap", measured.
2. **Row 4 (declared POMDP, true POMDP):** k = 1 selected (a finite
   difference of positions reconstructs the velocities); materiality margins
   `|contrast(1) − contrast(0)| > w_0` and `|contrast(2) − contrast(1)| ≤ w_1`.
3. **Row 3 (declared MDP, true POMDP):** the L5 record at the served lag 0
   CONTRADICTS the declaration on every leaf (p ≤ 0.05, ΔR² ≳ 1e-3,
   capacity-shrink < 1) and the `window_sufficient` diagnostic reads False
   (`WINDOW-TOO-SHORT(warn)`); GRACE serves as declared; its critic accuracy
   (`q1_contrast_error`) is WORSE than row 4's on the same data — the
   degradation is observable next to the warning (contract row 3).
4. **Row 1 (declared MDP, true MDP):** the L5 record rejects at floor
   effect sizes on most leaves (S18) with capacity-shrink > 1 — reported as
   the S18 floor behaviour, NOT a defect; `window_sufficient` True.
5. **Return:** grace ≥ base on the confounded point within column 1 (the
   pilot's d100 result); in column 2 the memoryless learners are worse than
   in column 1 in absolute terms (a POMDP has a worse achievable policy —
   never read across columns). Grace losing to base on return anywhere is
   reported with the decomposition, not treated as a failure.
6. **Abstentions** (fit-health, L4) are tabulated separately; the σ = 0.25
   d100 fits did not abstain in the pilot, so the prediction is 0 abstentions
   in column 1; column 2 unknown (first fits on masked views) — any
   abstention there is reported with its reason.
7. **Critic axis (Phase 6):** oracle_u ≤ grace ≤ observational on
   `q1_contrast_error` in column 1 (ceiling / floor); proximal ≈ grace on
   D-D (both point-ID via the same proxies).

**Null-calibration anchor (decided 2026-09-03 22:30, peer's flag):** every
σ = 0.25 cell declares `basic: false`, so the strategy critics'
null-calibration gate would have no anchor. Options were (a) σ = 0
companion points per truth column, (b) the stored fixed-denominator
reference (`null_cal_reference.yaml`, historical CartPole cql/iql
`noise_refs`), (c) declare the grid un-calibrated. **(a), base arm only:**
(b) is a stored per-environment constant (A2, stop-condition §7.5), (c)
leaves the critic axis ungated on the headline grid. `c1_tmdp_base_s0.yaml`
(seeds [0, 2, 3] — the full-view σ = 0 s1 failed the preflight at
generation; s3 is its certified substitute) and `c1_tpomdp_base_s0.yaml`
(seeds [0, 1, 2], certified today), same regime tags, so the leaves land at
`beta_000_sigma_000` under the σ = 0.25 cells' tags. +36 runs → 144.

**Cost projection is reported before launch** (Phase 2's speedup and the
k = 2 ratio enter it); launch is pre-authorised below 60 GPU-hours.

### PHASE 2 — SPEED, MEASURED FIRST (2026-09-03 22:54 → 00:15, the profile)

`tools/profile_grace_fit.py` on d100 σ = 0 s0 (49,125 rows, GPU, ALONE on
the card — load 2; cProfile overhead included): **TOTAL 4832 s = 1.34 h**
for one `fit_reward_transform` (23 fits: observed + 2 init seeds + 19
bootstrap replicates + the served fit). Where the time goes:

| phase | calls | total s | share |
|---|---|---|---|
| bootstrap replicates (19 refits) | 19 | 3974 | **82%** |
| all fits (23) | 23 | 4169 | 86% |
| M-step, total | 893 | 4152 | 86% |
| M-step: proxy nodes Z, W, V (MDN `fit_local`) | 893 × 3 | 1016 + 1010 + 1006 | **63%** |
| M-step: R, A (`neural_categorical`) | 893 × 2 | 555 + 554 | 23% |
| `interventional_sweep` (L4's contrast targets) | 552 | 662 | 14% |
| E-step | 1031 | 13 | 0.3% |
| U, S nodes | 893 × 2 | 7 | 0.1% |

Under the hood (cProfile): `run_backward` 1343 s, MDN `_log_prob` 941 s,
nbn `likelihood_weighting` 660 s (= the sweep), ~39 M-steps per fit (30
iterations + backtracks), 4.65 s per M-step. The ruling's tree, applied:

1. **bootstrap dominates → `n_jobs > 1`, gated by the free-memory rule.**
   Wired as a BUDGET (`grace_n_jobs`; threads, seeds by index, results by
   index; not in the cache key). Gate: measure one replicate's peak GPU
   memory (the reference run reports `max_memory_allocated`), divide the
   free memory by it, cap there. Adopted ONLY if the served rewards, lo, hi
   are BITWISE the reference's (`tools/phase2_speed.py parallel N`).
2. **M-step: the proxy nodes are 63% of everything.** Their step budget is
   the same 400 per node as R/A's (`_epochs` derives epochs from the fixed
   step budget), on 3,000 episode-rows vs 49k transition-rows, with the
   MDN's per-step cost ~1.8× the categorical's. Whether 400 steps over-spend
   on a 1-D proxy channel is a FIT-QUALITY question: any budget change
   alters the served numbers, so by the rule it is NOT a speed change and
   is not applied here — recorded as the next candidate, to be measured on
   its own (fit quality vs steps), never smuggled in.
3. **`interventional_sweep` 14% → batch it**: `sweep_chunk` is now a budget
   (default 4096, unchanged); `tools/phase2_speed.py sweep` measures the
   full-buffer chunk against 4096 for bitwise identity and time — adopted
   only if identical.
4. **Cache**: the c1 grace cells share `results/grace_cache`; the Phase 2
   reference run (`phase2_speed.py reference`, n_jobs = 1, d100 σ = 0.25
   s0) is the grid's FIRST entry (tmdp ds0 declared-MDP) as well as the
   bitwise reference.

**OUTCOME (2026-09-04 01:35).** The reference fit (d100 σ = 0.25 s0, alone
on the card, no profiler): **wall 4123 s = 1.15 h**; peak GPU memory
ALLOCATED **4776 MiB**, reserved 7416 MiB, of 8188 MiB. Stored at
`results/grace_cache/2abafc2ad4a30825` with `code_version 0cbf18a5…` ==
the launch tree's (verified; the package is FROZEN until the waves finish —
any edit under `src/rl/offline/grace/` or `nbn/` invalidates every entry by
design). Interval [+0.4803, +0.5080], rewards sha256 `a66476ed…` — the
bitwise reference for any later budget change.

* **n_jobs gate → 1.** One replicate's measured peak is 4.8 GB; with a
  512 MiB headroom, floor((8188 − 512) / 4776) = 1. Two concurrent
  replicates would need 9.6 GB on an 8.2 GB card. The lever is wired
  (`grace_n_jobs`) and stays at 1 on this hardware; said so, moved on. (On
  a 24 GB card the same gate gives 4 and the bitwise test would run then.)
* **Sweep chunk → stays 4096; it IS the memory budget.** The full buffer
  (49k rows) needed 7.5 GB and OOM'd; 12,288 rows tried to allocate 6 GB
  and OOM'd (≈ 0.5 MB per row inside nbn's likelihood weighting); 4096
  rows ≈ 2 GB and took 14.6 s per action-sweep (matches the profile's 1.2 s
  per chunk × 12). A larger chunk cannot be bitwise-tested because it
  cannot run; the lever is rejected on this card.
* **Achieved speedup for the grid: ×1.0 on the fit** (no lever passed the
  gate at this memory) — reported as the factor it is. The material saving
  is the CACHE (already built): 21.8 fit-units for 144 runs instead of 72
  fits, and the k = 0 collapse making every declared-POMDP-on-true-MDP fit
  a hit.

**PROJECTION, reported before launch** (`tools/project_c1_cost.py
--fit-hours 1.145`): fits 21.8 × 1.15 h = **24.9 GPU-h**; training 144
runs from the pilot's quiet medians (cql 391 s, iql 625 s; base cells
× 1.3 for the critic heads — a GUESS, the one unmeasured input) =
**23.4 GPU-h**; **TOTAL ≈ 48 GPU-h < 60 → GO**, on the condition that the
grid runs ALONE on the card (the contended numbers give 78). L5 records
(~4 min per (dataset, k), content-cached) run on the CPU alongside.

### C1 LAUNCHED 2026-09-04 01:35:44 — EARLY COST SIGNAL (02:20), the projection's guessed input was wrong ×5

First base leaf group (c1_tmdp_base/cql/base/ds0_ts0 → 4 per-critic
leaves, explosion correct): **2620 s** for the four-critic base run vs the
pilot's ~370 s single-critic median — ×7.1 against the projection's ×1.3
GUESS for the critic heads. Mechanism (presumed, to be confirmed on the
next two base leaves): proximal / oracle_u / sensitivity are FITTED
estimators on the shared stream, evaluated per checkpoint, not cheap
heads. Corrected projection if the slope holds (72 base runs across the 4
base cells incl. the σ = 0 companions × 2620 s + 72 grace × 500 s + 25.0
GPU-h of fits): **87 GPU-h — above the 60 GPU-h stop line (§7.4).**
Options costed (fits unchanged at 25.0):

| option | shape | GPU-h |
|---|---|---|
| as launched | 4 base cells × 18 four-critic runs | 87.4 |
| iv | critic axis at ts0 only (12 runs × 2620) + base observational-only at 3 ts (72 × 400) + grace | 51.7 |
| iv′ | critic axis at ts0 only, on the σ = 0.25 cells (6 × 2620) AND the σ = 0 companions (6 × 2620, the anchors keep their critic set; 1 ts suffices per (env, algo, critic)); base observational-only at 3 ts (36 × 400) | 47.7 |
| iii | base cells at 1 ts (breaks base-vs-grace pairing at ts1/ts2) | 52.4 |
| i | observational only everywhere (no Phase 6) | 43.0 |

**Measured (03:20): cql 2620 s, iql 3609 s for the four-critic base run
(mean 3114 s vs the single-critic mean 508 s: ×6.1).** Mechanism confirmed
in `critic_ablation.py`: proximal / oracle_u / sensitivity are FULL
LEARNERS (`build_<critic>_<base>`), so a four-critic run trains four
learners. Re-costed with the measured means (fits 25.0 unchanged): as
launched **97 GPU-h**; iv′ **51**; iv 56; iii 56; i 45.
The chain was stopped by the peer at the iql ds0_ts0 boundary and
restarted GRACE-ONLY at 03:20:50 (tmdp_grace_dmdp → tmdp_grace_dpomdp →
tpomdp_grace_dmdp → tpomdp_grace_dpomdp; every option keeps all 72), the
four base cells wait for Giovanni's ruling on their shape (§7.4) — a
projection-input correction, not a scope change by us. Option iv′ drafts
are in the session scratchpad, not in the tree.

### C1 — FIRST ROW-1 LEAF READ AGAINST THE PRE-REGISTRATION (2026-09-04 ~06:10)

`c1_tmdp_grace_dmdp/cql/grace/ds0_ts0` (declared MDP, true MDP; 9971 s =
the fresh k = 1 sufficient? fit ~2 h + training): `transform_applied True`,
coverage 1.0 (no silent no-op); `window[k=0|declared-mdp]`;
**`window_sufficient True` with delta 0.0000 vs w = 0.0139** — the k = 1
refit returned the IDENTICAL contrast (+0.4943 at k = 0 and k = 1, half
widths 0.0139 / 0.0145): "over-assumption is cheap" at its sharpest, and
the k = 1 entry now serves row 2's ds0 selection as a cache hit. Interval
[+0.4803, +0.5080] straddles the M = 0.5 truth; pessimism 0.014. Return
623.7 vs 499.0 (paired base column). L5 record: p = 0.010, ΔR² = 6.7e-9,
`rejected True` — **S18's floor behaviour, as predicted; reported, nothing
branched.** Reward-channel improvement 0.025, `serving_material False`.

**Prediction 4 over-stated one thing — characterised, not a defect:** the
capacity-shrink ratio on this leaf is **0.08**, not > 1. On the S18 null
rows shrink is < 1 in 3 of 25, and those are exactly the rows with the
SMALLEST effect sizes (5.3e-9 → 0.018, 2.1e-8 → 0.074, 7.5e-8 → 0.29):
below ~1e-7 both the 64-RFF and the 256-RFF statistics sit at the
numerical floor (base R² = 1.0000 on every obs dim here) and their ratio is
noise. Above the floor the nulls shrink (median 6.9, up to 665) and the
masked rows never do (max 0.62, always with ΔR² ≥ 5e-4). **So the
separator is the PAIR (effect size, shrink):** shrink is the mechanistic
read when the effect size is above the floor; at the floor the effect size
alone — five orders below any masked value — is the evidence. The record
carries both; the report reads them together. `l5_stat_hi` = 8.2e-8 is on
the record for exactly this reason.

**Cross-algorithm cache reuse CONFIRMED on the grid (06:40):** the peer
read `iql/grace/ds0_ts0` as a fresh fit from a 7.6 GB GPU footprint; the
footprint was the in-process driver's caching allocator holding cql's
reserved memory. Measured instead: the runner's fill reproduced outside the
runner gives exactly the cql entry's `data_sha256` (dd772333…, next_obs and
dones included) and the fill has no algorithm input — prediction written
before the read: iql hits both entries. It did: 891 s (training only),
`transform_cache_hit True`, no fourth entry, served numbers identical to
cql's (ΔR² 6.68e-9, interval [+0.4803, +0.5080], k = 0, sufficient True,
49,762 rewards). The fit count is per (dataset, k), algorithm-independent,
as the 21.8 fit-units assumed; the one-time reference miss is the only
cache cost. Lesson for the record: reserved GPU memory in a persistent
process is not evidence of work — measure the artifact, not the footprint.

### Open threads

* **RULED 2026-09-03 — (a): the selector's features EQUAL the served state's
  features (S19).** The finding: L5 selected k with history blocks carrying
  lagged (O, A, R); `pomdp_branch._augmented_cols` augments with lagged
  (A, S) only — validated-≠-served, one level in. The obvious repair (lagged
  R into the state) was checked against the served estimand and REJECTED:
  the catalogue's Q2 derivation (D-B Steps 1–3; D-D inherits) defines
  `g(s,a) = E_{U~P(U)}[E[R|s,a,U]]` over the EXOGENOUS MARGINAL with the
  stated reason (Step 1) that under do(pi) at every step the trajectory is
  independent of U — no U->S' edge (fact 3), pi does not read U — so
  P(U|s) = P(U) in the deployment regime, and Step 2 names the observational
  P(U|S_t) != P(U) as the WRONG distribution to integrate.
  `estimator.interventional_sweep` implements exactly this
  (`sum_k fit.prior[k] * E[R | S=s, do(A=a), do(U=k)]`, `prior` the (K,)
  mixing weights; the per-episode responsibilities never enter). Derivation
  and code agree. That reason holds for an (A, S)-augmented state (at
  deployment A_{t-1} = pi(.) does not read U; S never depends on U) and
  FAILS for an R-augmented one (U -> R is intact at deployment, so R_{t-1}
  is U-informative and P(U|s) != P(U)): R-in-state is a belief-state critic
  with posterior-weighted serving, pulling L4's contrast interval and the
  pessimism rule with it — a different method, not a repair.
  **Done:** `l5._build_design(history_reward=False)` is the family (the
  served columns); the reward-channel diagnostic keeps `history_reward=True`
  (shared base fit at lag 0; the selector requests it at stage 0 only). Module
  statement: *the selector certifies observation-channel sufficiency for the
  exact features served; reward-channel dependence is reported, not selected
  on.* The blind spot reintroduced — a hidden state visible ONLY through
  past rewards is invisible to the selector — is now characterised, not
  hidden: it is a reward-channel phenomenon by construction and lands in
  `reward_channel` / `serving_material`. Two tests pin it
  (`test_selector_history_features_equal_the_served_augmentation`,
  `test_reward_only_visible_hidden_state_is_reported_not_selected_on`).
  **Calibration consequence — MOOT the same evening:** the sweep was stopped
  and its rows serve only as S18 evidence (ruling below); no re-score. The
  shared-basis truncation stands on its own reason (one coherent basis for
  the family and the reward-channel blocks). Original note kept for the
  record: the in-flight sweep's rows were scored with
  the (O, A, R) family; on constant-per-step-reward datasets (R untestable:
  d_a_null CartPole, Acrobot) the lagged-R column is constant and the family
  statistic is unchanged; on variable-R datasets (d_b_prime) it can differ —
  measured below in this section's follow-up, and those rows are re-scored
  under the new code before the report is read as the gate.

* **(b) RECORDED as the D-F/D-G path — the belief-state critic.** Posterior-
  weighted serving `sum_u P(u | s_aug) E[R|s,a,u]` with a DEPLOYMENT-regime
  posterior (prior x reward channel; the behaviour channel must NOT enter —
  it is the confounding) is what confounded dynamics need anyway: `U ->
  S_next` breaks the prior-marginalisation reason for the same cause as
  R-in-state, only more severely (the occupancy itself becomes
  U-dependent, Step 1's first clause). Whoever builds D-F/D-G starts from
  this derivation (catalogue D-B Step 1–3 + the paragraph above), not from
  rediscovery; see `docs/grace_observability_contract_plan.md` "What stays
  future work".

* **NEXT GATE (ruled 2026-09-03): the true-POMDP column has never executed.**
  The undeclared `behavior_mask_indices` (near-miss (1) above) means the
  masked-behaviour path through `build_generator_agent` had never run once,
  so the true-POMDP datasets are both ungenerated and end-to-end untested.
  Before the contract grid: generate ONE true-POMDP dataset and CERTIFY it —
  the information-set stamp must record the masked behaviour and the
  preflight must pass on a dataset whose logged actions depend on the masked
  view only. Report the certification stamps, not "it ran". Scheduled after
  the L5 sweep and the cost-probe report (CPU/GPU contention).

* **`--resume` NEVER reuses, and "correcting" it naively DELETES THE WHOLE
  STORE. Read this entire item before touching it.** `generate_diagram_arms.py`
  offers `--resume` to "keep datasets whose generation_fingerprint already
  matches". It cannot match. `generate_offline_dataset` stamps the fingerprint
  WITHOUT forwarding `n_proxies`, so it always hashes the default 2, while the
  resume site passes the real value out of `arm_generator_kwargs` (3 for every
  V-carrying cell, i.e. every cell E1 uses). Measured 2026-09-01 on
  `d_d_sweep_d100` sigma = 0 seed 0: stored `480f5c317dfbe9b1`; recomputed with
  the real `n_proxies=3` gives `7cb1aab77247ba9b`; recomputed with `n_proxies=2`
  reproduces the stored hash BIT-IDENTICALLY. So every `--resume` run falls
  through to `minari.delete_dataset(did)` and regenerates.

  **Two things follow, and the second is the dangerous one.**

  (a) The stored fingerprint is WRONG, not merely mismatched. Its stated purpose
  is to prove that regenerating would reproduce the dataset; it omits a real
  generation input, so a two-proxy and a three-proxy dataset agreeing on
  everything else hash IDENTICALLY. This project holds both -- the frozen
  two-proxy `d_d` arms and the three-proxy `-d100` regenerations -- so that
  collision is in the direction that matters.

  (b) **The obvious fix is a trap.** Correcting the store site alone changes
  every FUTURE fingerprint while all 163 existing datasets keep the old wrong
  one. The next `--resume` then mismatches on the ENTIRE STORE and deletes it --
  campaign-wide, the catastrophe that `--sigmas` narrowly contained on
  2026-09-01, when the only thing standing between a routine generation command
  and the loss of 15 certified pinned datasets was a filter added that morning
  for an unrelated reason. The fix is therefore THREE steps, in order:
  (1) correct the store site to forward `n_proxies`; (2) BACKFILL the corrected
  hash into existing datasets' metadata, or add an explicit compatibility path
  that recognises the legacy hash; (3) only then rely on `--resume` again.

  **Root cause is S6's, applied to the fingerprint rather than the id:** two
  construction sites that disagree. Identity got consolidated to one site after
  three collision bugs; the fingerprint never did. Same fix shape when it is
  time. **Until then the operative guard is `--sigmas` (and not running
  generation against a live cell at all) -- NOT `--resume`, which does nothing.**


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
* **BEFORE LAUNCHING ANY GPU JOB, check free memory against the job's known peak, and default to SERIALISING ad-hoc GPU work unless the footprints are measured.** Both GPU failures this session had one shape: a second job launched onto a card whose occupant's footprint was unknown. (a) Parallel sweep tags launched while the serial driver was still looping its own list — two processes wrote the same output file for 8 hours. (b) Two Q2-A entry tickets launched together; the first took 6.89 of 7.62 GiB and the second died of CUDA OOM inside `query_batch`, producing NO report — which would have read as "the cell fails its entry ticket" had the log not been checked. The runner already serialises within a sweep; this rule is for the AD-HOC launches, which is where both errors happened. Same species as the `pgrep` self-match below: a check that was available and not run.
* **One agent per worktree.** Kill by **PID**, never a bare pattern (`pkill -f regime_sweep` matches `test_regime_sweep.py`). Note also that `pgrep -f <own pattern>` **matches its own shell**, so "still running" can be entirely self-reference — check `ps -eo args | grep -v grep` before concluding a job survived a kill.
  **It bit anyway, twice in one day (2026-08-22):** a watcher shell watching
  `pgrep -f generate_diagram_arms` matched ITSELF and reported a dead
  generation as alive for 24 hours; the replacement monitor had the same bug
  and was caught only on re-read. The guard that works: put a bracket in the
  pattern — `pgrep -f 'generate_diagram_[a]rms'` — so the watcher's own
  command line no longer contains the literal it greps for. Rule-form: any
  self-spawned watcher must quote its target pattern in a form that does not
  match the watcher.
* **`cd X && ... &` BACKGROUNDS THE `cd` TOO — a silent write into a sibling checkout.** The `&` applies to the whole `&&` chain, so any *following* line in the same tool call runs in the ORIGINAL cwd. A heredoc written that way landed a script in the frozen `benchmarking_causal_rl` checkout instead of this worktree, with no error anywhere. Guards: parenthesise, `(cd X && ...) &`, or use absolute paths in anything backgrounded. **The sharper risk is the one that did not happen this time:** it created an *untracked* file, which was harmless and visible in `git status`. The same slip onto an *existing* path would have been a silent modification to the frozen v1 branch with nothing to signal it. Check the sibling checkout is clean after any backgrounded write.
  **UPDATE (2026-08-19): the sharper risk then happened, by a different route.**
  Post-crash assessment found the frozen checkout with 25 *modified tracked
  files* under `nbn/` plus one staged addition — a half-finished wholesale copy
  of upstream over the vendored tree (made to source a corrected LICENSE, of
  which only the LICENSE was committed), sitting uncommitted on the frozen
  branch for four days. Not a backgrounded write at all. The general form of
  the rule: **after ANY operation that touches a sibling checkout — copies,
  syncs, licence fixes, anything — run `git status` there before leaving it.**
  Looking for stray files is not enough; the worse failure is modifications to
  tracked files, which `ls` cannot see. (Resolved: discarded as premature per
  the sync procedure; the licence commit was completed by fixing NOTICE.md's
  stale GPLv3 line in `b513ba3`.)
* The full test suite exceeds a 10-minute tool timeout; run in chunks with `-k`.
* Sampling from an **unfitted** model raises `assert self._bias is not None` — that is "sample before fit", unrelated to any do-semantics issue.
