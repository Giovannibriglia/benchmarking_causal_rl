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
| **S1b** | **Match the statistic's unit of observation to the unit at which the tested quantity varies.** A quantity constant within an episode (`U`, `Z`, `W`, `I`) enters statistics as **one row per episode**. Quantities that genuinely vary within an episode stay at transition level, with the null still clustered by episode. | **In RL, episode length is an OUTCOME.** Pooling an episode-constant quantity at transition level weights each episode by its length, so the weighting correlates with anything influencing behaviour and manufactures dependence. Measured: `corr(I,U)` = **−0.590** at transition level against **−0.034** at episode level, on an instrument drawn from its own Bernoulli that never reads `U`. The permutation null does not rescue it — permuting whole episodes destroys the length-weighting in the null while the observed statistic keeps it, so it surfaces as a huge z rather than as noise. |
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

### ⚠ S1b — the rule, and what it invalidates

S1 (episode-level *nulls*) was necessary and not sufficient. An episode-level
null over a length-weighted *statistic* is still wrong, and that is what broke
the D-E arm: mean episode length by `(U, I)` cell was 19.5 / 59.0 / 67.4 / 15.4,
so the long "disagreeing" episodes dominated the pooled correlation.

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
5. **Fix the skewed-tail cutoff** — the null of the maximum is built correctly
   then read with a 3-SD z-score, but a maximum's distribution is right-skewed,
   so use a *quantile* of the permutation draws. Likely secondary to the
   granularity bug, but wrong independently of it.
6. **Then**: fixed-step-budget M-step measurement (O(steps) not O(n·epochs),
   legitimate under GEM, symmetric-safe), constraints-per-diagram count
   (measured, not the estimated 4), and the V-D re-projection — on an idle
   machine, since every timing so far is a contended upper bound.

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
