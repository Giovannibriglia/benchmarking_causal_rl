# GRACE reframed: the observability contract — Phase 1 plan (2026-09-02)

Supersedes the *priorities* of `grace_first_class_plan.md`; its integration
work (YAML-driven driver, transform cache, deployed plotting) survives and is
assumed below. No code changed in this phase.

**The contract.** GRACE is a critic architecture with one user knob: the
declared observability, MDP or POMDP. It derives what is identifiable from the
declaration (L1/L2), estimates that (L3), carries uncertainty and abstention
(L4), and — the reorganising requirement — **falsifies the declaration when the
data contradict it** (L5), serving AS DECLARED with the contradiction reported
as a condition on the value (ruled 2026-09-03: the declaration is an input
GRACE never overrules; a warned-but-served value, never a silent one).

| declared | true | required | served by |
|---|---|---|---|
| MDP | MDP | correct | L2→L3→L4 + reward-transform seam — **built, piloted** |
| POMDP | MDP | cheap over-assumption | the window selector collapses to k=0 (§1) — the cost is one diagnostic |
| MDP | POMDP | **detected** | L5's Markov falsifier (§2) → SERVE AS DECLARED + warn + C3 condition (ruled 2026-09-03: the declaration is an input, never overruled) |
| POMDP | POMDP | correct | history-window branch (§1) — the main build |

---

## 1 — The POMDP branch design

### What already exists (found, not assumed)

- **L2 is substantially built.** `cell_graph.py` declares D-F ("POMDP, latent
  state, no U") and D-G ("POMDP + U") with both queries; `identify.py` derives
  the verdicts: D-F is **q1 point-ID** (conditioning on the observed emission
  O_t blocks the state path) and **q2 not identified in general**; D-G is q1
  bounds-only, q2 non-ID. The catalogue names the escape route explicitly:
  q2 identification "requires additional proxy structure (**past/future
  observations as proxies for the latent state**)".
- **The masking apparatus is mature**: `MaskedObservationWrapper` (Box-only,
  per-env velocity indices, `last_unmasked_obs` kept for per-context eval),
  `mask_indices` wired through online wrapping and offline load-time
  projection, `offline_dqn_recurrent` as an lstm-capable base, and the
  `offline_pomdp` cell YAML already declaring CartPole `[1,3]` /
  Acrobot `[4,5]`.

### ⚠ Finding: the current POMDP construction asserts the wrong edge

Datasets store FULL observations; the mask is applied at **load** (`np.delete`
in `minari_loader`). Generation never masks, so the behavior policy acted on
the full state. In the masked view, the logged action then depends on the
*hidden* components — the data realise `S → A`, not D-F's declared `O → A`.
That is a legitimate (and harder) diagram — partial observability manifesting
as state-side confounding — but it is **not the diagram the catalogue entry
asserts**. Consequence for the grid: the true-POMDP datasets must be generated
with the mask applied *during collection* (generator trained and rolled out on
the masked env), so the constructed cell matches D-F/D-G as declared. The
load-time-mask variant is kept as a documented harder condition, not the
default. Generation cost is minutes per dataset (measured: 3 certified
datasets in 2.6 min from an existing checkpoint; a masked-behavior generator
adds one short online DQN training per (env, seed)).

### The branch, proposed

**State = a fixed-lag history window; the window is selected by the same
statistic L5 uses to falsify.** Under a POMDP declaration, GRACE augments the
observation with the last k observations and actions
(`Õ_t = (O_t, A_{t-1}, O_{t-1}, …, O_{t-k})`) — exactly the "past observations
as proxies for the latent state" structure the catalogue names — and chooses
the smallest k whose augmented view **passes the Markov test** (§2). Then:

- the dynamics over `Õ` are (testably) Markov, `U` remains episode-constant in
  every constructed cell, and no `U → Õ_next` edge exists — so **the
  reward-transform reduction becomes valid again on the augmented state**, and
  the POMDP branch reduces to: window selection → the existing MDP branch
  (L3 fit, L4 interval, reward substitution) on `Õ`;
- if no k ≤ k_max passes, GRACE **abstains with reason** (a fit-mechanism
  budget condition in the L4 family — distinct from L5 falsification, which
  never stops serving; ruled 2026-09-03)
  (`window-exhausted: process not Markov at lag k_max`) — the existing
  abstention machinery, new reason string;
- the C3 label carries `window=k` plus the Markov-test evidence, so a served
  value's observability assumption travels with it.

**Why this and not a recurrent/belief estimator:** deterministic, cheap,
environment-invariant (no per-env architecture), auditable (k is a measured
quantity with a stated test, not a hyperparameter), and it reuses the entire
MDP stack — `EpisodeData`, the mechanisms, the annealed GEM fit, the episode
bootstrap, L4, the serving seam, the C3 discipline. The lstm base
(`offline_dqn_recurrent`) stays as a *base-algorithm* comparison arm, not as
the estimator's state.

**Degradation symmetry (contract rows 2 and 4):** declared POMDP on a true
MDP → k=0 already passes the Markov test → the branch collapses to the MDP
branch; the over-assumption costs one diagnostic run. That symmetry — the same
statistic detects under-assumption and prices over-assumption — is the
design's central economy: **one construction site** for the test and the
selector, the same move the catalogue already prescribes for the rank
constraint and `u_card`.

### What stays future work, said plainly

Truly confounded dynamics (`U → S_next`) or a per-step latent confounder are
NOT served by this branch — that is the **sequential causal critic**, and the
model-exploitation failure measured for it stands: the model-based `Q*_do`
diverged on the null cell (V 1.11 → 251 over 60 sweeps, truth ≈ 9; **+37,504%**
— 375× — overestimate on d100). Solving it means uncertainty-penalised backups
(ensemble-disagreement penalty on model samples), support-constrained
maximisation (the argmax restricted to behavioral support, BCQ-style), and the
already-wired sup-change C3 gate as the fail-loud — a research work-package
(~2–3 weeks), scheduled on the roadmap, not smuggled into this grid. Until
then D-G's q2 row serves bounds or abstains, which is what the catalogue's own
verdict licenses.

**The shape of that work, derived rather than guessed (ruled 2026-09-03,
handoff S19 item).** The MDP branch and this POMDP branch serve
`g(s,a) = E_{U~P(U)}[E[R|s,a,U]]` over the exogenous marginal, warranted by
the catalogue's Q2 Step 1: under `do(π)` the trajectory is independent of
`U` (no `U → S'`, `π` does not read `U`), so `P(U | s) = P(U)` at deployment.
That warrant survives an `(A, S)`-augmented state and fails for any state
that carries `U`-information at deployment — lagged `R` (since `U → R` is
intact) or, more severely, any `U → S_next` edge (the occupancy itself
becomes `U`-dependent). What those cases need is a **belief-state critic**:
posterior-weighted serving `Σ_u P(u | s_aug) E[R | s, a, u]` with a
DEPLOYMENT-regime posterior built from the prior and the reward channel only
(the behaviour channel is the confounding and must not enter), and L4's
contrast interval and the pessimism rule re-derived on that object. That is
D-F/D-G's method, not a repair to this branch; it starts from this
derivation.

---

## 2 — L5, promoted to headline

### The testable implication per declaration

- **Declared MDP** ⇒ the observed process is Markov:
  `(O_{t+1}, R_t) ⊥ history_{<t} | (O_t, A_t)`. This is the implication a
  masked environment violates, and the one contract row 3 needs.
- **Declared POMDP** ⇒ the HMM-like rank implication (cross-time observation
  matrices bound the latent cardinality — the catalogue's stated shadow).
  v1 scope: computed and **reported as a diagnostic**, not a gate — rank
  statistics are brittle, and the binding contract row is row 3. It graduates
  to a gate only after its own calibration study.

### The statistic, and its null

**Statistic:** episode-blocked cross-validated one-step predictive improvement
of a history-augmented model over the memoryless model — fit
`p(O_{t+1} | O_t, A_t)` and `p(O_{t+1} | O_t, A_t, O_{t-1}, A_{t-1})` (same
mechanism class, same budget), score both on held-out **episodes**, take the
per-dimension improvement family and judge the **family statistic** (max over
observation dimensions) against **the null of that family statistic** (S3 —
L5 tests many things at once, so this is its default case, not an edge case).

**Null:** permute the *history* features across episodes — marginals preserved,
within-episode alignment severed, the `(O_t, A_t) → O_{t+1}` pairs intact —
at **episode granularity** (S1/S1b; reusing `arm_preflight`'s
`_episode_constant`/`_episode_mean` discipline and the shared bootstrap
machinery that L4 already uses). Tails are read as **quantiles of the
permutation draws**, never z-scores (family maxima are right-skewed — the
standing rule, written before this design).

**Calibration and the stated rates (S9, S14, S17):**

- **FPR:** p-value uniformity verified on ≥20 true-MDP datasets per
  environment (KS against uniform), α stated at 0.05. A statistic that cannot
  vary is reported *untestable*, never passed.
- **Detection rate:** measured on ≥20 constructed-POMDP datasets per
  environment. **This is cheap by construction**: L5 consumes datasets, not
  RL runs — generation is ~1 min/dataset and the statistic is one small model
  fit — so the detection-rate and FPR curves are measured at n ≥ 20 per cell
  without touching the RL training budget. The 2×2's RL runs stay at seed
  scale; the *detection* claim gets real statistics.
- **Positive control before anything is reported:** on each environment, the
  masked arm must be detected and the unmasked arm's p-values must be uniform,
  before any grid cell citing L5 runs (S17: the control sits on the reported
  endpoint).

**Wiring:** L5 runs in the serving seam at fit time. On falsification the
label carries `L5-FALSIFIED(markov, p=…)`, GRACE abstains with that reason and
falls back to the base critic — the existing abstention path, which the pilot
already exercises end-to-end (the d100s0 s3 abstention propagated correctly
through provenance, plotting, and the paired report).

---

## 3 — The POMDP construction, certified

**Wrapper:** the existing `MaskedObservationWrapper`; per-env masked
components are the velocity-like dimensions (below). **Generation applies the
mask during collection** (§1 finding) so the behavior policy's information set
matches the declared diagram; the dataset stamp records the information set
(`behavior_view: masked|full`) so the two constructions can never be
conflated.

**Certification that it is genuinely partially observed, not merely noisier**
(each check is a measurement, run per environment before the grid):

1. **The L5 positive control itself** — the masked view must fail the Markov
   test at the stated α (a POMDP the falsifier cannot see is not a usable
   POMDP cell). Note iid observation noise does NOT create history
   dependence, so this check separates "partial" from "noisy" by construction.
2. **Memory pays**: a history-augmented (or recurrent) learner must beat the
   memoryless learner on the masked env by a seed-resolved margin — hiding
   dimensions the policy never needed would pass check 1 on dynamics grounds
   while making the POMDP cell vacuous for the return comparison.
3. **The hidden component is recoverable from history** (a small probe
   regressing the masked dims from the window): establishes the window
   structure the branch relies on actually carries the information — and its
   failure predicts where the branch's abstention should fire.
4. **Per-context return spread** via the existing `eval_per_context` writer
   (`last_unmasked_obs`): return must vary with the hidden component.

---

## 4 — The environment set

CartPole-v1, Acrobot-v1, LunarLander-v3 — all discrete-action (the cql/iql
constraint), Box observations (the mask wrapper's constraint), all reachable
by the existing dqn generator pipeline. The axes they separate:

| | CartPole | Acrobot | LunarLander |
|---|---|---|---|
| horizon | short–mid (T̄ ≈ 16–130, cap 500) | long (cap 500, s1 never terminates) | variable, long (land/crash/timeout) |
| obs dim | 4 | 6 | 8 |
| reward under the gate | two-valued categorical | two-valued categorical | **continuous shaped** |
| dynamics | near-linear | nonlinear underactuated | thrust + contact |
| masked dims | [1, 3] (velocities) | [4, 5] (angular velocities) | [2, 3, 5] (vx, vy, ω) |

LunarLander is the load-bearing addition: its continuous reward exercises the
`_resolve_reward_type` → MDN-R path — historically the failure-prone branch
(the scale-floor family of bugs) — where CartPole and Acrobot are both
categorical-R. Acrobot brings the long-horizon/annealing regime and the known
hard fit. An environment set whose third member is "CartPole with different
constants" would not test invariance; these differ on every axis the estimator
has ever broken on. (LunarLander needs `uv sync --extra box2d`; MuJoCo-class
continuous-action envs are out of scope while the offline algos under test are
discrete.)

Per S17, each new environment gets **one calibration leaf first** — a single
(base, confounded, s0) run whose deployment return must move and sit in a
plausible range — before its grid is launched.

---

## 5 — The revised grid, costed

**Per environment** (offline first, one confounded operating point — the
audited defaults, one configuration everywhere, per the D-D generalisation
result):

- **Datasets:** 3 dataset seeds × {true-MDP, true-POMDP(masked-behavior)} =
  6 certified datasets. Generation + certification: well under an hour total.
- **RL runs** at 3 ds × 3 ts (the `ds{d}_ts{t}` layout from the integration
  plan): base is declaration-independent → 2 truths × 2 algos × 9 = 36 runs;
  grace → 2 declarations × 2 truths × 2 algos × 9 = 72 runs. 108 runs ×
  ~500 s (measured cql 380 s / iql 620 s) ≈ **15 h**.
- **Fits (cached, fit once per (dataset, declaration))**: 12 fits. CartPole
  ≈ 1 h each (measured) → ~12 h; Acrobot's fixed-step M-step measured much
  cheaper per fit (~43 s/fit ⇒ transform ≈ 23 fits ≈ 20–40 min) → ~3 h;
  LunarLander unknown until its calibration leaf. Envelope **3–12 h/env**.
- **L5 calibration**: ~40 extra datasets (~1 h generation) + 40 statistic
  runs (minutes each) — noise next to the RL budget.

**Per env ≈ 20–27 h GPU; three environments ≈ 60–80 h ≈ 2.5–3.5 GPU-days**,
staged: CartPole (harvesting the pilot, below) → Acrobot → LunarLander, each
gated by its calibration leaf and L5 positive control. At 3 ds × 2 ts the
full set fits in ≈ 45–55 h. This replaces the previously proposed n=15 seed
expansion of the old grid, which is **not launched**.

**Projection with the MEASURED augmentation ratio (2026-09-03, before
scheduling — the discipline that caught the 17 h → 36 h error).** The cost
probe (d100 σ=0 seed 0, 49k rows, GPU, under sweep contention) measured
selection 912 s, unaugmented fit 7709 s, augmented fit 13184 s at k = 1:
**ratio 1.71**. The absolutes are contaminated by contention (the plan's
uncontended CartPole fit is ~1 h); the ratio is the usable number. Fits are
counted under the cache, one per `(cell, dataset seed, declared
observability)` — with one correction to the "12 fits" above: the
POMDP-declared arm at **k = 0 collapses to the MDP branch** on the SAME
buffer, cache dir and dataset id, so it is a content-address HIT on the
MDP-declared fit, not a second fit. Under the cut, k = 0 was selected on
27/27 true-MDP calibration rows, so that is the expected case.

| per environment | fits (fit-units) | selections | RL runs |
|---|---|---|---|
| MDP-declared, 6 datasets | 6 × 1.00 | — | — |
| POMDP-declared on true-MDP (k = 0 expected) | 0 (cache hit) | 3 | — |
| POMDP-declared on true-POMDP (k = 1 expected) | 3 × 1.71 = 5.13 | 3 | — |
| training | — | — | 108 × ~500 s |
| **expected** | **11.1 fit-units** | **6** | **15 h** |
| worst case at k = 1 everywhere | 16.3 fit-units | 6 | 15 h |

- **CartPole** (1 h/fit uncontended; 0.25 h/selection contended): fits
  11.1 h (worst 16.3 h), selection 1.5 h, training 15 h → **≈ 27.6 h
  expected, ≈ 32.8 h worst**. If the fits run under contention as the probe
  did (2.14 h/fit) add ~12.7 h.
- **Acrobot** (transform ≈ 23 fits × 43 s ≈ 0.3–0.6 h; take 0.5 h): fits
  5.6 h (worst 8.1 h), selection ≤ 3 h (longer episodes, more rows;
  unmeasured), training 15 h → **≈ 24 h expected, ≈ 26 h worst**.
- **LunarLander**: fit cost unknown until its calibration leaf (the MDN-R
  path); formula `6·F + 3·1.71·F + 6·S + 108·T`, with training alone ≥ 15 h.
- **Three environments: ≥ 67 h expected, ≥ 74 h worst at k = 1**, plus
  LunarLander's fits — consistent with the 60–80 h envelope above only if
  LunarLander's fit is ≤ ~1 h. **k = 2** (state 4 + 5k = 14 dims, not 9) is
  possible on some datasets and its ratio is NOT measured; the probe re-run
  (peer session) is asked for a forced-k = 2 timing to bound the worst case.

**Revised the same evening for the materiality selector (dr2_cut stripped,
ruled 2026-09-03).** Selection now fits k and k+1 and stops when the served
contrast moves by no more than L4's half-width; a supplied k (declared MDP
included) buys the *sufficient?* fit at k+1 and, for k >= 1, the
*necessary?* fit at k-1 (`grace_k_diagnostics`, a budget switch). Per
environment, diagnostics ON:

| arm | fits (fit-units) |
|---|---|
| MDP-declared, 6 datasets: k = 0 + sufficient? at k = 1 | 6 × (1 + 1.71) = 16.3 |
| POMDP-declared on true-MDP (delegated): k = 0, k = 1 — both cache HITS | 0 |
| POMDP-declared on true-POMDP (delegated): k = 0, k = 1 hits; k = 2 needed to close at k = 1 | 3 × r₂ |
| **expected** | **16.3 + 3 r₂** (r₂ = the k = 2 ratio, being measured) |

With r₂ ≈ 2.4 (state 14 vs 9 dims, linear guess pending the measurement):
CartPole ≈ 23.5 h fits + 15 h training ≈ **38 h**; Acrobot ≈ 12 h fits +
15 h ≈ **27 h**; LunarLander training ≥ 15 h + unknown fits. Diagnostics OFF
on the MDP-declared arms returns to the earlier table (11.1 fit-units + 3
(r₂ − 1.71) for the delegated close). The selection pass (912 s × 6) is gone.
Assumptions: (i) the materiality criterion stops at k = 0 on true-MDP data
(first empirical point: identical k = 0 / k = 1 intervals on d100s0 s0);
(ii) both declared arms share one cache directory (the address is content);
(iii) the ratio measured under contention transfers (both fits were
contended alike).

**Pilot harvest** (all CartPole, all (declared MDP, true MDP)): `danull`
(unconfounded control), `d100` (confounded), `d100s0` (σ=0 no-harm control,
with its two abstained seeds reported as abstentions) — these populate the
(MDP, MDP) row's return and critic-accuracy columns at 3 seeds, and the σ=0
IQL grace-below-base gap (~7 steps) flagged during the pilot carries into the
no-harm analysis. `d025` / `d010asym` (finishing now) are **archived as
instrument design** — they established where proximal identification is
load-bearing and do not appear in the contract grid.

**Analysis plan — comparisons are WITHIN-COLUMN, fixed before any result
exists (ruled 2026-09-03).** Masked-behaviour generation changes more than
observability: a policy acting on the partial view is genuinely less
competent, so the true-POMDP datasets differ from the true-MDP ones in data
quality and distribution as well. Honest — a POMDP really does have a worse
achievable policy — and confounding for any cross-column reading. Therefore:

- **row 1 vs row 2** (declared MDP vs declared POMDP, both on TRUE-MDP data)
  — identical dataset, only the declaration differs. This is where
  "over-assumption is cheap" is measured.
- **row 3 vs row 4** (declared MDP vs declared POMDP, both on TRUE-POMDP
  data) — same argument. This is where "degradation, but detected" is
  measured.
- **across columns** (true-MDP vs true-POMDP) is confounded by achievable
  performance and MUST NOT be read as a GRACE effect. No figure or table
  lines the two columns up on a shared performance axis.

**Scope statement (ruled 2026-09-03): the demonstrated POMDP class is
finite-memory and SHORT-WINDOW by construction.** Hiding velocities makes
the hidden state recoverable from one lag (a finite difference of positions
reconstructs it), so k = 1 is the EXPECTED selection, not a discovery, and
no environment in this set exercises the "no k <= k_max passes -> abstain"
path — that path is verified in unit fixtures only, and the paper says so.
A non-finite-window construction (e.g. a slowly-switching latent affecting
dynamics) belongs to the D-F/D-G work, not this grid.

**Cell reporting** (each cell, per seed, via the deployed report):
return vs base, critic accuracy (`q1_contrast_pred/error`), the return
decomposition; the (MDP-declared, POMDP-true) cell adds **detection rate at
stated FPR**; the (POMDP-declared, MDP-true) cell adds the **cost of
over-assumption** (selected k distribution + paired return delta vs the
MDP-declared arm).

---

## 6 — Retired, plainly

Instrument design, question answered, archived — not the product:

- the d-separation sweep: `d_d_sweep_{d005,d010,d050}.yaml` and their
  datasets' role in future grids;
- the asymmetric point (`d_d_sweep_d010_asym`, `e1_d010asym*`), and `e1_d025*`;
- the σ grid beyond the single audited operating point;
- the planned n=15 seed expansion of the E1 grid (superseded by §5);
- `tools/merge_dd_sweep_seeds.py`, `tools/render_dd_sweep_s1c.py` and the
  sweep-specific renderers (kept in-tree, marked archive in their docstrings;
  nothing deleted mid-campaign).

Kept, with their roles restated: L1 (the catalogue IS the user contract),
L2 (both queries, now consulted — the L2-drives-the-estimator roadmap item
becomes part of the branch dispatch), L3's MDP branch, L4 (abstention is how
"something must be done" is served), the reward-transform seam (the MDP
branch's serving path, and the POMDP branch's after augmentation), C3, and
the integration work (YAML driver — now carrying `declared:` and
`true_observability:` as first-class config axes; the transform cache — whose
content-addressed key covers the masked/augmented views automatically; the
deployed plotting — gaining the detection-rate table).

## Build estimate before the grid runs

| piece | estimate |
|---|---|
| L5 Markov statistic + episode-permutation null + calibration harness | 3–4 days |
| window augmentation + selection + serving/abstention wiring | 2–3 days |
| masked-behavior generation + information-set stamp + certification checks | 1–2 days |
| YAML driver + cache + seed split + plotting (from the integration plan, re-scoped to the new grid) | 2–3 days |
| **total** | **≈ 2 weeks of build**, then 2.5–3.5 GPU-days of grid |
