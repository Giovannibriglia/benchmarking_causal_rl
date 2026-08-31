# GRACE — General Regime-Adaptive Causal Estimator (as built)

Branch: `feat/grace-critic`. This documents the **implemented** architecture
and config surface (not the proposal). Tabular-first, discrete-latent slice;
the continuous-native extension is scoped (not started) in [Phase 3](#phase-3-seams-a10).

## What it is

A critic arm on the benchmark's identification axis (`CausalCritic = base ×
IdentificationStrategy`). GRACE = a **declared-graph template CBN** with two
likelihood channels, an **exact belief filter** over the discrete latent
block, a **null-calibrated regime router** with channel-split detection
statistics, and a **bootstrap-ensemble interval** — served per the rule:

| detected | served head |
|---|---|
| no defect | `Q_obs` (the trainable observational floor head) |
| coverage defect only | `Q⁻` (ensemble lower bound) |
| confounding, identification healthy | `Q_do` (interventional, mutilated channel) |
| confounding, identification unhealthy | `Q⁻` |
| router uncalibrated | `Q_obs` — **never a silent causal route** |

Training is exactly the observational floor's: the `Grace` strategy's
`critic_value` is a pass-through to the trainable head (`net.q_obs`), and the
learn-wrapper toggles routing OFF inside `learn`. Only the **serving surface**
(`act`, eval rollouts, the ablation's scoring hook) sees the routed estimand.
That construction is the no-harm/reduction guarantee (B6), unit-tested in
`tests/test_grace_reduction.py`.

## Module map (`src/rl/offline/grace/`)

| module | contents |
|---|---|
| `cell_graph.py` | `CellGraph` declarations per (cell, arm) — the A9 single source of graph structure; `identification_report` (the A9.3 graphical precondition); `CELL_GRAPHS` registry |
| `discretizer.py` | per-dim quantile bins fit on the dataset (state-conditional); reward class codec |
| `cbn.py` | `TemplateCBN`: EM fits, A3 router components, `q_do` value iteration (the single target-computation boundary), bootstrap ensemble, NBN mirror |
| `filter.py` | `BeliefFilter`: exact forward filter over (S×U) [POMDP] / (U,) [MDP] |
| `router.py` | `RegimeRouter`: thresholds + the serving rule; `calibrate()` (null stats → thresholds) |
| `machinery.py` | `GraceMachinery`: five-keys ingestion, fit lifecycle, coverage statistic, serving |
| `heads.py` | `GraceQNetwork` / `GraceRecurrentQNetwork`: the route-toggled serving wrappers |
| `builders.py` | base-parity builders (floor builder verbatim + strategy + wrap + install) |

Plus `Grace` in `src/rl/off_policy/identification.py` (the protocol hooks).

## The declared graphs (A9)

`CellGraph` encodes, per (cell, arm), exactly the SCM the generation-time
gates enforce: `U → A` **observational channel only** (declared on the
confounded arm and the fitting *template*; absent on basic — the
marginally-matched σ=0 construction makes A⊥U exactly; absent on biased — no
U at all); the **action-gated** `U → R` (`r += c_r·U·1[a=a_bad]`, present
wherever U is, including basic — `c_r_for` keeps c_r>0 at σ=0 where the noise
is action-independent); `S → O` emission (identity in MDP cells, real in
POMDP cells where S is latent); `S,A → S′`; the optional `U → U′` persistence
edge behind `rho` (declared, off, fit not implemented in this slice); **proxy
nodes declared explicitly absent** (`proxy_nodes=()` in every current cell —
no proximal-style point-identification claim is ever made, and the router
reflects it).

The mutilated channel is *derived*, never re-declared:
`CellGraph.mutilated_edges()` = the declared list minus in-edges of A. The
NBN mirror builds both channels from these lists with **shared
`CategoricalTableMechanism` modules** for every non-A node (A7; asserted at
object and storage level in `tests/test_grace_mutilation.py`).

**Paper cross-reference (A9.4):** the per-cell SCMs correspond to the
formalization in `docs/experimental_design.md` (the taxonomy's
per-cell structural equations; the action-dependent confounder mechanism is
the one merged at the cell-9 arc). Exact figure numbers in the paper
manuscript: TODO-verify against the camera-ready draft (the manuscript is not
in this repository).

## Estimation

* **MDP cells** (S observed as the joint state bin): exact-enumeration EM
  over the episode-static U. E-step: per-episode posterior from the behavior
  and reward channels (`P_b(A|S,U)`, `P(Rc|S,A,U)`), exact softmax over
  strata. M-step: responsibility-weighted Dirichlet-smoothed counts.
  Dynamics/initial-state/emission never carry U (the declared graphs have no
  `U → S′` edge).
* **POMDP cells** (S latent, O = binned masked obs): Baum-Welch over the
  latent chain × exact enumeration over U — one forward-backward per
  (EM-iteration, stratum), gamma/xi expected counts, all in log-domain.
* **Canonicalization**: the proximal label-swap convention, verbatim
  semantics (`_CANON_EPS = 0.05`): no residual spread → symmetric prior;
  guarded flip so "stratum 1 = higher mean state-conditional reward residual"
  — every E-step, and re-anchored on every online refresh (Gate-B label
  persistence). Consequence: **stratum index 0 is the U=0 (clean-world)
  reference stratum**.
* **`q_do`** (the target-computation boundary): exact value iteration on the
  mutilated channel, `Q(u,s,a) = E[R|s,a,u] + γ(1−p_done)·E_{s′}[max_{a′}Q]`.
  Transitions: dense joint CPT when the joint bin space ≤ `joint_cap` (2048;
  CartPole 6⁴=1296), else the **A8 factored per-dimension approximation**
  (per-dim next-bin CPTs whose product approximates the joint law; Acrobot
  4⁶=4096). The factored mode is an explicit, documented approximation
  validated by the reduction test — per A8, if it fails that gate on Acrobot
  the cell stops and reports rather than shipping silently.
* **Ensemble**: K episode-level bootstrap refits → `[Q⁻, Q⁺]` = min/max of
  the deployed `Q_do` across members (B4's minimal interval mode; no MSM
  ball in this slice).
* **Unvisited state bins fall back to `Q_obs`** at serve time (tabular
  coverage holes must not serve stale zeros).

## The router (A3)

Data-derivable statistics only (leakage rule R5 — the generation-time gate
metadata is computed *with* the recorded U and is ground truth for scoring,
never router input; asserted in `tests/test_grace_router.py`):

* `delta_a` — held-out per-transition log-likelihood improvement of the
  **U→A-only** restricted fit over the U-free fit: the *confounding-specific*
  component.
* `delta_r` — same for **U→R-only**: the heterogeneity / PO component. It
  legitimately fires on the basic arm (c_r>0 action-independent U reward
  noise) — which is exactly why the split exists: a "confounded" verdict
  requires `delta_a`.
* `coverage` — state-conditional action coverage: the 10th percentile over
  sufficiently-visited state bins of `min_a π̂_b(a|s)`. Never marginal.
* `width` — mean deployed ensemble width over visited bins; `separability`,
  `belief_entropy` — inference-health telemetry.

Thresholds are **null-calibrated on basic-cell runs** with the repo's
`k·noise` convention (`k = NULL_CALIBRATION_K = 1.5`): fire-above stats get
`mean + k·sd`, the coverage defect (fire-below) `mean − k·sd`. Stored per env
in `reproducibility/rl_regimes/_base/grace_router_reference.yaml` (populated
by the E0 calibration block). A missing reference ⇒ **uncalibrated** ⇒ serve
`Q_obs` — mirroring the null-calibration gate's missing-reference convention.
The A9.3 graph gate additionally blocks the confounded serving mode when the
declared graph has no `U → A` edge, whatever the statistics say.

## Config surface

Critic arms (cell YAML `critics:` blocks): `grace` (router on) and
`grace_no_router` (always-causal ablation switch — serves `Q_do`
unconditionally; **exempt** from the null-calibration gate like sensitivity,
while `grace` is gate-judged like proximal). `grace_obs_only` is a
**test-verified alias** of `observational` (A1) — parameter-identical by
construction, not a burned sweep arm.

Cell-level options block (all optional):

```yaml
grace:
  u_card: 2          # |U| ∈ {2, 4, ...}
  rho: 0.0           # U persistence — declared, off; >0 raises (not this slice)
  ensemble_k: 5      # bootstrap members
  n_bins: null       # per-dim bins; null = auto (6 if obs_dim<=4 else 4)
  n_latent: 32       # POMDP latent-S cardinality
  em_iters: 20       # MDP EM iterations (pomdp_em_iters: 30)
  alpha: 0.5         # Dirichlet smoothing
  joint_cap: 2048    # joint-transition cap; above -> factored (A8)
  seed: 0            # grace-internal generator (never the global stream)
```

The arm-defining switches (`router`, `interval`, `deploy`) live on the
`CriticSpec` and always win over the cell block (a cell cannot silently turn
`grace_no_router` back into `grace`).

Algo variants (classical simulation / return-level comparisons):
`offline_dqn_grace`, `bcq_grace`, `cql_grace`, `iql_grace`,
`offline_dqn_recurrent_grace`, `online_dqn_grace` (refresh-cadence CBN,
online mutilated channel). These default to `deploy: u0` — the **U=0
reference-stratum deploy**, the return-correct choice under the action-gated
confounder (cell-9 semantics), while the ablation arms deploy the marginal
`E_U[Q]` (the oracle-comparable convention).

## Metrics added to the ablation (D3)

`critic_ablation_metrics.csv` gains additive columns:

* `value_mse_to_mc` / `mc_rtg_mean` — Q(s, a_data) against the **absolute MC
  return-to-go** computed from the dataset episodes (the budget-independent
  anchor ported from `tools/probe_offline_budget_v2.py`, identical iteration
  order and 4000-point subsample).
* `value_mse_to_mc_u0` / `mc_rtg_u0_mean` — restricted to U=0 episodes.
  **Interpretation caveat (approved D3):** the U=0-stratum RTG's continuation
  is π_b|U=0, *not* the target policy — it is a data-consistent
  deployment-stratum reference (the clean world the eval env realizes), not
  an interventional oracle. Uses the *logged* U evaluation-side only.
* `router_verdict`, `router_serve`, `router_delta_a`, `router_delta_r`,
  `router_coverage`, `ensemble_width`, `grace_separability`,
  `grace_belief_entropy` — grace telemetry (blank for other critics).
* `gap_closed_fraction` now also covers the grace arms (was proximal-only).

The reporting layer (`regime_report.py`) judges `grace` with the same
fixed-denominator null-calibration gate as proximal (an additive per-(env,
algo) row; `grace_no_router` exempt), and aggregates the new numeric columns.

## NBN usage and the vendored snapshot (A5/A6)

`nbn/` is a **pinned vendored snapshot** of
[NeuralBayesianNetworks](https://github.com/Giovannibriglia/NeuralBayesianNetworks)
at `0.7.dev37+ga5f97d5d7` (see `nbn/NOTICE.md`; GPLv3, `nbn/LICENSE`).
GRACE uses NBN for: graph representation (`DAG` built from the `CellGraph`
edge lists), CPT storage (`CategoricalTableMechanism`, shared module objects
across the two channels), and **exact VE cross-checks** of the mutilation
(`tests/test_grace_mutilation.py`). The EM, filter and value iteration run on
tabulated tensors in GRACE — this snapshot has no EM, no temporal machinery,
and no graph-mutilation API (Phase-1 audit), so those algorithms live here by
design.

Snapshot sharp edges worked around (do not "fix" the vendored tree):

* `model.intervene()` breaks exact VE and VE silently ignores `do=` — GRACE
  never uses either; **mutilation is graph surgery at construction**
  (`CellGraph.mutilated_edges()`; conditioning on the mutilated net's root A
  *is* do()).
* VE caches are instance-level and keyed by `id(model)` —
  `TemplateCBN.invalidate_nbn_cache()` after every rebuild or in-place CPT
  change (A7-tested).
* `model.query(engine=...)` takes an engine *instance*; strings only resolve
  via `default_engine` at construction.
* `NeuralBayesianNetwork.load()` does not restore the state_dict; mechanisms
  are (re)populated the `from_bif` way (`_logits`, `_n_classes`,
  `_parent_cards`, `_parent_strides`, `_class_values`).

**Deferred-sync TODO (A5(c)):** upstream has moved past this snapshot (v0.14
line, with first-class IS proposals + ESS-gated fallback + per-query
diagnostics). Syncing is the **first commit of Phase 3**, with a re-audit of
the Phase-1 §1A.3 API surface (entry points, LW `return_ess`/`return_psis_k`
kwargs, `fit`/Dirichlet-update signatures, every sharp edge above) and an
updated `nbn/NOTICE.md` provenance block.

## Known limitations (this slice)

* **Recurrent serving without a reward stream**: eval rollouts thread
  (trunk state, belief, last greedy action); the belief updates on emissions
  and the *previous greedy action* (exact when eval is greedy), but the
  reward channel is not available on the serving path, so U-evidence stays at
  the prior there. Offline scoring (flat single-obs eval set) uses the
  emission-conditioned prior belief (QMDP-style belief-averaged Q_do).
* **Factored transitions (A8) are UNEXERCISED at production scale.** The
  factored per-dimension transition mode engages only when the joint bin space
  exceeds `joint_cap` (2048). CartPole at 6 bins × 4 dims = 1296 stays *under*
  the cap, so it runs the exact-joint path; Acrobot (4⁶ = 4096) is the only
  wired env that would exercise the factored path — and Acrobot is **deferred
  to a background job** under the Block-A re-scope (it has no stored
  independent `noise_ref`, so its gate verdicts would be provisional
  regardless). Consequently the factored approximation is covered *only* by
  the unit-level reduction test, never at production budget, and the **R1
  state-space-blowup risk is not closed** by the offline_mdp blocks. The
  deferred Acrobot job is its validation; until that job runs, treat any
  factored-mode claim as untested at scale, and honour the A8 stop rule if it
  then fails the reduction gate at the chosen binning.
* **`rho` persistence** is declared but its fit is out of scope (raises if
  enabled).
* **POMDP latent-S EM** is a random-init Baum-Welch (seeded, canonicalized);
  its identification quality is reported via separability/belief-entropy
  telemetry, not assumed.
* The NBN mirror covers the joint-transition MDP mode (the factored path has
  no single S′ CPT to mirror).

## Phase-3 seams (A10)

Kept deliberately: (1) the **discrete latent block** (U enumeration:
responsibilities, belief, the mixture in the backup) is separated from
observable-evidence evaluation (`_step_loglik_mdp` / `_local_evidence_pomdp`
are the only places observable likelihoods are computed — continuous
mechanisms swap there); (2) **`q_do` is the single target-computation
boundary** (the continuous slice replaces its interior with mutilated-net
ancestral sampling, same signature); (3) nothing tabular leaks into the
`Grace` / `RegimeRouter` public signatures. Continuous actions are out of
scope at every phase (the ablation host is discrete-action by construction).

## Cost of the method (measured)

GRACE's machinery is cheap; its cost to the benchmark is the extra *arms*.

* **Per-fit CBN cost** (one `fit_from_buffer`: main EM + K=5 bootstrap + the
  three restricted router fits + value iteration): **~1.8 s** at CartPole
  scale, **~13.2 s** at half-Acrobot scale (factored mode). Negligible beside
  a 50k-step learner.
* **Marginal benchmark cost**: the critic set grows 4 → 6 arms, and an
  ablation task costs `(1 base actor + N critic arms) × grad_steps × algos`,
  so a basic/confounded point costs **~+50%**. That is the honest price of an
  always-on ensemble plus router: the interval needs K refits and the router
  needs its restricted fits, both per arm, on every point.

The E0 router calibration is cheaper than it looks: its components are
*dataset* quantities (no base-learner training), so it costs dataset
generation plus CBN fits only.

## Experiment scope (Block A onward)

Experiments run as **blocks**, each ending in a push and a relay pause.
Block A (offline_mdp E1) is scoped to **CartPole-v1 × {cql, iql}**, all 7
sweep points, all 5 seeds — those being the only (env, algo) pairs with
stored *independent* `noise_ref` values and therefore the only ones whose
G1/G2 verdicts can be non-provisional. Acrobot and the `offline_dqn` / `bcq`
bases are deferred to background jobs that gate no report; see the A8
limitation above for what the Acrobot deferral leaves untested.

## Results

Populated per block (E0 calibration; E1 ablations; E2 return-level runs; E4 RE
ladder; E3 online) — see the per-block reports and `results/_report/`.
