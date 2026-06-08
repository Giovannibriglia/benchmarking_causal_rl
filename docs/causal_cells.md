# Causal cells — design notes

## Cell 2: what kind of partial observability the mask creates

**[flag for the paper]** Velocity-masking is *history-recoverable* partial
observability: the hidden coordinates (q̇ / ẋ) are deterministic functions of
a short window of past observations, so an agent with even minimal memory can
restore full observability by construction. That is exactly why the Cell-2
variant works: frame-stack PPO (4 stacked masked frames) gives the policy the
finite-difference information needed to re-estimate velocities, literally
implementing strategy S3 ("restore observability") from the taxonomy. The
Cell-2 basic (feed-forward PPO on the masked observation) cannot, and pays
the regret.

The *irrecoverable* / genuinely epistemic flavor of Cell 2 — a per-episode
hidden dynamics parameter (e.g. randomized pole length or actuator gain,
never emitted in any observation), where no amount of observation history
identifies the context and the agent must act under posterior uncertainty
(Ghosh et al. 2021) — is **noted as future work and intentionally not built**
(mid-Phase-2 gate decision 3, 2026-06-05). When interpreting the headline
plot: Cell-2 recovery by the variant demonstrates restorable identification,
not a general solution to epistemic POMDPs.

## Cell-2 variant choice

Frame-stack PPO was chosen over recurrent PPO (both are canonical) because it
restores the masked information with zero new algorithm code — the variant is
an env-id (`causal/<anchor>-cell2fs`), keeping the basic-vs-variant comparison
free of optimizer/architecture confounds beyond input width.

## Attribution note (Phase-2 gate)

The HalfCheetah jump from ≈0 to J≈2629 between the first and final Phase-2
runs is attributed **primarily to the training-budget extension (0.5M → 2M
steps)**; HalfCheetah truncates in lockstep, so the autoreset fix likely
mattered little there. The fix's correctness credit belongs to the
CartPole-family results (variable-length episodes), where it took the Cell-2
variant from J≈30 (corrupted training data) to J≈497.

## Cell 3: the horizon ablation (medium50) — graphical vs statistical identifiability

**[flag for the paper]** Within Cell 3 the backdoor criterion holds at every
horizon — the effect is *graphically* identifiable — yet IPW with EXACT
logged propensities transitions from valid to degenerate as the horizon
grows. Three regimes, same cell:

1. **Analytic fixture, H=6**: IPW(known) matches closed-form DP values and
   DR within 0.25 on a ±6 J scale (exact agreement).
2. **Real data, H=50** (`causal/cartpole/medium50-v0`): IPW(known) CI
   [8.0, 55.1] overlaps DR [51.7, 65.5]; self-normalized IPW point-matches
   the truth (50.0 vs naive/DM ≈ 50). Valid but variance-dominated.
3. **Real data, H=200+** (standard tiers): full-episode importance products
   collapse to 0/∞ — *statistical* identifiability is lost (curse of
   horizon) even though nothing about the causal graph changed.

`medium50` is therefore the HORIZON ABLATION of Cell 3, not a workaround:
identification ≠ estimability, and the cells story should separate the two.

## Cell 3: the expert-tier inversion (S4 / coverage result)

**[flag for the paper]** On the expert tier (ε=0.05, narrow support) the
basic/variant ordering INVERTS: BC achieves norm regret 0.06 while CQL
collapses to 0.51 (medium tier: BC 0.60 vs CQL 0.42). This is the known
BC-vs-offline-RL support phenomenon — value-based offline RL needs coverage
to evaluate counterfactual actions and its conservatism actively hurts on
narrow expert data, while imitation inherits the logging policy's quality
directly. Present as the S4/coverage result of the taxonomy, not a failure:
which strategy is correct in Cell 3 depends on the support axis.

## Continuous Cell-1 reference — ON HOLD (Phase-3 gate flag 1)

The stock `mujoco/halfcheetah/medium-v0` behavior policy (J≈12,089) is far
stronger than the paper_mini PPO reference (J≈2,629), so offline algorithms
can show negative regret on the continuous anchor. AWAITING AUTHOR CHOICE:
(A) add SAC as the continuous Cell-1 reference trainer in Phase 6
[gate-recommended] vs (B) reframe the reference as max(J_online, J_behavior).
**No Phase-6 HalfCheetah work starts before this lands.** Phase 6 must also
make references reproducible in-pipeline (train-or-load keyed by the cell
YAML; no dependencies on gitignored run dirs).

## Cells 7–8 paired result: sampling uncertainty ≠ delphic uncertainty

**[flag for the paper]** The Cell-7 variant is *ensemble-pessimistic offline
RL (delphic-inspired, Pace et al. 2024)*: FQI on rewards penalized by
bootstrap-ensemble disagreement, with a Kallus–Zhou-flavored sensitivity cap.
In Cell 7 (Z observed) it wins decisively (norm regret 0.24 vs BC 0.63). In
Cell 8 the κ-ablation (κ ∈ {0, 1, 3} → identical J; ensemble std = 0.022)
demonstrates WHY the simplification fails there: bootstrap disagreement
captures **sampling** uncertainty, which vanishes as data grows, whereas
delphic uncertainty is **variation across confounding-compatible models** —
on masked confounded data all bootstrap members agree with each other and
are all equally wrong. Present the Cell-7 success and the Cell-8 blindness
as a paired result; the Cell-8 variant is the Kallus–Zhou sensitivity
INTERVAL instead (gate-approved switch).

## Cell 6 honesty note: what the latent world model actually demonstrates

**[flag for the paper]** In our velocity-masked instantiation, Cell 6 is the
history-RECOVERABLE flavor of hidden state, so the GRU world model is doing
state restoration (S3) — the same mechanism as the Cell-2 frame-stack — and
its J ≈ 212 ceiling is the IMITATION-quality cap of its BC head (it matches
the full-information BC on the same data, norm regret 0.60), not the
unidentifiability bound. The unidentifiability claim genuinely binds only in
the epistemic variant (hidden per-episode dynamics parameter, future work).
Corollary TESTED (belief+CQL probe, `tools/probe_belief_cql.py`): reusing
the trained GRU encoder with a CQL head was expected to recover the Cell-5
result (~299). **Measured: J = 210.9 ± 43.3 — the reduction did NOT
materialize**; CQL on the learned beliefs performs exactly like the BC head.
Reading: the encoder's latent space, shaped by world-model + imitation
losses, restores enough state for imitation but not the geometry offline
value learning needs — so in practice the Cell-6 variant is capped twice
(imitation head AND encoder quality), even where restoration is possible in
principle. Caveat: single-seed probe with untuned CQL hyperparameters — the
direction of the result is informative, its magnitude is not load-bearing.
Paper-flag this as a negative result alongside the honesty note above.

## Methods note: the confounding gate needs a conditional test

**[flag for the paper]** A state-symmetric U→action bias
(``logits' = logits + β·U·e_{a*}``) leaves the MARGINAL action distribution
unchanged — U boosts whichever action is preferred at each state, so
marginal A–U tests (contingency/TV) have zero power against it. The gate's
condition (ii) therefore tests **conditional** dependence via the propensity
residual ``log π_b(a|s,U) − log π̄(a|s)`` (logged propensities minus the
U-blind behavior clone), which isolates "A depends on U given s".

## Methods note: the sign of naive-OPE bias is coupling-dependent

**[flag for the paper]** "Naive OPE inflated" is not a law: in our Cell-7
instantiation all data-side estimators (naive/DM/DR ≈ 72–81) UNDERestimate
true deployment values (BC 198, variant 384) by 2.5–5×. The U→action bias
degrades the logging policy's survival (shorter episodes → lower logged
returns), and the additive ±δ reward shifts cancel in expectation — so the
bias direction flows through the dominant coupling channel (here: episode
length), not through a universal inflation. The robust statement is
"confounding makes OPE wrong regardless of estimator"; the sign is
DGP-specific.

## Evaluation: window returns vs per-episode J

The benchmark-mode eval metric accumulates rewards over a fixed
`rollout_len`-step window and saturates for CartPole-like tasks
(`512 − #failures`), hiding most of the Cell-2 regret. Acceptance and the
Phase-3 regret protocol use TRUE per-episode returns
(`tools/quick_j_eval.py`; ≥100 full episodes per checkpoint).

## Phase-6A grid findings

**[flag for the paper — three results]**

1. **Gate-statistic factorization** (heatmap iii, the gate-validation
   figure): over the β×δ grid the A–U conditional z-score depends ONLY on β
   (3.1–6.4 → 13.2 → 18.1) and the R–U z-score ONLY on δ — the two gate
   conditions cleanly isolate their designed causal pathways.
2. **Gate power boundary**: β=0.5 confounding is undetectable at 300
   episodes (z = 2.34 < 3) and decisively detected at 600 (z = 6.4) — the
   gate's sensitivity is a sample-size statement, quotable as an operating
   characteristic.
3. **Non-monotone naive-OPE bias** (heatmap ii, caption-level claim only):
   |naive − true J| peaks at weak-β/strong-δ (175 at β=0.5, δ=0.5) rather
   than at maximal confounding — the bias channel mixes action-pathway
   survival effects with reward shifts, so "stronger confounding" does not
   monotonically mean "more biased naive OPE".

Dataset ruling (6A gate, decision A): the 600-episode confounded datasets
are canonical for cells 7/8; June-6 (300-episode) values are archived in the
run dirs; the within-CI drift is explained-and-resolved. Collection tools
now hard-error on existing dataset ids (no silent overwrites), embed the
full collection config in Minari metadata, and cell YAMLs can pin
`dataset_expect:` blocks asserted at load time.

## Continuous anchor: the gate is condition (ii) ∧ (iii), condition (i) is a diagnostic

**[flag for the paper]** The confounding gate's three conditions —
(i) |J_naive − J_IPW| > τ, (ii) A ⊥̸ U | s, (iii) R ⊥̸ U — were designed for
the discrete short-horizon anchor. On the continuous HalfCheetah anchor
(H ≈ 1000, 6-dim actions) condition (i) becomes **non-discriminative**: the
Gaussian behavior clone underfits the structured SAC-based logging policy by
a systematic ≈ −2.87 nats/step, so even a 50-step importance product
saturates the weight clamp and the self-normalized IPW collapses onto naive
(gap ≡ 0.000) — INDEPENDENT of whether the data is confounded. This is the
curse of horizon (documented for IPW estimation above) striking the *gate*
one level earlier than the OPE.

Resolution (Phase-6C ruling, Option A): on the continuous anchor the gate
ACCEPTS on the two horizon-independent conditions (ii) A ⊥̸ U | s (propensity
residual, z ≈ 14.5) and (iii) R ⊥̸ U (z ≈ 712); the neutered control still
fails both. Condition (i) is COMPUTED and REPORTED as a diagnostic, never
pass/fail. The gate's causal meaning — "U functionally drives both action
and reward" — is identical across anchors; only the *estimability* of one
redundant symptom differs. This is itself a clean cross-anchor result: the
identification status is graph-determined and anchor-invariant, but the
finite-sample detectability of its symptoms is horizon-dependent.

## Continuous Cell-1 reference and the regret normalizer

The continuous Cell-1 reference is SAC (ReLU nets, UTD 1.0) at 2M env steps:
J = 9,516 ± 120, curve rising-but-decelerating (the 12-checkpoint curve is
the continuous Cell-1 learning-curve panel). It sits below the stock medium
demonstrator (J_behavior(medium) = 12,451), so the **regret normalizer for
all continuous cells is the single constant J_ref^cont = max(9516, 12451) =
12451**, keeping normalized regret ≥ 0. SAC's 9.5k is reported honestly as
the *online* result against the ~12.5k demonstrator — the gap is itself a
continuous-anchor datapoint (online RL from scratch under-performs the
logged demonstrator within a 2M-step budget), not hidden by the normalizer
choice.

**[flag for the paper — methods]** Two on-policy-defaults-leak lessons from
building the continuous reference: (a) tanh-trunk actor/critic nets (the
repo's PPO default) cost ≈ 4× final return on HalfCheetah (3.2k vs 9.5k) vs
standard ReLU — activation defaults do not transfer across algorithm
families; (b) the shared off-policy loop calls update() once per vector
step (UTD ≈ 1/n_envs), so SAC needs an internal update-to-data multiplier to
reach the canonical UTD = 1.0 that MuJoCo SAC results assume.

## Minari fixed-info-schema quirk

MuJoCo envs vary their native `info` keys between reset and step
(`reward_run`, `x_position`, … appear only post-step), but Minari's
`record_infos` requires an IDENTICAL info-key set on every step. The
collection callback therefore emits a FIXED schema — only the load-bearing
`behavior_logprob` (and `confounder_u` when present) — discarding the env's
variable native infos (full observations are stored separately, so nothing
is lost). Without this, continuous-anchor collection raises a Minari
"Dict key mismatch".

## Reading the merged 8×2 money plot: anchors are NOT comparable in height

**[binding figure-interpretation rule]** Continuous normalized regret is
computed against a DEMONSTRATOR-level normalizer (J_ref^cont = 12,451, the
medium-tier behavior policy), which 100k-step offline learners do not
approach even in the IDENTIFIED cells (3, 5). Consequently the ABSOLUTE
regret levels are not comparable across anchors — a continuous cell sitting
higher than its discrete counterpart does NOT mean it is "harder". Only
WITHIN-anchor cell patterns and basic-vs-variant orderings carry meaning.

This is the same effect already visible on the discrete anchor, where
identified-cell regret is nonzero because medium-tier data caps achievable
performance below the Cell-1 optimum — the normalizer measures distance to
the reference, not to the best offline-achievable policy. The merged money
plot caption states this explicitly; the two anchors are drawn in separate
side-by-side panels (never a shared y-axis comparison) precisely to prevent
a height-offset misreading.

## Per-anchor variant choices (Phase-6C value-tiered ruling)

The Phase-4/5 variant algorithms `proximal`, `latent_world_model`, and
`ens_pessimistic` were implemented discrete-only (categorical heads /
discrete Q). Rather than risk fragile continuous ports late in the project,
the continuous anchor uses a value-tiered hybrid:

| cell | discrete variant | continuous variant | note |
|------|------------------|--------------------|------|
| 3, 5 | CQL / CQL | **IQL** | continuous-native (d3rlpy) |
| 4    | proximal | **basic only** | continuous proximal = future work |
| 6    | latent_wm | **basic only** | continuous recurrent latent-WM = future work |
| 7    | ens_pessimistic | **CQL** | CQL = continuous-native pessimistic offline RL, substitutes the bootstrap-ensemble proxy |
| 8    | kz_select / ens_pess | **kz_select / {BC, CQL}** | K-Z interval is anchor-agnostic |

All eight continuous BASICS are BC (anchor-agnostic). The
history-recoverable recovery story (cells 4, 6) is fully established on the
discrete anchor; the continuous basic-only rows still populate the regret
matrix and the confounding axis (cells 7, 8) is covered on BOTH anchors. The
substitutions are documented here so the cross-anchor comparison is read as
"different canonical variant per anchor", not a silent omission.

## Continuous-anchor completion notes (Phase-6C)

- **3-seed vs 5-seed asymmetry**: continuous cells use 3 seeds (IQM + bootstrap
  CI), discrete 5 — a compute trade documented here so cross-anchor CI widths
  are read correctly. The continuous claim is "the within-anchor pattern
  generalizes", carried at 3-seed resolution; the discrete anchor carries the
  full 5-seed statistics and all estimator relations.
- **Continuous OPE is naive-only** (fqe_iters=0): DM/IPW/DR are
  diagnostic-only and degenerate at H≈1000 (curse of horizon), so FQE is
  skipped. Regret uses true-env J vs the constant normalizer, unaffected. The
  discrete cells×estimators table carries the IPW≈DR / DM relations.
- **Continuous learning-curve grid is reference-only**: offline-cell curves
  were turned off for compute; the continuous Cell-1 panel comes from the
  relu8 SAC reference run (12-checkpoint curve). Discrete carries the full
  offline learning-curve grid.
- **Cell-8 KZ containment (HC)**: the Γ=2 interval is [940, 2938] in the
  LOGGED (confounded) reward space, while clean-env true J ≈ 868 — the
  interval does NOT contain true J. This is honest: (a) the interval is
  computed on confounded returns (which carry the +δ·U shift) while true J is
  measured in the clean env, so they live in offset reward spaces; (b) at
  H≈1000 the importance weights are extreme, widening/shifting the interval.
  Reported as an interval-calibration result at long horizon, not hidden — it
  does not change the cell-8 conclusion (nothing closes the regret; variant
  ties basic).

## The continuous matrix mirrors the discrete story (within-anchor)

Both anchors, grouped by identification regime:
- **identified** (3, 5): variant (IQL) beats basic (BC) — 0.77 vs 0.88.
- **confounded** (7): variant (CQL pessimism) beats basic decisively —
  0.60 vs 0.82.
- **hidden state** (4, 6): basic-only on continuous (variants are
  discrete-anchor-only); masked BC collapses toward random.
- **cell 8** (hidden + confounded + unknown π_b): variant (KZ-selected) ties
  basic at 0.91 — nothing closes the regret, the matrix's hardest cell on
  both anchors.
