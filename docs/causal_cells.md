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
