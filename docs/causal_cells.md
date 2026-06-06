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

## Evaluation: window returns vs per-episode J

The benchmark-mode eval metric accumulates rewards over a fixed
`rollout_len`-step window and saturates for CartPole-like tasks
(`512 − #failures`), hiding most of the Cell-2 regret. Acceptance and the
Phase-3 regret protocol use TRUE per-episode returns
(`tools/quick_j_eval.py`; ≥100 full episodes per checkpoint).
