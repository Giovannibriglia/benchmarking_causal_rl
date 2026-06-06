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

## Evaluation: window returns vs per-episode J

The benchmark-mode eval metric accumulates rewards over a fixed
`rollout_len`-step window and saturates for CartPole-like tasks
(`512 − #failures`), hiding most of the Cell-2 regret. Acceptance and the
Phase-3 regret protocol use TRUE per-episode returns
(`tools/quick_j_eval.py`; ≥100 full episodes per checkpoint).
