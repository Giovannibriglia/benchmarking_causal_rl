"""Measure the null-calibration separation to pin k (PR 6 follow-up, feat/pin-k).

null_calibrated = gap < k * noise. We have the correct-pipeline endpoint; this
measures BOTH endpoints so k is set from the SEPARATION, not the passing number:

  A. CORRECT (control): the observational floor built on the SAME base class as the
     oracle (CQL vs CQL).
  B. BROKEN: the observational floor forced to a BARE DQN while the oracle stays CQL
     — the historical bare-DQN-vs-CQL base confound the gate exists to catch (DQN
     max-overestimation inflates apparent-Q above the conservative oracle). This
     reuses the EXISTING bare-DQN build (``_build_strategy_critic("observational",
     "offline_dqn", ...)`` -> the bare-DQN branch); no new forcing mechanism.

Both configs are measured in ONE harness per seed: oracle_u/proximal/observational
are all CQL, and a bare-DQN observational is injected alongside — all trained on the
SAME episode-grouped stream and scored against the SAME CQL oracle, so A and B differ
ONLY in the observational's base class. The training loop matches the runner's
offline-grouped budget (batch 128, seq_len 8, offline_grad_steps updates).

Measurement only — imports the shipped machinery, changes no source. Run:
    uv run python tools/measure_null_calibration_k.py

Budget overridable via env: OGS (offline_grad_steps), RE (rollout_episodes), ALGO
(cql|iql|...). The training loop MIRRORS the runner's new flat offline step-loop
(runner._offline_grouped_step_loop): fill the buffer, check can_sample(SEQ) ONCE, then
run EXACTLY offline_grad_steps critic updates via
mgr.update_strategy(buf.sample_sequences(BATCH, SEQ)) — no per-epoch break. This is
the runner's critic sub-call (self.critic_ablation.update_strategy(batch) on
batch = sample_sequences(batch_size, seq_len)); the base actor's self.agent.update is
intentionally OMITTED because value_mse_to_oracle is a critic-only quantity (the base
actor is a separate network that never enters the critic scoring), and noise_ref has
always been measured critic-only.

The shipped null_cal_reference.yaml (feat/noise-ref-v3, 2026-07-20) is measured at the
NEW production budget (offline_grad_steps=50_000, rollout_episodes=3000), CartPole-v1,
5 seeds, for cql and iql:
    OGS=50000 RE=3000 ALGO=cql uv run python tools/measure_null_calibration_k.py
    OGS=50000 RE=3000 ALGO=iql uv run python tools/measure_null_calibration_k.py
BATCH/SEQ track the runner (offpolicy_batch_size=128 / offpolicy_seq_len=8).
"""

from __future__ import annotations

import math
import os

import minari
import torch
from src.benchmarking.critic_ablation import (
    CRITIC_LIBRARY,
    CriticAblationConfig,
    CriticAblationManager,
    StrategyCritic,
)
from src.benchmarking.registry import register_default_algorithms
from src.config.device import detect_device
from src.config.seeding import set_seed
from src.envs.offline.generate import build_generator_agent, generate_offline_dataset
from src.envs.offline.minari_loader import fill_sequence_buffer_from_minari
from src.envs.registry import register_default_env_wrappers
from src.rl.off_policy.sequence_replay_buffer import SequenceReplayBuffer

SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2,3,4").split(",")]
# The offline budget, mirroring the runner: OFFLINE_GRAD_STEPS optimiser steps against
# a rollout_episodes=RE dataset. Defaults = the NEW production budget (50_000, 3000).
OFFLINE_GRAD_STEPS = int(os.environ.get("OGS", "50000"))
RE = int(os.environ.get("RE", "3000"))
BATCH, SEQ = 128, 8  # = runner.offpolicy_batch_size / offpolicy_seq_len
ALGO = os.environ.get("ALGO", "cql")  # the CORRECT base class (broken stays bare DQN)
OBS_DIM, ACT_DIM = 4, 2


def _measure_seed(seed: int, dev: torch.device):
    set_seed(seed, deterministic=True)
    agent, _ = build_generator_agent("CartPole-v1", "dqn", "random", seed=seed)
    did = f"pink/basic-seed{seed}-v0"
    try:
        minari.delete_dataset(did)
    except Exception:
        pass
    set_seed(seed, deterministic=True)
    ds = generate_offline_dataset(
        env_id="CartPole-v1",
        generator_algo="dqn",
        tier="random",
        behavior_policy="bias_confounded_action",
        behavior_strength=0.0,
        pi_basic_epsilon=0.5,
        confounder_c_r=1.0,
        rollout_episodes=RE,
        seed=seed,
        dataset_id=did,
        agent=agent,
    )
    ds.storage.update_metadata({"behavior_strength_sigma": 0.0})  # σ=0 basic anchor
    buf = SequenceReplayBuffer(capacity=1_000_000, device=dev)
    fill_sequence_buffer_from_minari(did, buf, dev, load_u=True)  # oracle_u needs U

    torch.manual_seed(seed)
    mgr = CriticAblationManager(
        obs_dim=OBS_DIM,
        device=dev,
        config=CriticAblationConfig(critics=["observational", "proximal", "oracle_u"]),
        base_algo=ALGO,
        action_dim=ACT_DIM,
        encoder="mlp",
    )
    # inject the BROKEN observational: bare DQN vs the CQL oracle (the historical bug).
    mgr.strategy_critics["observational_dqn"] = StrategyCritic(
        "observational",
        CRITIC_LIBRARY["observational"],
        "offline_dqn",
        OBS_DIM,
        ACT_DIM,
        dev,
        "mlp",
    )
    mgr.set_sequence_buffer(buf)
    # MIRROR runner._offline_grouped_step_loop: check can_sample ONCE, then run EXACTLY
    # offline_grad_steps critic updates with no per-epoch break (sampling is
    # with-replacement, so a buffer that can-sample once can serve every step).
    if not buf.can_sample(SEQ):
        raise ValueError(
            f"buffer has no episode >= seq_len={SEQ}; cannot run the "
            f"{OFFLINE_GRAD_STEPS}-step offline_grad_steps budget."
        )
    for _ in range(OFFLINE_GRAD_STEPS):
        mgr.update_strategy(buf.sample_sequences(BATCH, SEQ))
    rows = {
        r["critic"]: r
        for r in mgr.checkpoint_rows_strategy(1, ALGO, "CartPole-v1", 0.0)
    }
    try:
        minari.delete_dataset(did)
    except Exception:
        pass

    def mse(name: str) -> float:
        v = rows[name]["value_mse_to_oracle"]
        return float(v) if v not in ("", None) else float("nan")

    return mse("observational"), mse("observational_dqn"), mse("proximal")


def _stats(xs):
    n = len(xs)
    m = sum(xs) / n
    v = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(v)


def _report(tag, obs, prox):
    om, osd = _stats(obs)
    pm, psd = _stats(prox)
    gap = abs(om - pm)
    noise = math.sqrt((osd**2 + psd**2) / 2)  # pooled seed-sd (regime_report defn)
    print(
        f"CONFIG {tag}: obs_mean={om:.5f}(sd {osd:.5f}) prox_mean={pm:.5f}(sd {psd:.5f}) "
        f"| gap={gap:.5f} noise={noise:.5f} ratio={gap / noise:.4f}"
    )
    return gap, noise, gap / noise


def main() -> int:
    register_default_algorithms()
    register_default_env_wrappers()
    dev = detect_device()
    obs_cql, obs_dqn, prox = [], [], []
    for s in SEEDS:
        a, b, p = _measure_seed(s, dev)
        obs_cql.append(a)
        obs_dqn.append(b)
        prox.append(p)
        print(f"seed {s}: obs_cql={a:.5f} obs_dqn={b:.5f} prox={p:.5f}")
    print()
    gA, nA, rA = _report(f"A CORRECT (obs={ALGO})   ", obs_cql, prox)
    gB, nB, rB = _report("B BROKEN  (obs=bareDQN)", obs_dqn, prox)
    print(
        f"\nratio_A={rA:.3f} ratio_B={rB:.3f}  noise_A={nA:.5f} noise_B={nB:.5f} "
        f"noise_B/noise_A={nB / nA:.2f}"
    )
    # probe C: gap in noise-units of the CORRECT cell (robust to the bug's own noise)
    print(f"gap_A/noise_A={gA / nA:.3f}  gap_B/noise_A={gB / nA:.3f}")
    if rB < 2.0:
        print(
            "\nSTOP: ratio_B < 2 — the ratio gate does NOT separate correct from "
            "broken at this budget (the bug inflates NOISE as well as the mean, so "
            "gap/noise understates the separation). k not pinned; see "
            "docs/null_calibration_k.md."
        )
    else:
        print(
            f"\nk = sqrt(ratio_A*ratio_B) = {math.sqrt(rA * rB):.3f} "
            f"-> {round(math.sqrt(rA * rB), 1)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
