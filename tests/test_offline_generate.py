"""B2 offline GENERATE pipeline — train -> snapshot-by-return -> rollout -> write.

(iii) select_tier_episode: deterministic, given a controlled return trajectory
      (the medium checkpoint is the one at the range fraction) — not at the
      mercy of small-budget training noise.
(ii)  guards: generating with an offline algo (regime) and a generator/env
      action-space mismatch are each rejected BEFORE training.
(iv)  provenance: a collection policy composes into the rollout; bias_confounded
      wraps the rollout env -> a confounded offline dataset.
(i)   e2e: generate a tiny tiered dataset -> load via B1 -> train offline_dqn ->
      CSV (frozen schema). Proves train->snapshot->rollout->write->load->train.
"""

from __future__ import annotations

import csv

import pytest
from src.envs.offline.generate import (
    assert_action_space_match,
    assert_online_generator,
    build_rollout_env,
    dataset_name,
    generate_offline_dataset,
    select_tier_episode,
)

CPU = "cpu"


# --------------------------------------------------------------------------
# (iii) tier selection — pure / deterministic
# --------------------------------------------------------------------------
def test_select_tier_episode_by_return():
    returns = {0: 20.0, 5: 100.0, 10: 300.0, 15: 500.0}
    assert select_tier_episode(returns, "random") is None
    assert select_tier_episode(returns, "expert") == 15
    # medium target = 20 + (1/3)*(500-20) = 180 -> first checkpoint >= 180 is ep 10.
    assert select_tier_episode(returns, "medium") == 10


def test_select_tier_episode_negative_returns():
    # Pendulum-like: range-based target works where 1/3*R_expert would be > expert.
    neg = {0: -1200.0, 5: -800.0, 10: -150.0}
    # target = -1200 + (1/3)*(1050) = -850 -> first >= -850 is ep 5 (-800).
    assert select_tier_episode(neg, "medium") == 5
    assert select_tier_episode(neg, "expert") == 10


def test_dataset_name_convention():
    assert dataset_name("CartPole-v1", "expert") == "generated/cartpole/expert-v0"
    assert (
        dataset_name("Pendulum-v1", "medium", "bias_confounded")
        == "generated/pendulum/medium-bias_confounded-v0"
    )
    assert dataset_name("ALE/Pong-v5", "random") == "generated/ale-pong/random-v0"


# --------------------------------------------------------------------------
# (ii) guards — fire before training
# --------------------------------------------------------------------------
def test_regime_guard_rejects_offline_algo():
    from src.benchmarking.registry import register_default_algorithms

    register_default_algorithms()
    with pytest.raises(ValueError, match="offline algo"):
        assert_online_generator("offline_dqn")
    assert_online_generator("dqn")  # online -> no raise


def test_action_space_guard_both_directions():
    with pytest.raises(ValueError, match="discrete-only"):
        assert_action_space_match("dqn", "continuous")
    with pytest.raises(ValueError, match="continuous-only"):
        assert_action_space_match("sac", "discrete")
    assert_action_space_match("dqn", "discrete")  # match -> no raise
    assert_action_space_match("sac", "continuous")


def test_generate_rejects_offline_generator():
    # regime guard fires at the top, before any training/rollout/minari import.
    with pytest.raises(ValueError, match="offline algo"):
        generate_offline_dataset("CartPole-v1", "offline_dqn", "random")


def test_generate_rejects_action_space_mismatch():
    with pytest.raises(ValueError, match="discrete-only"):
        generate_offline_dataset("Pendulum-v1", "dqn", "random")


# --------------------------------------------------------------------------
# Provenance — confounder wraps the rollout env (no minari needed)
# --------------------------------------------------------------------------
def test_build_rollout_env_wraps_confounded_only():
    from src.envs.registry import register_default_env_wrappers
    from src.envs.wrappers.confounded import ConfoundedCollectionWrapper

    register_default_env_wrappers()
    conf = build_rollout_env("CartPole-v1", 1, CPU, 0, "bias_confounded", 1.0)
    assert isinstance(conf, ConfoundedCollectionWrapper)
    conf.close()
    clean = build_rollout_env("CartPole-v1", 1, CPU, 0, "agent")
    assert not isinstance(clean, ConfoundedCollectionWrapper)
    clean.close()


# --------------------------------------------------------------------------
# Generate / load / train (need the offline extra)
# --------------------------------------------------------------------------
pytest.importorskip("minari")
pytest.importorskip("h5py")


def test_generate_provenance_writes_dataset(tmp_path, monkeypatch):
    # random tier -> no training (fast); anti_reward + confounded both compose.
    monkeypatch.setenv("MINARI_DATASETS_PATH", str(tmp_path / "minari"))
    for i, bp in enumerate(["anti_reward", "bias_confounded"]):
        ds = generate_offline_dataset(
            "CartPole-v1",
            "dqn",
            "random",
            behavior_policy=bp,
            rollout_episodes=3,
            dataset_id=f"gen/cartpole-{bp}-v0",
            seed=i,
        )
        assert ds.total_episodes == 3 and ds.total_steps > 0


def test_generate_expert_then_b1_load_and_train(tmp_path, monkeypatch):
    monkeypatch.setenv("MINARI_DATASETS_PATH", str(tmp_path / "minari"))
    # Full pipeline: train (tiny) -> snapshot expert -> rollout -> write.
    ds = generate_offline_dataset(
        "CartPole-v1",
        "dqn",
        "expert",
        train_episodes=2,
        n_checkpoints=2,
        rollout_episodes=4,
        run_dir=str(tmp_path / "gen"),
        dataset_id="gen/cartpole-expert-v0",
    )

    # Consume the generated dataset through B1's load path.
    from src.benchmarking.registry import register_default_algorithms, registry
    from src.benchmarking.runner import BenchmarkRunner, TRAIN_COLUMNS
    from src.config.defaults import EnvConfig, RunConfig, TrainingConfig
    from src.config.device import detect_device
    from src.envs.registry import register_default_env_wrappers

    register_default_algorithms()
    register_default_env_wrappers()
    run_dir = tmp_path / "run"
    env_cfg = EnvConfig(
        env_id="CartPole-v1",
        n_train_envs=2,
        n_eval_envs=2,
        rollout_len=5,
        seed=0,
        offline_dataset=ds.id,
    )
    train_cfg = TrainingConfig(
        n_episodes=1,
        n_checkpoints=1,
        device=str(detect_device()),
        algorithm="offline_dqn",
        aggregation="mean",
    )
    BenchmarkRunner(
        env_cfg,
        train_cfg,
        RunConfig(run_dir=str(run_dir), timestamp="t"),
        registry.get("offline_dqn"),
    ).run()
    with (run_dir / "train_metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == TRAIN_COLUMNS and rows[0]["q_loss"] != ""
