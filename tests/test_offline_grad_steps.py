"""feat/offline-budget-key: the offline learner's step count is an EXPLICIT
offline_grad_steps budget, not the on-policy n_episodes x rollout_len product.

T1  offline optimiser-step count == offline_grad_steps (flat AND grouped paths),
    NOT n_episodes x rollout_len. Asserts the ACTUAL number of steps taken.
T2  the ONLINE path is unchanged when offline_grad_steps is set (regression).
T3  a missing offline_grad_steps key WARNS and falls back (never silently the old
    product).
T4  offline checkpoints are evenly spaced across offline_grad_steps, n_checkpoints
    of them, last == offline_grad_steps.
T5  both sweep_smoke.yaml resolve to a small rollout_episodes (and a small
    offline_grad_steps); production resolves to (3000, 50_000).
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

pytest.importorskip("minari")
pytest.importorskip("h5py")

from src.benchmarking.critic_ablation import CriticAblationConfig  # noqa: E402
from src.benchmarking.registry import (  # noqa: E402
    register_default_algorithms,
    registry,
)
from src.benchmarking.runner import BenchmarkRunner  # noqa: E402
from src.config.defaults import EnvConfig, RunConfig, TrainingConfig  # noqa: E402
from src.config.device import detect_device  # noqa: E402
from src.envs.registry import register_default_env_wrappers  # noqa: E402

_REPO = Path(__file__).resolve().parent.parent
_DEV = str(detect_device())
DATASET_ID = "cartpole/gradsteps-test-v0"


@pytest.fixture
def cartpole_dataset(tmp_path, monkeypatch):
    monkeypatch.setenv("MINARI_DATASETS_PATH", str(tmp_path / "minari"))
    from tools.make_cartpole_offline import make_cartpole_dataset

    make_cartpole_dataset(dataset_id=DATASET_ID, n_episodes=12, seed=0)
    return DATASET_ID


def _env_cfg(offline_dataset=None, **kw):
    return EnvConfig(
        env_id="CartPole-v1",
        n_train_envs=2,
        n_eval_envs=2,
        rollout_len=5,
        seed=0,
        offline_dataset=offline_dataset,
        **kw,
    )


def _read_episodes(run_dir: Path, csv_name: str) -> list[int]:
    path = run_dir / csv_name
    with path.open() as f:
        return [int(r["episode"]) for r in csv.DictReader(f)]


# --------------------------------------------------------------------------- #
# T4 (pure) — the checkpoint schedule                                          #
# --------------------------------------------------------------------------- #
def test_t4_offline_checkpoint_steps_uniform():
    tc = TrainingConfig(n_checkpoints=25, offline_grad_steps=50_000)
    steps = tc.offline_checkpoint_steps()
    assert steps == list(range(2000, 50_001, 2000))  # every 2000, last == 50_000
    assert len(steps) == 25
    # uniform spacing
    diffs = {b - a for a, b in zip(steps, steps[1:])}
    assert diffs == {2000}
    # a non-round total still lands its last checkpoint exactly on the total.
    tc2 = TrainingConfig(n_checkpoints=4, offline_grad_steps=137)
    s2 = tc2.offline_checkpoint_steps()
    assert len(s2) == 4 and s2[-1] == 137 and s2 == sorted(s2)


# --------------------------------------------------------------------------- #
# T1 — exactly offline_grad_steps optimiser steps (flat + grouped)             #
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_t1_flat_offline_step_count(cartpole_dataset, tmp_path):
    register_default_algorithms()
    register_default_env_wrappers()
    run_dir = tmp_path / "run"
    # 137 is prime: it is NOT n_episodes x rollout_len for the cfg below (2 x 5 = 10).
    train_cfg = TrainingConfig(
        n_episodes=2,
        n_checkpoints=4,
        device=_DEV,
        algorithm="offline_dqn",
        aggregation="iqm",
        offline_grad_steps=137,
    )
    runner = BenchmarkRunner(
        _env_cfg(cartpole_dataset),
        train_cfg,
        RunConfig(run_dir=str(run_dir), timestamp="t"),
        registry.get("offline_dqn"),
        progress_label="offline",
    )
    runner.run()
    # the ACTUAL number of optimiser steps, not 2 x 5 = 10.
    assert runner._offline_steps_taken == 137
    # T4 end-to-end: checkpoints land on the uniform step schedule, n_checkpoints of them.
    assert (
        _read_episodes(run_dir, "eval_metrics.csv")
        == train_cfg.offline_checkpoint_steps()
    )


@pytest.mark.slow
def test_t1_grouped_offline_step_count(cartpole_dataset, tmp_path):
    register_default_algorithms()
    register_default_env_wrappers()
    run_dir = tmp_path / "run"
    train_cfg = TrainingConfig(
        n_episodes=2,
        n_checkpoints=3,
        device=_DEV,
        algorithm="offline_dqn",
        aggregation="iqm",
        offline_grad_steps=101,  # prime; grouped path via the strategy ablation
    )
    runner = BenchmarkRunner(
        _env_cfg(cartpole_dataset),
        train_cfg,
        RunConfig(run_dir=str(run_dir), timestamp="t"),
        registry.get("offline_dqn"),
        critic_ablation_cfg=CriticAblationConfig(critics=["observational"]),
        progress_label="offline-grouped",
    )
    runner.run()
    assert runner._offline_steps_taken == 101
    assert (
        _read_episodes(run_dir, "eval_metrics.csv")
        == train_cfg.offline_checkpoint_steps()
    )


# --------------------------------------------------------------------------- #
# T3 — missing key warns + falls back (never silent)                           #
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_t3_missing_key_warns_and_falls_back(cartpole_dataset, tmp_path):
    register_default_algorithms()
    register_default_env_wrappers()
    run_dir = tmp_path / "run"
    train_cfg = TrainingConfig(
        n_episodes=2,
        n_checkpoints=2,
        device=_DEV,
        algorithm="offline_dqn",
        aggregation="iqm",
        # offline_grad_steps intentionally UNSET
    )
    runner = BenchmarkRunner(
        _env_cfg(cartpole_dataset),
        train_cfg,
        RunConfig(run_dir=str(run_dir), timestamp="t"),
        registry.get("offline_dqn"),
        progress_label="offline",
    )
    with pytest.warns(UserWarning, match="offline_grad_steps"):
        runner.run()
    # fell back to the legacy loop (still produced output), never silently continued.
    assert (run_dir / "eval_metrics.csv").exists()
    assert not hasattr(runner, "_offline_steps_taken")  # the new-path attr is not set


# --------------------------------------------------------------------------- #
# T2 — the ONLINE path is unchanged by offline_grad_steps (regression)         #
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_t2_online_path_unchanged(tmp_path):
    register_default_algorithms()
    register_default_env_wrappers()

    def _run(run_dir, ogs):
        tc = TrainingConfig(
            n_episodes=2,
            n_checkpoints=2,
            device=_DEV,
            algorithm="dqn",  # ONLINE off-policy
            aggregation="iqm",
            deterministic=True,
            offline_grad_steps=ogs,
        )
        BenchmarkRunner(
            _env_cfg(None),
            tc,
            RunConfig(run_dir=str(run_dir), timestamp="t"),
            registry.get("dqn"),
            progress_label="online",
        ).run()
        return (run_dir / "eval_metrics.csv").read_text()

    base = _run(tmp_path / "none", None)
    withkey = _run(tmp_path / "withkey", 999_999)
    # setting offline_grad_steps must not perturb the online path at all.
    assert base == withkey


# --------------------------------------------------------------------------- #
# T5 — smokes stay small; production resolves to the calibrated values          #
# --------------------------------------------------------------------------- #
def test_t5_budget_resolution():
    from src.benchmarking.regime_sweep import load_sweep_spec

    for regime in ("offline_mdp", "offline_pomdp"):
        smoke = load_sweep_spec(
            _REPO / "reproducibility" / "rl_regimes" / regime / "sweep_smoke.yaml"
        )
        assert smoke.budget("rollout_episodes", 0) == 40, regime
        assert smoke.budget("offline_grad_steps", 0) == 40, regime
    prod = load_sweep_spec(
        _REPO / "reproducibility" / "rl_regimes" / "offline_mdp" / "sweep.yaml"
    )
    assert prod.budget("rollout_episodes", 0) == 3000
    assert prod.budget("offline_grad_steps", 0) == 50_000
