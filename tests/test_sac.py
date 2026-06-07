"""SAC tests (Phase 6B): registration, determinism, Pendulum smoke."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
from src.rl.base import Algorithm
from tests.conftest import REPO_ROOT


def test_sac_registered_off_policy_continuous():
    """Additive registry entry (the frozen six-algo test stays untouched)."""
    import gymnasium as gym
    from src.benchmarking.registry import register_default_algorithms, registry

    register_default_algorithms()
    spec = registry.get("sac")
    assert spec.kind == "off_policy"
    env = gym.make("Pendulum-v1")
    policy, agent = spec.builder(
        obs_dim=3,
        action_dim=1,
        action_type="continuous",
        device=torch.device("cpu"),
        action_space=env.action_space,
    )
    env.close()
    assert isinstance(agent, Algorithm)
    assert agent.action_type == "continuous"
    assert agent.action_scale == 2.0  # Pendulum bounds
    # runner-compat kwargs
    det = agent.act(torch.zeros(2, 3), noise=False)
    det2 = agent.act(torch.zeros(2, 3), deterministic=True)
    assert torch.equal(det.action, det2.action)


def _run_sac_job(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tests" / "_phase0_seed_probe.py"),
            "--envs",
            "Pendulum-v1",
            "--algos",
            "sac",
            "--n-episodes",
            "3",
            "--rollout-len",
            "128",
            "--n-train-envs",
            "4",
            "--n-eval-envs",
            "4",
            "--deterministic",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return sorted((tmp_path / "runs").iterdir())[0]


@pytest.mark.golden
def test_sac_deterministic_bitwise(tmp_path):
    """Two pre-seeded short runs must be bitwise identical (gate condition)."""
    run1 = _run_sac_job(tmp_path / "a")
    run2 = _run_sac_job(tmp_path / "b")
    for name in ("train_metrics.csv", "eval_metrics.csv"):
        a = (run1 / name).read_text()
        b = (run2 / name).read_text()
        assert a == b, f"SAC {name} not reproducible"


@pytest.mark.golden
def test_sac_learns_pendulum(tmp_path):
    """Window eval return must improve substantially within a short budget."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tests" / "_phase0_seed_probe.py"),
            "--envs",
            "Pendulum-v1",
            "--algos",
            "sac",
            "--n-episodes",
            "30",
            "--n-checkpoints",
            "6",
            "--rollout-len",
            "256",
            "--n-train-envs",
            "4",
            "--n-eval-envs",
            "4",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    run = sorted((tmp_path / "runs").iterdir())[0]
    ev = pd.read_csv(run / "eval_metrics.csv")
    first, last = ev.eval_return_mean.iloc[0], ev.eval_return_mean.iloc[-1]
    assert last > first + 200, f"SAC did not learn: {first:.0f} -> {last:.0f}"
