"""Golden-VALUE regression tests for the off-policy paths (DQN, DDPG).

Goldens were generated on the PRE-refactor tree (commit d844f15, parent of the
Phase-1 changes) with the same pre-seeded probe as the PPO goldens, and each
verified bitwise reproducible across two runs. They pin the exact RNG
consumption order of the off-policy collection/update loop through the
Phase-1 relocation into ``OnlineSource.collect_off_policy``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from tests.conftest import GOLDEN_DIR, REPO_ROOT

JOBS = {
    "offpolicy_dqn": [
        "--envs",
        "CartPole-v1",
        "--algos",
        "dqn",
    ],
    "offpolicy_ddpg": [
        "--envs",
        "Pendulum-v1",
        "--algos",
        "ddpg",
    ],
}

COMMON = [
    "--n-episodes",
    "3",
    "--rollout-len",
    "256",
    "--n-train-envs",
    "8",
    "--n-eval-envs",
    "8",
    "--deterministic",
]


def _run_job(tmp_path: Path, args: list[str]) -> Path:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tests" / "_phase0_seed_probe.py")]
        + args
        + COMMON,
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0, f"golden job failed:\n{proc.stdout}\n{proc.stderr}"
    run_dirs = sorted((tmp_path / "runs").iterdir())
    assert len(run_dirs) == 1, run_dirs
    return run_dirs[0]


@pytest.mark.golden
@pytest.mark.parametrize("golden_subdir", sorted(JOBS), ids=["ddpg", "dqn"])
def test_offpolicy_golden_values(tmp_path, golden_subdir):
    run_dir = _run_job(tmp_path, JOBS[golden_subdir])
    for name in ("train_metrics.csv", "eval_metrics.csv"):
        produced = (run_dir / name).read_text(encoding="utf-8")
        golden = (GOLDEN_DIR / golden_subdir / name).read_text(encoding="utf-8")
        assert produced == golden, (
            f"{name} deviates from tests/golden/{golden_subdir}/{name}. "
            "NEVER absorb a numeric deviation silently (§3.1)."
        )
