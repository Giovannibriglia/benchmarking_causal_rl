"""Golden-VALUE regression tests (§3.1 of the refactor contract).

Each test launches the pinned deterministic job in a subprocess (in a temp cwd
so `runs/` artifacts don't pollute the repo) and requires the produced CSVs to
match the golden files in tests/golden/ EXACTLY, byte for byte (modulo the
header-only timestamp fields, which do not appear in the CSVs).

Notes
-----
* The job is launched through ``tests/_phase0_seed_probe.py``, which calls
  ``set_seed(42, deterministic=True)`` BEFORE ``main()``. On unmodified master
  this is required because policy weights are initialized in
  ``BenchmarkRunner.__init__`` while ``set_seed`` only runs later in
  ``BenchmarkRunner.run()`` — without the pre-seed, runs are not reproducible
  at all (verified in Phase 0). Once the approved Phase-1 seeding fix lands,
  the pre-seed becomes redundant but harmless, so these tests stay valid
  before and after.
* Golden values are machine-pinned: generated on this workstation
  (RTX 4070 Laptop, torch 2.10.0+cu130, CUDA device). Exact equality is only
  expected on the same device/stack.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from tests.conftest import GOLDEN_DIR, REPO_ROOT

GOLDEN_ARGS = [
    "--envs",
    "CartPole-v1",
    "--algos",
    "ppo",
    "--n-episodes",
    "5",
    "--deterministic",
]


def _run_golden_job(tmp_path: Path, extra_args: list[str]) -> Path:
    """Run the pinned job in tmp_path; return the produced run directory."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tests" / "_phase0_seed_probe.py")]
        + GOLDEN_ARGS
        + extra_args,
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


def _assert_artifact_layout(run_dir: Path, expect_critic_csv: bool) -> None:
    """Characterize the §2 run-artifact layout invariant."""
    assert (run_dir / "config.yaml").is_file()
    assert (run_dir / "metadata.json").is_file()
    assert (run_dir / "train_metrics.csv").is_file()
    assert (run_dir / "eval_metrics.csv").is_file()
    assert (run_dir / "checkpoints").is_dir()
    assert (run_dir / "videos").is_dir()
    assert (run_dir / "critic_ablation_metrics.csv").is_file() == expect_critic_csv
    # checkpoints written per env/algo/seed with per-episode files
    ckpt_dirs = list((run_dir / "checkpoints").iterdir())
    assert ckpt_dirs, "no checkpoint subdirectory written"
    assert any(d.name == "CartPole-v1_ppo_seed42" for d in ckpt_dirs)
    assert list(ckpt_dirs[0].glob("ckpt_ep*.pt"))


def _assert_csv_equals_golden(run_dir: Path, golden_subdir: str, name: str) -> None:
    produced = (run_dir / name).read_text(encoding="utf-8")
    golden = (GOLDEN_DIR / golden_subdir / name).read_text(encoding="utf-8")
    assert produced == golden, (
        f"{name} deviates from tests/golden/{golden_subdir}/{name}. "
        "NEVER absorb a numeric deviation silently: stop and present the cause "
        "for explicit sign-off (§3.1)."
    )


@pytest.mark.golden
def test_benchmark_golden_values(tmp_path):
    run_dir = _run_golden_job(tmp_path, extra_args=[])
    _assert_artifact_layout(run_dir, expect_critic_csv=False)
    _assert_csv_equals_golden(run_dir, "benchmark", "train_metrics.csv")
    _assert_csv_equals_golden(run_dir, "benchmark", "eval_metrics.csv")


@pytest.mark.golden
def test_critic_ablation_golden_values(tmp_path):
    run_dir = _run_golden_job(tmp_path, extra_args=["--ablation"])
    _assert_artifact_layout(run_dir, expect_critic_csv=True)
    _assert_csv_equals_golden(run_dir, "critic_ablation", "train_metrics.csv")
    _assert_csv_equals_golden(run_dir, "critic_ablation", "eval_metrics.csv")
    _assert_csv_equals_golden(run_dir, "critic_ablation", "critic_ablation_metrics.csv")
