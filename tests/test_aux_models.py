"""Auxiliary reward/next-state models — RNG-isolation regression tests.

The headline test is the RNG-isolation PROOF as a permanent regression: an
aux-enabled run of the pinned golden job must produce train/eval CSVs that are
byte-for-byte identical to the aux-DISABLED golden. This is what catches a
future change that lets the aux path leak into the global RNG stream.
"""

from __future__ import annotations

import csv
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from src.benchmarking.aux_models import (
    AUX_METRICS_COLUMNS,
    AuxModelConfig,
    AuxModelManager,
)
from src.config.device import detect_device
from tests.conftest import GOLDEN_DIR, REPO_ROOT

# Same pinned job as tests/test_regression.py::test_benchmark_golden_values.
GOLDEN_ARGS = [
    "--envs",
    "CartPole-v1",
    "--algos",
    "ppo",
    "--n-episodes",
    "5",
    "--deterministic",
]


def test_aux_construction_is_rng_neutral():
    """Building the manager must not advance the global CPU/CUDA RNG."""
    device = detect_device()
    cpu_before = torch.get_rng_state()
    cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    AuxModelManager(
        obs_dim=4,
        obs_shape=(4,),
        action_dim=2,
        action_type="discrete",
        device=device,
        config=AuxModelConfig(),
    )

    assert torch.equal(torch.get_rng_state(), cpu_before)
    if cuda_before is not None:
        for a, b in zip(torch.cuda.get_rng_state_all(), cuda_before):
            assert torch.equal(a, b)


def _run(tmp_path: Path, extra_args: list[str]) -> Path:
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
    assert proc.returncode == 0, f"job failed:\n{proc.stdout}\n{proc.stderr}"
    run_dirs = sorted((tmp_path / "runs").iterdir())
    assert len(run_dirs) == 1, run_dirs
    return run_dirs[0]


@pytest.mark.golden
def test_aux_enabled_matches_benchmark_golden_bitwise(tmp_path):
    """RNG-isolation proof: aux-ON train/eval == aux-OFF golden, byte-for-byte."""
    run_dir = _run(tmp_path, extra_args=["--aux-models"])
    for name in ("train_metrics.csv", "eval_metrics.csv"):
        produced = (run_dir / name).read_text(encoding="utf-8")
        golden = (GOLDEN_DIR / "benchmark" / name).read_text(encoding="utf-8")
        assert produced == golden, (
            f"{name} diverged from the aux-DISABLED golden with --aux-models on. "
            "The aux path leaked into the global RNG stream — fix the isolation, "
            "do not regenerate the golden."
        )

    # aux metrics land in their own CSV with the expected columns + finite losses.
    aux_csv = run_dir / "aux_metrics.csv"
    assert aux_csv.is_file()
    with aux_csv.open() as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == AUX_METRICS_COLUMNS
    assert {r["model"] for r in rows} == {"reward", "transition"}
    for r in rows:
        assert math.isfinite(float(r["train_loss"]))
        assert math.isfinite(float(r["mse"]))


@pytest.mark.golden
def test_default_run_writes_no_aux_csv(tmp_path):
    """Default (aux off) run dir is unchanged — no aux_metrics.csv."""
    run_dir = _run(tmp_path, extra_args=[])
    assert not (run_dir / "aux_metrics.csv").exists()
