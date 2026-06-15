"""feat/eval-per-context-csv (PR2) — Cell 2 per-context eval breakdown.

eval_per_context.csv is written ONLY when --mask-indices is set (strict opt-in,
so a no-mask run is byte-identical to a pre-PR2 run). These tests cover the
gate-closed contract, the frozen schema, and well-formed binning. CSV content of
the bins is checked structurally; the offline-value-trace writer and dataset
metadata are out of scope (PR3).
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

from src.benchmarking.runner import EVAL_PER_CONTEXT_COLUMNS
from tests.conftest import REPO_ROOT

# Minimal online CartPole/PPO job (fast: a couple of checkpoints, short rollout,
# full eval-env axis so the bins have something to spread over).
BASE_ARGS = [
    "--envs",
    "CartPole-v1",
    "--algos",
    "ppo",
    "--n-episodes",
    "2",
    "--n-checkpoints",
    "2",
    "--rollout-len",
    "64",
    "--n-train-envs",
    "4",
    "--n-eval-envs",
    "16",
]


def _run(tmp_path: Path, extra_args: list[str]) -> Path:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "main.py")] + BASE_ARGS + extra_args,
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


def _read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def test_eval_per_context_not_written_without_mask(tmp_path):
    """Gate closed: no --mask-indices -> the file must not exist at all."""
    run_dir = _run(tmp_path, extra_args=[])
    assert not (run_dir / "eval_per_context.csv").exists()


def test_eval_per_context_schema_frozen(tmp_path):
    """Gate open: file exists, header matches the pinned schema, has data."""
    run_dir = _run(tmp_path, extra_args=["--mask-indices", "1,3"])
    path = run_dir / "eval_per_context.csv"
    assert path.exists()
    with path.open(newline="") as f:
        header = next(csv.reader(f))
    assert header == EVAL_PER_CONTEXT_COLUMNS
    assert len(_read_rows(path)) >= 1


def test_eval_per_context_bins_are_well_formed(tmp_path):
    run_dir = _run(tmp_path, extra_args=["--mask-indices", "1,3"])
    rows = _read_rows(run_dir / "eval_per_context.csv")
    assert rows

    # (a) every bin holds at least one episode; (b) bin interval is non-empty.
    for row in rows:
        assert int(row["n_episodes_in_bin"]) >= 1
        assert float(row["context_value_low"]) < float(row["context_value_high"])

    # (c) bin indices are unique within a single checkpoint episode.
    by_episode: dict[str, list[int]] = {}
    for row in rows:
        by_episode.setdefault(row["episode"], []).append(int(row["context_bin"]))
    for ep, bins in by_episode.items():
        assert len(bins) == len(set(bins)), f"duplicate bin in episode {ep}"

    # (d) per-context episodes are a subset of the eval_metrics checkpoint episodes.
    eval_eps = {r["episode"] for r in _read_rows(run_dir / "eval_metrics.csv")}
    assert set(by_episode) <= eval_eps
