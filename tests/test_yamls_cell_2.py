"""feat/yamls-cell-2 (PR5) — Cell 2 masked-online YAMLs + per-env mask_indices.

Cell 2 (Invisible Gene) is the online, hidden-Z, unconfounded cell: each env
masks its velocity components per §5. One YAML covers several envs with DIFFERENT
masks, so the reproduce loader resolves mask_indices as a per-env map and threads
the right indices into each env's EnvConfig (PR1's mask path). These tests cover
parse + map-matches-§5, per-env threading, strict-mode rejection, and an
end-to-end smoke that confirms eval_per_context.csv (PR2) activates.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
import yaml
from main import _resolve_mask_indices_map
from tests.conftest import REPO_ROOT

CELL_2 = REPO_ROOT / "reproducibility" / "rl_regimes" / "cell_2"

# §5 per-env mask spec (the source of truth).
DISCRETE_MASKS = {
    "CartPole-v1": [1, 3],
    "LunarLander-v3": [2, 3, 5],
    "Acrobot-v1": [4, 5],
}
CONTINUOUS_MASKS = {
    "Pendulum-v1": [2],
    "HalfCheetah-v5": [8, 9],
    "Hopper-v5": [5, 6],
    "Walker2d-v5": [9, 10],
}


def _load(name: str) -> dict:
    return yaml.safe_load((CELL_2 / name).read_text())


def test_cell_2_yamls_parse():
    discrete = _load("online_masked_discrete.yaml")
    continuous = _load("online_masked_continuous.yaml")

    assert discrete["mask_indices"] == DISCRETE_MASKS
    assert continuous["mask_indices"] == CONTINUOUS_MASKS
    for cfg in (discrete, continuous):
        assert cfg["seed"] == 0
        assert cfg["n_train_envs"] == 16


def test_per_env_mask_threading():
    cfg = _load("online_masked_discrete.yaml")
    resolved = _resolve_mask_indices_map(
        cfg["mask_indices"], cfg["envs"], "online_masked_discrete.yaml"
    )
    # Different envs get DIFFERENT indices — the per-env (not uniform) contract.
    assert resolved["CartPole-v1"] == (1, 3)
    assert resolved["Acrobot-v1"] == (4, 5)
    assert resolved["CartPole-v1"] != resolved["Acrobot-v1"]


def test_strict_mode_raises_on_missing_env():
    raw = {"CartPole-v1": [1, 3]}  # FakeEnv-v0 deliberately absent
    with pytest.raises(ValueError, match="FakeEnv-v0") as exc:
        _resolve_mask_indices_map(
            raw, ["CartPole-v1", "FakeEnv-v0"], "strict_mode_test.yaml"
        )
    msg = str(exc.value)
    # Error must point the user at a fix.
    assert "mask_indices map" in msg and "strict_mode_test.yaml" in msg


def test_cell_2_discrete_smoke(tmp_path):
    # Tiny mirror YAML (same per-env map) at the nested path; --reproduce
    # overrides CLI, so settings are shrunk in the YAML itself.
    repro_dir = tmp_path / "reproducibility" / "rl_regimes" / "cell_2"
    repro_dir.mkdir(parents=True)
    (repro_dir / "online_masked_discrete.yaml").write_text(
        yaml.safe_dump(
            {
                "envs": ["CartPole-v1"],
                "algos": ["ppo"],
                "mask_indices": {"CartPole-v1": [1, 3]},
                "n_episodes": 1,
                "rollout_len": 8,
                "n_train_envs": 2,
                "n_eval_envs": 2,
                "n_checkpoints": 2,
                "aggregation": "iqm",
                "deterministic": True,
                "seed": 0,
            }
        )
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "main.py"),
            "--reproduce",
            "rl_regimes/cell_2/online_masked_discrete",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0, f"smoke run failed:\n{proc.stdout}\n{proc.stderr}"

    train_csvs = list((tmp_path / "runs").rglob("train_metrics.csv"))
    eval_csvs = list((tmp_path / "runs").rglob("eval_metrics.csv"))
    # eval_per_context.csv must be written because mask_indices is set (PR2 gate).
    per_ctx = list((tmp_path / "runs").rglob("eval_per_context.csv"))
    assert len(train_csvs) == 1 and train_csvs[0].stat().st_size > 0
    assert len(eval_csvs) == 1 and eval_csvs[0].stat().st_size > 0
    assert len(per_ctx) == 1 and per_ctx[0].stat().st_size > 0
