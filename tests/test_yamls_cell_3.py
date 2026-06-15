"""feat/yamls-cell-3 (PR6) — Cell 3 offline tier-sweep YAMLs + per-env dataset map.

Cell 3 (Perfect Archive) is the fully-observed offline cell: each env trains on a
B2-generated dataset at one of three tiers (random/medium/expert). One YAML covers
several envs with DIFFERENT dataset ids, so the reproduce loader resolves
offline_dataset as a per-env map and threads the right id into each env's
EnvConfig (PR3's offline path). These tests cover parse + map-matches-convention,
per-env threading, strict-mode rejection, and an end-to-end smoke on an in-test
dataset (the real Cell 3 datasets are NOT required to exist).
"""

from __future__ import annotations

import os
import subprocess
import sys
import warnings

import minari
import pytest
import yaml
from main import _resolve_offline_dataset_map
from src.envs.offline.generate import generate_offline_dataset
from tests.conftest import REPO_ROOT

warnings.filterwarnings("ignore")

CELL_3 = REPO_ROOT / "reproducibility" / "rl_regimes" / "cell_3"

DISCRETE_ENVS = ["CartPole-v1", "LunarLander-v3", "Acrobot-v1"]
CONTINUOUS_ENVS = ["Pendulum-v1", "HalfCheetah-v5", "Hopper-v5", "Walker2d-v5"]


def _slug(env: str) -> str:
    # Mirrors src/envs/offline/generate.py::dataset_name slug computation.
    return env.split("-v")[0].lower().replace("/", "-")


def _load(name: str) -> dict:
    return yaml.safe_load((CELL_3 / name).read_text())


def test_cell_3_yamls_parse():
    for tier in ("random", "medium", "expert"):
        for arm, envs in (("discrete", DISCRETE_ENVS), ("continuous", CONTINUOUS_ENVS)):
            cfg = _load(f"offline_{tier}_{arm}.yaml")
            assert cfg["seed"] == 0
            assert cfg["n_train_envs"] == 16
            # offline_dataset map matches generated/<slug>/<tier>-v0 for every env.
            assert cfg["offline_dataset"] == {
                e: f"generated/{_slug(e)}/{tier}-v0" for e in envs
            }


def test_per_env_offline_dataset_threading():
    cfg = _load("offline_random_discrete.yaml")
    resolved = _resolve_offline_dataset_map(
        cfg["offline_dataset"], cfg["envs"], "offline_random_discrete.yaml"
    )
    # Different envs get DIFFERENT dataset ids — the per-env (not uniform) contract.
    assert resolved["CartPole-v1"] == "generated/cartpole/random-v0"
    assert resolved["Acrobot-v1"] == "generated/acrobot/random-v0"
    assert resolved["CartPole-v1"] != resolved["Acrobot-v1"]


def test_strict_mode_raises_on_missing_env_dataset():
    raw = {"CartPole-v1": "foo/bar-v0"}  # FakeEnv-v0 deliberately absent
    with pytest.raises(ValueError, match="FakeEnv-v0") as exc:
        _resolve_offline_dataset_map(
            raw, ["CartPole-v1", "FakeEnv-v0"], "strict_mode_test.yaml"
        )
    msg = str(exc.value)
    assert "offline_dataset map" in msg and "strict_mode_test.yaml" in msg


def test_cell_3_discrete_smoke(tmp_path):
    # Generate a tiny in-test dataset (random tier, agent behavior = unconfounded,
    # like a Cell 3 dataset) and point a mirror YAML at it. The real Cell 3
    # dataset ids are NOT required to exist.
    did = "test/cell3-smoke-v0"
    try:
        minari.delete_dataset(did)
    except Exception:
        pass
    generate_offline_dataset(
        env_id="CartPole-v1",
        generator_algo="dqn",
        tier="random",
        behavior_policy="agent",
        rollout_episodes=30,
        seed=0,
        dataset_id=did,
    )
    try:
        repro_dir = tmp_path / "reproducibility" / "rl_regimes" / "cell_3"
        repro_dir.mkdir(parents=True)
        (repro_dir / "offline_random_discrete.yaml").write_text(
            yaml.safe_dump(
                {
                    "envs": ["CartPole-v1"],
                    "algos": ["cql"],
                    "offline_dataset": {"CartPole-v1": did},
                    "n_episodes": 1,
                    "rollout_len": 2,
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
                "rl_regimes/cell_3/offline_random_discrete",
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
        assert len(train_csvs) == 1 and train_csvs[0].stat().st_size > 0
        assert len(eval_csvs) == 1 and eval_csvs[0].stat().st_size > 0
    finally:
        minari.delete_dataset(did)
