"""feat/yamls-cell-7 (PR8) — Cell 7 confounded-offline σ sweep + σ-encoded ids.

Cell 7 (Shadowed Vitals) is the confounded offline σ-sweep: each YAML trains on a
B2 dataset rolled out under bias_confounded at one σ, and σ is encoded in the
dataset id so distinct σ → distinct datasets. These tests cover the σ-suffix
helper (incl. the rounding edge), the dataset_name round-trip (confounded vs
agent), that all ten YAMLs' ids round-trip through the helper, and an end-to-end
smoke where the PR3 confounded gate fires (offline_value_trace.csv written).
"""

from __future__ import annotations

import os
import subprocess
import sys
import warnings

import minari
import yaml
from src.envs.offline.generate import (
    _sigma_suffix,
    dataset_name,
    generate_offline_dataset,
)
from tests.conftest import REPO_ROOT

warnings.filterwarnings("ignore")

CELL_7 = REPO_ROOT / "reproducibility" / "rl_regimes" / "cell_7"
DISCRETE_ENVS = ["CartPole-v1", "LunarLander-v3", "Acrobot-v1"]
CONTINUOUS_ENVS = ["Pendulum-v1", "HalfCheetah-v5", "Hopper-v5", "Walker2d-v5"]


def _load(name: str) -> dict:
    return yaml.safe_load((CELL_7 / name).read_text())


def test_sigma_encoding():
    assert _sigma_suffix(0.0) == "-sigma000"
    assert _sigma_suffix(0.25) == "-sigma025"
    assert _sigma_suffix(0.5) == "-sigma050"
    assert _sigma_suffix(0.75) == "-sigma075"
    assert _sigma_suffix(1.0) == "-sigma100"
    # rounding (not truncation): 0.3 * 100 = 29.999... -> 030, not 029.
    assert _sigma_suffix(0.3) == "-sigma030"


def test_dataset_name_confounded_round_trip():
    assert (
        dataset_name("CartPole-v1", "random", "bias_confounded", behavior_strength=0.5)
        == "generated/cartpole/random-bias_confounded-sigma050-v0"
    )
    # agent path unchanged (pre-PR8 convention, no -sigma suffix).
    assert dataset_name("CartPole-v1", "expert") == "generated/cartpole/expert-v0"


def test_cell_7_yamls_parse_and_match_convention():
    for nnn, sigma in (
        ("000", 0.0),
        ("025", 0.25),
        ("050", 0.5),
        ("075", 0.75),
        ("100", 1.0),
    ):
        for arm, envs in (("discrete", DISCRETE_ENVS), ("continuous", CONTINUOUS_ENVS)):
            cfg = _load(f"confounded_sigma_{nnn}_{arm}.yaml")
            assert cfg["behavior_policy"] == "bias_confounded"
            assert cfg["behavior_strength"] == sigma
            assert cfg["seed"] == 0 and cfg["n_train_envs"] == 16
            # every id round-trips through dataset_name at this σ.
            assert cfg["offline_dataset"] == {
                e: dataset_name(e, "random", "bias_confounded", sigma) for e in envs
            }


def test_cell_7_discrete_smoke(tmp_path):
    # Confounded σ=0.5 random-tier dataset, auto-named so the real σ-encoded
    # convention id is exercised. PR3 verified this gates True (marginal=0.450).
    did = dataset_name("CartPole-v1", "random", "bias_confounded", 0.5)
    assert did == "generated/cartpole/random-bias_confounded-sigma050-v0"
    existed = did in minari.list_local_datasets()
    if not existed:
        generate_offline_dataset(
            env_id="CartPole-v1",
            generator_algo="dqn",
            tier="random",
            behavior_policy="bias_confounded",
            behavior_strength=0.5,
            rollout_episodes=30,
            seed=0,
        )
    try:
        repro_dir = tmp_path / "reproducibility" / "rl_regimes" / "cell_7"
        repro_dir.mkdir(parents=True)
        (repro_dir / "confounded_sigma_050_discrete.yaml").write_text(
            yaml.safe_dump(
                {
                    "envs": ["CartPole-v1"],
                    "algos": ["cql"],
                    "offline_dataset": {"CartPole-v1": did},
                    "behavior_policy": "bias_confounded",
                    "behavior_strength": 0.5,
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
                "rl_regimes/cell_7/confounded_sigma_050_discrete",
            ],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
        )
        # exit 0 => the dataset was NOT rejected on the gate-test-passed check.
        assert proc.returncode == 0, f"smoke run failed:\n{proc.stdout}\n{proc.stderr}"

        run_dirs = list((tmp_path / "runs").rglob("config.yaml"))
        assert len(run_dirs) == 1
        run_dir = run_dirs[0].parent
        assert (run_dir / "train_metrics.csv").stat().st_size > 0
        assert (run_dir / "eval_metrics.csv").stat().st_size > 0
        # confounded + offline => the PR3 gate writer produces offline_value_trace.csv.
        assert (run_dir / "offline_value_trace.csv").stat().st_size > 0
    finally:
        if not existed:
            minari.delete_dataset(did)
