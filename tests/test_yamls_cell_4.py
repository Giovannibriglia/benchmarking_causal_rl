"""feat/yamls-cell-4 (PR7) — Cell 4 masked-offline tier sweep (both gates compose).

Cell 4 (Burned Files) is Cell 3's offline tier sweep with Cell 2's per-env masks
applied on top: the runner loads each env's B2 dataset and the loader projects the
hidden-Z indices off obs/next_obs at load time, so backdoor adjustment fails for
lack of the conditioning variables. One YAML composes BOTH per-env maps
(offline_dataset from PR6, mask_indices from PR5). These tests cover parse + both
maps, simultaneous per-env composition, strict-mode on either partial map, and an
end-to-end smoke where both gates fire on the same config.
"""

from __future__ import annotations

import os
import subprocess
import sys
import warnings

import minari
import pytest
import yaml
from main import _resolve_mask_indices_map, _resolve_offline_dataset_map
from src.envs.offline.generate import generate_offline_dataset
from tests.conftest import REPO_ROOT

warnings.filterwarnings("ignore")

CELL_4 = REPO_ROOT / "reproducibility" / "rl_regimes" / "cell_4"

DISCRETE_ENVS = ["CartPole-v1", "LunarLander-v3", "Acrobot-v1"]
CONTINUOUS_ENVS = ["Pendulum-v1", "HalfCheetah-v5", "Hopper-v5", "Walker2d-v5"]
# §5 per-env mask spec (source of truth).
MASKS = {
    "CartPole-v1": [1, 3],
    "LunarLander-v3": [2, 3, 5],
    "Acrobot-v1": [4, 5],
    "Pendulum-v1": [2],
    "HalfCheetah-v5": [8, 9],
    "Hopper-v5": [5, 6],
    "Walker2d-v5": [9, 10],
}


def _slug(env: str) -> str:
    return env.split("-v")[0].lower().replace("/", "-")


def _load(name: str) -> dict:
    return yaml.safe_load((CELL_4 / name).read_text())


def test_cell_4_yamls_parse():
    for tier in ("random", "medium", "expert"):
        for arm, envs in (("discrete", DISCRETE_ENVS), ("continuous", CONTINUOUS_ENVS)):
            cfg = _load(f"offline_{tier}_masked_{arm}.yaml")
            assert cfg["seed"] == 0
            assert cfg["n_train_envs"] == 16
            assert cfg["offline_dataset"] == {
                e: f"generated/{_slug(e)}/{tier}-v0" for e in envs
            }
            assert cfg["mask_indices"] == {e: MASKS[e] for e in envs}


def test_both_gates_compose():
    cfg = _load("offline_random_masked_discrete.yaml")
    ds = _resolve_offline_dataset_map(
        cfg["offline_dataset"], cfg["envs"], "offline_random_masked_discrete.yaml"
    )
    mask = _resolve_mask_indices_map(
        cfg["mask_indices"], cfg["envs"], "offline_random_masked_discrete.yaml"
    )
    # Same config, two envs: BOTH maps vary per-env, simultaneously.
    assert ds["CartPole-v1"] == "generated/cartpole/random-v0"
    assert mask["CartPole-v1"] == (1, 3)
    assert ds["Acrobot-v1"] == "generated/acrobot/random-v0"
    assert mask["Acrobot-v1"] == (4, 5)
    assert ds["CartPole-v1"] != ds["Acrobot-v1"]
    assert mask["CartPole-v1"] != mask["Acrobot-v1"]


def test_strict_mode_raises_on_partial_maps():
    envs = ["CartPole-v1", "LunarLander-v3"]
    # mask map missing LunarLander-v3 -> PR5 strict mode raises.
    with pytest.raises(ValueError, match="LunarLander-v3") as exc_mask:
        _resolve_mask_indices_map({"CartPole-v1": [1, 3]}, envs, "partial_mask.yaml")
    assert "mask_indices map" in str(exc_mask.value)

    # symmetric: dataset map missing LunarLander-v3 -> PR6 strict mode raises.
    with pytest.raises(ValueError, match="LunarLander-v3") as exc_ds:
        _resolve_offline_dataset_map(
            {"CartPole-v1": "foo/bar-v0"}, envs, "partial_ds.yaml"
        )
    assert "offline_dataset map" in str(exc_ds.value)


def test_cell_4_discrete_smoke(tmp_path):
    # In-test dataset (random tier, agent behavior); both gates on the same config.
    did = "test/cell4-smoke-v0"
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
        repro_dir = tmp_path / "reproducibility" / "rl_regimes" / "cell_4"
        repro_dir.mkdir(parents=True)
        (repro_dir / "offline_random_masked_discrete.yaml").write_text(
            yaml.safe_dump(
                {
                    "envs": ["CartPole-v1"],
                    "algos": ["cql"],
                    "offline_dataset": {"CartPole-v1": did},
                    "mask_indices": {"CartPole-v1": [1, 3]},
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
                "rl_regimes/cell_4/offline_random_masked_discrete",
            ],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
        )
        assert proc.returncode == 0, f"smoke run failed:\n{proc.stdout}\n{proc.stderr}"

        run_dirs = list((tmp_path / "runs").rglob("config.yaml"))
        assert len(run_dirs) == 1
        run_dir = run_dirs[0].parent
        # Both gates fired: eval_per_context.csv (mask) + offline dataset in config.
        assert (run_dir / "train_metrics.csv").stat().st_size > 0
        assert (run_dir / "eval_metrics.csv").stat().st_size > 0
        assert (run_dir / "eval_per_context.csv").stat().st_size > 0
        snap = yaml.safe_load((run_dir / "config.yaml").read_text())
        assert snap["env"]["offline_dataset"] == {"CartPole-v1": did}
        assert snap["env"]["mask_indices"] == {"CartPole-v1": [1, 3]}
    finally:
        minari.delete_dataset(did)
