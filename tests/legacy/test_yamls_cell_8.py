"""feat/yamls-cell-8 (PR9) — Cell 8 confounded+masked offline (all three gates).

Cell 8 (Dark Ages) is the worst case: Z hidden (per-env velocity masks, PR5/PR1)
+ confounded offline data (σ-encoded datasets, PR6/PR8) + the σ-aware confounded
gate (PR3/PR8.5), all on one config. It reuses Cell 7's σ-encoded datasets with
masking applied at load time. These tests cover parse + all three per-env
mechanisms, three-gates-compose per env, strict-mode on either partial map, and
an end-to-end smoke where mask + confounded gates both fire simultaneously.
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
from src.envs.offline.generate import dataset_name, generate_offline_dataset
from tests.conftest import REPO_ROOT

warnings.filterwarnings("ignore")

CELL_8 = REPO_ROOT / "reproducibility" / "rl_regimes" / "_legacy" / "cell_8"
DISCRETE_ENVS = ["CartPole-v1", "LunarLander-v3", "Acrobot-v1"]
CONTINUOUS_ENVS = ["Pendulum-v1", "HalfCheetah-v5", "Hopper-v5", "Walker2d-v5"]
MASKS = {
    "CartPole-v1": [1, 3],
    "LunarLander-v3": [2, 3, 5],
    "Acrobot-v1": [4, 5],
    "Pendulum-v1": [2],
    "HalfCheetah-v5": [8, 9],
    "Hopper-v5": [5, 6],
    "Walker2d-v5": [9, 10],
}


def _load(name: str) -> dict:
    return yaml.safe_load((CELL_8 / name).read_text())


def test_cell_8_yamls_parse_and_match_convention():
    for nnn, sigma in (
        ("000", 0.0),
        ("025", 0.25),
        ("050", 0.5),
        ("075", 0.75),
        ("100", 1.0),
    ):
        for arm, envs in (("discrete", DISCRETE_ENVS), ("continuous", CONTINUOUS_ENVS)):
            cfg = _load(f"confounded_sigma_{nnn}_masked_{arm}.yaml")
            assert cfg["behavior_policy"] == "bias_confounded"
            assert cfg["behavior_strength"] == sigma
            assert cfg["seed"] == 0 and cfg["n_train_envs"] == 16
            assert cfg["mask_indices"] == {e: MASKS[e] for e in envs}
            assert cfg["offline_dataset"] == {
                e: dataset_name(e, "random", "bias_confounded", sigma) for e in envs
            }


def test_three_gates_compose_per_env():
    cfg = _load("confounded_sigma_050_masked_discrete.yaml")
    ds = _resolve_offline_dataset_map(cfg["offline_dataset"], cfg["envs"], "x.yaml")
    mask = _resolve_mask_indices_map(cfg["mask_indices"], cfg["envs"], "x.yaml")
    # CartPole: dataset + mask + confounded behavior all present and correct.
    assert ds["CartPole-v1"] == "generated/cartpole/random-bias_confounded-sigma050-v0"
    assert mask["CartPole-v1"] == (1, 3)
    assert (
        cfg["behavior_policy"] == "bias_confounded" and cfg["behavior_strength"] == 0.5
    )
    # Acrobot: different mask AND different dataset; behavior stays uniform.
    assert ds["Acrobot-v1"] == "generated/acrobot/random-bias_confounded-sigma050-v0"
    assert mask["Acrobot-v1"] == (4, 5)
    assert mask["CartPole-v1"] != mask["Acrobot-v1"]
    assert ds["CartPole-v1"] != ds["Acrobot-v1"]


def test_strict_mode_raises_when_any_partial_map():
    envs = ["CartPole-v1", "LunarLander-v3"]
    # (1) mask map missing an env -> PR5 strict mode raises.
    with pytest.raises(ValueError, match="LunarLander-v3"):
        _resolve_mask_indices_map({"CartPole-v1": [1, 3]}, envs, "partial_mask.yaml")
    # (2) dataset map missing an env -> PR6 strict mode raises.
    with pytest.raises(ValueError, match="LunarLander-v3"):
        _resolve_offline_dataset_map(
            {"CartPole-v1": "foo/bar-v0"}, envs, "partial_ds.yaml"
        )
    # (3) KNOWN LIMITATION: both maps complete but behavior_policy mismatched with
    # σ-encoded ids is NOT statically rejected — the loader can't cross-check
    # behavior against dataset id, so it resolves fine (a runtime concern only).
    ds = _resolve_offline_dataset_map(
        {e: "generated/x/random-bias_confounded-sigma050-v0" for e in envs}, envs, "x"
    )
    assert set(ds) == set(envs)  # loads; mismatch would surface at runtime, not here


def test_cell_8_discrete_smoke(tmp_path):
    # Confounded σ=0.5 dataset (gate-passing); all three per-env mechanisms set.
    # Use a unique test id + manual_seed before generation. After issue #36 the
    # confounder's U is drawn from an isolated per-instance generator (seeded from
    # generate's seed=0), so the global manual_seed(0) no longer governs U. It is
    # KEPT (not redundant at σ<1): at σ=0.5 the agent action and the confounded
    # mixture coin still draw on the GLOBAL RNG, so without it the gate-pass would
    # stay suite-order dependent (the σ=0.5 marginal can dip near the 0.2 gate
    # threshold). seed=0 pins a deterministic gate-passing dataset, and the unique
    # id avoids collisions with the convention-id dataset used elsewhere.
    import torch

    did = "test/cell8-smoke-v0"
    try:
        minari.delete_dataset(did)
    except Exception:
        pass
    torch.manual_seed(0)
    ds = generate_offline_dataset(
        env_id="CartPole-v1",
        generator_algo="dqn",
        tier="random",
        behavior_policy="bias_confounded",
        behavior_strength=0.5,
        rollout_episodes=30,
        seed=0,
        dataset_id=did,
    )
    assert (
        ds.storage.metadata["gate_test_passed"] is True
    ), "seed=0 σ=0.5 must gate-pass"
    try:
        repro_dir = tmp_path / "reproducibility" / "rl_regimes" / "_legacy" / "cell_8"
        repro_dir.mkdir(parents=True)
        (repro_dir / "confounded_sigma_050_masked_discrete.yaml").write_text(
            yaml.safe_dump(
                {
                    "envs": ["CartPole-v1"],
                    "algos": ["cql"],
                    "offline_dataset": {"CartPole-v1": did},
                    "mask_indices": {"CartPole-v1": [1, 3]},
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
                "rl_regimes/_legacy/cell_8/confounded_sigma_050_masked_discrete",
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
        # All three gates fired: mask (eval_per_context) + confounded (value_trace).
        assert (run_dir / "train_metrics.csv").stat().st_size > 0
        assert (run_dir / "eval_metrics.csv").stat().st_size > 0
        assert (run_dir / "eval_per_context.csv").stat().st_size > 0
        assert (run_dir / "offline_value_trace.csv").stat().st_size > 0
        snap = yaml.safe_load((run_dir / "config.yaml").read_text())
        assert snap["env"]["offline_dataset"] == {"CartPole-v1": did}
        assert snap["env"]["mask_indices"] == {"CartPole-v1": [1, 3]}
    finally:
        minari.delete_dataset(did)
