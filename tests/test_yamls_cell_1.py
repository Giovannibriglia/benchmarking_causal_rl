"""feat/yamls-cell-1 (PR4) — Cell 1 baseline-anchor YAMLs + nested --reproduce.

Cell 1 (Crystal Clear Clinic) is the online, fully-observed, unconfounded
baseline. These tests check that the two YAMLs parse with the §3 common settings,
that the nested --reproduce path resolves to the real files, and that a Cell-1-
shaped config is consumable end-to-end by the runner.

Note on the smoke test: reproduce-YAML values OVERRIDE CLI args in main.py
(precedence: --reproduce > CLI), so the heavy 250-episode settings in the real
files cannot be shrunk via CLI flags. The smoke test therefore writes a tiny
YAML at the SAME nested path (in a temp reproducibility/ tree) to exercise the
nested-resolution + flat-key + runner pipeline in seconds. The real files' content
is checked by test_cell_1_yamls_parse; their resolvability by the resolution test.
"""

from __future__ import annotations

import os
import subprocess
import sys

import yaml
from tests.conftest import REPO_ROOT

CELL_1 = REPO_ROOT / "reproducibility" / "rl_regimes" / "cell_1"


def _load(name: str) -> dict:
    # Mirrors main.py's loader: yaml.safe_load of the resolved reproduce file.
    return yaml.safe_load((CELL_1 / name).read_text())


def test_cell_1_yamls_parse():
    discrete = _load("online_discrete.yaml")
    continuous = _load("online_continuous.yaml")

    # (b) expected envs / algos per §4 Cell 1.
    assert discrete["envs"] == ["CartPole-v1", "LunarLander-v3", "Acrobot-v1"]
    assert discrete["algos"] == ["ppo", "dqn"]
    assert continuous["envs"] == [
        "Pendulum-v1",
        "HalfCheetah-v5",
        "Hopper-v5",
        "Walker2d-v5",
    ]
    assert continuous["algos"] == ["sac", "ppo"]

    # (c) common §3 settings landed on both arms.
    for cfg in (discrete, continuous):
        assert cfg["seed"] == 0
        assert cfg["n_train_envs"] == 16
        assert cfg["n_eval_envs"] == 16
        assert cfg["n_episodes"] == 250
        assert cfg["rollout_len"] == 1024
        assert cfg["n_checkpoints"] == 25
        assert cfg["aggregation"] == "iqm"
        assert cfg["deterministic"] is True


def test_reproduce_nested_path_resolution():
    # Mirrors main.py:197-201 inline resolution (no public resolver fn to import):
    # a name without an extension gets ".yaml" and is joined under reproducibility/.
    name = "rl_regimes/cell_1/online_discrete"
    repro_name = name if name.endswith((".yaml", ".yml")) else f"{name}.yaml"
    resolved = REPO_ROOT / "reproducibility" / repro_name
    assert resolved == CELL_1 / "online_discrete.yaml"
    assert resolved.exists()


def test_cell_1_discrete_smoke(tmp_path):
    # Tiny YAML at the SAME nested path the real file uses (CLI can't shrink the
    # real file because reproduce-YAML overrides CLI). Exercises nested
    # resolution + flat-key parsing + runner consumption, fast.
    repro_dir = tmp_path / "reproducibility" / "rl_regimes" / "cell_1"
    repro_dir.mkdir(parents=True)
    (repro_dir / "online_discrete.yaml").write_text(
        yaml.safe_dump(
            {
                "envs": ["CartPole-v1"],
                "algos": ["ppo"],
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
            "rl_regimes/cell_1/online_discrete",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0, f"smoke run failed:\n{proc.stdout}\n{proc.stderr}"

    # Run dir is nested under runs/ because the reproduce tag carries the path;
    # find the CSVs anywhere beneath runs/.
    train_csvs = list((tmp_path / "runs").rglob("train_metrics.csv"))
    eval_csvs = list((tmp_path / "runs").rglob("eval_metrics.csv"))
    assert len(train_csvs) == 1 and train_csvs[0].stat().st_size > 0
    assert len(eval_csvs) == 1 and eval_csvs[0].stat().st_size > 0
