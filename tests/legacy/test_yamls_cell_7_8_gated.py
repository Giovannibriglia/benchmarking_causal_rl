"""feat/yamls-cell-7-8-gated — Cell 7/8 σ-sweep restricted to gate-passing envs.

A diagnosis recon (2026-06-16) found the absolute σ·U confounding mechanism only
induces detectable A→R structure for unit-reward-scale envs (CartPole, Acrobot);
dense-reward (LunarLander) and continuous (MuJoCo) envs gate-fail at all σ≤1. The
`_gated` YAMLs restrict Cell 7/8 to the two gate-passing envs; the originals are
preserved. These tests check parse + both maps + dataset-id round-trip.
"""

from __future__ import annotations

import yaml
from src.envs.offline.generate import dataset_name
from tests.conftest import REPO_ROOT

ENVS = ["CartPole-v1", "Acrobot-v1"]
MASKS = {"CartPole-v1": [1, 3], "Acrobot-v1": [4, 5]}  # §5
SIGMAS = [("000", 0.0), ("025", 0.25), ("050", 0.5), ("075", 0.75), ("100", 1.0)]

CELL_7 = REPO_ROOT / "reproducibility" / "rl_regimes" / "_legacy" / "cell_7"
CELL_8 = REPO_ROOT / "reproducibility" / "rl_regimes" / "_legacy" / "cell_8"


def _gated_files():
    """(yaml_path, sigma, is_masked) for all ten gated YAMLs."""
    out = []
    for nnn, s in SIGMAS:
        out.append((CELL_7 / f"confounded_sigma_{nnn}_discrete_gated.yaml", s, False))
        out.append(
            (CELL_8 / f"confounded_sigma_{nnn}_masked_discrete_gated.yaml", s, True)
        )
    return out


def test_gated_yamls_parse_and_match_convention():
    for path, sigma, is_masked in _gated_files():
        cfg = yaml.safe_load(path.read_text())
        assert cfg["envs"] == ENVS, path.name
        assert cfg["behavior_policy"] == "bias_confounded"
        assert cfg["behavior_strength"] == sigma
        assert cfg["seed"] == 0 and cfg["n_train_envs"] == 16
        # offline_dataset map has exactly the two gate-passing envs.
        assert set(cfg["offline_dataset"]) == set(ENVS)
        if is_masked:
            assert cfg["mask_indices"] == {e: MASKS[e] for e in ENVS}
        else:
            assert "mask_indices" not in cfg


def test_gated_yamls_dataset_ids_round_trip():
    for path, sigma, _ in _gated_files():
        cfg = yaml.safe_load(path.read_text())
        assert cfg["offline_dataset"] == {
            e: dataset_name(e, "random", "bias_confounded", sigma) for e in ENVS
        }
