"""feat/cells-7-8-online-variants — online variants of Cells 7/8.

Cells 7 (Shadowed Vitals) and 8 (Dark Ages) gain ONLINE arms alongside their
existing offline σ-sweeps. The online variants live in the same cell_7/ and
cell_8/ folders, distinguished by the ``online_confounded_`` filename prefix —
online vs offline is an axis-value, not a new cell number, so the eight-cell
taxonomy is unchanged.

Each online YAML is online + bias_confounded over the σ grid {0.0..1.0}, gated to
CartPole + Acrobot (PR #34 dense-reward finding), algos = [ppo, dqn] where DQN
gets full σ·U confounding and PPO is the on-policy structural control. These
tests check parse, both behavior fields, the online shape (no offline_dataset),
and Cell 8's masks. No run smoke test (live confounded path is out of scope here).
"""

from __future__ import annotations

import yaml
from tests.conftest import REPO_ROOT

ENVS = ["CartPole-v1", "Acrobot-v1"]
MASKS = {"CartPole-v1": [1, 3], "Acrobot-v1": [4, 5]}  # §5, matches offline Cell 8
SIGMAS = [("000", 0.0), ("025", 0.25), ("050", 0.5), ("075", 0.75), ("100", 1.0)]

CELL_7 = REPO_ROOT / "reproducibility" / "rl_regimes" / "cell_7"
CELL_8 = REPO_ROOT / "reproducibility" / "rl_regimes" / "cell_8"


def _online_files():
    """(yaml_path, sigma, is_masked) for all ten online-variant YAMLs."""
    out = []
    for nnn, s in SIGMAS:
        out.append(
            (CELL_7 / f"online_confounded_sigma_{nnn}_discrete_gated.yaml", s, False)
        )
        out.append(
            (
                CELL_8 / f"online_confounded_sigma_{nnn}_masked_discrete_gated.yaml",
                s,
                True,
            )
        )
    return out


def test_online_variant_yamls_parse_and_match_convention():
    for path, sigma, is_masked in _online_files():
        cfg = yaml.safe_load(path.read_text())
        assert cfg["envs"] == ENVS, path.name
        assert cfg["behavior_policy"] == "bias_confounded", path.name
        assert cfg["behavior_strength"] == sigma, path.name
        # Online arm: ppo (on-policy control) + dqn (off-policy, fully confounded)
        # + their LSTM recurrent variants (dict form, PR #49 schema).
        assert cfg["algos"] == [
            "ppo",
            "dqn",
            {"name": "ppo", "networks": {"actor": "lstm", "critic": "lstm"}},
            {"name": "dqn", "networks": {"actor": "lstm", "critic": "lstm"}},
        ], path.name
        # Online => no offline dataset (the offline siblings have one; these don't).
        assert "offline_dataset" not in cfg, path.name
        assert cfg["seed"] == 0 and cfg["n_train_envs"] == 16
        assert cfg["n_episodes"] == 250 and cfg["aggregation"] == "iqm"
        if is_masked:
            assert cfg["mask_indices"] == {e: MASKS[e] for e in ENVS}, path.name
        else:
            assert "mask_indices" not in cfg, path.name


def test_online_variant_grid_consistency():
    expected = {s for _, s in SIGMAS}
    for cell, prefix in (
        (CELL_7, "online_confounded_sigma_{}_discrete_gated.yaml"),
        (CELL_8, "online_confounded_sigma_{}_masked_discrete_gated.yaml"),
    ):
        present = set()
        for nnn, s in SIGMAS:
            cfg = yaml.safe_load((cell / prefix.format(nnn)).read_text())
            present.add(cfg["behavior_strength"])
        assert present == expected, f"{cell.name}: {present}"
