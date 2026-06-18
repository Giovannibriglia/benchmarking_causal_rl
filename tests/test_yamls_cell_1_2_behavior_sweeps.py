"""feat/cells-1-2-behavior-sweeps — curiosity / anti_reward strength sweeps.

Cells 1 (online, fully observed) and 2 (online, hidden Z) gain two new behavior
policies — curiosity (intrinsic-disagreement bonus) and anti_reward (critic-
pessimal argmin-Q) — swept across a shared strength grid {0.0, 0.25, 0.5, 0.75,
1.0}, the same grid Cells 7/8 use for bias_confounded's σ. 0.0 = pure agent
(baseline anchor), 1.0 = fully active.

Behavior policies bias OFF-POLICY collection only, so each sweep YAML pairs the
off-policy collector (dqn discrete / sac continuous) with on-policy PPO as a
structural control: PPO ignores behavior_strength, so its curve is a baseline
that isolates the behavior's effect on off-policy collection (matching Cells 7/8's
online σ-sweep, PR #45). These tests check parse, the two behavior fields, the
[collector, ppo] algo list, and grid completeness. No run smoke test (that would
exercise live policy/wrapper code — out of scope here).
"""

from __future__ import annotations

import yaml
from tests.conftest import REPO_ROOT

# (cell dir, baseline stem, off-policy algo, is_masked)
ARMS = [
    ("cell_1", "online_discrete", "dqn", False),
    ("cell_1", "online_continuous", "sac", False),
    ("cell_2", "online_masked_discrete", "dqn", True),
    ("cell_2", "online_masked_continuous", "sac", True),
]
BEHAVIORS = ["curiosity", "anti_reward"]
STRENGTHS = [("000", 0.0), ("025", 0.25), ("050", 0.5), ("075", 0.75), ("100", 1.0)]


def _path(cell: str, stem: str, behavior: str, nnn: str):
    return (
        REPO_ROOT
        / "reproducibility"
        / "rl_regimes"
        / cell
        / f"{stem}_{behavior}_{nnn}.yaml"
    )


def _check_behavior(behavior: str):
    for cell, stem, algo, is_masked in ARMS:
        for nnn, sval in STRENGTHS:
            path = _path(cell, stem, behavior, nnn)
            cfg = yaml.safe_load(path.read_text())
            assert cfg["behavior_policy"] == behavior, path.name
            assert cfg["behavior_strength"] == sval, path.name
            # Off-policy collector + on-policy PPO (the structural control): the
            # behavior policy biases off-policy collection, while PPO ignores
            # behavior_strength and serves as a baseline across the sweep.
            assert cfg["algos"] == [algo, "ppo"], path.name
            # §3 common settings carry over unchanged.
            assert cfg["seed"] == 0 and cfg["n_train_envs"] == 16
            assert cfg["n_episodes"] == 250 and cfg["aggregation"] == "iqm"
            if is_masked:
                assert isinstance(cfg["mask_indices"], dict) and cfg["mask_indices"]
            else:
                assert "mask_indices" not in cfg


def test_curiosity_yamls_parse():
    _check_behavior("curiosity")


def test_anti_reward_yamls_parse():
    _check_behavior("anti_reward")


def test_strength_grid_consistency():
    expected = {sval for _, sval in STRENGTHS}
    for behavior in BEHAVIORS:
        for cell, stem, _, _ in ARMS:
            present = set()
            for nnn, sval in STRENGTHS:
                cfg = yaml.safe_load(_path(cell, stem, behavior, nnn).read_text())
                present.add(cfg["behavior_strength"])
            assert present == expected, f"{cell}/{stem} {behavior}: {present}"
