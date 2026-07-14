"""feat/recurrent-baselines-yamls — verify recurrent LSTM variants are present
in the cell baselines and online σ-sweep YAMLs.

The paper's POMDP demonstration requires LSTM PPO/DQN/SAC baselines in specific
cells: Cell 1 (control, unmasked), Cell 2 (primary, masked), Cell 7 online
(control with confounding), Cell 8 online (primary masked with confounding).
This pins those variants are present and use separate-but-same-type trunks.
"""

from __future__ import annotations

import pytest
import yaml
from tests.conftest import REPO_ROOT

CELLS = {
    "cell_1": ["online_discrete.yaml", "online_continuous.yaml"],
    "cell_2": ["online_masked_discrete.yaml", "online_masked_continuous.yaml"],
}

CELLS_ONLINE_SIGMA = {
    "cell_7": [
        f"online_confounded_sigma_{s}_discrete_gated.yaml"
        for s in ("000", "025", "050", "075", "100")
    ],
    "cell_8": [
        f"online_confounded_sigma_{s}_masked_discrete_gated.yaml"
        for s in ("000", "025", "050", "075", "100")
    ],
}


def _load(cell, name):
    return yaml.safe_load(
        (
            REPO_ROOT / "reproducibility" / "rl_regimes" / "_legacy" / cell / name
        ).read_text()
    )


def _has_lstm_variant(algos, algo_name):
    """True iff algos contains {name: algo_name, networks: {actor: lstm,
    critic: lstm}}."""
    for entry in algos:
        if not isinstance(entry, dict) or entry.get("name") != algo_name:
            continue
        nets = entry.get("networks", {})
        if nets.get("actor") == "lstm" and nets.get("critic") == "lstm":
            return True
    return False


@pytest.mark.parametrize(
    "cell,name,algo_name",
    [
        ("cell_1", "online_discrete.yaml", "ppo"),
        ("cell_1", "online_discrete.yaml", "dqn"),
        ("cell_1", "online_continuous.yaml", "ppo"),
        ("cell_1", "online_continuous.yaml", "sac"),
        ("cell_2", "online_masked_discrete.yaml", "ppo"),
        ("cell_2", "online_masked_discrete.yaml", "dqn"),
        ("cell_2", "online_masked_continuous.yaml", "ppo"),
        ("cell_2", "online_masked_continuous.yaml", "sac"),
    ],
)
def test_cell_1_2_baseline_has_lstm_variant(cell, name, algo_name):
    cfg = _load(cell, name)
    assert _has_lstm_variant(
        cfg["algos"], algo_name
    ), f"{cell}/{name} missing LSTM {algo_name} variant"


@pytest.mark.parametrize(
    "cell,name",
    [(cell, fname) for cell, fnames in CELLS_ONLINE_SIGMA.items() for fname in fnames],
)
def test_cell_7_8_online_sigma_has_lstm_variant(cell, name):
    cfg = _load(cell, name)
    assert _has_lstm_variant(cfg["algos"], "ppo"), f"{cell}/{name} missing LSTM ppo"
    assert _has_lstm_variant(cfg["algos"], "dqn"), f"{cell}/{name} missing LSTM dqn"


def test_lstm_variants_use_separate_same_type_trunks():
    """Every LSTM entry has actor==critic==lstm (Pattern X: separate-but-same-type
    trunks); no asymmetric (actor: lstm, critic: mlp) baselines in the paper set."""
    for cell, names in {**CELLS, **CELLS_ONLINE_SIGMA}.items():
        for name in names:
            for entry in _load(cell, name)["algos"]:
                if not isinstance(entry, dict):
                    continue
                nets = entry.get("networks", {})
                if "lstm" in (nets.get("actor"), nets.get("critic")):
                    assert (
                        nets.get("actor") == "lstm" and nets.get("critic") == "lstm"
                    ), f"{cell}/{name} asymmetric trunks: {nets}"
