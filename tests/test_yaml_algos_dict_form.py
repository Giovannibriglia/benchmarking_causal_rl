"""feat/on-policy-recurrent-integration — algos dict-form parsing.

The algos field accepts both the plain string form (``ppo`` -> all-MLP) and the
dict form (``{name: ppo, networks: {actor: lstm, critic: lstm, hidden_dim: 128}}``).
These test the normalizer + the canonical-id builder in main.py.
"""

from __future__ import annotations

import pytest
from main import _canonical_algo_id, _normalize_algo
from src.benchmarking.registry import register_default_algorithms

register_default_algorithms()  # populate registry so _canonical_algo_id sees kinds


def test_plain_string_normalizes_to_all_mlp():
    assert _normalize_algo("ppo") == {
        "name": "ppo",
        "actor": "mlp",
        "critic": "mlp",
        "network_kwargs": {},
    }


def test_dict_form_parses_networks_and_trunk_kwargs():
    spec = _normalize_algo(
        {
            "name": "ppo",
            "networks": {
                "actor": "lstm",
                "critic": "gru",
                "hidden_dim": 128,
                "num_layers": 2,
            },
        }
    )
    assert spec["name"] == "ppo"
    assert spec["actor"] == "lstm" and spec["critic"] == "gru"
    assert spec["network_kwargs"] == {"hidden_dim": 128, "num_layers": 2}


def test_dict_form_defaults_missing_components_to_mlp():
    spec = _normalize_algo({"name": "ppo", "networks": {"actor": "lstm"}})
    assert spec["actor"] == "lstm" and spec["critic"] == "mlp"
    spec2 = _normalize_algo({"name": "dqn"})  # no networks key at all
    assert spec2["actor"] == "mlp" and spec2["critic"] == "mlp"


def test_dict_missing_name_raises():
    with pytest.raises(ValueError, match="missing required 'name'"):
        _normalize_algo({"networks": {"actor": "lstm"}})


def test_networks_not_a_map_raises():
    with pytest.raises(ValueError, match="must be a map"):
        _normalize_algo({"name": "ppo", "networks": ["lstm", "lstm"]})


def test_bad_entry_type_raises():
    with pytest.raises(ValueError, match="must be a string or a dict"):
        _normalize_algo(42)


def test_canonical_id_on_policy_gets_network_suffix():
    assert _canonical_algo_id(_normalize_algo("ppo")) == "ppo__mlp__mlp"
    assert (
        _canonical_algo_id(
            _normalize_algo(
                {"name": "ppo", "networks": {"actor": "lstm", "critic": "mlp"}}
            )
        )
        == "ppo__lstm__mlp"
    )


def test_canonical_id_off_policy_stays_bare():
    # dqn/sac keep bare names -> off-policy goldens & run-dirs unaffected.
    assert _canonical_algo_id(_normalize_algo("dqn")) == "dqn"
    assert _canonical_algo_id(_normalize_algo("sac")) == "sac"


def test_canonical_id_off_policy_all_mlp_stays_bare():
    # Explicit all-MLP dict form is equivalent to the bare string (goldens green).
    spec = _normalize_algo(
        {"name": "dqn", "networks": {"actor": "mlp", "critic": "mlp"}}
    )
    assert _canonical_algo_id(spec) == "dqn"


def test_canonical_id_off_policy_non_default_is_suffixed():
    # The canonical id is computed for recurrent off-policy (used by PR-1C2),
    # even though this config is rejected at builder time in PR-1C1.
    spec = _normalize_algo(
        {"name": "dqn", "networks": {"actor": "lstm", "critic": "lstm"}}
    )
    assert _canonical_algo_id(spec) == "dqn__lstm__lstm"
