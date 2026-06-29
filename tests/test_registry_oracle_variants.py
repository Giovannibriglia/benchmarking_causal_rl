"""Oracle-U variants are first-class registry algos (selected via --algos).

After the named-variant refactor there is no --causal-method run mode: the four
discrete deconfounding variants are ordinary registry entries flagged
requires_confounder_u=True (the oracle reference line), the base offline algos
keep requires_confounder_u=False, and no continuous oracle variant exists.
"""

from __future__ import annotations

import pytest
from src.benchmarking.registry import register_default_algorithms, registry

_VARIANTS = ["offline_dqn_oracle_u", "bcq_oracle_u", "cql_oracle_u", "iql_oracle_u"]
_BASES = ["offline_dqn", "bcq", "cql", "iql"]


@pytest.mark.parametrize("name", _VARIANTS)
def test_variant_registered_as_oracle(name):
    register_default_algorithms()
    spec = registry.get(name)
    assert spec.requires_confounder_u is True
    assert spec.kind == "off_policy"
    assert spec.data_regime == "offline"


@pytest.mark.parametrize("name", _BASES)
def test_base_algo_is_not_oracle(name):
    register_default_algorithms()
    spec = registry.get(name)
    assert spec.requires_confounder_u is False


@pytest.mark.parametrize(
    "name", ["sac_oracle_u", "cql_continuous_oracle_u", "ppo_oracle_u"]
)
def test_unregistered_variant_raises_keyerror(name):
    register_default_algorithms()
    with pytest.raises(KeyError):
        registry.get(name)
