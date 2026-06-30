"""Proximal STUB scaffolding (PR-1, hard gate 4).

The Proximal strategy is registered (*_proximal) and exercises the PR-0
episode-grouped sequence path end to end WITHOUT estimator math: it consumes
fill_sequence_buffer_from_minari, computes BOTH gate surrogates from a sampled
(B,T) batch, and degrades to a bound. No EM, no Q(s,a,u) fit.
"""

from __future__ import annotations

import warnings

import pytest
import torch
from src.benchmarking.registry import register_default_algorithms, registry
from src.rl.off_policy.identification import Observational, OracleU, Proximal

warnings.filterwarnings("ignore")

_CPU = torch.device("cpu")


def test_proximal_registered_as_episode_grouping_non_oracle():
    register_default_algorithms()
    for name in (
        "offline_dqn_proximal",
        "bcq_proximal",
        "cql_proximal",
        "iql_proximal",
    ):
        spec = registry.get(name)
        assert spec.needs_episode_grouping is True
        assert spec.requires_confounder_u is False  # infers U, never reads it
        assert spec.kind == "off_policy" and spec.data_regime == "offline"


def test_proximal_agent_carries_proximal_strategy():
    register_default_algorithms()
    _, agent = registry.get("cql_proximal").builder(
        obs_dim=4,
        action_dim=2,
        action_type="discrete",
        device=_CPU,
        action_space=None,
        obs_shape=(4,),
    )
    assert isinstance(agent._strategy, Proximal)
    assert agent.is_oracle_u is False  # not an oracle; never reads U


def test_proximal_critic_value_routes_through_the_q_su_hook():
    """PR-2b: Proximal.critic_value is the OracleU q_su hook with an INFERRED-and-
    sampled confounder_u (the E-step writes it), NOT a degrade-to-floor net(x).
    Identical routing to OracleU; only the provenance of confounder_u differs."""
    from src.rl.offline.oracle_u import UMarginalizedQ

    net = UMarginalizedQ(4, 2)
    obs = torch.randn(5, 4)
    u = torch.bernoulli(torch.full((5,), 0.5))
    batch = {"obs": obs, "confounder_u": u}
    assert torch.equal(
        Proximal().critic_value(net, obs, batch),
        OracleU().critic_value(net, obs, batch),  # same hook, q_su(x, u)
    )
    # And NOT the Observational floor (it deconfounds via q_su, not plain net(x)).
    assert not torch.equal(
        Proximal().critic_value(net, obs, batch),
        Observational().critic_value(net, obs, batch),
    )


# --------------------------------------------------------------------------
# End-to-end: consume the PR-0 sequence fill, compute both gate surrogates.
# --------------------------------------------------------------------------
pytest.importorskip("minari")
pytest.importorskip("h5py")


def test_proximal_consumes_sequence_fill_and_computes_diagnostic(tmp_path, monkeypatch):
    monkeypatch.setenv("MINARI_DATASETS_PATH", str(tmp_path / "minari"))
    from src.envs.offline.generate import generate_offline_dataset
    from src.envs.offline.minari_loader import (
        fill_sequence_buffer_from_minari,
        load_minari_dataset,
    )
    from src.envs.registry import register_default_env_wrappers
    from src.rl.off_policy.sequence_replay_buffer import SequenceReplayBuffer

    torch.manual_seed(0)
    register_default_algorithms()
    register_default_env_wrappers()
    did = "proximal/conf-v0"
    generate_offline_dataset(
        env_id="CartPole-v1",
        generator_algo="dqn",
        tier="random",
        behavior_policy="bias_confounded",
        behavior_strength=0.5,
        rollout_episodes=12,
        seed=0,
        dataset_id=did,
        device="cpu",
    )

    # PR-0 path: episode-grouped fill, then same-episode windows.
    seq = SequenceReplayBuffer(capacity=1_000_000, device=_CPU)
    n = fill_sequence_buffer_from_minari(did, seq, _CPU)  # proximal INFERS U: no load_u
    assert n > 0
    T = min(int(len(ep.rewards)) for ep in load_minari_dataset(did).iterate_episodes())
    seq_batch = seq.sample_sequences(batch_size=8, seq_len=T)

    diag = Proximal().statistical_diagnostic(seq_batch)
    assert set(diag) == {"separability", "action_overlap"}
    # Both surrogates are real, finite numbers (placeholder posterior, but computed).
    assert 0.5 <= diag["separability"] <= 1.0
    assert 0.0 <= diag["action_overlap"] <= 1.0
