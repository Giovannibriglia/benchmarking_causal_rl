"""Unit tests for the Phase-1 abstractions: Algorithm ABC, ExperienceSource."""

from __future__ import annotations

import pytest
import torch
from src.benchmarking.registry import register_default_algorithms, registry
from src.data.experience_source import ExperienceSource, OnlineSource, validate_pairing
from src.rl.base import ActionOutput, Algorithm

DEVICE = torch.device("cpu")


def _build(name: str):
    register_default_algorithms()
    spec = registry.get(name)
    action_type = "continuous" if name == "ddpg" else "discrete"
    policy, agent = spec.builder(
        obs_dim=4,
        action_dim=2,
        action_type=action_type,
        device=DEVICE,
        action_space=None,
    )
    return spec, policy, agent


ALL_ALGOS = ["vanilla", "a2c", "ppo", "trpo", "dqn", "ddpg"]


@pytest.mark.parametrize("name", ALL_ALGOS)
def test_algorithms_implement_abc(name):
    spec, policy, agent = _build(name)
    assert isinstance(agent, Algorithm)
    assert isinstance(agent, torch.nn.Module)
    # registry kind and class paradigm must agree
    assert agent.paradigm == spec.kind
    assert agent.action_type in ("discrete", "continuous", "both")


@pytest.mark.parametrize("name", ALL_ALGOS)
def test_unified_act_contract(name):
    _, policy, agent = _build(name)
    obs = torch.zeros(3, 4)
    out = agent.act(obs)
    assert isinstance(out, ActionOutput)
    assert out.action.shape[0] == 3
    det = agent.act(obs, deterministic=True)
    assert isinstance(det, ActionOutput)
    # deterministic action selection is repeatable
    det2 = agent.act(obs, deterministic=True)
    assert torch.equal(det.action, det2.action)


@pytest.mark.parametrize("name", ALL_ALGOS)
def test_update_is_process_then_learn(name):
    """Algorithm.update must equal process_batch -> learn (identity default)."""
    _, policy, agent = _build(name)
    assert type(agent).process_batch is Algorithm.process_batch
    assert type(agent).update is Algorithm.update
    assert callable(agent.learn)


def test_set_agent_group_is_noop():
    _, _, agent = _build("ppo")
    assert agent.set_agent_group({"agents": ["a0"]}) is None


class _FakeOfflineSource(ExperienceSource):
    is_online = False


def test_validate_pairing_rejects_on_policy_with_offline_source():
    with pytest.raises(ValueError, match="online experience source"):
        validate_pairing("on_policy", _FakeOfflineSource())


def test_validate_pairing_rejects_offline_algo_with_online_source():
    with pytest.raises(ValueError, match="offline dataset source"):
        validate_pairing("offline", OnlineSource(env=None, device=DEVICE))


def test_validate_pairing_accepts_valid_combos():
    validate_pairing("on_policy", OnlineSource(env=None, device=DEVICE))
    validate_pairing("off_policy", OnlineSource(env=None, device=DEVICE))
    validate_pairing("off_policy", _FakeOfflineSource())
    validate_pairing("offline", _FakeOfflineSource())


def test_offline_stubs_raise():
    src = OnlineSource(env=None, device=DEVICE)
    with pytest.raises(NotImplementedError):
        src.sample(32)
    with pytest.raises(NotImplementedError):
        src.as_mdpdataset()
    assert src.behavior_logprob is None
