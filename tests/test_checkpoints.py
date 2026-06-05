"""Checkpoint compatibility: old-style (pre-Phase-1) checkpoints load fine.

Pre-Phase-1, agents were plain objects without ``state_dict``, so
``BenchmarkRunner._save_checkpoint`` stored ``agent_state == {}``. Since
Phase 1 agents are ``nn.Module``s, ``agent_state`` now contains the full
agent (including the policy as a submodule). Both formats must stay loadable:
consumers (regret protocol, Phase 3+) treat ``policy_state`` as the canonical
weights and load ``agent_state`` with ``strict=False``.
"""

from __future__ import annotations

import torch
from src.benchmarking.checkpoints import load_checkpoint, save_checkpoint
from src.benchmarking.registry import register_default_algorithms, registry

DEVICE = torch.device("cpu")


def _fresh_ppo():
    register_default_algorithms()
    spec = registry.get("ppo")
    return spec.builder(
        obs_dim=4,
        action_dim=2,
        action_type="discrete",
        device=DEVICE,
        action_space=None,
    )


def test_old_style_checkpoint_loads(tmp_path):
    """A pre-Phase-1 checkpoint (agent_state == {}) loads without error."""
    policy, agent = _fresh_ppo()
    path = str(tmp_path / "old_style.pt")
    save_checkpoint(
        path,
        {
            "episode": 0,
            "policy_state": policy.state_dict(),
            "agent_state": {},  # what getattr(agent, "state_dict", lambda: {})() produced
            "config": {"env": {}, "training": {}},
        },
    )
    ckpt = load_checkpoint(path)
    policy2, agent2 = _fresh_ppo()
    policy2.load_state_dict(ckpt["policy_state"])  # strict: canonical weights
    # empty agent_state must be accepted (non-strict) without touching weights
    agent2.load_state_dict(ckpt["agent_state"], strict=False)
    for k, v in policy.state_dict().items():
        assert torch.equal(policy2.state_dict()[k], v)


def test_new_style_checkpoint_round_trips(tmp_path):
    """A Phase-1 checkpoint (agent_state = full nn.Module state) round-trips."""
    policy, agent = _fresh_ppo()
    path = str(tmp_path / "new_style.pt")
    save_checkpoint(
        path,
        {
            "episode": 0,
            "policy_state": policy.state_dict(),
            "agent_state": agent.state_dict(),
            "config": {"env": {}, "training": {}},
        },
    )
    ckpt = load_checkpoint(path)
    assert ckpt["agent_state"], "Phase-1 agents must serialize their modules"

    policy2, agent2 = _fresh_ppo()
    policy2.load_state_dict(ckpt["policy_state"])
    agent2.load_state_dict(ckpt["agent_state"])  # strict round-trip
    for k, v in agent.state_dict().items():
        assert torch.equal(agent2.state_dict()[k], v)
    # policy_state and the agent's embedded policy.* entries agree
    for k, v in policy.state_dict().items():
        assert torch.equal(ckpt["agent_state"][f"policy.{k}"], v)
