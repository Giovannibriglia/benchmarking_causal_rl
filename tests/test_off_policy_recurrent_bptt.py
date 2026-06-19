"""feat/off-policy-recurrent-integration — sequence-batch BPTT in DQN/SAC.

Verifies that the recurrent learn paths consume (B, T, ...) sequence batches and
that gradients propagate through the recurrent-cell (hidden-to-hidden) weights —
i.e. the truncated BPTT actually flows.
"""

from __future__ import annotations

import torch
from src.benchmarking.registry import register_default_algorithms, registry
from src.rl.off_policy.sequence_replay_buffer import SequenceReplayBuffer

register_default_algorithms()
DEV = torch.device("cpu")
B, T = 4, 6


class _Box:
    low = [-1.0]
    high = [1.0]


def test_dqn_recurrent_learn_bptt_grad():
    buf = SequenceReplayBuffer(capacity=1000, device=DEV)
    _, dqn = registry.get("dqn").builder(
        obs_dim=3,
        action_dim=2,
        action_type="discrete",
        device=DEV,
        action_space=None,
        obs_shape=(3,),
        critic_network="lstm",
        buffer=buf,
    )
    batch = {
        "obs": torch.randn(B, T, 3),
        "actions": torch.randint(0, 2, (B, T)),
        "rewards": torch.randn(B, T),
        "next_obs": torch.randn(B, T, 3),
        "dones": torch.zeros(B, T),
    }
    dqn.learn(batch)  # dispatches to _learn_recurrent
    grad = dqn.q_network.lstm.weight_hh_l0.grad
    assert grad is not None and grad.abs().sum() > 0


def test_sac_recurrent_learn_bptt_grad():
    buf = SequenceReplayBuffer(capacity=1000, device=DEV)
    actor, sac = registry.get("sac").builder(
        obs_dim=3,
        action_dim=1,
        action_type="continuous",
        device=DEV,
        action_space=_Box(),
        obs_shape=(3,),
        actor_network="lstm",
        critic_network="lstm",
        buffer=buf,
    )
    batch = {
        "obs": torch.randn(B, T, 3),
        "actions": torch.rand(B, T, 1) * 2 - 1,
        "rewards": torch.randn(B, T),
        "next_obs": torch.randn(B, T, 3),
        "dones": torch.zeros(B, T),
    }
    sac._learn_one_recurrent(batch)  # one step (skip the resample loop)
    # critic recurrent weights got gradients from the critic loss backward.
    qgrad = sac.q1.lstm.weight_hh_l0.grad
    assert qgrad is not None and qgrad.abs().sum() > 0
    # actor recurrent weights got gradients from the actor loss backward.
    agrad = actor.trunk.lstm.weight_hh_l0.grad
    assert agrad is not None and agrad.abs().sum() > 0
