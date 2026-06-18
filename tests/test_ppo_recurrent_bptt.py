"""feat/on-policy-recurrent-integration — PPO truncated-BPTT update.

Tests that PPO.learn dispatches recurrent batches to the BPTT path, that the
update runs and propagates gradients through the recurrent cell connections, and
that the flat MLP path is untouched (no recurrent_seq_shape -> flat update).
"""

from __future__ import annotations

import torch
from src.data.experience_source import OnlineSource
from src.rl.on_policy.actor_critic import ActorCritic
from src.rl.on_policy.ppo import PPO
from tests.test_ppo_recurrent_rollout import _FixedDoneVecEnv


def _recurrent_batch(n_steps=6, n_envs=4):
    env = _FixedDoneVecEnv(n_envs=n_envs, period=2)
    src = OnlineSource(env, torch.device("cpu"))
    policy = ActorCritic(
        (3,),
        3,
        2,
        "discrete",
        torch.device("cpu"),
        actor_network="lstm",
        critic_network="lstm",
    )
    agent = PPO(policy, device=torch.device("cpu"), train_iters=2)
    batch, _ = src.rollout_recurrent(policy, agent, n_steps=n_steps, n_envs=n_envs)
    return policy, agent, batch


def test_recurrent_learn_runs_and_returns_metrics():
    policy, agent, batch = _recurrent_batch()
    metrics = agent.learn(batch)
    assert {"loss", "policy_loss", "value_loss", "entropy"} <= set(metrics)
    assert all(isinstance(v, float) for v in metrics.values())


def test_bptt_updates_recurrent_cell_weights():
    policy, agent, batch = _recurrent_batch()
    lstm = policy.actor_trunk.trunk.lstm
    before = lstm.weight_hh_l0.detach().clone()
    agent.learn(batch)
    after = lstm.weight_hh_l0.detach()
    # The recurrent (hidden-to-hidden) weights changed -> BPTT gradient flowed
    # through the recurrence and the optimizer stepped them.
    assert not torch.allclose(before, after)


def test_recurrent_dispatch_only_when_seq_shape_present():
    # A batch without recurrent_seq_shape must take the flat (MLP) path. Build a
    # flat batch via the non-recurrent rollout and confirm PPO.learn handles it.
    env = _FixedDoneVecEnv(n_envs=4, period=2)
    src = OnlineSource(env, torch.device("cpu"))
    mlp_policy = ActorCritic((3,), 3, 2, "discrete", torch.device("cpu"))
    agent = PPO(mlp_policy, device=torch.device("cpu"), train_iters=1, batch_size=8)
    batch, _ = src.rollout(mlp_policy, agent, n_steps=6, n_envs=4)
    assert batch.recurrent_seq_shape is None
    metrics = agent.learn(batch)  # flat path; should not raise
    assert "loss" in metrics
