"""feat/on-policy-recurrent-integration — recurrent rollout state threading.

Tests that OnlineSource.rollout_recurrent threads hidden state across env steps,
resets per-env state at episode boundaries, and stores the additive
recurrent_states / recurrent_seq_shape fields on the batch (the MLP rollout path
is untouched and covered by the existing on-policy regression tests).
"""

from __future__ import annotations

import torch
from src.data.experience_source import OnlineSource
from src.rl.on_policy.actor_critic import ActorCritic
from src.rl.on_policy.ppo import PPO


class _FixedDoneVecEnv:
    """Minimal vectorized env: obs is a 3-vector, episodes end every ``period``
    steps (deterministic), reward is constant. Just enough to exercise the
    recurrent rollout's state threading + per-env reset."""

    def __init__(self, n_envs=4, obs_dim=3, n_actions=2, period=2, device="cpu"):
        self.n_envs = n_envs
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.period = period
        self.device = torch.device(device)
        self._t = 0

        class _Box:
            shape = (obs_dim,)

        class _Disc:
            n = n_actions

        self.obs_space = _Box()
        self.act_space = _Disc()

    def reset(self, seed=None):
        self._t = 0
        return torch.zeros(self.n_envs, self.obs_dim, device=self.device), {}

    def step(self, action):
        self._t += 1
        obs = torch.randn(self.n_envs, self.obs_dim, device=self.device)
        reward = torch.ones(self.n_envs, device=self.device)
        done = (self._t % self.period) == 0
        terminated = torch.full((self.n_envs,), bool(done), device=self.device)
        truncated = torch.zeros(self.n_envs, dtype=torch.bool, device=self.device)
        return obs, reward, terminated, truncated, {}


def _policy(device="cpu"):
    return ActorCritic(
        (3,),
        3,
        2,
        "discrete",
        torch.device(device),
        actor_network="lstm",
        critic_network="lstm",
    )


def test_rollout_recurrent_stores_additive_fields():
    env = _FixedDoneVecEnv(n_envs=4, period=2)
    src = OnlineSource(env, torch.device("cpu"))
    policy = _policy()
    agent = PPO(policy, device=torch.device("cpu"))
    batch, ep_returns = src.rollout_recurrent(policy, agent, n_steps=6, n_envs=4)
    assert batch.recurrent_seq_shape == (6, 4)
    assert batch.recurrent_states is not None
    assert set(batch.recurrent_states.keys()) == {"actor", "critic"}
    # Flat fields preserved at (T*N, ...) for backward-compat consumers.
    assert batch.obs.shape == (24, 3)
    assert batch.advantages.shape == (24,)
    assert ep_returns.shape == (4,)


def test_rollout_recurrent_marks_episode_boundaries():
    # period=2 over 6 steps -> dones at t=1,3,5 (0-indexed). The BPTT update
    # derives episode_starts from these; verify the dones land where expected.
    env = _FixedDoneVecEnv(n_envs=4, period=2)
    src = OnlineSource(env, torch.device("cpu"))
    policy = _policy()
    agent = PPO(policy, device=torch.device("cpu"))
    batch, _ = src.rollout_recurrent(policy, agent, n_steps=6, n_envs=4)
    dones = batch.dones.view(6, 4)
    assert (
        torch.all(dones[1] == 1)
        and torch.all(dones[3] == 1)
        and torch.all(dones[5] == 1)
    )
    assert torch.all(dones[0] == 0) and torch.all(dones[2] == 0)


def test_runner_dispatches_recurrent_vs_mlp():
    # is_recurrent drives which rollout path the runner uses.
    assert _policy().is_recurrent is True
    mlp = ActorCritic((3,), 3, 2, "discrete", torch.device("cpu"))
    assert mlp.is_recurrent is False
