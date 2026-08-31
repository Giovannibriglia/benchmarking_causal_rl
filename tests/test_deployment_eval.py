"""Deployment evaluation: U -> R intact, U -> A severed.

``do(a)`` removes the incoming edges to A and leaves everything else, so the
deployment environment IS the mutilated graph and its value is
``E_U[R | do(a), s]`` — the causal estimand. These tests pin the two halves of
that: the reward path survives at eval, and the action path does not exist
there because the learned policy acts.
"""

from __future__ import annotations

import torch
from src.envs.wrappers.confounded import ConfoundedCollectionWrapper


class _StubEnv:
    """Minimal vectorised env surface the wrapper needs."""

    def __init__(self, n=1, reward=1.0):
        self.n_envs = n
        self.num_envs = n
        self._reward = reward
        self.device = torch.device("cpu")
        self.obs_space = None
        self.action_space = None

    def reset(self, *a, **k):
        return torch.zeros(self.n_envs, 4), {}

    def step(self, action):
        obs = torch.zeros(self.n_envs, 4)
        rew = torch.full((self.n_envs,), self._reward)
        done = torch.zeros(self.n_envs, dtype=torch.bool)
        return obs, rew, done, done.clone(), {}

    def close(self):
        pass


def _gated_reward(u_value: int, action: int, c_r: float = 3.0) -> float:
    """Step a gated wrapper with U pinned, return the reward it emits."""
    env = ConfoundedCollectionWrapper(
        _StubEnv(), c_a=0.0, c_r=c_r, confounder_kind="action_gated"
    )
    env.reset(seed=0)
    env.current_u = torch.full((1, 1), float(u_value))
    _, rew, _, _, _ = env.step(torch.tensor([action]))
    return float(torch.as_tensor(rew).reshape(-1)[0])


def test_eval_reward_carries_the_gated_bonus_only_when_u_is_one():
    """The whole point of the deployment env: the gate can still fire.

    With a clean eval env (U absent) the gate NEVER fires, so the reward
    channel the d_d cell is built around is missing from the metric and
    neither critic's estimand is being measured.
    """
    a_bad = 1  # the wrapper's gated action
    assert _gated_reward(1, a_bad) > _gated_reward(0, a_bad)
    # U = 0 pays no bonus, whatever the action
    assert _gated_reward(0, a_bad) == _gated_reward(0, 1 - a_bad)


def test_the_bonus_is_gated_on_the_action_not_paid_unconditionally():
    """`r += c_r*U*1[a == a_bad]`: U=1 alone must not pay."""
    a_bad = 1
    assert _gated_reward(1, 1 - a_bad) == _gated_reward(0, 1 - a_bad)


def test_c_a_zero_means_the_wrapper_never_biases_the_action():
    """U -> A is severed at deployment because the LEARNED POLICY acts; the
    wrapper is only asked for the reward path. c_a=0 makes that explicit, so
    a future reader cannot mistake the eval wrap for a behaviour wrap."""
    env = ConfoundedCollectionWrapper(
        _StubEnv(), c_a=0.0, c_r=1.0, confounder_kind="action_gated"
    )
    assert float(env.c_a) == 0.0


def test_the_flag_is_off_by_default_so_existing_evals_are_untouched():
    from src.config.defaults import EnvConfig

    assert EnvConfig(env_id="CartPole-v1").eval_confounded_reward is False
