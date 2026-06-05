"""Interventional oracles over REAL Gymnasium environments (§6.4).

The oracle answers ``P_true(R | do(A=a), S=s)`` by snapshot/restore of the
simulator's physical state: snapshot the state, apply ``do(A=a)``, step once,
read the true reward; Monte-Carlo over restarts gives the interventional
reward distribution. CartPole uses the analytic ``env.unwrapped.state``;
MuJoCo envs use ``set_state(qpos, qvel)``.

Standalone module (never imported by ``critic_ablation``); estimators in
``src/eval/ope.py`` must NEVER consume oracle outputs — the oracle is only
for ground-truth gap reporting and validation tests.
"""

from __future__ import annotations

import abc
from typing import Any

import gymnasium as gym
import numpy as np
import torch


class InterventionalOracle(abc.ABC):
    """Snapshot/restore-based do-operator for a single (non-vector) env."""

    def __init__(self, env: gym.Env) -> None:
        self.env = env

    @abc.abstractmethod
    def snapshot(self) -> Any:
        """Capture the env's full physical state."""

    @abc.abstractmethod
    def restore(self, state: Any) -> None:
        """Restore a previously captured physical state exactly."""

    def do_step(self, action) -> float:
        """Apply do(A=action) from the CURRENT state; return the true reward.

        Mutates the env — callers restore() afterwards if they need the
        original state back.
        """
        _, reward, _, _, _ = self.env.step(action)
        return float(reward)

    def reward_samples(self, action, n_samples: int = 1) -> torch.Tensor:
        """Monte-Carlo samples of R | do(A=action), S=current state.

        Restores the starting state between draws and afterwards, so the
        env is left where it started.
        """
        start = self.snapshot()
        out = []
        for _ in range(int(n_samples)):
            out.append(self.do_step(action))
            self.restore(start)
        return torch.tensor(out, dtype=torch.float32)


class CartPoleOracle(InterventionalOracle):
    """Analytic-state oracle for CartPole-style classic-control envs."""

    def snapshot(self) -> np.ndarray:
        state = self.env.unwrapped.state
        return np.array(state, dtype=np.float64, copy=True)

    def restore(self, state: np.ndarray) -> None:
        self.env.unwrapped.state = np.array(state, dtype=np.float64, copy=True)
        # classic-control envs track termination via steps_beyond_terminated
        if hasattr(self.env.unwrapped, "steps_beyond_terminated"):
            self.env.unwrapped.steps_beyond_terminated = None


class MuJoCoOracle(InterventionalOracle):
    """(qpos, qvel) snapshot/restore oracle for MuJoCo v5 envs."""

    def snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        data = self.env.unwrapped.data
        return (np.array(data.qpos, copy=True), np.array(data.qvel, copy=True))

    def restore(self, state: tuple[np.ndarray, np.ndarray]) -> None:
        qpos, qvel = state
        self.env.unwrapped.set_state(qpos, qvel)


def make_oracle(env: gym.Env) -> InterventionalOracle:
    """Pick the oracle implementation for ``env`` (single env, unwrapped
    through standard wrappers)."""
    unwrapped = env.unwrapped
    if hasattr(unwrapped, "set_state") and hasattr(unwrapped, "data"):
        return MuJoCoOracle(env)
    if hasattr(unwrapped, "state"):
        return CartPoleOracle(env)
    raise TypeError(
        f"No interventional oracle available for {type(unwrapped).__name__}: "
        "needs MuJoCo set_state or an analytic .state."
    )
