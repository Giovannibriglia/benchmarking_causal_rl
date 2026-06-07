"""Oracle validation (Phase 6): restore exactness on both anchors and the
oracle-vs-empirical sanity check on confounded data."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
from src.eval.oracle import CartPoleOracle, make_oracle, MuJoCoOracle

# ---------------------------------------------------------------------------
# Restore exactness
# ---------------------------------------------------------------------------


def test_cartpole_restore_exactness():
    env = gym.make("CartPole-v1")
    env.reset(seed=3)
    oracle = make_oracle(env)
    assert isinstance(oracle, CartPoleOracle)
    snap = oracle.snapshot()
    # drift the env, then restore
    for _ in range(5):
        env.step(env.action_space.sample())
    oracle.restore(snap)
    np.testing.assert_array_equal(np.array(env.unwrapped.state, dtype=np.float64), snap)
    # deterministic continuation: same state + same action twice -> identical
    obs1, r1, *_ = env.step(0)
    oracle.restore(snap)
    obs2, r2, *_ = env.step(0)
    np.testing.assert_allclose(obs1, obs2, rtol=0, atol=0)
    assert r1 == r2
    env.close()


def test_halfcheetah_set_state_round_trip():
    env = gym.make("HalfCheetah-v5")
    env.reset(seed=3)
    oracle = make_oracle(env)
    assert isinstance(oracle, MuJoCoOracle)
    for _ in range(10):  # drift into a non-trivial state
        env.step(env.action_space.sample())
    snap = oracle.snapshot()
    qpos0, qvel0 = snap
    for _ in range(7):
        env.step(env.action_space.sample())
    oracle.restore(snap)
    np.testing.assert_allclose(env.unwrapped.data.qpos, qpos0, rtol=0, atol=0)
    np.testing.assert_allclose(env.unwrapped.data.qvel, qvel0, rtol=0, atol=0)
    # deterministic continuation through the restored state
    a = np.zeros(env.action_space.shape, dtype=np.float32)
    obs1, r1, *_ = env.step(a)
    oracle.restore(snap)
    obs2, r2, *_ = env.step(a)
    np.testing.assert_allclose(obs1, obs2, rtol=0, atol=1e-12)
    assert abs(r1 - r2) < 1e-12
    env.close()


def test_reward_samples_leaves_state_unchanged():
    env = gym.make("HalfCheetah-v5")
    env.reset(seed=0)
    oracle = make_oracle(env)
    snap = oracle.snapshot()
    samples = oracle.reward_samples(
        np.zeros(env.action_space.shape, dtype=np.float32), n_samples=3
    )
    assert samples.shape == (3,)
    assert torch.allclose(samples, samples[0].expand(3))  # deterministic env
    qpos, qvel = oracle.snapshot()
    np.testing.assert_allclose(qpos, snap[0], rtol=0, atol=0)
    np.testing.assert_allclose(qvel, snap[1], rtol=0, atol=0)
    env.close()


# ---------------------------------------------------------------------------
# Oracle vs empirical on confounded data: do(A) rewards differ from logged
# rewards by exactly the confounder shift delta*U.
# ---------------------------------------------------------------------------


@pytest.mark.golden  # needs the locally collected cell-7 dataset
def test_oracle_detects_confounded_reward_shift():
    import minari

    if "causal/cartpole/cell7-b1-d0p5-v0" not in minari.list_local_datasets():
        pytest.skip("cell-7 dataset not collected on this machine")
    from src.data.minari_io import to_offline_source

    src = to_offline_source(
        "causal/cartpole/cell7-b1-d0p5-v0",
        torch.device("cpu"),
        behavior_policy="known",
    )
    env = gym.make("CartPole-v1")
    env.reset(seed=0)
    oracle = make_oracle(env)

    diffs, us = [], []
    for ep in src.episodes[:40]:
        T = int(ep["rewards"].shape[0])
        t = min(3, T - 1)
        # full observations are stored; CartPole's analytic state == obs
        oracle.restore(ep["obs"][t].numpy().astype(np.float64))
        r_oracle = oracle.do_step(int(ep["actions"][t]))
        diffs.append(float(ep["rewards"][t]) - r_oracle)
        us.append(float(ep["confounder_u"][t]))
    diffs, us = np.array(diffs), np.array(us)
    # logged r' = r + delta*U with delta = 0.5: difference == 0.5*U exactly
    np.testing.assert_allclose(diffs, 0.5 * us, atol=1e-9)
    env.close()
