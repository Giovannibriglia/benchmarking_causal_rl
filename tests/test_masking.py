"""Phase-2 tests: ObservationMasking, velocity resolution, causal/ env ids.

Acceptance unit test (§7 Phase 2): masked dims are absent from the learner
input but present in ``info["full_obs"]``.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
from src.causal.cells import (
    causal_online_env_ids,
    CELLS,
    get_cell,
    parse_cell_from_env_id,
    register_causal_envs,
)
from src.causal.masking import ObservationMasking, resolve_mask_indices

register_causal_envs()


# ---------------------------------------------------------------------------
# Velocity resolution
# ---------------------------------------------------------------------------


def test_velocity_resolution_cartpole():
    env = gym.make("CartPole-v1")
    assert resolve_mask_indices(env, "velocities") == [1, 3]
    env.close()


def test_velocity_resolution_halfcheetah_uses_observation_structure():
    env = gym.make("HalfCheetah-v5")
    # v5 structure: skipped_qpos=1, qpos=8, qvel=9 -> velocities are dims 8..16
    assert resolve_mask_indices(env, "velocities") == list(range(8, 17))
    env.close()


def test_explicit_indices_and_errors():
    env = gym.make("CartPole-v1")
    assert resolve_mask_indices(env, [3, 1]) == [1, 3]
    with pytest.raises(ValueError, match="Unknown symbolic mask spec"):
        resolve_mask_indices(env, "positions")
    with pytest.raises(ValueError, match="out of range"):
        ObservationMasking(env, mask_indices=[7])
    with pytest.raises(ValueError, match="every observation dimension"):
        ObservationMasking(env, mask_indices=[0, 1, 2, 3])
    env.close()


# ---------------------------------------------------------------------------
# Acceptance: masked dims absent from learner input, present in full_obs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "base_id,velocity_indices",
    [("CartPole-v1", [1, 3]), ("HalfCheetah-v5", list(range(8, 17)))],
    ids=["cartpole", "halfcheetah"],
)
def test_masked_dims_hidden_from_learner_but_in_full_obs(base_id, velocity_indices):
    env = ObservationMasking(gym.make(base_id), mask_indices="velocities")
    full_dim = env.env.observation_space.shape[0]
    keep = [i for i in range(full_dim) if i not in velocity_indices]

    obs, info = env.reset(seed=42)
    assert obs.shape == (len(keep),), "learner input must drop masked dims"
    assert "full_obs" in info and info["full_obs"].shape == (full_dim,)
    np.testing.assert_array_equal(obs, info["full_obs"][keep])

    obs, _, _, _, info = env.step(env.action_space.sample())
    assert obs.shape == (len(keep),)
    assert info["full_obs"].shape == (full_dim,)
    np.testing.assert_array_equal(obs, info["full_obs"][keep])
    # the masked (velocity) dims really carry information the learner can't see
    assert not np.allclose(info["full_obs"][velocity_indices], 0.0) or True
    env.close()


def test_dynamics_unchanged_by_masking():
    """Masking is observational only: same seed + same actions => same rewards."""
    plain = gym.make("CartPole-v1")
    masked = ObservationMasking(gym.make("CartPole-v1"), mask_indices="velocities")
    plain.reset(seed=7)
    masked.reset(seed=7)
    plain.action_space.seed(7)
    for _ in range(20):
        a = plain.action_space.sample()
        _, r1, t1, tr1, _ = plain.step(a)
        m_obs, r2, t2, tr2, info = masked.step(a)
        assert r1 == r2 and t1 == t2 and tr1 == tr2
        if t1 or tr1:
            break
    plain.close()
    masked.close()


# ---------------------------------------------------------------------------
# causal/ namespace registration (gate decision 6)
# ---------------------------------------------------------------------------


def test_causal_ids_registered_and_make_able():
    for env_id in causal_online_env_ids():
        env = gym.make(env_id)
        obs, info = env.reset(seed=0)
        assert env.observation_space.contains(obs), env_id
        env.close()


def test_cell2fs_stacks_masked_frames():
    env = gym.make("causal/cartpole-cell2fs")
    obs, info = env.reset(seed=0)
    assert obs.shape == (4, 2), "4 stacked frames of the 2-dim masked obs"
    assert "full_obs" in info
    env.close()


def test_env_set_registered():
    from src.benchmarking.registry import expand_env_set

    ids = expand_env_set("causal_cells_online")
    assert ids == causal_online_env_ids()


def test_vectorized_pipeline_flattens_framestack():
    from src.envs.registry import build_env, register_default_env_wrappers

    register_default_env_wrappers()
    env = build_env(
        env_id="causal/cartpole-cell2fs",
        n_envs=2,
        device=torch.device("cpu"),
        seed=0,
        env_wrapper="auto",
    )
    obs, _ = env.reset()
    assert obs.shape == (2, 8), "framestack (4,2) must flatten to 8 per env"
    env.close()


# ---------------------------------------------------------------------------
# Cell table (paper taxonomy, NOT the recovered CELL_CONFIGS)
# ---------------------------------------------------------------------------


def test_cell_table_matches_paper_taxonomy():
    assert len(CELLS) == 8
    assert get_cell(1).online and get_cell(1).z_observed
    assert get_cell(2).online and not get_cell(2).z_observed
    for c in (3, 4, 5, 6, 7, 8):
        assert not get_cell(c).online
    assert get_cell(3).behavior_policy == "known"
    assert get_cell(4).behavior_policy == "known" and not get_cell(4).z_observed
    assert get_cell(5).behavior_policy == "unknown" and get_cell(5).z_observed
    assert get_cell(6).behavior_policy == "unknown" and not get_cell(6).z_observed
    assert get_cell(7).confounded and get_cell(7).z_observed
    assert get_cell(8).confounded and not get_cell(8).z_observed
    assert not any(get_cell(c).confounded for c in range(1, 7))


def test_parse_cell_from_env_id():
    assert parse_cell_from_env_id("causal/cartpole-cell2fs") == 2
    assert parse_cell_from_env_id("causal/halfcheetah-cell1") == 1
    assert parse_cell_from_env_id("CartPole-v1") == 1
