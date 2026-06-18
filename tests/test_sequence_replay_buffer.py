"""feat/off-policy-sequence-buffer — episode-aware SequenceReplayBuffer.

Buffer-level unit tests: add/sample shapes, episode-boundary integrity,
multi-env storage, short-episode skipping, episode-preserving eviction, len
semantics, and the no-long-enough-episode edge case. No env training — fast.
"""

from __future__ import annotations

import pytest
import torch
from src.rl.off_policy.sequence_replay_buffer import SequenceReplayBuffer

D = 3


def _t(value: float, env: int = 0, step: int = 0):
    """A transition whose obs encodes (value) and next_obs encodes (env, step) so
    tests can verify which env/episode/step a sampled sequence came from."""
    return {
        "obs": torch.full((D,), float(value)),
        "actions": torch.tensor(0),
        "rewards": torch.tensor(float(value)),
        "next_obs": torch.tensor([float(env), float(step), 0.0]),
        "dones": torch.tensor(0.0),
    }


def test_add_and_sample_basic():
    buf = SequenceReplayBuffer(capacity=10_000, device=torch.device("cpu"))
    for s in range(100):
        buf.add(0, _t(s, env=0, step=s))
    batch = buf.sample_sequences(batch_size=4, seq_len=16)
    assert batch["obs"].shape == (4, 16, D)
    assert batch["rewards"].shape == (4, 16)
    assert len(buf) == 100


def test_episode_boundary_respected():
    buf = SequenceReplayBuffer(capacity=10_000, device=torch.device("cpu"))
    # Episode 1: obs value 100..109 ; Episode 2: obs value 200..219.
    for s in range(10):
        buf.add(0, _t(100 + s))
    buf.mark_episode_end(0)
    for s in range(20):
        buf.add(0, _t(200 + s))
    # seq_len 15 only fits episode 2 (len 20); episode 1 (len 10) is skipped.
    for _ in range(50):
        seq = buf.sample_sequences(batch_size=8, seq_len=15)["obs"][:, :, 0]
        # Every value must be within a single episode's range (no boundary span):
        # episode 2 is the only eligible one, so all values in [200, 219].
        assert torch.all((seq >= 200) & (seq <= 219))


def test_multi_env_storage():
    buf = SequenceReplayBuffer(capacity=10_000, device=torch.device("cpu"))
    # Interleave 4 envs; each env gets a single 20-step episode, obs value = env*1000+step.
    for step in range(20):
        for env in range(4):
            buf.add(env, _t(env * 1000 + step, env=env, step=step))
    seq = buf.sample_sequences(batch_size=16, seq_len=10)
    obs = seq["obs"][:, :, 0]  # (16, 10)
    # Each sampled sequence must come from ONE env: value//1000 constant per row,
    # and steps contiguous (value%1000 increments by 1).
    for row in obs:
        envs = (row // 1000).long()
        assert torch.all(envs == envs[0]), "sequence mixed envs"
        steps = (row % 1000).long()
        assert torch.all(steps[1:] - steps[:-1] == 1), "sequence not contiguous"


def test_short_episodes_skipped():
    buf = SequenceReplayBuffer(capacity=10_000, device=torch.device("cpu"))
    for s in range(5):  # short episode (value 0..4)
        buf.add(0, _t(s))
    buf.mark_episode_end(0)
    for s in range(30):  # long episode (value 1000..1029)
        buf.add(0, _t(1000 + s))
    seq = buf.sample_sequences(batch_size=16, seq_len=16)["obs"][:, :, 0]
    assert torch.all(seq >= 1000), "sampled from the too-short episode"


def test_buffer_eviction_preserves_episodes():
    buf = SequenceReplayBuffer(capacity=100, device=torch.device("cpu"))
    # 10 episodes of 30 transitions each -> 300 added, capacity 100.
    for ep in range(10):
        for s in range(30):
            buf.add(0, _t(ep * 1000 + s, env=0, step=s))
        buf.mark_episode_end(0)
    assert len(buf) <= 100
    # Whole episodes only: every retained episode has its full 30 transitions and
    # contiguous steps (no partial episode left behind).
    for ep_obj in buf.episodes:
        assert len(ep_obj) == 30
        steps = [int(tr["next_obs"][1]) for tr in ep_obj.transitions]
        assert steps == list(range(30))


def test_len_returns_transition_count():
    buf = SequenceReplayBuffer(capacity=10_000, device=torch.device("cpu"))
    buf.add(0, _t(0))
    buf.add(1, _t(0))
    buf.mark_episode_end(0)
    buf.add(0, _t(0))
    assert len(buf) == 3  # sum across envs/episodes


def test_sample_with_no_long_enough_episodes_raises():
    buf = SequenceReplayBuffer(capacity=10_000, device=torch.device("cpu"))
    for s in range(5):
        buf.add(0, _t(s))
    assert buf.can_sample(seq_len=4) is True
    assert buf.can_sample(seq_len=16) is False
    with pytest.raises(ValueError, match="no episode of length >= seq_len"):
        buf.sample_sequences(batch_size=2, seq_len=16)
