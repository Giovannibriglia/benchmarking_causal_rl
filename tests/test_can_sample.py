"""feat/off-policy-recurrent-integration — SequenceReplayBuffer.can_sample.

The recurrent train loop gates sampling on can_sample(seq_len) (sample_sequences
raises when no episode is long enough). These pin that gate's semantics.
"""

from __future__ import annotations

import torch
from src.rl.off_policy.sequence_replay_buffer import SequenceReplayBuffer


def _tr():
    return {
        "obs": torch.zeros(2),
        "actions": torch.tensor(0),
        "rewards": torch.tensor(0.0),
        "next_obs": torch.zeros(2),
        "dones": torch.tensor(0.0),
    }


def test_can_sample_returns_true_when_long_enough_episode_exists():
    buf = SequenceReplayBuffer(capacity=1000, device=torch.device("cpu"))
    for _ in range(100):
        buf.add(0, _tr())
    buf.mark_episode_end(0)
    assert buf.can_sample(seq_len=80) is True
    assert buf.can_sample(seq_len=200) is False


def test_can_sample_returns_false_for_short_episodes_only():
    buf = SequenceReplayBuffer(capacity=1000, device=torch.device("cpu"))
    for _ in range(20):
        buf.add(0, _tr())
    buf.mark_episode_end(0)
    for _ in range(15):
        buf.add(1, _tr())
    buf.mark_episode_end(1)
    assert buf.can_sample(seq_len=15) is True  # env 0's episode is long enough
    assert buf.can_sample(seq_len=25) is False  # neither is
