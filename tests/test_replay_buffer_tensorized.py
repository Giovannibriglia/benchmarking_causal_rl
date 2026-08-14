"""Tensorized ReplayBuffer — byte-identity with the deque it replaced.

The buffer was rewritten from a ``deque`` of per-transition dicts to contiguous
per-key tensors for speed (docs/offline_training_profile.md: sample 433 us ->
88 us). The whole point is that this is a REPRESENTATION change only: the same
seed must draw the same transitions, so offline goldens are untouched. These
tests pin that against a reference implementation of the old buffer.
"""

from __future__ import annotations

import random
from collections import deque

import pytest
import torch
from src.rl.off_policy.replay_buffer import ReplayBuffer


class _DequeBuffer:
    """The pre-tensorization implementation, verbatim, as the oracle."""

    def __init__(self, capacity, device):
        self.capacity, self.device = capacity, device
        self.storage = deque(maxlen=capacity)

    def add(self, tr):
        self.storage.append({k: v.detach().cpu() for k, v in tr.items()})

    def sample(self, b):
        batch = random.sample(self.storage, b)
        return {k: torch.stack([x[k] for x in batch]).to(self.device) for k in batch[0]}

    def __len__(self):
        return len(self.storage)


def _transition(i: int):
    g = torch.Generator().manual_seed(i)
    return {
        "obs": torch.randn(4, generator=g),
        "actions": torch.tensor(i % 2),
        "rewards": torch.tensor(float(i)),
        "next_obs": torch.randn(4, generator=g),
        "dones": torch.tensor(1.0 if i % 17 == 0 else 0.0),
    }


def _fill(buf, n):
    for i in range(n):
        buf.add(_transition(i))


@pytest.mark.parametrize(
    "capacity,n,wraps",
    [
        (100_000, 3000, False),  # normal offline fill
        (500, 2000, True),  # ring wraparound (online off-policy)
        (64, 64, False),  # exactly full, no wrap
    ],
)
def test_sample_is_identical_to_the_deque_buffer(capacity, n, wraps):
    dev = torch.device("cpu")
    old, new = _DequeBuffer(capacity, dev), ReplayBuffer(capacity, dev)
    _fill(old, n)
    _fill(new, n)
    assert len(old) == len(new) == min(capacity, n)
    assert (n > capacity) is wraps

    for seed in range(6):
        random.seed(seed)
        a = old.sample(32)
        random.seed(seed)
        b = new.sample(32)
        assert list(a.keys()) == list(b.keys())  # key order preserved
        for k in a:
            assert torch.equal(a[k], b[k]), (capacity, n, seed, k)


def test_storage_view_matches_deque_ordering_including_after_wraparound():
    dev = torch.device("cpu")
    old, new = _DequeBuffer(300, dev), ReplayBuffer(300, dev)
    _fill(old, 1000)
    _fill(new, 1000)
    # Logical index 0 is the OLDEST live transition in both.
    for i in (0, 1, 150, 299, -1):
        assert torch.equal(old.storage[i]["rewards"], new.storage[i]["rewards"]), i
    assert len(list(new.storage)) == 300


def test_gather_returns_the_requested_rows():
    dev = torch.device("cpu")
    buf = ReplayBuffer(10_000, dev)
    _fill(buf, 500)
    idx = [0, 7, 499, 42]
    rows = buf.gather(idx)
    assert rows["obs"].shape == (4, 4)
    for j, i in enumerate(idx):
        assert torch.equal(rows["rewards"][j], buf.storage[i]["rewards"])


def test_growth_preserves_contents_past_the_initial_allocation():
    dev = torch.device("cpu")
    buf = ReplayBuffer(50_000, dev)
    n = ReplayBuffer._INITIAL_ROWS * 3 + 7  # forces several geometric grows
    _fill(buf, n)
    assert len(buf) == n
    for i in (0, 1, ReplayBuffer._INITIAL_ROWS, n - 1):
        assert float(buf.storage[i]["rewards"]) == float(i)


def test_empty_buffer_reports_zero_length():
    assert len(ReplayBuffer(10, torch.device("cpu"))) == 0


# --------------------------------------------------------------------------
# Thread policy (the oversubscription precaution)
# --------------------------------------------------------------------------
def test_configure_intraop_threads_caps_and_respects_override(monkeypatch):
    from src.config import threads as th

    monkeypatch.delenv("BCRL_NUM_THREADS", raising=False)
    before = torch.get_num_threads()
    try:
        applied = th.configure_intraop_threads(2)
        assert applied == min(2, torch.get_num_threads() or 2)
        assert torch.get_num_threads() == applied
        # Never raises an already-lower pool.
        assert th.configure_intraop_threads(64) == applied
        # Env override wins.
        monkeypatch.setenv("BCRL_NUM_THREADS", "1")
        assert th.configure_intraop_threads() == 1
        # Never below 1, never above core count.
        monkeypatch.setenv("BCRL_NUM_THREADS", "0")
        assert th.configure_intraop_threads() == 1
    finally:
        torch.set_num_threads(before)
