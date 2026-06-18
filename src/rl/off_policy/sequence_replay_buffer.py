from __future__ import annotations

import random
from collections import deque
from typing import Deque, Dict, List, Optional

import torch


class _Episode:
    """One contiguous within-episode run of transitions for a single env."""

    __slots__ = ("env_id", "transitions")

    def __init__(self, env_id: int) -> None:
        self.env_id = env_id
        self.transitions: List[Dict[str, torch.Tensor]] = []

    def __len__(self) -> int:
        return len(self.transitions)


class SequenceReplayBuffer:
    """Episode-aware replay buffer for recurrent off-policy training.

    Unlike ``ReplayBuffer`` (a flat deque of independent single transitions that
    ``sample`` returns as shuffled singletons), this buffer preserves per-env
    episode structure so contiguous within-episode sequences can be sampled —
    the data shape recurrent learners (DQN-LSTM / SAC-LSTM, PR-1C2) need. The
    flat ``ReplayBuffer`` is left untouched; this is a parallel class.

    Storage model:
      * A single global ``deque`` of episodes in creation order (the eviction
        order). Each episode is a contiguous run of transitions from ONE env.
      * Per-env "open episode" pointers: ``add(env_id, t)`` appends to that env's
        open episode (creating one if needed); ``mark_episode_end(env_id)`` closes
        it so the next ``add`` for that env starts a fresh episode.

    Sampling (``sample_sequences(batch_size, seq_len)``):
      * Eligible episodes are those with ``len >= seq_len``. Each sampled sequence
        is a length-``seq_len`` slice from a uniformly-chosen eligible episode at a
        uniformly-chosen start offset — so a sequence NEVER spans an episode
        boundary (the hidden-state reconstruction stays valid).
      * Returns a dict of ``(B, T, *feat)`` tensors on ``device``.
      * **Short episodes (< seq_len) are skipped** (v1 semantic: simpler and
        well-defined; no zero-padding). If NO episode is long enough, raises
        ``ValueError`` (callers gate on ``len()``/warmup before sampling).

    Eviction: when the total transition count exceeds ``capacity``, whole oldest
    episodes are evicted (never partial episodes — episode integrity is
    preserved). Evicting an env's still-open episode also clears that env's open
    pointer so the next ``add`` starts cleanly.

    Hidden state: none is stored here. Recurrent learners zero-init at sequence
    start (reusing PR #49's episode_starts reset). Stored-state + R2D2 burn-in is
    a deferred future option that would extend this schema.
    """

    def __init__(self, capacity: int, device: torch.device) -> None:
        self.capacity = int(capacity)
        self.device = device
        self.episodes: Deque[_Episode] = deque()
        self._open: Dict[int, Optional[_Episode]] = {}
        self._count = 0

    def add(self, env_id: int, transition: Dict[str, torch.Tensor]) -> None:
        """Append a transition to ``env_id``'s open episode (starting one if the
        previous was closed/evicted). Transitions are detached to CPU, matching
        ``ReplayBuffer.add``."""
        ep = self._open.get(env_id)
        if ep is None:
            ep = _Episode(env_id)
            self.episodes.append(ep)
            self._open[env_id] = ep
        ep.transitions.append({k: v.detach().cpu() for k, v in transition.items()})
        self._count += 1
        self._evict_if_needed()

    def mark_episode_end(self, env_id: int) -> None:
        """Close ``env_id``'s open episode; the next ``add`` starts a fresh one."""
        self._open[env_id] = None

    def _evict_if_needed(self) -> None:
        while self._count > self.capacity and self.episodes:
            oldest = self.episodes.popleft()
            self._count -= len(oldest)
            # If we just evicted an env's still-open episode, clear the pointer so
            # the next add for that env starts a new (non-dangling) episode.
            if self._open.get(oldest.env_id) is oldest:
                self._open[oldest.env_id] = None

    def _eligible(self, seq_len: int) -> List[_Episode]:
        return [ep for ep in self.episodes if len(ep) >= seq_len]

    def can_sample(self, seq_len: int) -> bool:
        """True iff at least one episode is long enough for ``seq_len``."""
        return any(len(ep) >= seq_len for ep in self.episodes)

    def sample_sequences(
        self, batch_size: int, seq_len: int
    ) -> Dict[str, torch.Tensor]:
        """Sample ``batch_size`` contiguous within-episode sequences of length
        ``seq_len``. Returns ``{key: (B, T, *feat) tensor}`` on ``device``.

        Raises ``ValueError`` if no episode is long enough (skip-short semantic;
        callers should gate on ``can_sample``/warmup first)."""
        eligible = self._eligible(seq_len)
        if not eligible:
            raise ValueError(
                f"SequenceReplayBuffer: no episode of length >= seq_len={seq_len} "
                f"(longest is {max((len(e) for e in self.episodes), default=0)}). "
                "Collect longer episodes or reduce seq_len."
            )
        seqs: List[List[Dict[str, torch.Tensor]]] = []
        for _ in range(batch_size):
            ep = random.choice(eligible)
            start = random.randint(0, len(ep) - seq_len)
            seqs.append(ep.transitions[start : start + seq_len])

        keys = seqs[0][0].keys()
        out: Dict[str, torch.Tensor] = {}
        for k in keys:
            # (B, T, *feat): stack T within each sequence, then B across sequences.
            per_seq = [torch.stack([tr[k] for tr in seq]) for seq in seqs]
            out[k] = torch.stack(per_seq).to(self.device)
        return out

    def __len__(self) -> int:
        """Total stored transition count (so warmup gates work unchanged)."""
        return self._count
