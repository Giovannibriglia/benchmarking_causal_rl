from __future__ import annotations

from typing import Dict, Iterable, Iterator, List

import torch


class _TransitionView:
    """Read-only per-transition view of a ``ReplayBuffer``.

    Backward compatibility for call sites that expect ``buffer.storage`` to be a
    sequence of ``{key: tensor}`` dicts (the pre-tensorization layout). Rows are
    materialized on demand, so indexing one transition costs one dict build
    instead of keeping 48k dicts alive.
    """

    __slots__ = ("_buf",)

    def __init__(self, buf: "ReplayBuffer") -> None:
        self._buf = buf

    def __len__(self) -> int:
        return len(self._buf)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        n = len(self._buf)
        if i < 0:
            i += n
        if not 0 <= i < n:
            raise IndexError(i)
        p = self._buf._physical(i)
        return {k: v[p] for k, v in self._buf._data.items()}

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        for i in range(len(self._buf)):
            yield self[i]


class ReplayBuffer:
    """Flat transition buffer backed by contiguous per-key tensors.

    Replaces a ``deque`` of per-transition dicts. ``sample`` used to spend its
    time in ``torch.stack`` over ``batch_size`` one-element tensors per key
    (~277 us) plus an O(n) deque gather (~110 us); indexing one contiguous
    tensor per key measures **8.4x faster** end to end
    (docs/offline_training_profile.md), which is ~20-25% of a whole offline run.

    BYTE-IDENTICAL to the deque version, by construction:

      * ``sample`` still draws with ``random.sample(range(n), k)``. CPython's
        ``random.sample`` selects INDICES internally and maps them onto the
        population, so this consumes exactly the RNG the old
        ``random.sample(deque, k)`` did and selects the same transitions in the
        same order (verified elementwise).
      * Ordering matches deque semantics, including after wraparound: logical
        index 0 is the OLDEST live transition. Once the buffer is full the write
        cursor ``_pos`` points at the oldest slot, so logical ``i`` maps to
        physical ``(i + _pos) % capacity``.
      * Batch dict key order follows the first added transition, as before.

    Storage grows geometrically (1.5x) up to ``capacity`` rather than
    preallocating it: capacity is 1e6 across the offline builders while real
    fills are far smaller, and preallocating image observations at that size
    would be gigabytes.
    """

    _INITIAL_ROWS = 1024
    _GROWTH = 1.5

    def __init__(self, capacity: int, device: torch.device) -> None:
        self.capacity = int(capacity)
        self.device = device
        self._data: Dict[str, torch.Tensor] = {}
        self._rows = 0  # allocated rows (<= capacity)
        self._size = 0  # live transitions
        self._pos = 0  # next write slot

    # -- internals ---------------------------------------------------------
    def _allocate(self, transition: Dict[str, torch.Tensor], rows: int) -> None:
        self._data = {
            k: torch.empty((rows, *v.shape), dtype=v.dtype)
            for k, v in transition.items()
        }
        self._rows = rows

    def _grow(self) -> None:
        rows = min(self.capacity, max(self._rows + 1, int(self._rows * self._GROWTH)))
        for k, v in self._data.items():
            bigger = torch.empty((rows, *v.shape[1:]), dtype=v.dtype)
            bigger[: self._rows] = v
            self._data[k] = bigger
        self._rows = rows

    def _physical(self, logical):
        """Logical (oldest-first) index -> physical row. Identity until the
        buffer wraps; afterwards the oldest row is the write cursor."""
        if self._size < self.capacity:
            return logical
        return (logical + self._pos) % self.capacity

    # -- API ---------------------------------------------------------------
    @property
    def storage(self) -> _TransitionView:
        """Per-transition sequence view (compatibility; see _TransitionView)."""
        return _TransitionView(self)

    def add(self, transition: Dict[str, torch.Tensor]) -> None:
        cpu = {k: v.detach().cpu() for k, v in transition.items()}
        if not self._data:
            self._allocate(cpu, min(self._INITIAL_ROWS, self.capacity))
        if self._pos >= self._rows:
            self._grow()
        for k, v in cpu.items():
            self._data[k][self._pos] = v
        self._pos += 1
        if self._pos >= self.capacity:
            self._pos = 0
        self._size = min(self._size + 1, self.capacity)

    def gather(self, indices: Iterable[int]) -> Dict[str, torch.Tensor]:
        """Rows at the given LOGICAL indices, as stacked CPU tensors."""
        ix = self._physical(torch.as_tensor(list(indices), dtype=torch.long))
        return {k: v[ix] for k, v in self._data.items()}

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        import random

        idx: List[int] = random.sample(range(self._size), batch_size)
        ix = self._physical(torch.as_tensor(idx, dtype=torch.long))
        return {k: v[ix].to(self.device) for k, v in self._data.items()}

    def __len__(self) -> int:
        return self._size
