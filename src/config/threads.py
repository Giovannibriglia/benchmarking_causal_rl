"""Intra-op thread policy for this repo's workloads.

PyTorch defaults its CPU intra-op pool to the machine's core count. That is a
good default for large tensors and a BAD one here: the offline learner works on
128 x (4..192) batches through small MLPs, where the thread-pool barrier costs
far more than the arithmetic it parallelizes. Measured on a 20-core host
(us per update, CartPole offline learners, batch 128):

    threads      offline_dqn    cql      iql      bcq
    14 (default)      1061     1455    14681     3497
     4                 885     1162     2979     2117
     1                 874     1008     2481     1731

IQL, with the most ops per update, degrades 5.9x purely from oversubscription —
this is what made it look like "CUDA is faster than CPU for IQL" when the real
story is that every one of these algorithms is faster on CPU once the thread
count is sane (CUDA: 2480 / 3149 / 5307 / 3329 us for the same four).

Safety: capping intra-op threads does not change results for the paths this
repo runs. Production training runs on CUDA, where CPU thread count does not
touch kernel numerics, and the CPU-side tensor work here is indexing and copies
(no parallel reductions, so no summation-order change). It is still applied
explicitly rather than silently — override with ``BCRL_NUM_THREADS``.
"""

from __future__ import annotations

import os

import torch

#: Default ceiling. 1 measured fastest for the offline learner, but 4 keeps
#: headroom for genuinely parallel CPU work (image trunks, HDF5 decode) while
#: still avoiding the oversubscription cliff.
DEFAULT_MAX_INTRAOP_THREADS = 4


def configure_intraop_threads(max_threads: int | None = None) -> int:
    """Cap torch's intra-op thread pool; returns the value applied.

    ``BCRL_NUM_THREADS`` overrides (including upward). Never raises the count
    above the machine's core count, and never below 1. A pool already smaller
    than the cap — e.g. a sweep-supervisor child that inherited
    ``OMP_NUM_THREADS`` — is left alone.
    """
    env = os.environ.get("BCRL_NUM_THREADS")
    if env:
        try:
            max_threads = int(env)
        except ValueError:
            pass
    if max_threads is None:
        max_threads = DEFAULT_MAX_INTRAOP_THREADS
    cores = os.cpu_count() or 1
    target = max(1, min(int(max_threads), cores))
    if torch.get_num_threads() > target:
        torch.set_num_threads(target)
    return torch.get_num_threads()
