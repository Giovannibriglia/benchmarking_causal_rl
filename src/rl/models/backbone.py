from __future__ import annotations

from typing import Sequence

import torch.nn as nn

from src.rl.nets.mlp import MLP


def select_backbone(
    obs_shape: Sequence[int],
    input_dim: int,
    output_dim: int,
    **mlp_kwargs,
) -> nn.Module:
    """Pick a feature backbone by observation rank (shared seam for the image stack).

    Single rank -> backbone mapping, consumed by both the auxiliary models and
    the algorithm builders:

      * rank 1 (flat vector obs)  -> ``MLP`` over ``input_dim`` features.
      * rank 3 (image obs)        -> CNN, filled in PR6 Stage B.
      * rank 2 (sequence obs)     -> RNN/GRU, a documented future seam.

    The rank-1 branch forwards ``**mlp_kwargs`` VERBATIM to ``MLP`` and lets
    ``MLP`` own every default (``hidden_dims``, ``activation``,
    ``output_activation``). This is deliberate: ``select_backbone((d,), d, o,
    **kw)`` is then structurally indistinguishable from ``MLP(d, o, **kw)``, so
    routing an existing ``MLP(...)`` construction through the selector is a
    bitwise no-op (same module, same nn.Linear init order, same RNG draws).
    Redeclaring MLP's defaults here would be the one way to silently diverge.
    """
    rank = len(obs_shape)
    if rank == 1:
        return MLP(input_dim, output_dim, **mlp_kwargs)
    if rank == 2:
        # Seam: sequence observations -> RNN/GRU backbone (future work).
        raise NotImplementedError(
            "RNN/GRU backbone for rank-2 (sequence) observations is a documented "
            "seam; not implemented."
        )
    if rank == 3:
        raise NotImplementedError(
            "CNN backbone for rank-3 (image) observations: PR6 Stage B."
        )
    raise NotImplementedError(
        f"No backbone for observation rank {rank} (shape {tuple(obs_shape)})."
    )
