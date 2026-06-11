from __future__ import annotations

from typing import Sequence

import torch.nn as nn

from src.rl.nets.mlp import MLP


def select_backbone(
    obs_shape: Sequence[int],
    input_dim: int,
    output_dim: int,
    hidden_dims: tuple[int, ...] = (64, 64),
) -> nn.Module:
    """Pick a feature backbone by observation rank (shared seam for PR6).

    This is the single rank -> backbone mapping; PR6 fills the CNN branch once
    and both the auxiliary models here and the algorithm builders adopt it.

      * rank 1 (flat vector obs)  -> ``MLP`` over ``input_dim`` features
        (``input_dim`` already includes any concatenated action encoding).
      * rank 3 (image obs)        -> CNN, deferred to PR6.
      * rank 2 (sequence obs)     -> RNN/GRU, a documented future seam.

    Today the pipeline flattens observations to rank 1 before the builder, so
    the MLP branch is the live path; the others raise with a clear pointer.
    """
    rank = len(obs_shape)
    if rank == 1:
        return MLP(input_dim, output_dim, hidden_dims=hidden_dims)
    if rank == 2:
        # Seam: sequence observations -> RNN/GRU backbone (future work).
        raise NotImplementedError(
            "RNN/GRU backbone for rank-2 (sequence) observations is a documented "
            "seam; not implemented."
        )
    if rank == 3:
        raise NotImplementedError("CNN backbone for rank-3 (image) observations: PR6.")
    raise NotImplementedError(
        f"No backbone for observation rank {rank} (shape {tuple(obs_shape)})."
    )
