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
        # Image obs -> Nature-CNN. The CNN derives channels/spatial dims from
        # obs_shape and carries its own 512 feature width, so input_dim and the
        # MLP-only **mlp_kwargs (hidden_dims/activation/output_activation) are
        # vestigial here; only output_dim (the head) is used.
        from src.rl.nets.cnn import NatureCNN

        return NatureCNN(tuple(obs_shape), output_dim)
    raise NotImplementedError(
        f"No backbone for observation rank {rank} (shape {tuple(obs_shape)})."
    )


def build_trunk(
    network: str,
    obs_shape: Sequence[int],
    input_dim: int,
    output_dim: int,
    **kwargs,
) -> nn.Module:
    """Build a trunk network of the specified type.

    The ``network`` parameter selects the architecture ORTHOGONALLY to
    ``obs_shape``'s rank (which ``select_backbone`` uses to dispatch
    image-vs-vector). For rank-1 vector observations, network can be:
      - ``"mlp"`` (default; uses the existing MLP via ``select_backbone`` for
        backward compat).
      - ``"lstm"`` / ``"gru"`` / ``"rnn"`` (recurrent trunks).

    For non-rank-1 obs shapes, recurrent ``network`` types raise — CNN-vs-recurrent
    or higher-rank-recurrent combinations are out of scope. ``network="mlp"``
    keeps ``select_backbone``'s full rank dispatch (so image obs still route to
    the CNN).

    Returns the trunk module. Callers that need hidden state should check via
    ``isinstance(trunk, (LSTM, GRU, RNN))`` or ``hasattr(trunk, "initial_state")``.
    """
    if network == "mlp":
        # Delegate to select_backbone for byte-identical MLP backward compat.
        return select_backbone(obs_shape, input_dim, output_dim, **kwargs)

    if len(obs_shape) != 1:
        raise ValueError(
            f"Recurrent trunks (network={network!r}) require rank-1 vector "
            f"observations; got obs_shape={tuple(obs_shape)}."
        )

    if network == "lstm":
        from src.rl.nets.lstm import LSTM

        return LSTM(input_dim, output_dim, **kwargs)
    if network == "gru":
        from src.rl.nets.gru import GRU

        return GRU(input_dim, output_dim, **kwargs)
    if network == "rnn":
        from src.rl.nets.rnn import RNN

        return RNN(input_dim, output_dim, **kwargs)
    raise ValueError(
        f"Unknown network type: {network!r}. Supported: mlp, lstm, gru, rnn."
    )
