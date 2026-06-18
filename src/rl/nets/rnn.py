from __future__ import annotations

import torch
import torch.nn as nn


class RNN(nn.Module):
    """Recurrent trunk (vanilla Elman RNN, tanh) for vector observations.
    Maintains hidden state across sequential calls; designed to plug into the
    network pool alongside MLP.

    Forward signature differs from MLP: takes ``(obs, state)`` and returns
    ``(output, new_state)``. For non-recurrent callers, ``state`` can be None
    (initialized internally) and ``new_state`` can be ignored.

    Args:
        input_dim: dimension of input feature vector.
        output_dim: dimension of output feature vector (after final projection).
        hidden_dim: dimension of the RNN hidden state (default 128).
        num_layers: number of RNN layers (default 1).

    Notes on shape conventions:
        - Single-step forward: obs is ``(batch_size, input_dim)``. Returns
          ``(batch_size, output_dim)`` and new_state.
        - Sequence forward: obs is ``(batch_size, seq_len, input_dim)``. Returns
          ``(batch_size, seq_len, output_dim)`` and new_state from the last step.
        - State convention: RNN has a single hidden tensor ``h`` (not a tuple).
          Hidden state is initialized as zeros if state is None.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_dim: int = 128,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn = nn.RNN(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity="tanh",
        )
        self.proj = nn.Linear(hidden_dim, output_dim)

    def initial_state(self, batch_size: int, device=None):
        """Return zero-initialized ``h`` hidden state for a batch."""
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)

    def forward(self, obs, state=None):
        """Forward pass. obs shape is ``(batch, input_dim)`` for single-step or
        ``(batch, seq_len, input_dim)`` for sequence; returns
        ``(output, new_state)``."""
        if obs.dim() == 2:
            # Single step: insert seq dimension, then squeeze it out.
            obs = obs.unsqueeze(1)  # (batch, 1, input_dim)
            squeeze_seq = True
        else:
            squeeze_seq = False

        batch_size = obs.shape[0]
        if state is None:
            state = self.initial_state(batch_size, device=obs.device)

        out, new_state = self.rnn(obs, state)  # out: (batch, seq, hidden)
        out = self.proj(out)  # (batch, seq, output_dim)

        if squeeze_seq:
            out = out.squeeze(1)  # (batch, output_dim)

        return out, new_state
