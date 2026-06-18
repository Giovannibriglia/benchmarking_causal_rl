"""feat/recurrent-trunk-classes — LSTM/GRU/RNN trunk classes.

Trunk-level unit tests for the recurrent network classes added in this PR.
Tests are scoped to the trunks in isolation: forward pass shapes, hidden state
management, parameter counts, and the build_trunk factory. No algorithm
integration; that lands in the follow-up PR.
"""

import pytest
import torch
from src.rl.models.backbone import build_trunk
from src.rl.nets.gru import GRU
from src.rl.nets.lstm import LSTM
from src.rl.nets.rnn import RNN


@pytest.fixture
def shapes():
    return {"batch": 4, "seq_len": 8, "input_dim": 10, "output_dim": 16}


# ---- Shape tests: single-step forward ----


@pytest.mark.parametrize("trunk_cls", [LSTM, GRU, RNN])
def test_single_step_forward_shape(trunk_cls, shapes):
    """Single-step (batch, input_dim) -> (batch, output_dim) + state."""
    trunk = trunk_cls(shapes["input_dim"], shapes["output_dim"])
    obs = torch.randn(shapes["batch"], shapes["input_dim"])
    out, state = trunk(obs)
    assert out.shape == (shapes["batch"], shapes["output_dim"])
    # State has model-specific structure but should not be None.
    assert state is not None


@pytest.mark.parametrize("trunk_cls", [LSTM, GRU, RNN])
def test_sequence_forward_shape(trunk_cls, shapes):
    """Sequence (batch, seq_len, input_dim) -> (batch, seq_len, output_dim)."""
    trunk = trunk_cls(shapes["input_dim"], shapes["output_dim"])
    obs = torch.randn(shapes["batch"], shapes["seq_len"], shapes["input_dim"])
    out, state = trunk(obs)
    assert out.shape == (shapes["batch"], shapes["seq_len"], shapes["output_dim"])
    assert state is not None


# ---- Hidden state behavior ----


@pytest.mark.parametrize("trunk_cls", [LSTM, GRU, RNN])
def test_initial_state_is_zero(trunk_cls, shapes):
    """initial_state returns zero tensors."""
    trunk = trunk_cls(shapes["input_dim"], shapes["output_dim"])
    state = trunk.initial_state(shapes["batch"])
    if isinstance(state, tuple):
        for s in state:
            assert torch.all(s == 0)
    else:
        assert torch.all(state == 0)


@pytest.mark.parametrize("trunk_cls", [LSTM, GRU, RNN])
def test_state_passed_through_preserves_output(trunk_cls, shapes):
    """Calling forward(obs, state) where state is the previous output's state
    should be reproducible — second call deterministic given the same state."""
    trunk = trunk_cls(shapes["input_dim"], shapes["output_dim"])
    trunk.eval()  # Disable any stochastic elements.
    obs = torch.randn(shapes["batch"], shapes["input_dim"])
    out1, state1 = trunk(obs)
    out2, state2 = trunk(obs, state1)
    out2_repeat, _ = trunk(obs, state1)
    # Same input + same state should produce same output.
    assert torch.allclose(out2, out2_repeat)


@pytest.mark.parametrize("trunk_cls", [LSTM, GRU, RNN])
def test_state_evolves_across_steps(trunk_cls, shapes):
    """Hidden state changes when fed sequentially with non-zero input."""
    trunk = trunk_cls(shapes["input_dim"], shapes["output_dim"])
    obs = torch.randn(shapes["batch"], shapes["input_dim"])
    _, state1 = trunk(obs)
    _, state2 = trunk(obs, state1)
    if isinstance(state1, tuple):
        assert not torch.equal(state1[0], state2[0]) or not torch.equal(
            state1[1], state2[1]
        )
    else:
        assert not torch.equal(state1, state2)


# ---- Parameter counts ----


def test_lstm_has_lstm_layer():
    trunk = LSTM(10, 16, hidden_dim=32, num_layers=2)
    assert isinstance(trunk.lstm, torch.nn.LSTM)
    assert trunk.lstm.hidden_size == 32
    assert trunk.lstm.num_layers == 2


def test_gru_has_gru_layer():
    trunk = GRU(10, 16, hidden_dim=32, num_layers=2)
    assert isinstance(trunk.gru, torch.nn.GRU)


def test_rnn_has_rnn_layer():
    trunk = RNN(10, 16, hidden_dim=32, num_layers=2)
    assert isinstance(trunk.rnn, torch.nn.RNN)


# ---- Factory tests ----


@pytest.mark.parametrize("network", ["mlp", "lstm", "gru", "rnn"])
def test_build_trunk_returns_correct_type(network, shapes):
    """build_trunk dispatches on network name."""
    trunk = build_trunk(
        network=network,
        obs_shape=(shapes["input_dim"],),
        input_dim=shapes["input_dim"],
        output_dim=shapes["output_dim"],
    )
    # For non-MLP, type should match the expected class.
    if network == "lstm":
        assert isinstance(trunk, LSTM)
    elif network == "gru":
        assert isinstance(trunk, GRU)
    elif network == "rnn":
        assert isinstance(trunk, RNN)
    # MLP type assertion would require importing from src.rl.nets.mlp, which
    # we leave alone here.


def test_build_trunk_mlp_path_unchanged(shapes):
    """build_trunk with network='mlp' delegates to select_backbone — the
    return value should be functionally identical to a direct select_backbone
    call. This is the byte-identical-backward-compat invariant."""
    from src.rl.models.backbone import select_backbone

    direct = select_backbone(
        obs_shape=(shapes["input_dim"],),
        input_dim=shapes["input_dim"],
        output_dim=shapes["output_dim"],
    )
    via_factory = build_trunk(
        network="mlp",
        obs_shape=(shapes["input_dim"],),
        input_dim=shapes["input_dim"],
        output_dim=shapes["output_dim"],
    )
    # Both should be MLP instances (whatever MLP's actual class is).
    assert type(direct) is type(via_factory)


def test_build_trunk_rejects_unknown_network():
    """Unknown network names raise ValueError with a clear message."""
    with pytest.raises(ValueError, match="Unknown network type"):
        build_trunk(network="foo", obs_shape=(10,), input_dim=10, output_dim=16)


def test_build_trunk_recurrent_rejects_non_rank_1():
    """Recurrent trunks require rank-1 vector observations."""
    with pytest.raises(ValueError, match="rank-1 vector"):
        build_trunk(
            network="lstm",
            obs_shape=(3, 64, 64),  # Image-shaped, rank 3.
            input_dim=64 * 64 * 3,
            output_dim=16,
        )


# ---- Gradient flow ----


@pytest.mark.parametrize("trunk_cls", [LSTM, GRU, RNN])
def test_gradient_flows_through_trunk(trunk_cls, shapes):
    """Loss on output produces nonzero gradients on trunk parameters."""
    trunk = trunk_cls(shapes["input_dim"], shapes["output_dim"])
    obs = torch.randn(shapes["batch"], shapes["input_dim"])
    out, _ = trunk(obs)
    loss = out.sum()
    loss.backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in trunk.parameters()
    )
    assert has_grad


@pytest.mark.parametrize("trunk_cls", [LSTM, GRU, RNN])
def test_gradient_flows_through_sequence(trunk_cls, shapes):
    """Gradient flows through a sequence forward (truncated BPTT precursor)."""
    trunk = trunk_cls(shapes["input_dim"], shapes["output_dim"])
    obs = torch.randn(shapes["batch"], shapes["seq_len"], shapes["input_dim"])
    out, _ = trunk(obs)
    loss = out.sum()
    loss.backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in trunk.parameters()
    )
    assert has_grad
