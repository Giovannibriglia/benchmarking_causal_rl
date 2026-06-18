"""feat/on-policy-recurrent-integration — separate-trunk ActorCritic.

Unit tests for the new ActorCritic policy: separate actor/critic trunks
(per-component network), action_type-based head branching, the bare (stateless)
interface used by the MLP path, and the stateful interface used by recurrent PPO.
"""

from __future__ import annotations

import pytest
import torch
from src.rl.nets.lstm import LSTM
from src.rl.nets.mlp import MLP
from src.rl.on_policy.actor_critic import ActorCritic


def _ac(actor="mlp", critic="mlp", action_type="discrete", action_dim=3, obs_dim=5):
    return ActorCritic(
        (obs_dim,),
        obs_dim,
        action_dim,
        action_type,
        torch.device("cpu"),
        actor_network=actor,
        critic_network=critic,
    )


def test_separate_trunks_are_distinct_modules():
    ac = _ac()
    # Separate trunks: actor and critic do not share the encoder.
    assert ac.actor_trunk is not ac.critic_trunk
    assert isinstance(ac.actor_trunk.trunk, MLP) and isinstance(
        ac.critic_trunk.trunk, MLP
    )


def test_is_recurrent_flag():
    assert _ac("mlp", "mlp").is_recurrent is False
    assert _ac("lstm", "mlp").is_recurrent is True
    assert _ac("mlp", "gru").is_recurrent is True
    assert _ac("rnn", "rnn").is_recurrent is True


def test_recurrent_trunk_built_for_recurrent_component():
    ac = _ac("lstm", "mlp")
    assert isinstance(ac.actor_trunk.trunk, LSTM)
    assert isinstance(ac.critic_trunk.trunk, MLP)


@pytest.mark.parametrize("action_type,action_dim", [("discrete", 4), ("continuous", 2)])
def test_bare_interface_shapes_and_head_branching(action_type, action_dim):
    ac = _ac(action_type=action_type, action_dim=action_dim)
    obs = torch.randn(6, 5)
    dist = ac.distribution(obs)
    action, logp = ac.act(obs)
    value = ac.value(obs)
    assert value.shape == (6,)
    assert logp.shape == (6,)
    if action_type == "discrete":
        assert ac.log_std is None
        assert action.shape == (6,)
    else:
        assert ac.log_std is not None and ac.log_std.shape == (action_dim,)
        assert action.shape == (6, action_dim)


def test_mlp_state_is_none_passed_through():
    ac = _ac("mlp", "mlp")
    # MLP trunk adapter returns None state.
    assert ac.actor_trunk.initial_state(4) is None
    assert ac.initial_state(4) == {"actor": None, "critic": None}


def test_recurrent_state_threads_through_act_step():
    ac = _ac("lstm", "lstm")
    obs = torch.randn(4, 5)
    state = ac.initial_state(4)
    action, logp, value, new_state = ac.act_step(obs, state)
    assert action.shape == (4,) and value.shape == (4,)
    # LSTM state is an (h, c) tuple per component; it should advance from zeros.
    assert isinstance(new_state["actor"], tuple)
    assert not torch.equal(new_state["actor"][0], state["actor"][0])


def test_reset_state_where_zeros_selected_envs():
    ac = _ac("lstm", "gru")
    state = ac.initial_state(4)
    obs = torch.randn(4, 5)
    _, _, _, new_state = ac.act_step(obs, state)
    mask = torch.tensor([True, False, True, False])
    reset = ac.reset_state_where(new_state, mask)
    # actor is LSTM (h, c) tuple; critic is GRU (bare h). Reset envs -> zeros.
    assert torch.all(reset["actor"][0][:, mask, :] == 0)
    assert torch.all(reset["critic"][:, mask, :] == 0)
    # Non-reset envs keep their (non-zero) advanced state.
    assert not torch.all(reset["actor"][0][:, ~mask, :] == 0)


def test_evaluate_sequence_shapes_and_bptt_gradient():
    ac = _ac("lstm", "lstm", action_dim=3)
    T, N = 5, 4
    obs_seq = torch.randn(T, N, 5)
    actions = torch.randint(0, 3, (T, N))
    episode_starts = torch.zeros(T, N)
    episode_starts[0] = 1.0
    logp, values, entropy = ac.evaluate_sequence(
        obs_seq, actions, episode_starts, ac.initial_state(N)
    )
    assert logp.shape == (T, N) and values.shape == (T, N) and entropy.shape == (T, N)
    (logp.sum() + values.sum()).backward()
    # Gradient reaches the recurrent cell weights (the BPTT).
    grad = ac.actor_trunk.trunk.lstm.weight_hh_l0.grad
    assert grad is not None and grad.abs().sum() > 0
