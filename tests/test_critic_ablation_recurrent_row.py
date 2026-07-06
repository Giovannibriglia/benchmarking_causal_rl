"""Critic-ablation RNN row (encoder=lstm) — the Cell-8 recurrent strategy triad.

The MLP row (encoder=mlp) is the Cell-7 confounded-only ablation; this row swaps
the encoder to a recurrent trunk so the SAME triad {observational, proximal,
oracle_u} estimates on a confounded+MASKED (POMDP) stream. The wiring is purely
additive: ``_build_strategy_critic`` gains an ``encoder`` argument that forks the
three merged cell-8 recurrent builders; the MLP branch stays byte-frozen.

Covered here:
  * encoder=lstm builds the recurrent triad — all three fire ``is_recurrent`` and
    consume (B,T) windows natively (the batched q_su, no per-timestep loop);
  * encoder=mlp is unchanged (MLP / UMarginalizedQ, is_recurrent False);
  * five-keys — rnn-proximal's update is INVARIANT to the realized confounder_u
    (its m_step overwrites it with the E-step sample), rnn-oracle_u's is not;
  * the estimation-vs-oracle scoring path (predict_q_adj) unwraps the recurrent
    (q_all, state) tuple to a flat (N, A) tensor.
"""

from __future__ import annotations

import pytest
import torch
from src.benchmarking.critic_ablation import (
    _build_strategy_critic,
    CRITIC_LIBRARY,
    StrategyCritic,
)
from src.rl.offline.oracle_u import RecurrentUMarginalizedQ, UMarginalizedQ

_CPU = torch.device("cpu")
_OBS, _ACT, _B, _T = 4, 2, 8, 6
_TRIAD = ("observational", "proximal", "oracle_u")


def _window(u: torch.Tensor, gen: torch.Generator) -> dict:
    return {
        "obs": torch.randn(_B, _T, _OBS, generator=gen),
        "next_obs": torch.randn(_B, _T, _OBS, generator=gen),
        "actions": torch.randint(0, _ACT, (_B, _T), generator=gen),
        "rewards": torch.rand(_B, _T, generator=gen),
        "dones": torch.zeros(_B, _T),
        "confounder_u": u,
        "r_tau": torch.rand(_B, _T, generator=gen),
    }


def _critic(name: str, encoder: str) -> StrategyCritic:
    base = "offline_dqn_recurrent" if encoder != "mlp" else "offline_dqn"
    return StrategyCritic(name, CRITIC_LIBRARY[name], base, _OBS, _ACT, _CPU, encoder)


@pytest.mark.parametrize("name", _TRIAD)
def test_recurrent_triad_is_recurrent(name):
    """encoder=lstm -> every critic fires is_recurrent and consumes (B,T)."""
    c = _critic(name, "lstm")
    assert c.is_recurrent is True
    assert c.consumes_sequences is True


@pytest.mark.parametrize("name", _TRIAD)
def test_mlp_row_unchanged(name):
    """encoder=mlp -> the byte-frozen Cell-7 arm (MLP / UMarginalizedQ, flat)."""
    c = _critic(name, "mlp")
    assert c.is_recurrent is False
    if name == "observational":
        assert type(c.net).__name__ == "MLP"
        assert not hasattr(c.net, "q_su")
        assert c.consumes_sequences is False
    else:
        assert isinstance(c.net, UMarginalizedQ)
        # proximal consumes (B,T) (its m_step flattens); oracle is flat.
        assert c.consumes_sequences is (name == "proximal")


def test_recurrent_net_types():
    """observational floor = plain recurrent trunk (NO q_su); proximal/oracle =
    RecurrentUMarginalizedQ (the U-conditioned recurrent critic)."""
    obs_net = _critic("observational", "lstm").net
    assert not hasattr(obs_net, "q_su")  # observational floor: no U-conditioning
    assert obs_net.__class__.__name__ == "LSTM"
    for name in ("proximal", "oracle_u"):
        assert isinstance(_critic(name, "lstm").net, RecurrentUMarginalizedQ)


@pytest.mark.parametrize("name", _TRIAD)
def test_recurrent_critic_trains_on_bt_window(name):
    """All three recurrent critics consume a (B,T) window natively (no flatten)
    and return a finite loss."""
    c = _critic(name, "lstm")
    gen = torch.Generator().manual_seed(0)
    metrics = c.update(_window(torch.bernoulli(torch.full((_B, _T), 0.5)), gen))
    loss = metrics.get("loss", metrics.get("q_loss"))
    assert loss is not None and torch.isfinite(torch.tensor(loss))


def test_predict_q_adj_unwraps_recurrent_tuple():
    """The recurrent forward returns (q_all, state); the flat (N, obs_dim) eval
    set must score to a bare (N, A) tensor for estimation-vs-oracle."""
    obs_e = torch.randn(20, _OBS)
    for name in _TRIAD:
        q = _critic(name, "lstm").predict_q_adj(obs_e)
        assert isinstance(q, torch.Tensor)
        assert q.shape == (20, _ACT)


@pytest.mark.parametrize("name,invariant", [("proximal", True), ("oracle_u", False)])
def test_five_keys_recurrent(name, invariant):
    """Five-keys: rnn-proximal's update is INVARIANT to the realized confounder_u
    (m_step overwrites it with the inferred sample); rnn-oracle_u legitimately
    reads it (its update changes with U)."""
    losses = []
    for realized in (torch.zeros(_B, _T), torch.ones(_B, _T)):
        torch.manual_seed(123)
        gen = torch.Generator().manual_seed(7)
        c = _critic(name, "lstm")
        gen = torch.Generator().manual_seed(7)
        w = _window(realized, gen)
        torch.manual_seed(999)
        m = c.update(w)
        losses.append(m.get("loss", m.get("q_loss")))
    assert (abs(losses[0] - losses[1]) < 1e-9) is invariant


def test_recurrent_ablation_dqn_base_only():
    """The recurrent proximal/oracle builders are DQN-base only; a non-dqn base
    with a recurrent encoder is rejected (no recurrent cql/iql/bcq)."""
    with pytest.raises(ValueError, match="DQN-base only"):
        _build_strategy_critic("proximal", "cql", _OBS, _ACT, _CPU, "lstm")
