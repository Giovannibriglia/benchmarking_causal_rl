"""Strategy-collapse parity A/B (PR-1, hard gate 1).

The four merged OracleU* subclasses were re-expressed as (base_estimator x
OracleU strategy). This proves the re-expression is LOSSLESS: for each of the
four variants, the OLD subclass (frozen verbatim from 6a5bf0f in
tests/_oracle_u_reference.py) and the NEW (base + OracleU()) agent — started from
bitwise-identical weights, fed identical batches — produce torch.equal learn()
metrics, q_su(obs,u), Q_adj(obs), and act(obs) over N steps. IQL is mandatory
(its marginalized advantage is the likeliest drift). Not approximate: torch.equal.
"""

from __future__ import annotations

import warnings

import pytest
import torch
from src.benchmarking.registry import register_default_algorithms, registry
from src.rl.models.backbone import select_backbone
from src.rl.off_policy.replay_buffer import ReplayBuffer
from src.rl.offline.oracle_u import UMarginalizedQ
from tests._oracle_u_reference import OracleUBCQ, OracleUCQL, OracleUDQN, OracleUIQL

warnings.filterwarnings("ignore")

_CPU = torch.device("cpu")
_OBS, _ACT, _B = 4, 2, 16
_NET_ATTRS = ("q_network", "target_network", "value_net", "policy_net", "behavior_net")


def _buf():
    return ReplayBuffer(capacity=10_000, device=_CPU)


def _old_dqn():
    return OracleUDQN(
        UMarginalizedQ(_OBS, _ACT), UMarginalizedQ(_OBS, _ACT), _buf(), device=_CPU
    )


def _old_cql():
    return OracleUCQL(
        UMarginalizedQ(_OBS, _ACT), UMarginalizedQ(_OBS, _ACT), _buf(), device=_CPU
    )


def _old_iql():
    return OracleUIQL(
        select_backbone((_OBS,), _OBS, _ACT),
        UMarginalizedQ(_OBS, _ACT),
        UMarginalizedQ(_OBS, _ACT),
        select_backbone((_OBS,), _OBS, 1),
        _buf(),
        device=_CPU,
    )


def _old_bcq():
    return OracleUBCQ(
        UMarginalizedQ(_OBS, _ACT),
        UMarginalizedQ(_OBS, _ACT),
        select_backbone((_OBS,), _OBS, _ACT),
        _buf(),
        device=_CPU,
    )


_VARIANTS = {
    "offline_dqn": _old_dqn,
    "cql": _old_cql,
    "iql": _old_iql,
    "bcq": _old_bcq,
}


def _new(base: str):
    return registry.get(f"{base}_oracle_u").builder(
        obs_dim=_OBS,
        action_dim=_ACT,
        action_type="discrete",
        device=_CPU,
        action_space=None,
        obs_shape=(_OBS,),
    )[1]


def _sync(src, dst):
    """Copy every shared net's weights src -> dst so both start bitwise-identical
    (sidesteps any construction-RNG-order question)."""
    for attr in _NET_ATTRS:
        sm, dm = getattr(src, attr, None), getattr(dst, attr, None)
        if sm is not None and dm is not None:
            dm.load_state_dict(sm.state_dict())


def _batch(step: int):
    g = torch.Generator().manual_seed(100 + step)
    return {
        "obs": torch.randn(_B, _OBS, generator=g),
        "actions": torch.randint(0, _ACT, (_B,), generator=g),
        "rewards": torch.randn(_B, generator=g),
        "next_obs": torch.randn(_B, _OBS, generator=g),
        "dones": torch.zeros(_B),
        "confounder_u": torch.bernoulli(torch.full((_B,), 0.5), generator=g),
    }


@pytest.mark.parametrize("base", list(_VARIANTS))
def test_oracle_strategy_collapse_is_bitwise_parity(base):
    register_default_algorithms()
    new = _new(base)
    old = _VARIANTS[base]()
    _sync(new, old)  # old <- new weights: identical start

    eval_obs = torch.randn(7, _OBS)
    eval_u = torch.bernoulli(torch.full((7,), 0.5))

    for step in range(6):
        b = _batch(step)
        m_old = old.learn(b)
        m_new = new.learn(b)
        # 1. learn() return metrics, bitwise.
        assert set(m_old) == set(m_new)
        for k in m_old:
            assert m_old[k] == m_new[k], (base, step, k, m_old[k], m_new[k])
        # 2. U-conditioned row q_su(obs, u), bitwise.
        assert torch.equal(
            old.q_network.q_su(eval_obs, eval_u), new.q_network.q_su(eval_obs, eval_u)
        )
        # 3. deployed Q_adj = forward(obs), bitwise.
        assert torch.equal(old.q_network(eval_obs), new.q_network(eval_obs))
        # 4. greedy action, bitwise.
        assert torch.equal(
            old.act(eval_obs, deterministic=True).action,
            new.act(eval_obs, deterministic=True).action,
        )
