"""GRACE builders: base learner x ``Grace`` strategy x routed serving head.

Base-parity (the same requirement the observational floor and sensitivity
satisfy): every grace arm is the BASE ALGO'S OWN floor builder — same net
classes, same RNG draw order — with (1) the ``Grace`` strategy injected
(``critic_value -> net.q_obs`` — a literal pass-through to the trainable head,
so TRAINING is exactly the observational floor's), (2) the nets wrapped in
``GraceQNetwork`` (routed serving surface), and (3) the machinery installed on
the ``ProximalEM._install_*`` learn-wrapper seam.

The wrapped ``learn`` toggles ``route_enabled`` OFF for the duration of the
base update, so every net call a learner makes internally — including the
deliberately-unrouted plain target evals (e.g. ``iql.py``'s V-target) —
resolves to the plain trainable head. Serving (``act``, eval rollouts, the
ablation's ``predict_q_adj``) sees the routed estimand.

The CBN fit happens ONCE at the runner's ``set_sequence_buffer`` handoff
(offline datasets are static); online arms set ``online_refresh_interval`` so
the wrapped ``learn`` re-enters the fit on cadence (Gate-B label persistence:
the canonicalization re-runs every refresh).
"""

from __future__ import annotations

from typing import Optional

import torch

from src.rl.nets.mlp import MLP
from src.rl.off_policy.dqn import DQN
from src.rl.off_policy.identification import Grace
from src.rl.off_policy.replay_buffer import ReplayBuffer
from src.rl.offline.grace.cbn import GraceOptions
from src.rl.offline.grace.cell_graph import cell_graph, CellGraph
from src.rl.offline.grace.heads import GraceQNetwork, GraceRecurrentQNetwork
from src.rl.offline.grace.machinery import GraceMachinery

# Online arms refresh the CBN every this many optimiser steps (mirrors the
# online proximal's rolling-buffer cadence scale; offline arms never refresh).
_ONLINE_REFRESH_INTERVAL = 2000


def _options(kwargs) -> GraceOptions:
    """GraceOptions from builder kwargs. The cell YAML's ``grace:`` block (and
    the CriticSpec defaults) arrive as ``grace_options``; unknown keys are
    refused loudly."""
    raw = dict(kwargs.get("grace_options") or {})
    raw.pop("_env_id", None)
    valid = set(GraceOptions.__dataclass_fields__)
    unknown = set(raw) - valid
    if unknown:
        raise ValueError(f"unknown grace option(s): {sorted(unknown)}")
    return GraceOptions(**raw)


def _env_id(kwargs) -> Optional[str]:
    opts = kwargs.get("grace_options") or {}
    return opts.get("_env_id") or kwargs.get("env_id")


def _dims(kwargs):
    if kwargs.get("action_type", "discrete") != "discrete":
        raise NotImplementedError(
            "grace: discrete offline bases only (offline_dqn/bcq/cql/iql)."
        )
    return kwargs["obs_dim"], kwargs["action_dim"], kwargs["device"]


def _flatten_window(batch):
    return {
        k: (v.flatten(0, 1) if torch.is_tensor(v) and v.dim() >= 2 else v)
        for k, v in batch.items()
    }


def _install_grace(agent, machinery: GraceMachinery, nets) -> None:
    """The learn-wrapper seam (verbatim shape of ``_install_proximal_em``):
    route toggling + (B, T)-window flattening for flat bases + the online
    refresh cadence + the ``set_sequence_buffer`` handoff."""
    base_learn = agent.learn
    recurrent = bool(getattr(agent, "is_recurrent", False))

    def wrapped_learn(batch):
        machinery.tick_update()
        for net in nets:
            net.route_enabled = False
        try:
            if not recurrent:
                rew = batch.get("rewards")
                if torch.is_tensor(rew) and rew.dim() >= 2:
                    batch = _flatten_window(batch)
            return base_learn(batch)
        finally:
            for net in nets:
                net.route_enabled = True

    agent.learn = wrapped_learn
    agent.set_sequence_buffer = machinery.fit_from_buffer
    agent._grace_machinery = machinery  # keep a ref alive + testable


def _wrap_and_install(agent, graph: CellGraph, kwargs, recurrent: bool = False):
    obs_dim, action_dim, device = _dims(kwargs)
    opts = _options(kwargs)
    machinery = GraceMachinery(
        graph,
        opts,
        action_dim,
        device,
        gamma=float(getattr(agent, "gamma", 0.99)),
        env_id=_env_id(kwargs),
    )
    wrapper_cls = GraceRecurrentQNetwork if recurrent else GraceQNetwork
    nets = []
    if not isinstance(agent.q_network, wrapper_cls):
        agent.q_network = wrapper_cls(agent.q_network)
    nets.append(agent.q_network)
    tgt = getattr(agent, "target_network", None)
    if tgt is not None:
        if not isinstance(tgt, wrapper_cls):
            agent.target_network = wrapper_cls(tgt)
        nets.append(agent.target_network)
    for net in nets:
        net.attach(machinery)
    strategy = getattr(agent, "_strategy", None)
    if isinstance(strategy, Grace):
        strategy.bind(graph, machinery)
    _install_grace(agent, machinery, nets)
    return agent.q_network, agent


# ---------------------------------------------------------------- flat bases
def build_grace_dqn(**kwargs):
    obs_dim, action_dim, device = _dims(kwargs)
    graph = cell_graph("mdp", "template")
    # The observational DQN floor's exact nets in the exact RNG order
    # (wrapping adds no parameters and draws nothing).
    q = MLP(obs_dim, action_dim).to(device)
    tgt = MLP(obs_dim, action_dim).to(device)
    agent = DQN(
        q, tgt, ReplayBuffer(1_000_000, device), device=device, strategy=Grace()
    )
    return _wrap_and_install(agent, graph, kwargs)


def _build_flat(floor_builder, **kwargs):
    graph = cell_graph("mdp", "template")
    fkw = {k: v for k, v in kwargs.items() if k not in ("grace_options", "env_id")}
    fkw["strategy"] = Grace()
    _policy, agent = floor_builder(**fkw)
    return _wrap_and_install(agent, graph, kwargs)


def build_grace_cql(**kwargs):
    from src.rl.offline.cql import build_cql

    return _build_flat(build_cql, **kwargs)


def build_grace_iql(**kwargs):
    from src.rl.offline.iql import build_iql

    return _build_flat(build_iql, **kwargs)


def build_grace_bcq(**kwargs):
    from src.rl.offline.bcq import build_bcq

    return _build_flat(build_bcq, **kwargs)


# ------------------------------------------------------------ recurrent base
def build_grace_dqn_recurrent(**kwargs):
    """POMDP (offline_pomdp) grace arm — DQN base only, like every recurrent
    strategy arm. The floor is the plain recurrent trunk (``build_offline_
    dqn_recurrent`` semantics); the wrapper deliberately exposes NO ``q_su``
    so the recurrent learn path takes its plain observational branch."""
    from src.rl.models.backbone import build_trunk
    from src.rl.offline.oracle_u import _recurrent_trunk_kwargs

    obs_dim, action_dim, device = _dims(kwargs)
    obs_shape = kwargs.get("obs_shape", (obs_dim,))
    network = kwargs.get("critic_network", "lstm") or "lstm"
    if network == "mlp":
        raise ValueError(
            "build_grace_dqn_recurrent requires a recurrent critic_network "
            "(lstm/gru/rnn); use the flat grace builders for MLP bases."
        )
    graph = cell_graph("pomdp", "template")
    tk = _recurrent_trunk_kwargs(kwargs)
    q = build_trunk(network, obs_shape, obs_dim, action_dim, **tk).to(device)
    tgt = build_trunk(network, obs_shape, obs_dim, action_dim, **tk).to(device)
    agent = DQN(
        q,
        tgt,
        ReplayBuffer(1_000_000, device),
        device=device,
        strategy=Grace(),
    )
    return _wrap_and_install(agent, graph, kwargs, recurrent=True)


# ------------------------------------------------------------- online (E3)
def build_grace_online_dqn(**kwargs):
    """Online grace arm (the online mutilated-channel realization, Gate-B
    analog): the flat grace DQN with the refresh cadence enabled — the CBN
    refits from the rolling episode buffer, and the canonicalization re-runs
    every refresh (label persistence)."""
    net, agent = build_grace_dqn(**kwargs)
    agent._grace_machinery.online_refresh_interval = _ONLINE_REFRESH_INTERVAL
    return net, agent
