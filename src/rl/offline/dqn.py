from __future__ import annotations

from src.rl.models.backbone import build_trunk, select_backbone
from src.rl.off_policy.dqn import DQN
from src.rl.off_policy.replay_buffer import ReplayBuffer
from src.rl.offline.oracle_u import _recurrent_trunk_kwargs


def build_offline_dqn(**kwargs):
    """Build a DQN agent for offline (fixed-dataset) training.

    Reuses the online ``DQN`` agent unchanged: its TD update is identical
    whether the replay buffer is filled by live interaction or by a logged
    dataset. The offline-ness lives in the dispatch (``data_regime="offline"``)
    and in how the buffer is filled (``fill_replay_buffer_from_minari``), not in
    the learning rule. This module is the seam for later offline-only algorithms
    (BCQ / CQL / IQL), which will subclass or replace the update here.
    """
    if kwargs.get("action_type", "discrete") != "discrete":
        raise ValueError(
            "offline_dqn is discrete-only; use cql_continuous or iql_continuous "
            "for continuous action spaces. (Without this guard a continuous env "
            "would silently build a single-output discrete Q-net.)"
        )
    obs_dim = kwargs["obs_dim"]
    action_dim = kwargs["action_dim"]
    device = kwargs["device"]
    # Backbone by obs rank: vector -> MLP (bitwise-identical to the previous
    # MLP(obs_dim, action_dim)); image -> Nature-CNN.
    obs_shape = kwargs.get("obs_shape", (obs_dim,))
    q_net = select_backbone(obs_shape, obs_dim, action_dim)
    target_net = select_backbone(obs_shape, obs_dim, action_dim)
    buffer = ReplayBuffer(capacity=1_000_000, device=device)
    agent = DQN(
        q_net.to(device),
        target_net.to(device),
        buffer,
        device=device,
        strategy=kwargs.get("strategy"),
    )
    return q_net.to(device), agent


def build_offline_dqn_recurrent(**kwargs):
    """Cell 8 recurrent OBSERVATIONAL floor — the baseline the recurrent triad
    (floor / proximal / oracle_u) compares against.

    Same ``DQN`` class as ``build_offline_dqn``, but the Q-net is a RECURRENT trunk
    (``build_trunk`` from ``critic_network``) instead of an MLP, and there is NO
    identification strategy. Two consequences:

      * ``initial_state`` on the trunk -> ``DQN.is_recurrent`` fires -> the sequence
        ``_learn_recurrent`` path engages (POMDP estimation over (B,T), BPTT);
      * the net has NO ``q_su`` -> ``_learn_recurrent`` takes its BYTE-FROZEN
        else-branch (plain recurrent Q, no U-conditioning) = a recurrent
        OBSERVATIONAL critic. It reads ZERO ``confounder_u`` (no deconfounding).

    ``offline_dqn`` (MLP) and the ``_offpolicy_recurrent_guard`` are untouched;
    this is a separate, additive builder registered WITHOUT the guard. DQN base
    only — cql/iql/bcq recurrent remains a deferred workstream (not needed here,
    the whole floor/proximal/oracle stack is DQN-based).
    """
    if kwargs.get("action_type", "discrete") != "discrete":
        raise ValueError(
            "offline_dqn_recurrent is discrete-only (the Cell-8 recurrent arm); "
            "use cql_continuous or iql_continuous for continuous action spaces."
        )
    obs_dim = kwargs["obs_dim"]
    action_dim = kwargs["action_dim"]
    device = kwargs["device"]
    obs_shape = kwargs.get("obs_shape", (obs_dim,))
    # critic_network selects the recurrent architecture (lstm/gru/rnn). Default
    # lstm so a bare `offline_dqn_recurrent` (no networks: block) is still
    # recurrent; an explicit networks:{critic: gru} overrides. mlp would defeat
    # the purpose (that is what offline_dqn is for) -> reject it.
    network = kwargs.get("critic_network", "lstm") or "lstm"
    if network == "mlp":
        raise ValueError(
            "offline_dqn_recurrent requires a recurrent critic_network "
            "(lstm/gru/rnn); for an MLP observational floor use offline_dqn."
        )
    tk = _recurrent_trunk_kwargs(kwargs)
    q_net = build_trunk(network, obs_shape, obs_dim, action_dim, **tk)
    target_net = build_trunk(network, obs_shape, obs_dim, action_dim, **tk)
    buffer = ReplayBuffer(capacity=1_000_000, device=device)
    agent = DQN(
        q_net.to(device),
        target_net.to(device),
        buffer,
        device=device,
        strategy=None,  # observational floor: no q_su, no deconfounding, no U
    )
    return q_net.to(device), agent
