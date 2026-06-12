from __future__ import annotations

from src.rl.models.backbone import select_backbone
from src.rl.off_policy.dqn import DQN
from src.rl.off_policy.replay_buffer import ReplayBuffer


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
    agent = DQN(q_net.to(device), target_net.to(device), buffer, device=device)
    return q_net.to(device), agent
