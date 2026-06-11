from __future__ import annotations

from src.rl.nets.mlp import MLP
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
    obs_dim = kwargs["obs_dim"]
    action_dim = kwargs["action_dim"]
    device = kwargs["device"]
    q_net = MLP(obs_dim, action_dim)
    target_net = MLP(obs_dim, action_dim)
    buffer = ReplayBuffer(capacity=1_000_000, device=device)
    agent = DQN(q_net.to(device), target_net.to(device), buffer, device=device)
    return q_net.to(device), agent
