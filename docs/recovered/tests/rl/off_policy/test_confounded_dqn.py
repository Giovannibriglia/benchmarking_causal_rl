from __future__ import annotations

import torch
from src.envs.wrappers.block_mdp import BlockMDPEnv
from src.rl.nets.mlp import MLP
from src.rl.off_policy.confounded_dqn import ConfoundingRobustDQN
from src.rl.off_policy.dqn import DQN
from src.rl.off_policy.replay_buffer import ReplayBuffer


def _batch(device: torch.device, obs_dim: int, batch_size: int = 32):
    return {
        "obs": torch.randn(batch_size, obs_dim, device=device),
        "actions": torch.randint(0, 4, (batch_size,), device=device),
        "rewards": torch.randn(batch_size, device=device),
        "next_obs": torch.randn(batch_size, obs_dim, device=device),
        "dones": torch.zeros(batch_size, device=device),
    }


def test_lam_zero_matches_vanilla_dqn_loss() -> None:
    device = torch.device("cpu")
    torch.manual_seed(0)
    q1, t1 = MLP(6, 4).to(device), MLP(6, 4).to(device)
    q2, t2 = MLP(6, 4).to(device), MLP(6, 4).to(device)
    q2.load_state_dict(q1.state_dict())
    t2.load_state_dict(t1.state_dict())

    b1 = ReplayBuffer(capacity=128, device=device)
    b2 = ReplayBuffer(capacity=128, device=device)
    dqn = DQN(q1, t1, b1, device=device)
    env = BlockMDPEnv(
        env_id="causal-block-mdp-cell8", n_envs=32, device=device, seed=1, cell=8
    )
    env.reset(seed=1)
    conf = ConfoundingRobustDQN(q2, t2, b2, device=device, env_oracle=env, lam=0.0)

    batch = _batch(device, 6)
    m1 = dqn.update(batch)
    m2 = conf.update(batch)
    assert abs(m1["q_loss"] - m2["q_loss"]) < 1e-6
    env.close()


def test_delta_tv_finite_and_non_negative() -> None:
    device = torch.device("cpu")
    q, t = MLP(6, 4).to(device), MLP(6, 4).to(device)
    buf = ReplayBuffer(capacity=128, device=device)
    env = BlockMDPEnv(
        env_id="causal-block-mdp-cell8",
        n_envs=32,
        device=device,
        seed=3,
        cell=8,
        alpha=1.0,
    )
    env.reset(seed=3)
    agent = ConfoundingRobustDQN(q, t, buf, device=device, env_oracle=env, lam=0.1)
    metrics = agent.update(_batch(device, 6))
    assert metrics["delta_tv"] >= 0.0
    assert torch.isfinite(torch.tensor(metrics["delta_tv"]))
    env.close()
