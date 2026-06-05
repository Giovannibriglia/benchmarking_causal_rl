from __future__ import annotations

import torch
from src.causal_metrics.gap import compute_gap
from src.envs.wrappers.block_mdp import BlockMDPEnv


def test_block_mdp_alpha_zero_has_zero_tv_gap() -> None:
    env = BlockMDPEnv(
        env_id="causal-block-mdp-cell8",
        n_envs=16,
        device=torch.device("cpu"),
        seed=7,
        cell=8,
        alpha=0.0,
        d=3,
        D=8,
    )
    obs, _ = env.reset(seed=7)
    action = torch.randint(0, env.n_actions, (env.n_envs,), dtype=torch.long)
    next_obs, reward, _, _, _ = env.step(action)
    result = compute_gap(
        env=env,
        obs=obs,
        action=action,
        reward=reward,
        next_obs=next_obs,
        divergence="tv",
    )
    assert abs(result.delta) < 1e-6
    env.close()
