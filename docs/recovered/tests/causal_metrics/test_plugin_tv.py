from __future__ import annotations

import torch
from src.causal_metrics.gap import compute_gap
from src.envs.wrappers.block_mdp import BlockMDPEnv


def test_plugin_tv_block_mdp_alpha_zero_is_near_zero() -> None:
    env = BlockMDPEnv(
        env_id="causal-block-mdp-cell8",
        n_envs=32,
        device=torch.device("cpu"),
        seed=11,
        cell=8,
        alpha=0.0,
        d=3,
        D=8,
    )
    obs, _ = env.reset(seed=11)
    action = torch.randint(0, env.n_actions, (env.n_envs,), dtype=torch.long)
    next_obs, reward, _, _, _ = env.step(action)
    gap = compute_gap(
        env=env,
        obs=obs,
        action=action,
        reward=reward,
        next_obs=next_obs,
        divergence="tv",
    )
    assert gap.delta <= 1e-6
    env.close()
