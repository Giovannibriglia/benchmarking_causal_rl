from __future__ import annotations

import torch
from src.envs.causal_base import CausalEnv
from src.envs.wrappers.block_mdp import BlockMDPEnv
from src.envs.wrappers.sepsis import SepsisCausalEnv


def _exercise_env(env: CausalEnv) -> None:
    obs, _ = env.reset(seed=123)
    assert isinstance(obs, torch.Tensor)
    latent = env.latent_state()
    assert isinstance(latent, torch.Tensor)
    assert latent.shape[0] == env.n_envs
    action = torch.zeros(env.n_envs, dtype=torch.long, device=env.device)
    next_obs, reward, terminated, truncated, _ = env.step(action)
    assert isinstance(next_obs, torch.Tensor)
    assert isinstance(reward, torch.Tensor)
    assert isinstance(terminated, torch.Tensor)
    assert isinstance(truncated, torch.Tensor)
    latent2 = env.latent_state()
    assert isinstance(latent2, torch.Tensor)
    assert latent2.shape[0] == env.n_envs
    env.close()


def test_causal_subclasses_latent_state_shapes() -> None:
    device = torch.device("cpu")
    sepsis = SepsisCausalEnv(
        env_id="causal-sepsis-cell3", n_envs=3, device=device, seed=0, cell=3
    )
    block = BlockMDPEnv(
        env_id="causal-block-mdp-cell8", n_envs=3, device=device, seed=0, cell=8
    )
    _exercise_env(sepsis)
    _exercise_env(block)
