from __future__ import annotations

import torch
from src.benchmarking.offline_collector import collect_offline_dataset
from src.envs.wrappers.sepsis import SepsisCausalEnv
from src.rl.off_policy.biased_explorer import UniformExplorer
from src.rl.off_policy.replay_buffer import ReplayBuffer


def _collect(cell: int, expose_pi_b: bool, expose_latent: bool) -> ReplayBuffer:
    env = SepsisCausalEnv(
        env_id=f"causal-sepsis-cell{cell}",
        n_envs=2,
        device=torch.device("cpu"),
        seed=0,
        cell=cell,
    )
    explorer = UniformExplorer(n_actions=env.n_actions, device=env.device)
    buffer = ReplayBuffer(capacity=5000, device=env.device)
    collect_offline_dataset(
        env=env,
        explorer=explorer,
        n_episodes=1,
        expose_pi_b=expose_pi_b,
        expose_latent=expose_latent,
        buffer=buffer,
    )
    env.close()
    return buffer


def test_cell3_contains_behavior_logprobs() -> None:
    buffer = _collect(cell=3, expose_pi_b=True, expose_latent=True)
    batch = buffer.sample(8)
    assert batch["behavior_logprob"] is not None
    assert torch.isfinite(batch["behavior_logprob"]).all()


def test_cell4_latent_is_hidden() -> None:
    buffer = _collect(cell=4, expose_pi_b=True, expose_latent=False)
    batch = buffer.sample(8)
    assert batch["latent"] is None
    assert batch["behavior_logprob"] is not None


def test_cell8_hides_latent_and_behavior_policy() -> None:
    buffer = _collect(cell=8, expose_pi_b=False, expose_latent=False)
    batch = buffer.sample(8)
    assert batch["latent"] is None
    assert batch["behavior_logprob"] is None
