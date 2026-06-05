from __future__ import annotations

import torch
from src.envs.wrappers.sepsis import SepsisCausalEnv


def test_do_reward_cell3_matches_internal_oracle_formula() -> None:
    env = SepsisCausalEnv(
        env_id="causal-sepsis-cell3",
        n_envs=2,
        device=torch.device("cpu"),
        seed=42,
        cell=3,
    )
    env.reset(seed=42)
    action = torch.tensor([0, 1], dtype=torch.long)

    do_reward = env.do_reward(action)
    z = env._z  # noqa: SLF001 - test-level internal consistency check
    expected_u0 = torch.sigmoid(
        env.reward_logits[0, z, action] - env.alpha * env.behavior_gate[z, action]
    )
    expected_u1 = torch.sigmoid(
        env.reward_logits[1, z, action] + env.alpha * env.behavior_gate[z, action]
    )
    expected_p = 0.5 * (expected_u0 + expected_u1)
    expected = torch.stack([1.0 - expected_p, expected_p], dim=-1)

    assert torch.allclose(do_reward, expected, atol=1e-6)
    env.close()
