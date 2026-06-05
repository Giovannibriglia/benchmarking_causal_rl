from __future__ import annotations

import torch
import torch.nn as nn
from src.rl.off_policy.biased_explorer import (
    ConfoundedExplorer,
    EpsilonGreedyExplorer,
    UniformExplorer,
)


class _FixedQ(nn.Module):
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        _ = obs
        return torch.tensor([[1.0, 3.0, -1.0]], device=obs.device).repeat(
            obs.shape[0], 1
        )


def test_confounded_explorer_requires_latent() -> None:
    explorer = ConfoundedExplorer(
        logit_fn=lambda x: torch.zeros(x.shape[0], 3), alpha=1.0
    )
    obs = torch.zeros(4, 5)
    try:
        explorer.select_action(obs, latent=None)
    except ValueError:
        return
    raise AssertionError("Expected ValueError when latent is None.")


def test_uniform_explorer_logprob_matches_uniform() -> None:
    explorer = UniformExplorer(n_actions=5, device=torch.device("cpu"))
    obs = torch.zeros(32, 4)
    action, logp = explorer.select_action(obs)
    assert action.shape == (32,)
    assert logp.shape == (32,)
    assert torch.allclose(torch.exp(logp), torch.full((32,), 0.2), atol=1e-6)


def test_confounded_alpha_zero_recovers_epsilon_greedy_at_epsilon_zero() -> None:
    q = _FixedQ()
    eps = EpsilonGreedyExplorer(q_network=q, epsilon=0.0)
    conf = ConfoundedExplorer(logit_fn=q, alpha=0.0, beta=1.0, epsilon=0.0)
    obs = torch.randn(20, 4)
    latent = torch.ones(20, 2)

    a_eps, lp_eps = eps.select_action(obs, latent=None)
    a_conf, lp_conf = conf.select_action(obs, latent=latent)
    assert torch.equal(a_eps, a_conf)
    assert torch.allclose(lp_eps, lp_conf, atol=1e-6)
