"""Fixture-gated OPE correctness (Phase-3 gate hard condition).

The block-MDP fixture provides exact tabular dynamics; we compute the TRUE
target-policy value by dynamic programming and require Naive / DM(FQE) /
IPW(known & cloned) / DR to agree with the closed-form values within
tolerance — plus IPW(known) ≈ DR, the Cell-3 acceptance relation.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from src.data.experience_source import OfflineDatasetSource
from src.eval.ope import (
    DirectMethod,
    DoublyRobust,
    IPWEstimator,
    NaiveEstimator,
    TargetPolicy,
)
from tests.fixtures.block_mdp import BlockMDPEnv

DEVICE = torch.device("cpu")
HORIZON = 6
N_EPISODES = 4000
SEED = 7


# ---------------------------------------------------------------------------
# Tabular world from the fixture's exact tensors (alpha=0: unconfounded)
# ---------------------------------------------------------------------------


def _world():
    env = BlockMDPEnv(
        env_id="fixture-block-ope",
        n_envs=1,
        device=DEVICE,
        seed=SEED,
        cell=3,
        d=3,
        alpha=0.0,
        sigma2=0.0,
        n_actions=3,
        horizon=HORIZON,
    )
    S, A = env.n_states, env.n_actions
    T = env.transition_probs  # [S, A, S'] exact
    r_mean = 2.0 * torch.sigmoid(env.reward_logits) - 1.0  # E[r|z,a], r in {-1,1}
    p_plus = torch.sigmoid(env.reward_logits)  # P(r=+1|z,a)
    return S, A, T, r_mean, p_plus


def _dp_value(policy_probs: torch.Tensor, T, r_mean, horizon: int) -> float:
    """Exact J(pi) = E[sum_t r_t], uniform initial state."""
    S = T.shape[0]
    d = torch.full((S,), 1.0 / S)
    total = 0.0
    for _ in range(horizon):
        total += float((d.unsqueeze(-1) * policy_probs * r_mean).sum())
        d = torch.einsum("s,sa,saz->z", d, policy_probs, T)
    return total


class TabularPolicy(TargetPolicy):
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits  # [S, A]

    def _z(self, obs: torch.Tensor) -> torch.Tensor:
        return obs.argmax(dim=-1)  # one-hot state observations

    def log_prob(self, obs, actions):
        logp = torch.log_softmax(self.logits[self._z(obs)], dim=-1)
        return logp.gather(1, actions.long().unsqueeze(-1)).squeeze(-1)

    def act(self, obs):
        return self.logits[self._z(obs)].argmax(dim=-1)

    def action_probs(self, obs):
        return torch.softmax(self.logits[self._z(obs)], dim=-1)


def _collect(policy: TabularPolicy, T, p_plus, n_episodes: int, seed: int):
    """Roll episodes in the exact tabular world; log exact propensities."""
    g = torch.Generator().manual_seed(seed)
    S = T.shape[0]
    episodes = []
    for _ in range(n_episodes):
        z = torch.randint(0, S, (1,), generator=g).item()
        obs_list, act_list, rew_list, logp_list = [], [], [], []
        for _t in range(HORIZON):
            obs = F.one_hot(torch.tensor(z), num_classes=S).float()
            probs = torch.softmax(policy.logits[z], dim=-1)
            a = torch.multinomial(probs, 1, generator=g).item()
            r = 1.0 if torch.rand(1, generator=g).item() < float(p_plus[z, a]) else -1.0
            obs_list.append(obs)
            act_list.append(a)
            rew_list.append(r)
            logp_list.append(float(torch.log(probs[a])))
            z = torch.multinomial(T[z, a], 1, generator=g).item()
        obs_list.append(F.one_hot(torch.tensor(z), num_classes=S).float())
        episodes.append(
            {
                "obs": torch.stack(obs_list),
                "actions": torch.tensor(act_list, dtype=torch.long),
                "rewards": torch.tensor(rew_list),
                "terminations": torch.zeros(HORIZON, dtype=torch.bool),
                "truncations": torch.tensor(
                    [False] * (HORIZON - 1) + [True], dtype=torch.bool
                ),
                "behavior_logprob": torch.tensor(logp_list),
            }
        )
    return episodes


@pytest.fixture(scope="module")
def world():
    S, A, T, r_mean, p_plus = _world()
    g = torch.Generator().manual_seed(100)
    behavior_logits = torch.randn(S, A, generator=g)
    # target = behavior sharpened + perturbed (keeps overlap, changes value)
    target_logits = 1.5 * behavior_logits + 0.5 * torch.randn(S, A, generator=g)
    behavior = TabularPolicy(behavior_logits)
    target = TabularPolicy(target_logits)
    episodes = _collect(behavior, T, p_plus, N_EPISODES, seed=SEED)
    source = OfflineDatasetSource(episodes, DEVICE, behavior_policy="known")
    j_b = _dp_value(torch.softmax(behavior_logits, dim=-1), T, r_mean, HORIZON)
    j_t = _dp_value(torch.softmax(target_logits, dim=-1), T, r_mean, HORIZON)
    return source, behavior, target, j_b, j_t


TOL = 0.25  # absolute tolerance on J in [-6, 6]; ~4 sigma of the MC error


def test_dp_values_differ(world):
    _, _, _, j_b, j_t = world
    assert abs(j_b - j_t) > 0.05, "behavior/target values too close to test OPE"


def test_naive_matches_behavior_value_not_target(world):
    source, _, target, j_b, j_t = world
    res = NaiveEstimator().estimate(source, target)
    assert abs(res.value - j_b) < TOL
    assert res.ci_low <= j_b <= res.ci_high


def test_ipw_known_unbiased(world):
    source, _, target, _, j_t = world
    res = IPWEstimator(behavior="known").estimate(source, target)
    assert abs(res.value - j_t) < TOL, (res.value, j_t)


def test_ipw_cloned_close(world):
    source, _, target, _, j_t = world
    res = IPWEstimator(behavior="cloned", seed=1).estimate(source, target)
    assert abs(res.value - j_t) < 2 * TOL, (res.value, j_t)


def test_dm_fqe_close(world):
    source, _, target, _, j_t = world
    res = DirectMethod(gamma=1.0, n_iters=600, seed=1).estimate(source, target)
    assert abs(res.value - j_t) < 2 * TOL, (res.value, j_t)


def test_dr_close_and_matches_ipw(world):
    source, _, target, _, j_t = world
    dr = DoublyRobust(behavior="known", gamma=1.0, seed=1).estimate(source, target)
    ipw = IPWEstimator(behavior="known").estimate(source, target)
    assert abs(dr.value - j_t) < TOL, (dr.value, j_t)
    # Cell-3 acceptance relation at fixture level: IPW(known) ~= DR
    assert abs(dr.value - ipw.value) < TOL, (dr.value, ipw.value)


def test_ipw_known_refuses_unknown_propensities(world):
    source, _, target, _, _ = world
    unknown = OfflineDatasetSource(source.episodes, DEVICE, behavior_policy="unknown")
    with pytest.raises(ValueError, match="logged propensities"):
        IPWEstimator(behavior="known").estimate(unknown, target)
