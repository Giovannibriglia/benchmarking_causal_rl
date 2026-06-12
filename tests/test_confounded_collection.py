"""feat/confounded-collection — env-level unobserved confounding.

The centerpiece bar PROVES confounding rather than asserting it: a toy env where
the action has NO true effect on reward (reward = c_r*U + noise, action-
independent), so any action<->reward association is purely the confounding path
U->action, U->reward. We assert the operational signature:
  * marginal Corr(action, reward) != 0  (the spurious association is induced),
  * partial Corr(action, reward | U) ~= 0 (conditioning on U removes it).

Plus: U is per-episode (resamples only at done), U never enters the obs, the
eval env is never wrapped, and bias_confounded is opt-in/selectable end-to-end.
"""

from __future__ import annotations

import numpy as np
import torch
from gymnasium.spaces import Box
from src.envs.wrappers.confounded import ConfoundedCollectionWrapper
from src.rl.base import ActionOutput
from src.rl.policies.behavior_policy import ConfoundedBehaviorPolicy

CPU = torch.device("cpu")


class _NullActionEnv:
    """Vector toy env whose reward is INDEPENDENT of the action (action-free
    noise here; the wrapper adds c_r*U). Episodes of fixed length ``ep_len``."""

    def __init__(self, n_envs=1, obs_dim=2, ep_len=20, seed=0):
        self.n_envs = n_envs
        self.device = CPU
        self.ep_len = ep_len
        self.obs_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,))
        self.act_space = Box(low=-1.0, high=1.0, shape=(1,))
        self._obs_dim = obs_dim
        self._t = 0
        self._g = torch.Generator().manual_seed(seed)

    def reset(self, seed=None):
        self._t = 0
        return torch.randn(self.n_envs, self._obs_dim, generator=self._g), {}

    def step(self, action):
        self._t += 1
        obs = torch.randn(self.n_envs, self._obs_dim, generator=self._g)
        reward = 0.5 * torch.randn(self.n_envs, generator=self._g)  # action-free eps
        term = torch.zeros(self.n_envs, dtype=torch.bool)
        trunc = torch.full((self.n_envs,), self._t % self.ep_len == 0)
        return obs, reward, term, trunc, {}


class _NoiseAgent:
    """Continuous agent: independent N(0, 0.5) action noise (no U component)."""

    def act(self, obs, *a, **k):
        return ActionOutput(action=0.5 * torch.randn(obs.shape[0], 1))


# --------------------------------------------------------------------------
# THE bar — confounding signature: marginal != 0, partial-given-U ~= 0
# --------------------------------------------------------------------------
def _pearson(x, y):
    return float(np.corrcoef(x, y)[0, 1])


def test_confounding_signature_marginal_nonzero_partial_zero():
    torch.manual_seed(0)
    toy = _NullActionEnv(n_envs=1, ep_len=20)
    wrapper = ConfoundedCollectionWrapper(toy, c_a=1.0, c_r=1.0, u_dist="normal")
    pol = ConfoundedBehaviorPolicy(
        _NoiseAgent(), "continuous", toy.act_space, wrapper, strength=1.0, base_c_a=1.0
    )

    acts, rews, us = [], [], []
    obs, _ = wrapper.reset()
    for _ in range(5000):
        u = wrapper.current_u.clone()  # the U this step's action+reward share
        a = pol.act(obs).action
        obs, r, term, trunc, _ = wrapper.step(a)
        acts.append(a.squeeze().item())
        rews.append(r.squeeze().item())
        us.append(u.squeeze().item())
    a, r, u = map(np.asarray, (acts, rews, us))

    r_ar, r_au, r_ru = _pearson(a, r), _pearson(a, u), _pearson(r, u)
    partial = (r_ar - r_au * r_ru) / np.sqrt((1 - r_au**2) * (1 - r_ru**2))

    assert abs(r_ar) > 0.2, f"no spurious association induced (Corr(a,r)={r_ar:.3f})"
    assert (
        abs(partial) < 0.05
    ), f"association survives conditioning on U ({partial:.3f})"


# --------------------------------------------------------------------------
# U is per-episode: resamples exactly at done, constant otherwise
# --------------------------------------------------------------------------
def test_u_constant_within_episode_resamples_only_at_done():
    torch.manual_seed(0)
    wrapper = ConfoundedCollectionWrapper(
        _NullActionEnv(n_envs=1, ep_len=5), u_dist="normal"
    )
    wrapper.reset()
    for _ in range(20):
        prev_u = wrapper.current_u.clone()
        _, _, term, trunc, _ = wrapper.step(torch.zeros(1, 1))
        done = torch.logical_or(term, trunc)
        changed = wrapper.current_u != prev_u
        # U changes IFF the sub-env's episode just ended -> genuinely per-episode.
        assert torch.equal(changed, done)


# --------------------------------------------------------------------------
# U never enters the obs (the confounding-vs-observed-feature line)
# --------------------------------------------------------------------------
def test_u_absent_from_obs_present_in_info():
    toy = _NullActionEnv(n_envs=1, obs_dim=2)
    wrapper = ConfoundedCollectionWrapper(toy)
    assert wrapper.obs_space.shape == toy.obs_space.shape  # delegated, clean
    obs, info = wrapper.reset()
    assert obs.shape[-1] == 2 and "confounder_u" in info
    obs2, _, _, _, info2 = wrapper.step(torch.zeros(1, 1))
    assert obs2.shape[-1] == 2  # still no U appended to the obs
    assert "confounder_u" in info2  # U is exposed only via info (dropped by the loop)


def test_reward_is_perturbed_by_u():
    # With U=normal and c_r large, the wrapped reward differs from the inner one.
    torch.manual_seed(0)
    wrapper = ConfoundedCollectionWrapper(
        _NullActionEnv(n_envs=4), c_r=10.0, u_dist="normal"
    )
    wrapper.reset()
    _, reward, _, _, info = wrapper.step(torch.zeros(4, 1))
    # reward carries the +c_r*U term -> strongly correlated with the exposed U.
    assert _pearson(reward.numpy(), info["confounder_u"].numpy()) > 0.9


# --------------------------------------------------------------------------
# Runner integration — train-only wrap, eval clean, opt-in, runs e2e
# --------------------------------------------------------------------------
def test_runner_confounds_train_only_and_runs(tmp_path):
    import csv

    from src.benchmarking.registry import register_default_algorithms, registry
    from src.benchmarking.runner import BenchmarkRunner, EVAL_COLUMNS, TRAIN_COLUMNS
    from src.config.defaults import EnvConfig, RunConfig, TrainingConfig
    from src.config.device import detect_device
    from src.envs.registry import register_default_env_wrappers

    register_default_algorithms()
    register_default_env_wrappers()
    run_dir = tmp_path / "run"
    env_cfg = EnvConfig(
        env_id="CartPole-v1",
        n_train_envs=2,
        n_eval_envs=2,
        rollout_len=5,
        seed=0,
        behavior_policy="bias_confounded",
        behavior_strength=1.0,
    )
    train_cfg = TrainingConfig(
        n_episodes=1,
        n_checkpoints=1,
        device=str(detect_device()),
        algorithm="dqn",
        aggregation="mean",
    )
    runner = BenchmarkRunner(
        env_cfg,
        train_cfg,
        RunConfig(run_dir=str(run_dir), timestamp="t"),
        registry.get("dqn"),
    )
    # Train env is confounded; eval env is clean; the policy is the confounder.
    assert isinstance(runner.train_env, ConfoundedCollectionWrapper)
    assert not isinstance(runner.eval_env, ConfoundedCollectionWrapper)
    assert isinstance(runner.collection_policy, ConfoundedBehaviorPolicy)
    runner.run()
    with (run_dir / "train_metrics.csv").open() as f:
        train_rows = list(csv.DictReader(f))
    with (run_dir / "eval_metrics.csv").open() as f:
        eval_rows = list(csv.DictReader(f))
    assert list(train_rows[0].keys()) == TRAIN_COLUMNS
    assert list(eval_rows[0].keys()) == EVAL_COLUMNS
