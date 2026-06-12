"""A2 curiosity collection — ensemble-disagreement novelty.

Correctness bars (each fails on real behavior, not "it ran"):
  (a) the ensemble's disagreement is significantly LOWER on a trained region
      than on an untrained one — the epistemic-novelty signal is real, not
      constant/uniform (fails if the ensemble collapses).
  (b) the curiosity policy selects the HIGH-disagreement candidate — under a
      controlled ensemble where one action is novel and another familiar
      (fails if it selects by anything but disagreement).

Coverage (curiosity-collected data spanning more of the state space) is the
research payoff but too stochastic for a hard unit bar — exercised only as the
e2e "it's selectable and runs" integration check; the spread comparison is
deferred (same reason the Pendulum learning bars were calibrated, not assumed).
Off-policy golden bitwise-safety (default path untouched) is covered by the
regression suite.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.rl.base import ActionOutput
from src.rl.policies.behavior_policy import (
    build_collection_policy,
    CuriosityBehaviorPolicy,
)

CPU = torch.device("cpu")


class _DiscreteSpace:
    def __init__(self, n):
        self.n = n


class _BoxSpace:
    def __init__(self, low, high):
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)
        self.shape = self.low.shape


class _StubAgent:
    """Agent stub: fixed action, no buffer (so the ensemble never online-trains
    during these unit tests — they drive the models directly)."""

    buffer = None

    def __init__(self, action, action_type="discrete"):
        self._action = action
        self._type = action_type

    def act(self, obs, *a, **k):
        b = obs.shape[0]
        if self._type == "discrete":
            return ActionOutput(action=torch.full((b,), self._action, dtype=torch.long))
        return ActionOutput(action=torch.full((b, 1), float(self._action)))


# --------------------------------------------------------------------------
# Bar (a) — disagreement is lower on a trained region than an untrained one
# --------------------------------------------------------------------------
def test_disagreement_lower_on_trained_region():
    torch.manual_seed(0)
    pol = CuriosityBehaviorPolicy(
        _StubAgent(0, "continuous"), "continuous", _BoxSpace([-1.0], [1.0]), n_models=5
    )
    pol._build(torch.zeros(1, 3))  # ensemble for obs_dim=3, action_dim=1

    # Train every member to map (obs=0, a=0) -> next=1 (the familiar region R).
    r_obs = torch.zeros(64, 3)
    r_act = torch.zeros(64, 1)
    r_next = torch.ones(64, 3)
    for _ in range(400):
        for m, opt in zip(pol.models, pol.opts):
            loss = F.mse_loss(m(r_obs, r_act), r_next)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    dis_trained = pol._disagreement(torch.zeros(1, 3), torch.zeros(1, 1)).item()
    dis_novel = pol._disagreement(torch.full((1, 3), 5.0), torch.ones(1, 1)).item()
    # The members agree where they were trained, disagree (extrapolate apart)
    # where they weren't -> novelty signal is real and region-dependent.
    assert dis_novel > dis_trained
    assert dis_novel > 5 * dis_trained + 1e-6  # a clear margin, not noise


# --------------------------------------------------------------------------
# Bar (b) — the policy picks the high-disagreement candidate
# --------------------------------------------------------------------------
class _StubModel(nn.Module):
    """forward(obs, a) = a * k : action 0 -> 0 (members AGREE), action 1 -> k
    (members DISAGREE). So disagreement(a=1) >> disagreement(a=0)."""

    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, obs, actions):
        return actions.float().unsqueeze(-1) * self.k


def test_curiosity_selects_high_disagreement_action():
    pol = CuriosityBehaviorPolicy(
        _StubAgent(0, "discrete"), "discrete", _DiscreteSpace(2), strength=1.0
    )
    pol._build(torch.zeros(1, 3))
    pol.models = [_StubModel(k) for k in range(5)]  # controlled disagreement
    pol.action_dim = 2

    out = pol.act(torch.zeros(8, 3)).action
    assert (out == 1).all()  # action 1 is the high-disagreement one; strength=1


def test_curiosity_strength_zero_is_pure_agent():
    pol = CuriosityBehaviorPolicy(
        _StubAgent(0, "discrete"), "discrete", _DiscreteSpace(2), strength=0.0
    )
    pol._build(torch.zeros(1, 3))
    pol.models = [_StubModel(k) for k in range(5)]
    pol.action_dim = 2
    out = pol.act(torch.zeros(8, 3)).action
    assert (out == 0).all()  # strength=0 -> always the agent's action (0)


# --------------------------------------------------------------------------
# Registration / factory / RNG isolation
# --------------------------------------------------------------------------
def test_factory_builds_curiosity_and_maps_strength():
    pol = build_collection_policy(
        "curiosity", _StubAgent(0), "discrete", _DiscreteSpace(2), strength=0.7
    )
    assert isinstance(pol, CuriosityBehaviorPolicy) and pol.strength == 0.7


def test_training_sampler_uses_isolated_generator():
    # The ensemble's training-buffer sampler must be its own generator, not the
    # global stream the agent's ReplayBuffer.sample uses.
    pol = CuriosityBehaviorPolicy(_StubAgent(0), "discrete", _DiscreteSpace(2))
    assert isinstance(pol._gen, torch.Generator)


# --------------------------------------------------------------------------
# Integration — opt-in selection wires curiosity into collection, runs e2e
# --------------------------------------------------------------------------
def test_runner_selects_curiosity_and_runs(tmp_path):
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
        behavior_policy="curiosity",
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
    assert isinstance(runner.collection_policy, CuriosityBehaviorPolicy)
    runner.run()
    with (run_dir / "train_metrics.csv").open() as f:
        train_rows = list(csv.DictReader(f))
    with (run_dir / "eval_metrics.csv").open() as f:
        eval_rows = list(csv.DictReader(f))
    assert list(train_rows[0].keys()) == TRAIN_COLUMNS
    assert list(eval_rows[0].keys()) == EVAL_COLUMNS
