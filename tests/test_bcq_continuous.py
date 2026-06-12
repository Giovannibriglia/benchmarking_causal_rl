"""Continuous BCQ (CVAE-BCQ) — the net-new CVAE + perturbation + soft-clip.

The four correctness bars each fail on a named subtle bug:
  * CVAE ELBO — recon + KL present, KL >= 0, and the reparam gradient REACHES
    the encoder (recon-only backward; catches a detached / non-reparameterized z).
  * Perturbation — |applied - base| <= Phi elementwise and applied in [-1, 1]
    (catches an unscaled/unclipped perturbation).
  * act — returns the argmax-over-min-twin-Q candidate, sampling from the PRIOR
    (signature takes only obs; catches argmin / encoder-sampling at eval).
  * Target — _soft_clipped == lmbda*min + (1-lmbda)*max and != bare min.

Plus an end-to-end schema run and a learning-sanity check on the better-than-
random Pendulum dataset (threshold grounded against measured anchors).
"""

from __future__ import annotations

import csv

import pytest
import torch
import torch.nn as nn
from src.rl.offline.bcq_continuous import build_bcq_continuous, PerturbationNet

CPU = torch.device("cpu")


def _batch(b=64, obs_dim=3, act_dim=1, scale=2.0):
    return {
        "obs": torch.randn(b, obs_dim),
        "actions": torch.empty(b, act_dim).uniform_(-scale, scale),
        "rewards": torch.randn(b),
        "next_obs": torch.randn(b, obs_dim),
        "dones": torch.zeros(b),
    }


def _agent(obs_dim=3, act_dim=1):
    _, agent = build_bcq_continuous(
        obs_dim=obs_dim,
        action_dim=act_dim,
        action_type="continuous",
        device=CPU,
        action_space=None,
    )
    return agent


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------
def test_registered_bcq_continuous():
    from src.benchmarking.registry import register_default_algorithms, registry

    register_default_algorithms()
    spec = registry.get("bcq_continuous")
    assert spec.kind == "off_policy" and spec.data_regime == "offline"


# --------------------------------------------------------------------------
# Bar 1 — CVAE ELBO: recon + KL present, KL >= 0, reparam grad to encoder
# --------------------------------------------------------------------------
def test_cvae_elbo_present_and_kl_nonnegative():
    torch.manual_seed(0)
    agent = _agent()
    m = agent.update(_batch())
    assert "recon_loss" in m and "kl_loss" in m
    assert m["kl_loss"] >= 0.0  # analytic Gaussian KL is always >= 0
    assert m["recon_loss"] == m["recon_loss"]  # finite


def test_reparam_gradient_reaches_encoder():
    """RECON-only backward must reach the encoder — proves z = mu + std*eps is
    reparameterized (a detached z would leave the encoder grad-less from recon;
    KL is excluded here so it can't mask the bug via mu/logvar)."""
    torch.manual_seed(0)
    agent = _agent()
    agent.vae.zero_grad(set_to_none=True)
    obs = torch.randn(32, 3)
    act = torch.empty(32, 1).uniform_(-1, 1)
    recon, _, _ = agent.vae(obs, act)
    F_recon = ((recon - act) ** 2).mean()
    F_recon.backward()
    grads = [
        p.grad.abs().sum().item()
        for p in agent.vae.encoder.parameters()
        if p.grad is not None
    ]
    assert grads and sum(grads) > 0.0


# --------------------------------------------------------------------------
# Bar 2 — perturbation bounded by Phi and in [-1, 1]
# --------------------------------------------------------------------------
def test_perturbation_bounded_by_phi():
    torch.manual_seed(0)
    phi = 0.05
    net = PerturbationNet(3, 1, phi=phi)
    obs = torch.randn(64, 3)
    base = torch.empty(64, 1).uniform_(-1.0, 1.0)
    applied = net(obs, base)
    assert (applied - base).abs().max().item() <= phi + 1e-6
    assert applied.min().item() >= -1.0 and applied.max().item() <= 1.0


# --------------------------------------------------------------------------
# Bar 3 — act samples the prior, returns argmax-over-min-twin-Q
# --------------------------------------------------------------------------
class _ActionSumQ(nn.Module):
    """Q(s,a) = sum(a): larger actions score higher, so argmax-Q -> largest a."""

    def __init__(self, action_dim):
        super().__init__()
        self.ad = action_dim

    def forward(self, x):
        return x[:, -self.ad :].sum(dim=-1, keepdim=True)


class _IdentityPert(nn.Module):
    def forward(self, obs, a):
        return a


def test_act_returns_argmax_min_q_candidate():
    agent = _agent()
    agent.n_sampled = 5
    cand = torch.linspace(-1.0, 1.0, 5).reshape(5, 1)  # fixed candidate set
    # Deterministic decode + identity perturb + Q = sum(action).
    agent.vae.decode = lambda obs, z: cand
    agent.perturbation = _IdentityPert()
    agent.q1 = _ActionSumQ(agent.action_dim)
    agent.q2 = _ActionSumQ(agent.action_dim)
    out = agent.act(torch.randn(1, 3), noise=False).action
    # Best candidate under Q=sum(a) is the largest action (1.0); scale=1.0 here.
    assert torch.allclose(out, torch.tensor([[1.0]]), atol=1e-6)


# --------------------------------------------------------------------------
# Bar 4 — target uses the soft-clipped blend, not bare min
# --------------------------------------------------------------------------
def test_soft_clipped_is_lambda_blend_not_min():
    agent = _agent()
    q1 = torch.tensor([1.0, 5.0])
    q2 = torch.tensor([3.0, 2.0])
    sc = agent._soft_clipped(q1, q2)
    lam = agent.lmbda
    expected = lam * torch.min(q1, q2) + (1.0 - lam) * torch.max(q1, q2)
    assert torch.allclose(sc, expected)
    assert not torch.allclose(sc, torch.min(q1, q2))  # genuinely a blend


def test_continuous_bcq_update_finite():
    torch.manual_seed(0)
    agent = _agent()
    m = agent.update(_batch())
    for k in ("critic_loss", "actor_loss", "recon_loss", "kl_loss"):
        assert m[k] == m[k] and abs(m[k]) < float("inf")


# --------------------------------------------------------------------------
# End-to-end + learning (need the offline extra)
# --------------------------------------------------------------------------
pytest.importorskip("minari")
pytest.importorskip("h5py")

from src.benchmarking.registry import (  # noqa: E402
    register_default_algorithms,
    registry,
)
from src.benchmarking.runner import (  # noqa: E402
    BenchmarkRunner,
    EVAL_COLUMNS,
    TRAIN_COLUMNS,
)
from src.config.defaults import EnvConfig, RunConfig, TrainingConfig  # noqa: E402
from src.config.device import detect_device  # noqa: E402
from src.envs.registry import register_default_env_wrappers  # noqa: E402


def _build_dataset(tmp_path, monkeypatch, dataset_id, policy, n_episodes, seed):
    monkeypatch.setenv("MINARI_DATASETS_PATH", str(tmp_path / "minari"))
    from tools.make_pendulum_offline import make_pendulum_dataset

    make_pendulum_dataset(
        dataset_id=dataset_id, n_episodes=n_episodes, seed=seed, policy=policy
    )
    return dataset_id


def _run(dataset_id, epochs, grad_steps, run_dir, seed=0, n_eval=5):
    register_default_algorithms()
    register_default_env_wrappers()
    env_cfg = EnvConfig(
        env_id="Pendulum-v1",
        n_train_envs=1,
        n_eval_envs=n_eval,
        rollout_len=grad_steps,
        seed=seed,
        offline_dataset=dataset_id,
    )
    train_cfg = TrainingConfig(
        n_episodes=epochs,
        n_checkpoints=2,
        device=str(detect_device()),
        algorithm="bcq_continuous",
        aggregation="mean",
    )
    BenchmarkRunner(
        env_cfg,
        train_cfg,
        RunConfig(run_dir=str(run_dir), timestamp="t"),
        registry.get("bcq_continuous"),
    ).run()
    return run_dir


def test_bcq_continuous_end_to_end_schema(tmp_path, monkeypatch):
    ds = _build_dataset(tmp_path, monkeypatch, "pendulum/rand-v0", "random", 6, 0)
    run_dir = tmp_path / "run"
    _run(ds, epochs=2, grad_steps=10, run_dir=run_dir)

    with (run_dir / "train_metrics.csv").open() as f:
        train_rows = list(csv.DictReader(f))
    with (run_dir / "eval_metrics.csv").open() as f:
        eval_rows = list(csv.DictReader(f))
    assert list(train_rows[0].keys()) == TRAIN_COLUMNS
    assert list(eval_rows[0].keys()) == EVAL_COLUMNS
    assert float(train_rows[0]["q_loss"]) == float(train_rows[0]["q_loss"])  # finite
    ev = float(eval_rows[-1]["eval_return_mean"])
    assert ev == ev and ev < 0.0  # Pendulum return is negative


def test_bcq_continuous_learns_beats_random(tmp_path, monkeypatch):
    """Trained on swing-up data, eval clears the random Pendulum baseline
    (~= -1224) by a clear margin (threshold grounded by measurement)."""
    ds = _build_dataset(tmp_path, monkeypatch, "pendulum/heur-v0", "heuristic", 40, 0)
    run_dir = tmp_path / "learn"
    _run(ds, epochs=40, grad_steps=200, run_dir=run_dir, n_eval=20)

    with (run_dir / "eval_metrics.csv").open() as f:
        eval_rows = list(csv.DictReader(f))
    final_return = float(eval_rows[-1]["eval_return_mean"])
    assert final_return > -850.0, f"bcq eval={final_return:.0f} did not beat random"
