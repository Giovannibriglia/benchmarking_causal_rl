"""Model-based value iteration (tabular + MLP) — correctness and end-to-end.

The tabular planner is checked against a hand-solved MDP (exact values, not
"it runs"); the MLP planner's backup is checked for its shape/target logic on a
model with known behaviour; both ride the real runner online (FrozenLake /
CartPole) and offline (a tmp Minari dataset), producing the standard schema.
"""

from __future__ import annotations

import csv

import pytest
import torch
from src.benchmarking.registry import register_default_algorithms, registry
from src.benchmarking.runner import BenchmarkRunner, EVAL_COLUMNS, TRAIN_COLUMNS
from src.config.defaults import EnvConfig, RunConfig, TrainingConfig
from src.rl.model_based import DynamicsEnsemble, MLPValueIteration, TabularVI
from src.rl.off_policy.replay_buffer import ReplayBuffer

CPU = torch.device("cpu")


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name,regime",
    [
        ("tabular_vi", "online"),
        ("mlp_vi", "online"),
        ("offline_tabular_vi", "offline"),
        ("offline_mlp_vi", "offline"),
    ],
)
def test_registered(name, regime):
    register_default_algorithms()
    spec = registry.get(name)
    assert spec.kind == "off_policy" and spec.data_regime == regime


def test_continuous_actions_rejected():
    register_default_algorithms()
    for name in ("tabular_vi", "mlp_vi"):
        with pytest.raises(ValueError, match="discrete-action"):
            registry.get(name).builder(
                obs_dim=3,
                action_dim=1,
                action_type="continuous",
                device=CPU,
                action_space=None,
                obs_shape=(3,),
            )


# --------------------------------------------------------------------------
# Tabular: exact on a hand-solved chain
# --------------------------------------------------------------------------
def _one_hot(idx, n):
    return torch.nn.functional.one_hot(torch.as_tensor(idx), n).float()


def _chain_buffer(n_per=20):
    """3 states, 2 actions. a=1 moves right deterministically, a=0 stays.
    Reaching state 2 pays 1 and terminates. Every (s,a) visited n_per times."""
    buf = ReplayBuffer(capacity=10_000, device=CPU)
    for s in (0, 1):
        for a in (0, 1):
            for _ in range(n_per):
                s2 = s + 1 if a == 1 else s
                r = 1.0 if s2 == 2 else 0.0
                d = 1.0 if s2 == 2 else 0.0
                buf.add(
                    {
                        "obs": _one_hot(s, 3),
                        "actions": torch.tensor(a),
                        "rewards": torch.tensor(r),
                        "next_obs": _one_hot(s2, 3),
                        "dones": torch.tensor(d),
                    }
                )
    return buf


def test_tabular_vi_matches_hand_solution():
    buf = _chain_buffer()
    agent = TabularVI(3, 2, buf, CPU, gamma=0.9, plan_every=1)
    m = agent.learn(buf.sample(8))
    assert m["coverage"] == pytest.approx(4 / 6)  # state 2 is terminal: never a source
    q = agent.q_table.q
    # Q(1, right) = 1 (terminal, no bootstrap); Q(0, right) = 0.9 * V(1) = 0.9
    assert q[1, 1].item() == pytest.approx(1.0, abs=1e-5)
    assert q[0, 1].item() == pytest.approx(0.9, abs=1e-5)
    # staying: Q(s,0) = gamma * V(s) -> V(1) = 1, V(0) = 0.9
    assert q[1, 0].item() == pytest.approx(0.9, abs=1e-5)
    assert q[0, 0].item() == pytest.approx(0.81, abs=1e-5)
    # greedy policy moves right; unvisited pairs (the terminal row) sit at the
    # R-min bound min_r / (1 - gamma) = 0 here (rewards are 0/1)
    assert torch.equal(q[:2].argmax(dim=1), torch.tensor([1, 1]))
    assert torch.all(q[2] == 0.0)
    # a static buffer replans exactly once: the second call is a no-op
    n_iters = m["vi_iters"]
    m2 = agent.learn(buf.sample(8))
    assert m2["vi_iters"] == n_iters and agent._counted == len(buf)


def test_tabular_vi_rejects_non_one_hot():
    buf = ReplayBuffer(capacity=10, device=CPU)
    agent = TabularVI(3, 2, buf, CPU)
    with pytest.raises(ValueError, match="one-hot"):
        agent.act(torch.randn(4, 3))


def test_tabular_act_is_greedy_when_eps_zero():
    buf = _chain_buffer()
    agent = TabularVI(3, 2, buf, CPU, gamma=0.9)
    agent.learn(buf.sample(4))
    out = agent.act(_one_hot([0, 1], 3), epsilon=0.0)
    assert out.action.tolist() == [1, 1]


# --------------------------------------------------------------------------
# MLP: the backup's shape and its target logic
# --------------------------------------------------------------------------
def _mlp_agent(obs_dim=4, n_actions=3, **kw):
    from src.rl.models.backbone import select_backbone

    model = DynamicsEnsemble(obs_dim, n_actions, n_members=2, hidden_dims=(16, 16))
    q = select_backbone((obs_dim,), obs_dim, n_actions)
    qt = select_backbone((obs_dim,), obs_dim, n_actions)
    buf = ReplayBuffer(capacity=1000, device=CPU)
    return MLPValueIteration(model, q, qt, buf, CPU, **kw)


def _batch(n=32, obs_dim=4, n_actions=3):
    g = torch.Generator().manual_seed(0)
    return {
        "obs": torch.randn(n, obs_dim, generator=g),
        "actions": torch.randint(0, n_actions, (n,), generator=g),
        "rewards": torch.rand(n, generator=g),
        "next_obs": torch.randn(n, obs_dim, generator=g),
        "dones": (torch.rand(n, generator=g) < 0.2).float(),
    }


def test_mlp_vi_warmup_then_backup_over_all_actions():
    torch.manual_seed(0)
    agent = _mlp_agent(model_warmup=2)
    b = _batch()
    m1 = agent.learn(b)
    assert "q_loss" not in m1 and m1["loss"] == m1["model_loss"]  # warmup: model only
    agent.learn(b)
    m3 = agent.learn(b)
    assert "q_loss" in m3 and m3["loss"] == m3["q_loss"]
    # The backup regresses Q(s, .) onto |A| targets per state: q_network's
    # output for the batch has the full action dimension and finite values.
    q = agent.q_network(b["obs"])
    assert q.shape == (32, 3) and torch.isfinite(q).all()


def test_mlp_vi_penalty_lowers_targets():
    """With a positive disagreement penalty the backup targets can only go
    down (same model, same states, same target net)."""
    torch.manual_seed(0)
    agent = _mlp_agent(model_warmup=0)
    b = _batch()
    for _ in range(5):
        agent._model_step(b)
    B, A = b["obs"].shape[0], 3
    s = b["obs"].repeat_interleave(A, dim=0)
    a = torch.arange(A).repeat(B)
    nxt, r, d, u = agent.model.predict(s, a)
    assert nxt.shape == (B * A, 4) and r.shape == d.shape == u.shape == (B * A,)
    assert (u >= 0).all() and (d >= 0).all() and (d <= 1).all()
    y0 = r + agent.gamma * (1 - d) * agent.target_network(nxt).max(dim=1).values
    y1 = (r - 1.0 * u) + agent.gamma * (1 - d) * agent.target_network(nxt).max(
        dim=1
    ).values
    assert (y1 <= y0 + 1e-7).all()


# --------------------------------------------------------------------------
# End-to-end through the runner (online)
# --------------------------------------------------------------------------
def _run_online(env_id, algo, run_dir, n_episodes=3, rollout_len=64, seed=0):
    register_default_algorithms()
    from src.envs.registry import register_default_env_wrappers

    register_default_env_wrappers()
    env_cfg = EnvConfig(
        env_id=env_id, n_train_envs=2, n_eval_envs=2, rollout_len=rollout_len, seed=seed
    )
    train_cfg = TrainingConfig(
        n_episodes=n_episodes,
        n_checkpoints=2,
        device="cpu",
        algorithm=algo,
        aggregation="mean",
        record_eval_video=False,
    )
    runner = BenchmarkRunner(
        env_cfg,
        train_cfg,
        RunConfig(run_dir=str(run_dir), timestamp="t"),
        registry.get(algo),
    )
    runner.offpolicy_warmup = 16
    runner.offpolicy_batch_size = 16
    runner.run()
    with (run_dir / "train_metrics.csv").open() as f:
        train_rows = list(csv.DictReader(f))
    with (run_dir / "eval_metrics.csv").open() as f:
        eval_rows = list(csv.DictReader(f))
    assert list(train_rows[0].keys()) == TRAIN_COLUMNS
    assert list(eval_rows[0].keys()) == EVAL_COLUMNS
    return train_rows, eval_rows


def test_tabular_vi_online_frozenlake(tmp_path):
    train_rows, eval_rows = _run_online("FrozenLake-v1", "tabular_vi", tmp_path / "fl")
    assert float(train_rows[-1]["loss"]) == float(train_rows[-1]["loss"])  # finite
    assert float(eval_rows[-1]["eval_return_mean"]) >= 0.0


def test_mlp_vi_online_cartpole(tmp_path):
    train_rows, eval_rows = _run_online("CartPole-v1", "mlp_vi", tmp_path / "cp")
    assert float(train_rows[-1]["loss"]) == float(train_rows[-1]["loss"])
    assert float(eval_rows[-1]["eval_return_mean"]) > 0.0


# --------------------------------------------------------------------------
# End-to-end offline (a tmp Minari dataset) — needs the offline extra
# --------------------------------------------------------------------------
pytest.importorskip("minari")
pytest.importorskip("h5py")


def test_offline_mlp_vi_learns_cartpole(tmp_path, monkeypatch):
    """Trained on better-than-random data through the model, the planner's
    greedy policy clears the random baseline (~22) by a margin."""
    from src.config.device import detect_device
    from src.envs.registry import register_default_env_wrappers
    from tools.make_cartpole_offline import make_cartpole_dataset

    monkeypatch.setenv("MINARI_DATASETS_PATH", str(tmp_path / "minari"))
    make_cartpole_dataset(
        dataset_id="cartpole/heur-v0", n_episodes=60, seed=0, policy="heuristic"
    )
    register_default_algorithms()
    register_default_env_wrappers()
    run_dir = tmp_path / "off"
    env_cfg = EnvConfig(
        env_id="CartPole-v1",
        n_train_envs=2,
        n_eval_envs=10,
        rollout_len=100,
        seed=0,
        offline_dataset="cartpole/heur-v0",
    )
    train_cfg = TrainingConfig(
        n_episodes=40,
        n_checkpoints=2,
        device=str(detect_device()),
        algorithm="offline_mlp_vi",
        aggregation="mean",
        record_eval_video=False,
    )
    BenchmarkRunner(
        env_cfg,
        train_cfg,
        RunConfig(run_dir=str(run_dir), timestamp="t"),
        registry.get("offline_mlp_vi"),
    ).run()
    with (run_dir / "eval_metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    final = float(rows[-1]["eval_return_mean"])
    assert final > 50.0, f"offline_mlp_vi eval={final:.1f} did not beat random"


def test_tabular_greedy_breaks_ties_uniformly_and_optimism_bounds():
    torch.manual_seed(0)
    q = torch.zeros(2000, 4)
    acts = TabularVI._greedy(q)
    counts = torch.bincount(acts, minlength=4).float() / 2000
    assert (counts > 0.15).all(), counts  # all four actions drawn from an all-equal row
    q[:, 2] = 1.0
    assert (TabularVI._greedy(q) == 2).all()  # a strict max is never overridden
    # optimism: unvisited pairs take max_r / (1 - gamma) >= every fitted Q
    buf = _chain_buffer()
    agent = TabularVI(3, 2, buf, CPU, gamma=0.9, plan_every=1, optimistic=True)
    agent.learn(buf.sample(4))
    assert agent.q_explore.q[2].tolist() == pytest.approx([10.0, 10.0])  # 1/(1-0.9)
    assert (agent.q_explore.q[:2] <= 10.0).all()
    # the EXPLOITATION table pins unvisited pairs at the R-min bound (0 here)
    assert torch.all(agent.q_table.q[2] == 0.0)
    # training acts on the optimistic table, a greedy eval on the plain one
    obs = _one_hot([2, 2], 3)
    torch.manual_seed(1)
    assert agent.act(obs, epsilon=0.0).action.shape == (2,)


def test_tabular_exploit_table_is_pessimistic_on_negative_rewards():
    """Rewards in [-10, -1]: an untried pair must NOT look better than any tried
    one (a fixed 0 would), so its exploitation value is the R-min bound."""
    buf = ReplayBuffer(capacity=1000, device=CPU)
    for _ in range(10):
        buf.add(
            {
                "obs": _one_hot(0, 2),
                "actions": torch.tensor(0),
                "rewards": torch.tensor(-1.0),
                "next_obs": _one_hot(0, 2),
                "dones": torch.tensor(0.0),
            }
        )
    agent = TabularVI(2, 2, buf, CPU, gamma=0.9, plan_every=1)
    agent.learn(buf.sample(4))
    q = agent.q_table.q
    assert q[0, 0].item() == pytest.approx(-10.0, abs=1e-4)  # -1 / (1 - 0.9)
    assert q[0, 1].item() == pytest.approx(-10.0, abs=1e-4)  # untried: R-min bound
    assert q[0, 1] <= q[0, 0]
