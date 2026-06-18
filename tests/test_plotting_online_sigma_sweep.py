"""feat/cells-7-8-online-variants — online σ-sweep renderer (Cells 7/8 online).

The online analog of the value-trace σ-sweep: eval return vs σ across sibling
online_confounded_* runs, one figure per env, one line per algorithm. No
apparent-Q (online runs have no offline learner), so a single row.

These tests cover: it renders one figure per env with a line per algo; it skips
cleanly with no siblings; and — the key partition invariant — it reads ONLY the
online_confounded_* dirs and never the offline confounded_* siblings (those carry
offline_value_trace.csv and are the value-trace renderer's job).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless; must precede the pyplot import

import pandas as pd  # noqa: E402
from src.benchmarking.plotting import (  # noqa: E402
    _collect_online_sigma_sweep,
    render_online_sigma_sweep,
)

ALGOS = ["ppo", "dqn"]
ENVS = ["CartPole-v1", "Acrobot-v1"]
NNN = {0.0: "000", 0.25: "025", 0.5: "050", 0.75: "075", 1.0: "100"}


def _eval_df(sigma):
    """Final-checkpoint eval rows for both algos/envs. DQN degrades with σ
    (full confounding); PPO is flat (reward-noise only) — the structural wedge."""
    rows = []
    for ep in range(3):
        for env in ENVS:
            for a in ALGOS:
                base = 100.0 - (50.0 * sigma if a == "dqn" else 0.0)
                rows.append(
                    {
                        "episode": ep,
                        "algorithm": a,
                        "environment": env,
                        "eval_return_mean": base + ep,
                        "eval_return_std": 2.0,
                    }
                )
    return pd.DataFrame(rows)


def _write_online_run(run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    sigma = int(run_dir.name.split("sigma_")[1][:3]) / 100.0
    _eval_df(sigma).to_csv(run_dir / "eval_metrics.csv", index=False)


def _make_online_cell_dir(tmp_path, sigmas):
    """runs/rl_regimes/cell_7/online_confounded_sigma_NNN_discrete_gated_<ts>/."""
    cell = tmp_path / "runs" / "rl_regimes" / "cell_7"
    dirs = {}
    for i, s in enumerate(sorted(sigmas)):
        d = cell / f"online_confounded_sigma_{NNN[s]}_discrete_gated_2026061800000{i}"
        _write_online_run(d)
        dirs[s] = d
    return dirs


def test_online_sigma_sweep_renders_one_figure_per_env(tmp_path):
    dirs = _make_online_cell_dir(tmp_path, [0.0, 0.25, 0.5, 0.75, 1.0])
    outdir = tmp_path / "out"
    render_online_sigma_sweep(dirs[0.5], outdir, ["png"])
    pngs = sorted(
        p.name for p in (outdir / "plots" / "online_sigma_sweep").glob("*.png")
    )
    assert pngs == ["Acrobot-v1.png", "CartPole-v1.png"]


def test_online_sigma_sweep_collects_all_sigmas_and_algos(tmp_path):
    dirs = _make_online_cell_dir(tmp_path, [0.0, 0.5, 1.0])
    records, sigma_dirs = _collect_online_sigma_sweep(dirs[0.5])
    assert set(sigma_dirs) == {0.0, 0.5, 1.0}
    # One record series per (env, algo); each has all three σ points.
    assert set(records) == {(e, a) for e in ENVS for a in ALGOS}
    dqn_pts = records[("CartPole-v1", "dqn")]
    assert [p[0] for p in dqn_pts] == [0.0, 0.5, 1.0]
    # DQN return falls as σ grows (the confounding wedge); PPO stays flat.
    dqn_means = [p[1] for p in dqn_pts]
    ppo_means = [p[1] for p in records[("CartPole-v1", "ppo")]]
    assert dqn_means[0] > dqn_means[-1]
    assert ppo_means[0] == ppo_means[-1]


def test_online_sigma_sweep_skips_when_no_siblings(tmp_path):
    run_dir = (
        tmp_path / "runs" / "rl_regimes" / "cell_7" / "some_other_run_discrete_gated_x"
    )
    run_dir.mkdir(parents=True)
    _eval_df(0.0).to_csv(run_dir / "eval_metrics.csv", index=False)
    outdir = tmp_path / "out"
    render_online_sigma_sweep(run_dir, outdir, ["png"])
    assert not (outdir / "plots" / "online_sigma_sweep").exists()


def test_online_collector_ignores_offline_confounded_siblings(tmp_path):
    """Partition invariant: offline confounded_* runs (with value trace) must NOT
    be picked up by the online collector — even when they sit in the same dir."""
    dirs = _make_online_cell_dir(tmp_path, [0.0, 0.5, 1.0])
    cell = tmp_path / "runs" / "rl_regimes" / "cell_7"
    # An offline sibling at a σ NOT in the online set, with a value trace.
    offline = cell / "confounded_sigma_075_discrete_gated_20260617000000"
    offline.mkdir(parents=True)
    _eval_df(0.75).to_csv(offline / "eval_metrics.csv", index=False)
    pd.DataFrame(
        {"epoch": [0], "algorithm": ["dqn"], "environment": ["CartPole-v1"]}
    ).to_csv(offline / "offline_value_trace.csv", index=False)
    _records, sigma_dirs = _collect_online_sigma_sweep(dirs[0.5])
    assert set(sigma_dirs) == {0.0, 0.5, 1.0}  # 0.75 offline sibling excluded
