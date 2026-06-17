"""feat/plotting-value-trace — Cell 7/8 value-trace renderers + per-context snapshot.

Covers the two deferred plotting items: the offline_value_trace.csv renderers
(per-config apparent-vs-true overlay + σ-sweep panel) and the final-checkpoint
per-context snapshot. All gated on file existence; non-relevant runs skip.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless; must precede the plotting (pyplot) import

import pandas as pd  # noqa: E402
from src.benchmarking.plotting import (  # noqa: E402
    _build_sigma_sweep_env_figure,
    _collect_sigma_sweep,
    _resolve_x_label,
    render_eval_per_context_final,
    render_value_trace_per_config,
    render_value_trace_sigma_sweep,
)

ALGOS = ["offline_dqn", "cql", "iql"]
ENV = "CartPole-v1"


def _value_trace_df(epochs=10, algos=ALGOS, env=ENV, base=10.0):
    rows = []
    for ep in range(epochs):
        for a in algos:
            rows.append(
                {
                    "epoch": ep,
                    "algorithm": a,
                    "environment": env,
                    "apparent_value_iqm": base + ep,  # apparent inflates
                    "apparent_value_iqr_std": 1.0,
                }
            )
    return pd.DataFrame(rows)


def _eval_df(epochs=10, algos=ALGOS, env=ENV, base=50.0):
    rows = []
    for ep in range(epochs):
        for a in algos:
            rows.append(
                {
                    "episode": ep,
                    "algorithm": a,
                    "environment": env,
                    "eval_return_mean": base - ep,  # true decays
                    "eval_return_std": 2.0,
                }
            )
    return pd.DataFrame(rows)


def _write_run(run_dir, value_trace=True, eval_=True):
    run_dir.mkdir(parents=True, exist_ok=True)
    # train_metrics with all-empty train_return_mean = offline signature
    pd.DataFrame(
        {
            "episode": [0, 1],
            "algorithm": ["cql"] * 2,
            "environment": [ENV] * 2,
            "train_return_mean": [None, None],
        }
    ).to_csv(run_dir / "train_metrics.csv", index=False)
    if eval_:
        _eval_df().to_csv(run_dir / "eval_metrics.csv", index=False)
    if value_trace:
        _value_trace_df().to_csv(run_dir / "offline_value_trace.csv", index=False)


# --------------------------------------------------------------------------
# Renderer A — per-config overlay
# --------------------------------------------------------------------------
def test_value_trace_per_config_skips_when_file_absent(tmp_path):
    run_dir = tmp_path / "run"
    _write_run(run_dir, value_trace=False, eval_=True)
    outdir = tmp_path / "out"
    render_value_trace_per_config(
        run_dir, outdir, "episodes", ["png"], "Training epochs", 1024
    )
    assert not (outdir / "plots" / "value_trace").exists()
    assert not (outdir / "tables" / "value_trace").exists()


def test_value_trace_per_config_renders(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    _write_run(run_dir, value_trace=True, eval_=True)
    outdir = tmp_path / "out"

    # Capture the figures before the renderer closes them, to assert the
    # twin-row layout (apparent on top, true on bottom) rather than just files.
    captured = []
    real_close = matplotlib.pyplot.close

    def _capture_close(fig=None):
        if hasattr(fig, "axes"):
            captured.append(fig)
        return real_close(fig)

    monkeypatch.setattr(matplotlib.pyplot, "close", _capture_close)
    render_value_trace_per_config(
        run_dir, outdir, "episodes", ["png"], "Training epochs", 1024
    )
    pngs = list((outdir / "plots" / "value_trace" / "per_config").glob("*.png"))
    texs = list((outdir / "tables" / "value_trace" / "per_config").glob("*.tex"))
    assert len(pngs) == len(ALGOS)  # one per (env, algo)
    assert len(texs) == len(ALGOS)
    # twin-row: each figure has 2 axes with distinct, scale-explicit y-labels
    assert len(captured) == len(ALGOS)
    for fig in captured:
        assert len(fig.axes) == 2
        ylabels = {ax.get_ylabel() for ax in fig.axes}
        assert ylabels == {"apparent Q (discounted)", "true return (undiscounted)"}


# --------------------------------------------------------------------------
# Renderer B — σ-sweep
# --------------------------------------------------------------------------
def _make_cell_dir(tmp_path, sigmas):
    """Build runs/rl_regimes/cell_7/confounded_sigma_NNN_discrete_<ts>/ for each σ."""
    cell = tmp_path / "runs" / "rl_regimes" / "cell_7"
    nnn = {0.0: "000", 0.25: "025", 0.5: "050", 0.75: "075", 1.0: "100"}
    dirs = {}
    for i, s in enumerate(sorted(sigmas)):
        d = cell / f"confounded_sigma_{nnn[s]}_discrete_2026061600000{i}"
        _write_run(d, value_trace=True, eval_=True)
        dirs[s] = d
    return dirs


def _algo_to_pts(records):
    """Collapse records[(env, algo)] -> {algo: pts} (single-env test fixtures)."""
    return {algo: pts for (env, algo), pts in records.items()}


def test_sigma_sweep_handles_partial_and_full_sibling_sets(tmp_path):
    # --- partial: a single σ run (sweep incomplete) ---
    p = tmp_path / "partial"
    dirs = _make_cell_dir(p, [0.5])
    out_p = p / "out"
    render_value_trace_sigma_sweep(dirs[0.5], out_p, ["png"])
    pngs = list((out_p / "plots" / "value_trace" / "sigma_sweep").glob("*.png"))
    assert len(pngs) == 1  # one figure per env (not per (env, algo))
    assert pngs[0].name == f"{ENV}.png"
    records, sigma_dirs = _collect_sigma_sweep(dirs[0.5])
    assert len(sigma_dirs) == 1  # incomplete sweep

    # --- full: all five σ values ---
    f = tmp_path / "full"
    dirs = _make_cell_dir(f, [0.0, 0.25, 0.5, 0.75, 1.0])
    out_f = f / "out"
    render_value_trace_sigma_sweep(dirs[0.5], out_f, ["png"])
    pngs = list((out_f / "plots" / "value_trace" / "sigma_sweep").glob("*.png"))
    assert len(pngs) == 1 and pngs[0].name == f"{ENV}.png"  # per-env figure
    records, sigma_dirs = _collect_sigma_sweep(dirs[0.5])
    assert len(sigma_dirs) == 5
    # every (env, algo) has 5 σ points
    for (env, algo), pts in records.items():
        assert sorted(p[0] for p in pts) == [0.0, 0.25, 0.5, 0.75, 1.0]

    # twin-row figure shape: 2 rows × N(algo) columns, anchor on the top-left
    fig, axes = _build_sigma_sweep_env_figure(ENV, _algo_to_pts(records))
    assert axes.shape == (2, len(ALGOS))  # 2 rows (apparent / true), N columns
    assert any(tuple(ln.get_xdata()) == (0.0, 0.0) for ln in axes[0][0].lines)
    matplotlib.pyplot.close(fig)

    # σ-sweep table: scale-invariant ratio columns replace the contaminated gap
    tex = (
        out_f / "tables" / "value_trace" / "sigma_sweep" / f"cql_{ENV}.tex"
    ).read_text()
    assert "apparent\\_rel" in tex and "true\\_rel" in tex
    assert "\\textbf{gap}" not in tex
    assert tex.count("\\\\") >= 6  # header + 5 data rows (+ tabular close)

    # anchor absent -> no axvline at σ=0
    fig, axes = _build_sigma_sweep_env_figure(ENV, {"cql": [(0.25, 1, 2), (0.5, 3, 1)]})
    assert not any(tuple(ln.get_xdata()) == (0.0, 0.0) for ln in axes[0][0].lines)
    matplotlib.pyplot.close(fig)


def test_per_algo_y_axes_are_independent():
    # Two algos at very different apparent-Q scales (CQL-like ~10 vs DQN-like ~600).
    # A shared row axis would compress the small-scale algo onto the x-axis; the
    # twin-row layout gives each its own scale.
    small = [(0.0, 10.0, 500.0), (0.5, 12.0, 480.0), (1.0, 11.0, 460.0)]
    big = [(0.0, 100.0, 500.0), (0.5, 600.0, 480.0), (1.0, 300.0, 460.0)]
    fig, axes = _build_sigma_sweep_env_figure(ENV, {"cql": small, "offline_dqn": big})
    # columns are algo-sorted: ["cql", "offline_dqn"]; top row = apparent Q
    ylim_small = axes[0][0].get_ylim()
    ylim_big = axes[0][1].get_ylim()
    assert ylim_small != ylim_big  # each subplot scaled to its own data
    assert ylim_small[1] < ylim_big[1]  # small-scale algo not stretched up
    matplotlib.pyplot.close(fig)


def test_sigma_sweep_filters_by_arm_tag(tmp_path, capsys):
    # A cell dir holding both gated runs (the post-PR#34 Cell 7/8 case) and stray
    # non-gated runs (leftovers from the original sweep). They share the arm token
    # but have different env sets, so the σ-sweep must never aggregate across arms.
    cell = tmp_path / "runs" / "rl_regimes" / "cell_7"
    nnn = {0.0: "000", 0.25: "025", 0.5: "050", 0.75: "075", 1.0: "100"}
    gated = {}
    for i, s in enumerate([0.0, 0.25, 0.5, 0.75, 1.0]):
        d = cell / f"confounded_sigma_{nnn[s]}_discrete_gated_2026061700000{i}"
        _write_run(d, value_trace=True, eval_=True)
        gated[s] = d
    nongated = {}
    for i, s in enumerate([0.25, 0.5, 0.75]):
        d = cell / f"confounded_sigma_{nnn[s]}_discrete_2026061600000{i}"
        _write_run(d, value_trace=True, eval_=True)
        nongated[s] = d

    # current run gated -> only the 5 gated σ values aggregate
    _, sigma_dirs = _collect_sigma_sweep(gated[0.5])
    assert sorted(sigma_dirs) == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert all("_gated_" in d.name for d in sigma_dirs.values())
    assert not any(d in sigma_dirs.values() for d in nongated.values())
    assert "excluding 3 non-gated siblings" in capsys.readouterr().out

    # current run non-gated -> only the 3 non-gated σ values aggregate
    _, sigma_dirs = _collect_sigma_sweep(nongated[0.5])
    assert sorted(sigma_dirs) == [0.25, 0.5, 0.75]
    assert all("_gated_" not in d.name for d in sigma_dirs.values())
    assert "excluding 5 gated siblings" in capsys.readouterr().out


# --------------------------------------------------------------------------
# Renderer C — final-checkpoint per-context snapshot
# --------------------------------------------------------------------------
def _per_context_df(env=ENV, algos=("cql", "iql"), n_bins=5, episodes=(0, 10)):
    rows = []
    for ep in episodes:
        for a in algos:
            for b in range(n_bins):
                rows.append(
                    {
                        "episode": ep,
                        "algorithm": a,
                        "environment": env,
                        "context_bin": b,
                        "context_value_low": float(b),
                        "context_value_high": float(b + 1),
                        "n_episodes_in_bin": 3,
                        "return_iqm": (100.0 + b) if ep == max(episodes) else 0.0,
                        "return_iqr_std": 2.0,
                    }
                )
    return pd.DataFrame(rows)


def test_per_context_final_skips_and_renders(tmp_path):
    # absent -> skip
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    outdir = tmp_path / "out"
    render_eval_per_context_final(run_dir, outdir, ["png"])
    assert not (outdir / "plots" / "per_context_final").exists()

    # present, 2 checkpoints -> only the final checkpoint is plotted/tabulated
    _per_context_df().to_csv(run_dir / "eval_per_context.csv", index=False)
    render_eval_per_context_final(run_dir, outdir, ["png"])
    pngs = list((outdir / "plots" / "per_context_final").glob("*.png"))
    texs = list((outdir / "tables" / "per_context_final").glob("*.tex"))
    assert len(pngs) == 2 and len(texs) == 2  # 2 algos x 1 env
    # the table holds 5 bins (final checkpoint only, not 10 across both checkpoints)
    tex = (outdir / "tables" / "per_context_final" / f"cql_{ENV}.tex").read_text()
    data_rows = [ln for ln in tex.splitlines() if ln and ln[0].isdigit()]
    assert len(data_rows) == 5
    assert "100.000" in tex  # final-checkpoint return values, not the ep-0 zeros


# --------------------------------------------------------------------------
# x-axis label
# --------------------------------------------------------------------------
def test_x_axis_label_value_trace_uses_training_epochs():
    offline = pd.DataFrame({"train_return_mean": [None, None]})
    assert _resolve_x_label("episodes", offline) == "Training epochs"
