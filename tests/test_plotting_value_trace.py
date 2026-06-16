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
    _collect_sigma_sweep,
    _plot_sigma_sweep_ax,
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


def test_value_trace_per_config_renders(tmp_path):
    run_dir = tmp_path / "run"
    _write_run(run_dir, value_trace=True, eval_=True)
    outdir = tmp_path / "out"
    render_value_trace_per_config(
        run_dir, outdir, "episodes", ["png"], "Training epochs", 1024
    )
    pngs = list((outdir / "plots" / "value_trace" / "per_config").glob("*.png"))
    texs = list((outdir / "tables" / "value_trace" / "per_config").glob("*.tex"))
    assert len(pngs) == len(ALGOS)  # one per (env, algo)
    assert len(texs) == len(ALGOS)


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


def test_sigma_sweep_handles_partial_and_full_sibling_sets(tmp_path):
    # --- partial: a single σ run (sweep incomplete) ---
    p = tmp_path / "partial"
    dirs = _make_cell_dir(p, [0.5])
    out_p = p / "out"
    render_value_trace_sigma_sweep(dirs[0.5], out_p, ["png"])
    pngs = list((out_p / "plots" / "value_trace" / "sigma_sweep").glob("*.png"))
    assert len(pngs) == len(ALGOS)  # one per (env, algo), 1 data point each
    records, sigma_dirs = _collect_sigma_sweep(dirs[0.5])
    assert len(sigma_dirs) == 1  # incomplete sweep

    # --- full: all five σ values ---
    f = tmp_path / "full"
    dirs = _make_cell_dir(f, [0.0, 0.25, 0.5, 0.75, 1.0])
    out_f = f / "out"
    render_value_trace_sigma_sweep(dirs[0.5], out_f, ["png"])
    records, sigma_dirs = _collect_sigma_sweep(dirs[0.5])
    assert len(sigma_dirs) == 5
    # every (env, algo) has 5 σ points
    for (env, algo), pts in records.items():
        assert sorted(p[0] for p in pts) == [0.0, 0.25, 0.5, 0.75, 1.0]
    # σ-sweep table has 5 rows
    tex = (
        out_f / "tables" / "value_trace" / "sigma_sweep" / f"cql_{ENV}.tex"
    ).read_text()
    assert tex.count("\\\\") >= 6  # header + 5 data rows (+ tabular close)

    # anchor annotation: σ=0.0 present -> axvline drawn; absent -> not.
    fig, ax = matplotlib.pyplot.subplots()
    assert _plot_sigma_sweep_ax(ax, [(0.0, 1, 2), (0.5, 3, 1)], "cql") is True
    assert any(tuple(ln.get_xdata()) == (0.0, 0.0) for ln in ax.lines)  # axvline at 0
    matplotlib.pyplot.close(fig)
    fig, ax = matplotlib.pyplot.subplots()
    assert _plot_sigma_sweep_ax(ax, [(0.25, 1, 2), (0.5, 3, 1)], "cql") is False
    matplotlib.pyplot.close(fig)


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
