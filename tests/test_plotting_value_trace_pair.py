"""feat/value-trace-pair-overlay — floor-vs-ceiling overlay in the standard split.

When ONE run dir contains a base algo and its *_oracle_u sibling, the standard
value_trace per-config renderer collapses the two separate figures into one
overlaid figure (base solid, oracle Q_adj dashed, u=0 anchor dotted). With no
pair present it is byte-identical to the per-config output. No new split.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless; must precede the plotting (pyplot) import

import pandas as pd  # noqa: E402
from src.benchmarking.plotting import (  # noqa: E402
    render_value_trace_per_config,
    short_algo_label,
)

ENV = "CartPole-v1"


def _vt_rows(algo, epochs=8, base=10.0, with_u0=False):
    rows = []
    for ep in range(epochs):
        row = {
            "epoch": ep,
            "algorithm": algo,
            "environment": ENV,
            "apparent_value_iqm": base + ep,
            "apparent_value_iqr_std": 1.0,
        }
        if with_u0:
            row["apparent_value_u0_iqm"] = base + 0.4 * ep
            row["apparent_value_u0_iqr_std"] = 0.5
        rows.append(row)
    return rows


def _eval_rows(algo, epochs=8, base=50.0):
    return [
        {
            "episode": ep,
            "algorithm": algo,
            "environment": ENV,
            "eval_return_mean": base - ep,
            "eval_return_std": 2.0,
        }
        for ep in range(epochs)
    ]


def _write_run(run_dir, algos_with_u0):
    """algos_with_u0: dict {algo_name: writes_u0_columns}."""
    run_dir.mkdir(parents=True, exist_ok=True)
    vt, ev = [], []
    for algo, with_u0 in algos_with_u0.items():
        vt += _vt_rows(algo, with_u0=with_u0)
        ev += _eval_rows(algo)
    # train_metrics with all-empty train_return_mean = offline signature
    pd.DataFrame(
        {
            "episode": [0, 1],
            "algorithm": ["x"] * 2,
            "environment": [ENV] * 2,
            "train_return_mean": [None, None],
        }
    ).to_csv(run_dir / "train_metrics.csv", index=False)
    pd.DataFrame(ev).to_csv(run_dir / "eval_metrics.csv", index=False)
    pd.DataFrame(vt).to_csv(run_dir / "offline_value_trace.csv", index=False)
    return run_dir


def _capture(run_dir, outdir, monkeypatch):
    captured = []
    real_close = matplotlib.pyplot.close

    def _cap(fig=None):
        if hasattr(fig, "axes"):
            captured.append(fig)
        return real_close(fig)

    monkeypatch.setattr(matplotlib.pyplot, "close", _cap)
    render_value_trace_per_config(
        run_dir, outdir, "episodes", ["png"], "Training epochs", 1024
    )
    return captured


def _styles(fig):
    return {ln.get_linestyle() for ax in fig.axes for ln in ax.get_lines()}


def test_short_algo_label_brands_oracle_variant():
    assert short_algo_label("offline_dqn_oracle_u") == "offline_dqn (oracle-U, true U)"
    assert short_algo_label("offline_dqn") == "offline_dqn"  # base unchanged


def test_pair_renders_single_overlay_with_anchor(tmp_path, monkeypatch):
    run_dir = _write_run(
        tmp_path / "run",
        {"offline_dqn": False, "offline_dqn_oracle_u": True},
    )
    outdir = tmp_path / "out"
    captured = _capture(run_dir, outdir, monkeypatch)

    pdir = outdir / "plots" / "value_trace" / "per_config"
    pngs = {p.name for p in pdir.glob("*.png")}
    # ONE overlaid figure for the pair; NO separate per-config figures for either.
    assert f"offline_dqn__vs__oracle_u_{ENV}.png" in pngs
    assert f"offline_dqn_{ENV}.png" not in pngs
    assert f"offline_dqn_oracle_u_{ENV}.png" not in pngs
    assert len(captured) == 1
    styles = _styles(captured[0])
    assert "--" in styles  # oracle variant dashed
    assert ":" in styles  # u=0 anchor dotted


def test_base_only_run_is_unchanged_per_config(tmp_path, monkeypatch):
    run_dir = _write_run(tmp_path / "run", {"offline_dqn": False, "cql": False})
    outdir = tmp_path / "out"
    captured = _capture(run_dir, outdir, monkeypatch)

    pdir = outdir / "plots" / "value_trace" / "per_config"
    pngs = {p.name for p in pdir.glob("*.png")}
    # No pair -> one per-config figure per algo, no overlay file.
    assert pngs == {f"offline_dqn_{ENV}.png", f"cql_{ENV}.png"}
    assert not any("__vs__" in n for n in pngs)
    # No dotted anchor anywhere (no oracle variant present).
    assert all(":" not in _styles(fig) for fig in captured)
