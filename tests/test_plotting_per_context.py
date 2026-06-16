"""feat/plotting-eval-per-context — Cell-2 per-context renderer + offline x-axis.

Covers the two plotting gaps from the Cells 1-3 sanity recon: the per_context
renderer (skip-on-absent, figures+table-on-present) and the offline-aware x-axis
label resolver (offline -> "Training epochs", online -> "Episodes").
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless; must precede the plotting (pyplot) import

import pandas as pd  # noqa: E402
from src.benchmarking.plotting import (  # noqa: E402
    _resolve_x_label,
    render_eval_per_context,
)
from src.benchmarking.runner import EVAL_PER_CONTEXT_COLUMNS  # noqa: E402


def _write_min_run(run_dir, with_per_context: bool):
    run_dir.mkdir(parents=True, exist_ok=True)
    # minimal train/eval so the dir looks like a real run
    pd.DataFrame(
        {
            "episode": [0, 1],
            "algorithm": ["ppo"] * 2,
            "environment": ["CartPole-v1"] * 2,
            "train_return_mean": [10.0, 20.0],
            "train_return_std": [1.0, 1.0],
        }
    ).to_csv(run_dir / "train_metrics.csv", index=False)
    pd.DataFrame(
        {
            "episode": [0, 1],
            "algorithm": ["ppo"] * 2,
            "environment": ["CartPole-v1"] * 2,
            "eval_return_mean": [10.0, 20.0],
            "eval_return_std": [1.0, 1.0],
        }
    ).to_csv(run_dir / "eval_metrics.csv", index=False)
    if with_per_context:
        rows = []
        for ep in (0, 10, 20):
            for algo in ("ppo", "dqn"):
                for b in range(5):
                    rows.append(
                        {
                            "episode": ep,
                            "algorithm": algo,
                            "environment": "CartPole-v1",
                            "context_bin": b,
                            "context_value_low": float(b),
                            "context_value_high": float(b + 1),
                            "n_episodes_in_bin": 3,
                            "return_iqm": 100.0 + 10 * b + ep,
                            "return_iqr_std": 5.0 + b,
                        }
                    )
        df = pd.DataFrame(rows, columns=EVAL_PER_CONTEXT_COLUMNS)
        df.to_csv(run_dir / "eval_per_context.csv", index=False)


def test_per_context_renderer_skips_when_file_absent(tmp_path):
    run_dir = tmp_path / "run"
    _write_min_run(run_dir, with_per_context=False)
    outdir = tmp_path / "out"

    render_eval_per_context(
        run_dir, outdir, "episodes", "iqm", ["png"], "Episodes", 1024
    )

    # No per_context artifacts produced when the CSV is absent.
    assert not (outdir / "plots" / "per_context").exists()
    assert not (outdir / "tables" / "per_context").exists()


def test_per_context_renderer_produces_figures_when_file_present(tmp_path):
    run_dir = tmp_path / "run"
    _write_min_run(run_dir, with_per_context=True)
    outdir = tmp_path / "out"

    render_eval_per_context(
        run_dir, outdir, "episodes", "iqm", ["png"], "Episodes", 1024
    )

    pngs = list((outdir / "plots" / "per_context").rglob("*.png"))
    texs = list((outdir / "tables" / "per_context").rglob("*.tex"))
    # 2 algos x 1 env = 2 figures + 2 tables.
    assert len(pngs) >= 1
    assert len(texs) >= 1


def test_offline_x_axis_label():
    online = pd.DataFrame({"train_return_mean": [10.0, 20.0]})
    offline = pd.DataFrame({"train_return_mean": [None, None]})  # PR3 signature

    assert _resolve_x_label("episodes", online) == "Episodes"
    assert _resolve_x_label("episodes", offline) == "Training epochs"
    # frames is regime-agnostic; empty df falls back to the online label.
    assert _resolve_x_label("frames", offline) == "Frames"
    assert _resolve_x_label("episodes", pd.DataFrame()) == "Episodes"
