"""feat/critic-layout-and-aggregation-honoring — --aggregation flow coverage.

The aggregation audit (2026-06-17) found ZERO tests asserting that the
``--aggregation`` flag actually controls cross-dimension (cross-env / critic)
collapses — the center stat was hardcoded to mean at four sites regardless of
the flag. These tests close that gap:

- A/B/C: unit-test ``_collapse_envs`` directly (mean vs iqm center; spread is
  always mean-of-spreads).
- D: end-to-end — render the critic-overall figure via ``run_plotting`` with
  ``aggregation='iqm'`` and assert the drawn curve matches the IQM env-collapse,
  not the mean env-collapse.
- E: the new critic layout drops the critic dimension and overlays algos — assert
  the figure/file counts and per-figure line counts.

End-to-end tests capture matplotlib output by monkeypatching ``Figure.savefig``
(no PNGs are written), so they inspect the actual drawn ``Line2D`` y-data.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless; must precede the plotting (pyplot) import

import matplotlib.figure as mfig  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from src.benchmarking.plotting import _collapse_envs, _iqm, run_plotting  # noqa: E402

# Five env-centers chosen so IQM (mean of middle 50% = mean of the 3 smallest
# for n=4..5) differs sharply from the arithmetic mean. With <4 values _iqm
# falls back to mean, so the fixtures below always use >=4 collapsed values.
_CENTERS = [1.0, 2.0, 3.0, 4.0, 100.0]
_SPREADS = [5.0, 6.0, 7.0, 8.0, 9.0]


def _collapse_input(centers, spreads):
    """One algo, one x, N envs — the shape a cross-env collapse reduces."""
    return pd.DataFrame(
        {
            "environment": [f"Env-{i}" for i in range(len(centers))],
            "algorithm": ["ppo"] * len(centers),
            "x": [0] * len(centers),
            "center": list(centers),
            "spread": list(spreads),
        }
    )


# ---------------------------------------------------------------------------
# A/B/C — _collapse_envs unit tests
# ---------------------------------------------------------------------------
def test_cross_env_collapse_honors_aggregation_mean():
    df = _collapse_input(_CENTERS, _SPREADS)
    out = _collapse_envs(df, ["algorithm", "x"], "mean")
    assert out.shape[0] == 1
    assert out.iloc[0]["center"] == pytest.approx(float(np.mean(_CENTERS)))


def test_cross_env_collapse_honors_aggregation_iqm():
    df = _collapse_input(_CENTERS, _SPREADS)
    out = _collapse_envs(df, ["algorithm", "x"], "iqm")
    assert out.shape[0] == 1
    assert out.iloc[0]["center"] == pytest.approx(_iqm(np.asarray(_CENTERS)))
    # Guard the fixture: IQM must actually differ from mean, else the test is vacuous.
    assert _iqm(np.asarray(_CENTERS)) != pytest.approx(float(np.mean(_CENTERS)))


def test_cross_env_collapse_uses_mean_for_spread_regardless_of_flag():
    df = _collapse_input(_CENTERS, _SPREADS)
    out_mean = _collapse_envs(df, ["algorithm", "x"], "mean")
    out_iqm = _collapse_envs(df, ["algorithm", "x"], "iqm")
    expected = float(np.mean(_SPREADS))
    assert out_mean.iloc[0]["spread"] == pytest.approx(expected)
    assert out_iqm.iloc[0]["spread"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# D/E — end-to-end critic render (layout + aggregation flow)
# ---------------------------------------------------------------------------
# 4 envs so the env-collapse has >=4 values (IQM != mean); 2 algos, 2 critics.
_E2E_ENVS = ["Env-0", "Env-1", "Env-2", "Env-3"]
_E2E_ALGOS = ["algoA", "algoB"]
_E2E_EPISODES = [0, 1, 2]
_METRIC = "explained_variance"
# Per-env critic-averaged target per algo (both critics share the value, so the
# critic-average equals the target). Constant across episodes for predictability.
_TARGETS = {
    "algoA": {"Env-0": 10.0, "Env-1": 20.0, "Env-2": 30.0, "Env-3": 1000.0},
    "algoB": {"Env-0": 40.0, "Env-1": 50.0, "Env-2": 60.0, "Env-3": 2000.0},
}


def _write_critic_run(tmp_path):
    run_dir = tmp_path / "runs" / "critic_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(
        "env:\n  rollout_len: 10\ntraining:\n  mode: critic_ablation\n"
    )
    rows = []
    for ep in _E2E_EPISODES:
        for algo in _E2E_ALGOS:
            for env in _E2E_ENVS:
                for critic in ("c0", "c1"):
                    rows.append(
                        {
                            "episode": ep,
                            "algorithm": algo,
                            "environment": env,
                            "critic": critic,
                            _METRIC: _TARGETS[algo][env],
                        }
                    )
    pd.DataFrame(rows).to_csv(run_dir / "critic_ablation_metrics.csv", index=False)
    return run_dir


def _capture_savefig(monkeypatch):
    """Patch Figure.savefig to record {path: [(line_label, ydata), ...]} and
    skip writing any file. Returns the dict, populated as figures are saved."""
    captured: dict[str, list] = {}

    def fake_savefig(self, fname, *args, **kwargs):
        captured[str(fname)] = [
            (line.get_label(), list(line.get_ydata()))
            for ax in self.axes
            for line in ax.get_lines()
        ]

    monkeypatch.setattr(mfig.Figure, "savefig", fake_savefig)
    return captured


def _run_critic_plotting(tmp_path, monkeypatch, aggregation):
    _write_critic_run(tmp_path)
    monkeypatch.chdir(tmp_path)  # load_run reads runs/<name> relative to cwd
    captured = _capture_savefig(monkeypatch)
    run_plotting(
        run_name="critic_run",
        split="critic",
        x_axis="episodes",
        aggregation=aggregation,
        outdir=tmp_path / "outputs",
        formats=["png"],
    )
    return captured


def test_critic_overall_uses_aggregation_flag_end_to_end(tmp_path, monkeypatch):
    captured = _run_critic_plotting(tmp_path, monkeypatch, aggregation="iqm")
    overall = {p: v for p, v in captured.items() if "/overall/" in p}
    assert len(overall) == 1, f"expected one overall figure, got {list(overall)}"
    lines = dict(next(iter(overall.values())))
    assert "algoA" in lines

    centers = list(_TARGETS["algoA"].values())  # 4 env-centers for algoA
    expected_iqm = _iqm(np.asarray(centers))
    expected_mean = float(np.mean(centers))
    assert expected_iqm != pytest.approx(expected_mean)  # fixture sanity

    ydata = lines["algoA"]
    assert ydata == pytest.approx([expected_iqm] * len(_E2E_EPISODES))
    # And explicitly NOT the (old, buggy) mean-collapse.
    assert ydata != pytest.approx([expected_mean] * len(_E2E_EPISODES))


def test_critic_layout_drops_critic_dimension(tmp_path, monkeypatch):
    captured = _run_critic_plotting(tmp_path, monkeypatch, aggregation="iqm")
    per_env = {p: v for p, v in captured.items() if "/per_env/" in p}
    overall = {p: v for p, v in captured.items() if "/overall/" in p}

    n_metrics, n_envs, n_algos = 1, len(_E2E_ENVS), len(_E2E_ALGOS)
    # per_env: one figure per (metric, env); overall: one per metric.
    assert len(per_env) == n_metrics * n_envs
    assert len(overall) == n_metrics
    # One line per algo in every figure (critic dimension dropped, not per-critic).
    for lines in per_env.values():
        assert len(lines) == n_algos
        assert sorted(lbl for lbl, _ in lines) == sorted(_E2E_ALGOS)
    for lines in overall.values():
        assert len(lines) == n_algos
        assert sorted(lbl for lbl, _ in lines) == sorted(_E2E_ALGOS)
