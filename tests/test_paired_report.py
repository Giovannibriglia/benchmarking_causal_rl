"""The paired-report extensions — pinned because a mispairing is silent.

A comparison built from the wrong pairs produces a table of exactly the right
shape with meaningless numbers, which is the failure species this seam keeps
producing (S16). So the join verifies rather than assumes.
"""

from __future__ import annotations

import csv

import pytest
from src.benchmarking.regime_report import (
    aggregate_per_seed,
    build_paired_report,
    read_leaf_series,
)


def _leaf(root, regime, env, algo, arm, seed, rows):
    d = root / regime / "beta_000_sigma_025" / env / algo / arm / str(seed)
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.yaml").write_text("{}\n")
    with (d / "eval_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["episode", "eval_return_mean"])
        w.writeheader()
        for ep, v in rows:
            w.writerow({"episode": ep, "eval_return_mean": v})
    return d


def test_read_leaf_series_returns_every_checkpoint_not_just_the_last(tmp_path):
    """The per-leaf CSVs are the ONLY place a learning curve lives; the
    aggregate keeps just the final row, so a curve cannot be reconstructed
    after the fact."""
    d = _leaf(
        tmp_path,
        "r",
        "CartPole-v1",
        "cql",
        "base",
        0,
        [(10, 1.0), (20, 2.0), (30, 3.0)],
    )
    assert read_leaf_series(d, "eval_return_mean") == [(10, 1.0), (20, 2.0), (30, 3.0)]


def test_aggregate_per_seed_keeps_the_seed_axis(tmp_path):
    for sd in (0, 1, 2):
        _leaf(tmp_path, "r", "CartPole-v1", "cql", "base", sd, [(10, float(sd))])
    rows = aggregate_per_seed(tmp_path, "r", metrics=("eval_return_mean",))
    assert sorted(r["seed"] for r in rows) == [0, 1, 2]
    assert sorted(r["eval_return_mean"] for r in rows) == [0.0, 1.0, 2.0]


def test_paired_report_matches_on_configuration_and_computes_per_seed_deltas(tmp_path):
    for sd in (0, 1, 2):
        _leaf(tmp_path, "r", "CartPole-v1", "cql", "base", sd, [(10, 10.0)])
        _leaf(tmp_path, "r", "CartPole-v1", "cql", "grace", sd, [(10, 10.0 + sd)])
    pairs = build_paired_report(tmp_path, "r", metrics=("eval_return_mean",))
    assert len(pairs) == 3
    assert [p["delta_eval_return_mean"] for p in pairs] == [0.0, 1.0, 2.0]
    # seeds are never collapsed -- the D-D reporting constraint
    assert sorted(p["seed"] for p in pairs) == [0, 1, 2]


def test_an_unmatched_pair_raises_rather_than_being_dropped(tmp_path):
    """Silently dropping the unpartnered row would leave a table that looks
    complete and compares different things."""
    _leaf(tmp_path, "r", "CartPole-v1", "cql", "base", 0, [(10, 1.0)])
    _leaf(tmp_path, "r", "CartPole-v1", "cql", "base", 1, [(10, 1.0)])
    _leaf(tmp_path, "r", "CartPole-v1", "cql", "grace", 0, [(10, 2.0)])
    with pytest.raises(ValueError, match="UNMATCHED PAIRS"):
        build_paired_report(tmp_path, "r", metrics=("eval_return_mean",))
