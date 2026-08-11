"""Hosted-dataset behavior-policy sweep — driver, layout, resume, renderer.

Runs the whole hosted_sweep -> results tree -> render_hosted_report chain on
LOCAL Dict-obs fixture datasets (no network), with two arms so the cross-arm
comparison axis actually exists.
"""

from __future__ import annotations

import csv
import json

import pytest
import torch  # noqa: F401  (backend presence gates the runner deps)
import yaml

pytest.importorskip("minari")
pytest.importorskip("h5py")
pytest.importorskip("minigrid")

from tests.test_hosted_dict_obs import _make_dict_dataset  # noqa: E402


def _family_config(tmp_path):
    cfg = {
        "regime": "offline_mdp",
        "simulation": "hosted_test",
        "env_wrapper": "minigrid_symbolic",
        "datasets": {
            "low": {"MiniGrid-Empty-5x5-v0": "hostedsweep/low-v0"},
            # Structured arm form with per-arm env_kwargs (parse coverage; the
            # kwargs are inert for this env).
            "high": {
                "offline_dataset": {"MiniGrid-Empty-5x5-v0": "hostedsweep/high-v0"},
                "env_kwargs": {},
            },
        },
        "algos": ["offline_dqn"],
        "seeds": [0],
        "offline_grad_steps": 8,
        "n_checkpoints": 2,
        "n_train_envs": 2,
        "n_eval_envs": 2,
        "rollout_len": 4,
        "aggregation": "mean",
        "eval_count_terminal_reward": True,
    }
    path = tmp_path / "family.yaml"
    # sort_keys=False: arm order in the file IS the axis order.
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def test_parse_family_config(tmp_path, monkeypatch):
    from src.benchmarking.hosted_sweep import parse_family_config

    spec = parse_family_config(str(_family_config(tmp_path)))
    assert [a.name for a in spec.arms] == ["low", "high"]  # file order kept
    assert spec.envs == ["MiniGrid-Empty-5x5-v0"]
    assert spec.arms[1].offline_dataset == {
        "MiniGrid-Empty-5x5-v0": "hostedsweep/high-v0"
    }
    assert spec.algos[0]["name"] == "offline_dqn"
    assert spec.budgets["offline_grad_steps"] == 8


def test_parse_rejects_mismatched_arm_envs(tmp_path):
    from src.benchmarking.hosted_sweep import parse_family_config

    path = tmp_path / "bad.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "datasets": {
                    "a": {"CartPole-v1": "x/a-v0"},
                    "b": {"Acrobot-v1": "x/b-v0"},
                },
                "algos": ["offline_dqn"],
                "offline_grad_steps": 8,
            }
        )
    )
    with pytest.raises(ValueError, match="same env set"):
        parse_family_config(str(path))


def test_hosted_sweep_end_to_end_and_resume(tmp_path, monkeypatch):
    _make_dict_dataset(tmp_path, monkeypatch, "hostedsweep/low-v0")
    _make_dict_dataset(tmp_path, monkeypatch, "hostedsweep/high-v0")
    from src.benchmarking.hosted_sweep import run_hosted_sweep

    cfg = _family_config(tmp_path)
    results = tmp_path / "results"
    summary = run_hosted_sweep(str(cfg), results_root=str(results), device="cpu")
    assert "2 leaves run, 0 skipped" in summary

    family = results / "offline_mdp" / "hosted_test"
    manifest = json.loads((family / "manifest.json").read_text())
    assert manifest["arms"] == ["low", "high"]
    for arm in ("low", "high"):
        leaf = family / arm / "MiniGrid-Empty-5x5-v0" / "offline_dqn" / "seed0"
        rows = list(csv.DictReader((leaf / "eval_metrics.csv").open()))
        assert rows and rows[-1]["algorithm"] == "offline_dqn"

    # Resume: every complete leaf is skipped, nothing re-runs.
    summary2 = run_hosted_sweep(str(cfg), results_root=str(results), device="cpu")
    assert "0 leaves run, 2 skipped" in summary2

    # Renderer over the finished tree.
    from src.benchmarking.render_hosted_report import render

    report = render("offline_mdp", "hosted_test", results_root=str(results))
    agg = list(csv.DictReader((report / "aggregate.csv").open()))
    assert {r["arm"] for r in agg} == {"low", "high"}
    assert (report / "offline_mdp_hosted_test_MiniGrid-Empty-5x5-v0.png").exists()
