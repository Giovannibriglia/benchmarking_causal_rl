"""The classical / critic-ablation simulation split (composable cells).

Covers: (1) the ``simulation:`` key + the now-REAL ``sweep:``/``critics:`` blocks
(parsed & validated, previously documentation-only); (2) the classical results
tree (``{regime}/classical/``, no critic axis) and its walker isolation from the
ablation tree; (3) the online strategy->algo-variant resolution; (4) the
supervisor's simulation-aware leaf accounting; (5) tiny end-to-end runs of the
offline-classical, online-classical and online-ablation drivers; (6) every
shipped cell YAML parses and matches its filename.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from src.benchmarking.regime_report import (
    aggregate_classical,
    iter_classical_leaves,
    iter_leaves,
    parse_classical_leaf,
    parse_results_leaf,
)
from src.benchmarking.regime_sweep import (
    classical_results_leaf,
    load_sweep_spec,
    resolve_online_strategy_algo,
    results_leaf,
    run_cell,
    sweep_points,
)
from src.benchmarking.sweep_supervisor import _expected_leaves, _leaf_marker_files

_REPO = Path(__file__).resolve().parents[1]
_CELLS = ("offline_mdp", "offline_pomdp", "online_mdp", "online_pomdp")

_TINY = {
    "n_episodes": 1,
    "n_checkpoints": 2,
    "n_train_envs": 2,
    "n_eval_envs": 2,
    "rollout_len": 8,
    "rollout_episodes": 40,
    "offline_grad_steps": 4,
}


def _write_cell_yaml(tmp_path: Path, **overrides) -> Path:
    """A minimal cell YAML with a REDUCED L (1+1+1 = 3 points) so end-to-end
    driver tests stay fast. _base fragments are two levels up from the file, so
    park it in a nested dir with no _base sibling (defaults only)."""
    cell_dir = tmp_path / "cells" / "unit"
    cell_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "regime": "offline_mdp",
        "observability": "mdp",
        "data_regime": "offline",
        "simulation": "classical",
        "generator_algo": "dqn",
        "pi_basic_epsilon": 0.5,
        "confounder_c_r": 1.0,
        "envs": ["CartPole-v1"],
        "algos": ["cql"],
        "seeds": [0],
        "budgets": dict(_TINY),
        "sweep": {
            "basic": {"beta": 0.0, "sigma": 0.0},
            "biased": {"beta": [0.5], "sigma": 0.0},
            "confounded": {"beta": 0.0, "sigma": [1.0]},
        },
    }
    cfg.update(overrides)
    p = cell_dir / "cell.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


# --------------------------------------------------------------------------- #
# Spec parsing: simulation key + the now-REAL sweep/critics blocks              #
# --------------------------------------------------------------------------- #
def test_simulation_key_parses_and_defaults(tmp_path):
    spec = load_sweep_spec(_write_cell_yaml(tmp_path, simulation="classical"))
    assert spec.simulation == "classical"
    cfg_no_sim = _write_cell_yaml(tmp_path)
    yaml_doc = yaml.safe_load(cfg_no_sim.read_text())
    del yaml_doc["simulation"]
    cfg_no_sim.write_text(yaml.safe_dump(yaml_doc))
    # absent -> the historical default (every legacy sweep.yaml is an ablation)
    assert load_sweep_spec(cfg_no_sim).simulation == "critic_ablation"


def test_unknown_simulation_refused(tmp_path):
    with pytest.raises(ValueError, match="unknown simulation"):
        load_sweep_spec(_write_cell_yaml(tmp_path, simulation="benchmarkish"))


def test_sweep_block_is_real_config(tmp_path):
    spec = load_sweep_spec(_write_cell_yaml(tmp_path))
    assert spec.beta_arm == (0.5,) and spec.sigma_arm == (1.0,)
    assert spec.points() == [(0.0, 0.0), (0.5, 0.0), (0.0, 1.0)]
    # absent block -> the canonical 7-point L
    assert len(sweep_points()) == 7


@pytest.mark.parametrize(
    "bad_sweep",
    [
        {"basic": {"beta": 0.1, "sigma": 0.0}},  # origin off the L
        {"biased": {"beta": [0.5], "sigma": 0.5}},  # biased arm varies sigma
        {"confounded": {"beta": 0.5, "sigma": [0.5]}},  # confounded varies beta
        {"diagonal": {"beta": [0.5], "sigma": [0.5]}},  # unknown arm
        {"basic": False, "biased": False, "confounded": False},  # nothing left
        {"biased": "off"},  # an arm is a map or false, never a string
    ],
)
def test_off_L_sweep_blocks_refused(tmp_path, bad_sweep):
    with pytest.raises(ValueError):
        load_sweep_spec(_write_cell_yaml(tmp_path, sweep=bad_sweep))


def test_sweep_arm_false_excludes_the_arm(tmp_path):
    # ONLY explicit false shrinks the L; an ABSENT arm key falls back to the
    # canonical default (backward compatible — commenting out never removes).
    with pytest.warns(UserWarning, match="null-calibration anchor"):
        spec = load_sweep_spec(
            _write_cell_yaml(
                tmp_path,
                sweep={
                    "basic": False,
                    "biased": False,
                    "confounded": {"beta": 0.0, "sigma": [0.5]},
                },
            )
        )
    assert spec.points() == [(0.0, 0.5)]
    # absent biased -> canonical beta arm; basic true == absent == canonical.
    spec = load_sweep_spec(
        _write_cell_yaml(
            tmp_path, sweep={"basic": True, "confounded": {"sigma": [0.5]}}
        )
    )
    assert spec.points() == [
        (0.0, 0.0),
        (0.25, 0.0),
        (0.5, 0.0),
        (0.75, 0.0),
        (0.0, 0.5),
    ]


def test_critics_block_is_real_config(tmp_path):
    spec = load_sweep_spec(
        _write_cell_yaml(
            tmp_path,
            simulation="critic_ablation",
            critics={
                "basic": ["observational", "proximal"],
                "biased": ["observational"],
                "confounded": ["observational", "oracle_u"],
            },
        )
    )
    assert spec.critics_for("basic") == ["observational", "proximal"]
    assert spec.critics_for("confounded") == ["observational", "oracle_u"]


def test_critics_validation(tmp_path):
    with pytest.raises(ValueError, match="unknown critic strategy"):
        load_sweep_spec(_write_cell_yaml(tmp_path, critics={"basic": ["psychic"]}))
    with pytest.raises(ValueError, match="requires 'observational'"):
        load_sweep_spec(_write_cell_yaml(tmp_path, critics={"basic": ["sensitivity"]}))
    # offline-only strategies are refused in an ONLINE cell's critics block
    with pytest.raises(ValueError, match="online"):
        load_sweep_spec(
            _write_cell_yaml(
                tmp_path,
                data_regime="online",
                algos=["dqn"],
                critics={"basic": ["observational", "oracle_u"]},
            )
        )


# --------------------------------------------------------------------------- #
# Classical tree: leaf schema + walker isolation                                #
# --------------------------------------------------------------------------- #
def test_classical_leaf_roundtrip_and_isolation(tmp_path):
    root = tmp_path / "results"
    cl = classical_results_leaf(root, "offline_mdp", 0.0, 0.5, "CartPole-v1", "cql", 3)
    assert cl == (
        root
        / "offline_mdp"
        / "classical"
        / "beta_000_sigma_050"
        / "CartPole-v1"
        / "cql"
        / "3"
    )
    cl.mkdir(parents=True)
    (cl / "config.yaml").write_text("{}")
    rec = parse_classical_leaf(cl)
    assert (rec["beta"], rec["sigma"], rec["arm"]) == (0.0, 0.5, "confounded")
    assert (rec["env"], rec["algo"], rec["seed"]) == ("CartPole-v1", "cql", 3)
    # the ABLATION walker must NOT see classical leaves (3-segment tail != 4)...
    with pytest.raises(ValueError):
        parse_results_leaf(cl)
    assert iter_leaves(root, "offline_mdp") == []
    # ...and the CLASSICAL walker must not see ablation leaves.
    ab = results_leaf(
        root, "offline_mdp", 0.0, 0.5, "CartPole-v1", "cql", "observational", 3
    )
    ab.mkdir(parents=True)
    (ab / "config.yaml").write_text("{}")
    assert [r["path"] for r in iter_classical_leaves(root, "offline_mdp")] == [str(cl)]
    assert [r["critic"] for r in iter_leaves(root, "offline_mdp")] == ["observational"]


# --------------------------------------------------------------------------- #
# Algo entries: explicit trunks + designed-for-regime validation                #
# --------------------------------------------------------------------------- #
def test_parse_algo_entry_forms():
    from src.benchmarking.regime_sweep import parse_algo_entry

    # plain form: AUTO trunks (mlp on mdp, lstm critic on pomdp), id = bare name
    assert parse_algo_entry("cql", "mdp") == ("cql", "mlp", "mlp", "cql")
    assert parse_algo_entry("dqn", "pomdp") == ("dqn", "mlp", "lstm", "dqn")
    # explicit name__actor__critic: trunks pinned, id = entry VERBATIM (so two
    # rows of one base with different trunks never collide in the tree)
    assert parse_algo_entry("dqn__lstm__lstm", "pomdp") == (
        "dqn",
        "lstm",
        "lstm",
        "dqn__lstm__lstm",
    )
    assert parse_algo_entry("offline_dqn__mlp__mlp", "pomdp") == (
        "offline_dqn",
        "mlp",
        "mlp",
        "offline_dqn__mlp__mlp",
    )
    with pytest.raises(ValueError, match="name__actor__critic"):
        parse_algo_entry("dqn__lstm", "mdp")


def test_algos_validated_against_data_regime(tmp_path):
    # an OFFLINE learner in an ONLINE cell must be refused (and vice versa)
    with pytest.raises(ValueError, match="designed"):
        run_cell(
            _write_cell_yaml(
                tmp_path,
                regime="online_mdp",
                data_regime="online",
                simulation="classical",
                algos=["cql"],
            ),
            results_root=tmp_path / "results",
            device="cpu",
        )
    with pytest.raises(ValueError, match="designed"):
        run_cell(
            _write_cell_yaml(tmp_path, algos=["dqn"]),  # online algo, offline cell
            results_root=tmp_path / "results",
            device="cpu",
        )


# --------------------------------------------------------------------------- #
# Online strategy -> algo variant resolution                                    #
# --------------------------------------------------------------------------- #
def test_resolve_online_strategy_algo():
    from src.benchmarking.registry import register_default_algorithms

    register_default_algorithms()
    assert resolve_online_strategy_algo("dqn", "observational") == "dqn"
    assert resolve_online_strategy_algo("dqn", "proximal") == "online_dqn_proximal"
    with pytest.raises(ValueError, match="no online algo variant"):
        resolve_online_strategy_algo("dqn", "oracle_u")


# --------------------------------------------------------------------------- #
# Supervisor: simulation-aware leaf accounting                                  #
# --------------------------------------------------------------------------- #
def test_expected_leaves_per_simulation(tmp_path):
    classical = load_sweep_spec(_write_cell_yaml(tmp_path, simulation="classical"))
    # classical: one leaf per (point x algo), no critic segment
    leaves = _expected_leaves(classical, ["cql"], "CartPole-v1", 0, "results")
    assert len(leaves) == 3
    assert all("classical" in p.parts for p in leaves)
    assert _leaf_marker_files(classical) == ("config.yaml", "eval_metrics.csv")

    online_ab = load_sweep_spec(
        _write_cell_yaml(
            tmp_path, data_regime="online", simulation="critic_ablation", algos=["dqn"]
        )
    )
    # online ablation defaults: basic [obs, prox, grace] + biased [obs] +
    # confounded [obs, prox, grace] = 7 leaves for the reduced 3-point L
    # (feat/grace-critic: grace joined ONLINE_STRATEGIES — intentional pin
    # update).
    leaves = _expected_leaves(online_ab, ["dqn"], "CartPole-v1", 0, "results")
    assert len(leaves) == 7
    assert _leaf_marker_files(online_ab) == ("config.yaml", "eval_metrics.csv")

    offline_ab = load_sweep_spec(
        _write_cell_yaml(tmp_path, simulation="critic_ablation")
    )
    # offline ablation defaults: basic FULL(4) + biased obs(1) + confounded FULL(4)
    leaves = _expected_leaves(offline_ab, ["cql"], "CartPole-v1", 0, "results")
    assert len(leaves) == 9
    assert _leaf_marker_files(offline_ab) == (
        "config.yaml",
        "critic_ablation_metrics.csv",
    )


# --------------------------------------------------------------------------- #
# Shipped YAMLs: every cell ships both components + smokes, and they parse      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("regime", _CELLS)
@pytest.mark.parametrize(
    "stem", ["classical", "classical_smoke", "critic_ablation", "critic_ablation_smoke"]
)
def test_cells_ship_both_simulations(regime, stem):
    p = _REPO / "reproducibility" / "rl_regimes" / regime / f"{stem}.yaml"
    assert p.exists(), p
    spec = load_sweep_spec(p)
    assert spec.regime == regime
    assert spec.simulation == stem.replace("_smoke", "")
    expected_dr = "online" if regime.startswith("online") else "offline"
    assert spec.data_regime == expected_dr
    # algo selection is EXPLICIT in every file (never inherited from _base) and
    # every entry is DESIGNED for the cell's data regime (registry-validated).
    raw = yaml.safe_load(p.read_text())
    assert "algos" in raw and raw["algos"], f"{p} must declare algos explicitly"
    from src.benchmarking.regime_sweep import _validate_algos_for_regime
    from src.benchmarking.registry import register_default_algorithms

    register_default_algorithms()
    _validate_algos_for_regime(spec, spec.algos)  # raises on a mismatch
    if spec.simulation == "classical":
        # every classical config carries (at least) two algo rows
        assert len(spec.algos) >= 2, f"{p} classical must compare >= 2 algos"
    if expected_dr == "online" and spec.simulation == "critic_ablation":
        assert spec.algos == ["dqn"]
        # the critic sets only bind the ablation simulation
        for arm in ("basic", "confounded"):
            if regime == "online_pomdp":
                # no recurrent online proximal exists; lstm-vs-mlp across
                # strategies would confound the encoder axis
                assert spec.critics_for(arm) == ["observational"]
            else:
                # feat/grace-critic: grace resolves online (online_dqn_grace)
                # — intentional pin update.
                assert spec.critics_for(arm) == [
                    "observational",
                    "proximal",
                    "grace",
                ]
    if "smoke" in stem:
        assert spec.budget("n_episodes", 999) <= 20  # tiny budget baked in


# --------------------------------------------------------------------------- #
# End-to-end (tiny reduced L): the three new driver paths                       #
# --------------------------------------------------------------------------- #
def test_offline_classical_cell_end_to_end(tmp_path):
    root = tmp_path / "results"
    written = run_cell(
        _write_cell_yaml(tmp_path, simulation="classical"),
        results_root=root,
        dataset_prefix="test/clsplit",
        device="cpu",
    )
    assert len(written) == 3  # 3 points x 1 algo, no critic axis
    for leaf in written:
        assert "classical" in leaf.parts
        assert (leaf / "config.yaml").exists()
        assert (leaf / "eval_metrics.csv").exists()
        assert not (leaf / "critic_ablation_metrics.csv").exists()
        cfg = yaml.safe_load((leaf / "config.yaml").read_text())
        assert cfg["training"]["mode"] == "benchmark"
        assert cfg["sweep"]["simulation"] == "classical"
    # the classical aggregator sees exactly these cells, with derived arms
    agg = aggregate_classical(root, "offline_mdp")
    assert {(r["beta"], r["sigma"], r["arm"]) for r in agg} == {
        (0.0, 0.0, "basic"),
        (0.5, 0.0, "biased"),
        (0.0, 1.0, "confounded"),
    }
    # and the ablation walker sees NOTHING (tree isolation)
    assert iter_leaves(root, "offline_mdp") == []


def test_online_classical_cell_end_to_end(tmp_path):
    root = tmp_path / "results"
    written = run_cell(
        _write_cell_yaml(
            tmp_path,
            regime="online_mdp",
            data_regime="online",
            simulation="classical",
            algos=["dqn"],
        ),
        results_root=root,
        device="cpu",
    )
    assert len(written) == 3
    for leaf in written:
        assert (leaf / "eval_metrics.csv").exists()
        assert (leaf / "train_metrics.csv").exists()


def test_online_ablation_cell_end_to_end(tmp_path):
    root = tmp_path / "results"
    written = run_cell(
        _write_cell_yaml(
            tmp_path,
            regime="online_mdp",
            data_regime="online",
            simulation="critic_ablation",
            algos=["dqn"],
        ),
        results_root=root,
        device="cpu",
    )
    # basic [obs, prox, grace] + biased [obs] + confounded [obs, prox, grace]
    # = 7 leaves (feat/grace-critic: the online grace variant runs end-to-end
    # here too — intentional pin update).
    assert len(written) == 7
    recs = iter_leaves(root, "online_mdp")
    assert {(r["arm"], r["critic"]) for r in recs} == {
        ("basic", "observational"),
        ("basic", "proximal"),
        ("basic", "grace"),
        ("biased", "observational"),
        ("confounded", "observational"),
        ("confounded", "proximal"),
        ("confounded", "grace"),
    }
    # the {algo} segment stays the BASE name (dqn), never the variant
    assert {r["algo"] for r in recs} == {"dqn"}
