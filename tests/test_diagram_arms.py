"""The diagram must decide which generator channels exist — not the YAML.

v2's one assumption is the declared causal diagram. If a config could switch the
proxies on independently, the diagram would stop being the single assumption
surface: a cell could declare proxies the generator never emits (L2 would hand
out a proximal verdict on data that cannot support one) or emit proxies the
diagram does not declare (a channel nothing accounts for). Both directions are
errors, and both are tested here.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from src.benchmarking.regime_sweep import load_sweep_spec
from src.envs.offline.diagram_arms import arm_knobs, declared_channels

CELLS = Path("reproducibility/rl_regimes/diagrams")


def test_channels_are_read_off_the_catalogue():
    assert declared_channels("D-D") == {
        "proxy": True,
        "instrument": False,
        "drift": False,
        "latent": True,
        # 2026-08-21 revision: the third proxy V, and the declared licence for
        # the compensated gated-reward sweep.
        "gated_reward_sweep": True,
        "n_proxies": 3,
    }
    assert declared_channels("D-B-prime")["gated_reward_sweep"] is False
    assert declared_channels("D-E")["instrument"] is True
    assert declared_channels("D-B-prime")["drift"] is True
    assert declared_channels("D-A-null")["latent"] is False


def test_a_strength_for_an_undeclared_channel_is_refused():
    with pytest.raises(ValueError, match="declares no proxy channel"):
        arm_knobs(
            "D-E",
            sigma=0.5,
            instrument_strength=0.3,
            proxy_strength=1.5,
            gate_probs=(0.2, 0.8),
        )


def test_an_instrument_arm_must_make_its_exclusion_testable():
    """R2, route (a). Under the deterministic gate R is a function of (A, U), so
    residualising leaves zero variance and the exclusion check measures nothing
    while reporting a pass. An instrument arm declared without gate_probs is
    exactly that trap, so it is refused at resolve time."""
    with pytest.raises(ValueError, match="exclusion restriction cannot be tested"):
        arm_knobs("D-E", sigma=0.5, instrument_strength=0.3)
    # gate_probs on a cell with NEITHER an instrument NOR the declared
    # gated-reward-sweep licence is still refused (D-B-prime here; D-D stopped
    # being the example when its 2026-08-21 revision declared the licence).
    with pytest.raises(ValueError, match="neither an instrument nor a gated"):
        arm_knobs("D-B-prime", sigma=0.5, u_drift=0.05, gate_probs=(0.2, 0.8))


def test_the_gated_reward_sweep_licence_admits_and_derives():
    """D-D's declared licence: gate_probs is admitted as a reward
    parameterisation, and c_r is DERIVED as M / d -- the estimand-invariant
    compensated sweep. Declaring c_r alongside M is a contradiction and
    raises rather than letting one silently win."""
    k = arm_knobs(
        "D-D",
        sigma=0.25,
        proxy_strength=1.5,
        gate_probs=(0.1, 0.35),
        gate_mean_effect=1.0,
    )
    assert abs(k.confounder_c_r - 4.0) < 1e-9  # M / d = 1.0 / 0.25
    assert k.n_proxies == 3
    with pytest.raises(ValueError, match="both gate_mean_effect"):
        arm_knobs(
            "D-D",
            sigma=0.25,
            proxy_strength=1.5,
            gate_probs=(0.1, 0.35),
            gate_mean_effect=1.0,
            confounder_c_r=1.0,
        )
    with pytest.raises(ValueError, match="needs a separation"):
        arm_knobs("D-D", sigma=0.25, proxy_strength=1.5, gate_mean_effect=1.0)


def test_a_declared_channel_without_a_strength_is_refused():
    """The dangerous direction: the diagram promises proxies, the generator
    emits none, and L2 still returns point-ID by the proximal criterion."""
    with pytest.raises(ValueError, match="declares a proxy channel"):
        arm_knobs("D-D", sigma=0.5)


def test_the_null_arm_has_no_latent_edges_at_all():
    """D-A-null is where L5's false-positive rate is read, so it must be a world
    with genuinely nothing to find -- not a confounded world with the dial at 0.
    U is still drawn and logged (same code path), but c_r = 0 and sigma = 0 mean
    it touches neither the action nor the reward."""
    k = arm_knobs("D-A-null", sigma=0.0)
    assert k.gate_probs is None
    assert k.confounder_c_r == 0.0 and k.behavior_strength == 0.0
    assert k.proxy_strength is None and k.instrument_strength is None
    assert k.u_drift == 0.0
    with pytest.raises(ValueError, match="no latent"):
        arm_knobs("D-A-null", sigma=0.5)


@pytest.mark.parametrize("cell", sorted(p.name for p in CELLS.glob("*.yaml")))
def test_every_diagram_cell_resolves_at_every_sweep_point(cell):
    spec = load_sweep_spec(CELLS / cell)
    assert spec.diagram is not None
    declared = declared_channels(spec.diagram)
    for _, sigma in spec.points():
        kw = spec.arm_generator_kwargs(sigma)
        assert (kw["proxy_strength"] is not None) == declared["proxy"]
        assert (kw["instrument_strength"] is not None) == declared["instrument"]
        assert (kw["u_drift"] > 0.0) == declared["drift"]


def test_cell_files_cover_the_four_declared_arms():
    got = {yaml.safe_load(p.read_text())["diagram"] for p in CELLS.glob("*.yaml")}
    assert got == {"D-A-null", "D-D", "D-E", "D-B-prime"}


def test_historical_cells_declare_no_diagram_and_pass_no_channels():
    """The byte-frozen cells must reach generate_offline_dataset with exactly the
    kwargs they always did -- an empty dict, not a dict of Nones."""
    spec = load_sweep_spec(
        "reproducibility/rl_regimes/offline_mdp/critic_ablation.yaml"
    )
    assert spec.diagram is None
    assert spec.arm_generator_kwargs(0.5) == {}


# --------------------------------------------------------------------------- #
# Identity discipline. Three bugs in the V-B driver in one session were all the #
# same species -- identity and collision -- so the fix is a test that makes the #
# class unreachable, not a third careful launch.                                #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cell", sorted(p.stem for p in CELLS.glob("*.yaml")))
def test_diagram_arm_ids_are_injective(cell):
    """Every (cell, env, seed, sigma) must get its OWN dataset id.

    The seed omission that cost a full V-B run would have been caught the moment
    this was written: dataset_name carries env, tier, policy and sigma but not
    the seed, so all five seeds collided on one id, and because the fingerprint
    DOES include the seed each run deleted the previous one's data and
    regenerated. Pure and instant -- no generation involved.
    """
    from tools.generate_diagram_arms import grid_ids

    spec = load_sweep_spec(CELLS / f"{cell}.yaml")
    ids = grid_ids(cell, spec)
    expected = len(spec.envs) * len(spec.seeds) * len(spec.points())
    assert len(ids) == expected, (len(ids), expected)
    dupes = {i for i in ids if ids.count(i) > 1}
    assert not dupes, f"{len(dupes)} colliding ids, e.g. {sorted(dupes)[:2]}"


def test_ids_are_injective_across_cells_too():
    """Two cells sharing an env/seed/sigma must not share an id either -- they
    carry different channels, so serving one for the other would be silent."""
    from tools.generate_diagram_arms import grid_ids

    seen: dict = {}
    for p_ in sorted(CELLS.glob("*.yaml")):
        for i in grid_ids(p_.stem, load_sweep_spec(p_)):
            assert i not in seen, f"{i} produced by both {seen[i]} and {p_.stem}"
            seen[i] = p_.stem


def test_the_id_distinguishes_every_grid_axis():
    """Each axis must actually move the id -- an axis absent from it is exactly
    the seed bug in another coordinate."""
    from src.envs.offline.diagram_arms import arm_knobs
    from tools.generate_diagram_arms import dataset_id_for

    k = arm_knobs("D-D", sigma=0.5, proxy_strength=1.5)
    base = dataset_id_for("d_d", k, "CartPole-v1", 0, 0.5)
    assert base != dataset_id_for("d_d", k, "CartPole-v1", 1, 0.5), "seed"
    assert base != dataset_id_for("d_d", k, "Acrobot-v1", 0, 0.5), "env"
    assert base != dataset_id_for("d_d", k, "CartPole-v1", 0, 0.25), "sigma"
    assert base != dataset_id_for("d_e", k, "CartPole-v1", 0, 0.5), "cell"


def test_the_driver_constructs_ids_only_through_the_shared_helper():
    """The injectivity tests above cover ``dataset_id_for``. They are worthless
    if ``main()`` builds ids some other way -- which it did: an inline
    ``dataset_name(...)`` call omitted the seed and collided across seeds while
    every helper-level test passed. Identity must have exactly one construction
    site, so this pins that ``main()`` cannot reach ``dataset_name`` at all."""
    src = Path("tools/generate_diagram_arms.py").read_text()
    body = src[src.index("def main(") :]
    assert "dataset_name(" not in body, (
        "main() constructs a dataset id outside dataset_id_for; the injectivity "
        "tests cannot see that path"
    )
    assert "dataset_id_for(" in body
