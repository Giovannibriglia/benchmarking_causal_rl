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
    }
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
    with pytest.raises(ValueError, match="declares no instrument"):
        arm_knobs("D-D", sigma=0.5, proxy_strength=1.5, gate_probs=(0.2, 0.8))


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
