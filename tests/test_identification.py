"""Block V-A / gate V2 — L2 must reproduce every declared verdict.

Deterministic: no training, no sampling. This is 100% or it is a bug.

The test is non-circular by construction: ``identify`` derives from the
graph's nodes, edges and declared proxy/instrument roles and never reads the
catalogue's ``Verdict`` fields, so comparing the two is a real check.
"""

from __future__ import annotations

import pytest
from src.rl.offline.grace.cell_graph import CATALOGUE, catalogue_entry
from src.rl.offline.grace.identify import identify, identify_catalogue


@pytest.mark.parametrize("entry_id", sorted(CATALOGUE))
@pytest.mark.parametrize("query", ["q1", "q2"])
def test_L2_reproduces_the_declared_verdict(entry_id, query):
    """Gate V2."""
    g = catalogue_entry(entry_id)
    declared = getattr(g, query)
    got = identify(g, query)
    assert got.status == declared.status, (
        f"{entry_id}/{query}: L2 says {got.status}, catalogue declares "
        f"{declared.status} ({got.reason})"
    )
    assert got.effective_status == declared.effective_status


def test_L2_never_defaults_to_point_id():
    """A graph outside the decidable subset must come back bounds-only, never
    point-ID. Constructed as a latent confounder with no proxy, no instrument,
    and a NON-sink reward, so even the lagged route is unavailable."""
    from src.rl.offline.grace.cell_graph import CellGraph, GraphNode, Verdict

    stub = Verdict("bounds_only", "none")
    g = CellGraph(
        id="unknown-case",
        name="latent confounder, no identification device",
        nodes=(
            GraphNode("S", "state", observed=True),
            GraphNode("A", "action", observed=True),
            GraphNode("R", "reward", observed=True),
            GraphNode("S_next", "state", observed=True, lag=1),
            GraphNode("U", "latent_confounder", observed=False),
        ),
        # R is NOT a sink here: it feeds the next state, killing the lagged
        # negative-control-outcome argument.
        edges=(
            ("S", "A"),
            ("S", "R"),
            ("A", "R"),
            ("S", "S_next"),
            ("A", "S_next"),
            ("R", "S_next"),
            ("U", "A"),
            ("U", "R"),
        ),
        q1=stub,
        q2=stub,
    )
    res = identify(g, "q1")
    assert res.status == "bounds_only"
    assert res.criterion == "none"
    assert "unknown" in res.reason


def test_D_B_q2_stays_gated_off():
    """The contested lagged-proxy derivation must not serve point values,
    whatever the derivation concludes, until C2 is settled in writing."""
    g = catalogue_entry("D-B")
    for q in ("q1", "q2"):
        res = identify(g, q)
        assert res.gated is True
        assert res.status == "point_id"
        assert res.effective_status == "bounds_only"


def test_q2_needs_finite_K_wherever_a_latent_survives():
    """Sequential identification with a latent rests on strictly more than the
    per-step bridge: the exogenous P(U) must be recoverable."""
    for eid in ("D-B", "D-D"):
        q1, q2 = identify(catalogue_entry(eid), "q1"), identify(
            catalogue_entry(eid), "q2"
        )
        assert set(q1.assumptions) < set(q2.assumptions), eid
        assert "finite_K_latent_class" in set(q2.assumptions) - set(q1.assumptions)


def test_q2_is_non_id_under_confounded_dynamics():
    """The guard: with a latent into the next state, no per-step result
    composes — regardless of what q1 concluded."""
    for eid in ("D-F", "D-G"):
        g = catalogue_entry(eid)
        assert g.dynamics_confounded
        assert identify(g, "q2").status == "non_id"
    # and D-F is the sharp case: identified per-step, not sequentially
    assert identify(catalogue_entry("D-F"), "q1").status == "point_id"


def test_iv_is_bounds_never_point_id():
    res = identify(catalogue_entry("D-E"), "q1")
    assert res.status == "bounds_only" and res.criterion == "iv"


def test_backdoor_adjustment_sets_are_the_expected_ones():
    """D-A/D-C adjust on the observed state: `A <- S -> R` is a genuine
    back-door path, so the set is {S}, not empty. D-F adjusts on the emission
    instead, because the state itself is latent there — conditioning on O
    blocks `A <- O <- S -> R` as a chain."""
    assert identify(catalogue_entry("D-A"), "q1").adjustment_set == ("S",)
    assert identify(catalogue_entry("D-C"), "q1").adjustment_set == ("S",)
    assert identify(catalogue_entry("D-F"), "q1").adjustment_set == ("O",)


def test_catalogue_sweep_is_complete():
    out = identify_catalogue(CATALOGUE)
    assert set(out) == set(CATALOGUE)
    assert all(set(v) == {"q1", "q2"} for v in out.values())


def test_cross_stratum_linking_is_cited_where_the_views_are_covariate_conditional():
    """The lagged-proxy views are conditionally independent given U only
    TOGETHER WITH the (S, A) at the measurement times, so Kruskal applies per
    configuration and identifies the latent only up to a relabelling AT EACH
    (s, a). Nothing in the diagram links those labels -- U is NOT independent
    of the covariates here (S_t descends from U through past actions) -- so the
    linking comes from the shared mechanism family, a MODEL-CLASS assumption.

    D-D is the contrast: its declared proxies are covariate-FREE (parents =
    {U}), so P(Z|U) and P(W|U) are global and pin the labelling globally. It
    therefore does not need the assumption, which is part of why it is the
    clean point-ID case."""
    from src.rl.offline.grace.identify import _parents

    for eid in ("D-B", "D-B-prime"):
        g = catalogue_entry(eid)
        assert "cross_stratum_label_linking" in g.q1.assumptions, eid
        assert "cross_stratum_label_linking" in g.q2.assumptions, eid
        # ...and it is honestly recorded as having no observable shadow.
        assert g.assumption("cross_stratum_label_linking").untestable
    # L2 names it too, for the same criterion.
    assert (
        "cross_stratum_label_linking"
        in identify(catalogue_entry("D-B"), "q1").assumptions
    )

    d_d = catalogue_entry("D-D")
    assert "cross_stratum_label_linking" not in d_d.q1.assumptions
    for proxy in d_d.proxy_nodes:  # the reason why
        assert sorted(_parents(d_d, proxy)) == ["U"], "proxy must be covariate-free"
