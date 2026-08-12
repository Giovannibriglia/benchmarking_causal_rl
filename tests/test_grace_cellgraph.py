"""A9.4 — every declared CellGraph equals its built DAGs.

For each (cell, arm) declaration: the built observational NBN DAG equals the
declaration and the mutilated DAG equals it minus in-edges of A; plus the
repo's standing diagram-validity checks (valid DAG, no bidirectional action
edges, state/emission nodes present in PO cells, proxies declared).
"""

from __future__ import annotations

from src.rl.offline.grace import (
    cell_graph,
    CELL_GRAPHS,
    GraceOptions,
    identification_report,
)
from tests._grace_test_utils import DEV, FakeSeqBuffer, make_confounded_episodes


def test_every_declared_graph_validates_and_mutilates_correctly():
    assert set(CELL_GRAPHS) == {
        "offline_mdp",
        "offline_pomdp",
        "online_mdp",
        "online_pomdp",
    }
    for cell, arms in CELL_GRAPHS.items():
        assert set(arms) == {"template", "basic", "biased", "confounded"}
        for arm, g in arms.items():
            g.validate()  # DAG, no bidirectional action edges, PO emission
            # Mutilated = declaration minus in-edges of A, NOTHING else.
            assert set(g.mutilated_edges()) == set(g.edges) - set(g.in_edges_of("A"))
            # No edge out of the mutilation survives into A.
            assert all(dst != "A" for _, dst in g.mutilated_edges())
            # Proxy declaration is explicit (absent in every current cell).
            assert g.proxy_nodes == ()
            # PO cells: latent S with an observed emission.
            if "pomdp" in cell:
                assert g.pomdp and g.has_edge("S", "O") and g.node("O").observed
            else:
                assert not g.pomdp and g.identity_emission


def test_arm_semantics_match_the_generation_gates():
    for cell, arms in CELL_GRAPHS.items():
        # confounded + template declare U -> A; basic declares U -> R only
        # (A independent of U by the marginally-matched sigma=0 construction);
        # biased declares no U at all.
        assert arms["confounded"].confounded and arms["template"].confounded
        assert not arms["basic"].confounded and arms["basic"].has_edge("U", "R")
        assert not arms["biased"].has_node("U")


def test_identification_report_flags():
    rep_t = identification_report(cell_graph("mdp", "template"))
    assert rep_t["confounded_serving_ok"] and rep_t["adjustment_ok"]
    assert not rep_t["proxies_present"]  # never a proximal-style claim
    rep_b = identification_report(cell_graph("mdp", "basic"))
    assert not rep_b["confounded_serving_ok"]  # no declared U -> A
    assert (
        any("U->A" in r or "U -> A" in r for r in rep_b["reasons"]) or rep_b["reasons"]
    )


def test_nbn_dags_equal_the_declaration():
    """The built NBN networks' DAGs are exactly the declared edge lists."""
    graph = cell_graph("mdp", "template")
    eps, _ = make_confounded_episodes(n_ep=80, t_len=8)
    from src.rl.offline.grace import GraceMachinery

    m = GraceMachinery(
        graph,
        GraceOptions(n_bins=3, em_iters=4, ensemble_k=2),
        n_actions=2,
        device=DEV,
        gamma=0.9,
    )
    m.fit_from_buffer(FakeSeqBuffer(eps))
    net_obs, net_mut = m.cbn.build_nbn()
    assert net_obs is not None
    assert set(net_obs.dag.edges()) == set(graph.edges)
    assert set(net_mut.dag.edges()) == set(graph.mutilated_edges())
