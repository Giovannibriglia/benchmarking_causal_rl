"""The catalogue is GRACE v2's whole assumption surface, so the human-readable
document and the machine-readable declarations must not drift apart.

These tests are the agreement check required by Gate A2, plus the structural
invariants the catalogue's own derivations depend on.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from src.rl.offline.grace.cell_graph import CATALOGUE, catalogue_entry, STATUSES

_DOC = Path(__file__).resolve().parents[1] / "docs" / "diagram_catalogue.md"


def test_doc_exists_and_lists_every_code_entry():
    """Every declared diagram is documented, by its code id."""
    text = _DOC.read_text(encoding="utf-8")
    for entry_id in CATALOGUE:
        assert entry_id in text, f"{entry_id} declared in code but absent from the doc"


def test_doc_declares_no_entry_the_code_lacks():
    """The doc's id table cannot invent entries the code does not declare."""
    text = _DOC.read_text(encoding="utf-8")
    m = re.search(r"as used in code:\*\*(.+?)\n\n", text, re.S)
    assert m, "the doc must carry the 'Entry ids as used in code' line"
    documented = set(re.findall(r"`([A-Za-z0-9\-]+)`", m.group(1)))
    assert documented == set(
        CATALOGUE
    ), f"doc lists {sorted(documented)}, code declares {sorted(CATALOGUE)}"


def test_every_entry_validates():
    for entry in CATALOGUE.values():
        entry.validate()


def test_mutilation_is_edges_minus_in_edges_of_target():
    """The do()-surgery has exactly one definition."""
    for entry in CATALOGUE.values():
        expected = set(entry.edges) - set(entry.in_edges_of(entry.intervention_target))
        assert set(entry.mutilated_edges()) == expected, entry.id
        assert all(
            dst != entry.intervention_target for _, dst in entry.mutilated_edges()
        )


def test_two_verdicts_per_entry_with_valid_statuses():
    for entry in CATALOGUE.values():
        assert entry.q1.status in STATUSES and entry.q2.status in STATUSES, entry.id


def test_q2_never_point_id_under_confounded_dynamics():
    """The guard behind the sequential derivation: with U -> S_next the
    occupancy under do(pi) is not latent-independent, so no per-step result
    composes. validate() enforces it; assert it holds catalogue-wide."""
    for entry in CATALOGUE.values():
        if entry.dynamics_confounded:
            assert entry.q2.status != "point_id", entry.id


def test_D_F_splits_across_queries():
    """The entry that proves two verdicts are necessary: per-step identified,
    sequentially not."""
    d_f = catalogue_entry("D-F")
    assert d_f.q1.status == "point_id"
    assert d_f.q1.adjustment_set == ("O",)
    assert d_f.q2.status == "non_id"


def test_D_B_q2_assumes_strictly_more_than_q1():
    """Sequential identification rests on a STRICTLY stronger assumption set —
    the finite-K latent class — because V^pi integrates the exogenous P(U)
    while the per-step bridge delivers the behaviour-tilted P(U|X)."""
    d_b = catalogue_entry("D-B")
    q1, q2 = set(d_b.q1.assumptions), set(d_b.q2.assumptions)
    assert q1 < q2, "q2's assumptions must strictly contain q1's"
    assert "finite_K_latent_class" in q2 - q1


def test_D_B_ships_gated_off():
    """The contested derivation must not serve point values by default."""
    d_b = catalogue_entry("D-B")
    assert d_b.q1.gated_off_by_default and d_b.q2.gated_off_by_default
    assert d_b.q1.effective_status == "bounds_only"
    assert d_b.q2.effective_status == "bounds_only"


def test_basic_arm_asserts_D_C_not_D_A():
    """The correction that only code-verification found: at sigma=0 the basic
    arm still injects U (c_r > 0), so A is unconfounded but U -> R remains."""
    d_c = catalogue_entry("D-C")
    assert "offline_mdp/basic" in d_c.asserted_by
    assert d_c.has_edge("U", "R") and not d_c.has_edge("U", "A")
    assert not d_c.confounded  # identified despite carrying a latent
    d_a = catalogue_entry("D-A")
    assert "offline_mdp/basic" not in d_a.asserted_by
    assert not d_a.has_node("U")


def test_D_A_and_D_C_differ_by_exactly_one_implication():
    """What makes M1 detectable: D-A forbids cross-step reward dependence and
    D-C permits it; both forbid A_t depending on the past given S_t."""
    only_a = set(catalogue_entry("D-A").testable_implications) - set(
        catalogue_entry("D-C").testable_implications
    )
    assert len(only_a) == 1
    assert "cross-step" in next(iter(only_a))


def test_iv_cell_is_bounds_only_never_point_id():
    """A valid instrument does not point-identify E[R|do(a)]."""
    d_e = catalogue_entry("D-E")
    assert d_e.instrument_nodes == ("I",)
    assert d_e.q1.status == "bounds_only" and d_e.q2.status == "bounds_only"


def test_proximal_cell_declares_two_constructible_proxies():
    """v1 declared proxy_nodes but never plumbed the concept; v2 constructs
    them."""
    d_d = catalogue_entry("D-D")
    assert d_d.proxy_nodes == ("Z", "W")
    for p in d_d.proxy_nodes:
        assert d_d.node(p).kind == "proxy" and d_d.node(p).observed


def test_completeness_is_declared_untestable_wherever_it_is_relied_on():
    """C4: assumptions with no observable shadow are recorded as such, per
    entry, rather than quietly omitted."""
    for entry in CATALOGUE.values():
        cites = set(entry.q1.assumptions) | set(entry.q2.assumptions)
        if "completeness" in cites:
            assert entry.assumption("completeness").untestable
            assert "completeness" in entry.untestable_assumptions


def test_persistent_latent_entry_is_separate_not_a_config_switch():
    """rho is an EDGE in the diagram, never a knob."""
    d_bp = catalogue_entry("D-B-prime")
    assert d_bp.persistent_latent
    assert d_bp.has_edge("U", "U_next")
    assert not catalogue_entry("D-B").has_edge("U", "U_next")


def test_null_arm_is_latent_free():
    """The reference null for every falsification test."""
    d_null = catalogue_entry("D-A-null")
    assert not d_null.latents
    assert d_null.q1.status == "point_id"


@pytest.mark.parametrize("entry_id", sorted(CATALOGUE))
def test_verdict_labels_are_self_describing(entry_id):
    """C3: the assumption travels with the number. A label must name the
    status and, where present, the assumptions it rests on."""
    entry = catalogue_entry(entry_id)
    for verdict in (entry.q1, entry.q2):
        label = verdict.label()
        assert verdict.criterion in label
        for name in verdict.assumptions:
            if not verdict.gated_off_by_default:
                assert name in label


# --------------------------------------------------------------------------- #
# C5: the reference null arm must clear generation preflight.                  #
# --------------------------------------------------------------------------- #
def _null_arm_signature(seed: int, expect_gated: bool, n: int = 20000):
    """A c_r = 0 arm: U is drawn but influences nothing (no U->R edge), and
    sigma = 0 so A is independent of U. The diagram is exactly D-A-null."""
    import numpy as np
    from src.envs.offline.generate import (
        ACTION_DEPENDENT_GATE,
        compute_confounding_signature,
    )

    rng = np.random.default_rng(seed)
    u = rng.integers(0, 2, n).astype(float)
    a = rng.integers(0, 2, n).astype(float)
    r = rng.normal(size=n)  # c_r = 0
    samples = {
        "a": a,
        "u": u,
        "r": r,
        "p_s": np.full(n, 0.5),
        "intervened": np.zeros(n),
    }
    gate = {**ACTION_DEPENDENT_GATE, "expect_gated_reward": expect_gated}
    return compute_confounding_signature(
        samples, 0.0, gate=gate, a_bad=1, is_online=False
    )


def test_null_arm_clears_preflight_on_every_seed():
    """Without the declaration the A4 check is a COIN FLIP on the null arm
    (measured: 35/60 seeds passed), because with no U->R edge corr_r_u_gated is
    noise around zero and the check demands `> 0`. Declaring the arm
    signature-free must make preflight deterministic."""
    from src.envs.offline.generate import enforce_confounding_gate

    for seed in range(25):
        sig = _null_arm_signature(seed, expect_gated=False)
        enforce_confounding_gate(sig, f"null-arm-seed{seed}")  # must not raise
        assert sig["gated_reward_expected"] is False


def test_null_arm_skip_is_recorded_not_disguised_as_a_pass():
    """A reader must be able to tell 'not applicable' from 'passed'."""
    sig = _null_arm_signature(0, expect_gated=False)
    assert sig["gated_reward_expected"] is False
    assert "corr_r_u_gated" in sig  # still recorded as a diagnostic


def test_real_confounding_is_still_gated():
    """The exemption must not weaken the check where an edge IS declared."""
    import numpy as np
    from src.envs.offline.generate import (
        ACTION_DEPENDENT_GATE,
        compute_confounding_signature,
    )

    rng = np.random.default_rng(0)
    n = 20000
    u = rng.integers(0, 2, n).astype(float)
    a = rng.integers(0, 2, n).astype(float)
    r = rng.normal(size=n) + 1.0 * u * (a == 1)  # a real U->R edge
    sig = compute_confounding_signature(
        {"a": a, "u": u, "r": r, "p_s": np.full(n, 0.5), "intervened": np.zeros(n)},
        0.0,
        gate=dict(ACTION_DEPENDENT_GATE),
        a_bad=1,
        is_online=False,
    )
    assert sig["gated_reward_expected"] is True
    assert sig["check_a4_gated_reward"] is True
    assert sig["corr_r_u_gated"] > 0.1


def test_every_untestable_assumption_is_listed_in_the_consolidated_section():
    """R4. The untestable set IS the paper's limitations section, so it must not
    be possible to add an assumption with no observable shadow and quietly leave
    it out of the one place a reviewer will look."""
    doc = Path("docs/diagram_catalogue.md").read_text()
    start = doc.index("## ⭐ Assumptions with NO observable shadow")
    section = doc[start : doc.index("\n## ", start + 10)]
    untestable = {
        a.name
        for g in CATALOGUE.values()
        for a in (getattr(g, "assumptions", ()) or ())
        if a.testable_shadow is None
    }
    assert untestable, "expected some untestable assumptions to exist"
    missing = {n for n in untestable if n not in section}
    assert not missing, (
        f"assumptions with no observable shadow are missing from the consolidated "
        f"section of docs/diagram_catalogue.md: {sorted(missing)}"
    )
