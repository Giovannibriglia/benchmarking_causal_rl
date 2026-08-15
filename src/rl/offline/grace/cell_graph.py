"""Declared causal diagrams — GRACE v2's ENTIRE assumption surface.

v2 asserts exactly one thing per scenario: the causal diagram. Everything else
is derived from it (L2 identification, L5's testable implications), learned
from data (mechanism parameters), or selected by a held-out criterion
(``u_card``, mechanism class). A quantity that is none of those is a defect.

This module is therefore four things at once, and is authored as a reviewable
artifact rather than embedded implicitly in estimator code:

  1. L1 — the declaration itself, and L2's input;
  2. gate V2's ground truth (L2 must reproduce every ``Verdict`` below);
  3. L5's source of testable implications;
  4. the taxonomy's identifiability axis.

The human-readable companion is ``docs/diagram_catalogue.md``; a test asserts
the two agree entry-for-entry.

TWO VERDICTS PER ENTRY. ``q1`` (per-step ``E[R_t | do(a), s]``) and ``q2``
(sequential ``V^pi``) are declared separately BECAUSE THEY DIFFER, and a
single-verdict design would silently license a sequential point estimate from
a per-step argument:

  * D-F is q1 point-ID (conditioning on the observed emission blocks the
    state back-door) but q2 NON-ID (a history-dependent policy's value needs
    the latent-state trajectory law; per-step adjustment does not compose);
  * D-B is q1 point-ID on the nonparametric proximal bridge, but q2 point-ID
    only on the STRICTLY STRONGER finite-K latent-class assumption, because
    the sequential value integrates the exogenous P(U) while the per-step
    bridge delivers the behaviour-tilted P(U | X).

Each ``Verdict`` therefore carries the ``assumptions`` it rests on, and every
number produced under it must travel with that label (see ``Verdict.label``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Tuple

__all__ = [
    "Assumption",
    "CellGraph",
    "GraphNode",
    "Verdict",
    "CATALOGUE",
    "NODE_KINDS",
    "STATUSES",
    "catalogue_entry",
]

# Node roles. ``proxy`` and ``instrument`` are CONSTRUCTIBLE kinds in v2 (in
# v1 "proxy" existed only in a comment and a declared proxy node would have
# raised downstream).
NODE_KINDS: Tuple[str, ...] = (
    "state",
    "latent_confounder",
    "action",
    "reward",
    "observation",
    "proxy",
    "instrument",
)

STATUSES: Tuple[str, ...] = ("point_id", "bounds_only", "non_id")

CRITERIA: Tuple[str, ...] = (
    "none",
    "backdoor",
    "frontdoor",
    "proximal",
    "proximal_lagged",
    "iv",
    "sequential_composition",
)


@dataclass(frozen=True)
class GraphNode:
    """One node. ``lag`` is explicit rather than encoded in a ``_next`` naming
    convention, so temporal structure is data, not string parsing."""

    name: str
    kind: str
    observed: bool
    lag: int = 0

    def __post_init__(self) -> None:
        if self.kind not in NODE_KINDS:
            raise ValueError(f"unknown node kind {self.kind!r}")


@dataclass(frozen=True)
class Assumption:
    """A named assumption a verdict rests on.

    ``testable_shadow`` is the observable consequence L5 can test, or ``None``
    when the assumption has NO observable shadow. Recording ``None`` explicitly
    is the point: it is how the catalogue states which of its own assumptions
    are beyond falsification (C4).
    """

    name: str
    statement: str
    testable_shadow: str | None = None

    @property
    def untestable(self) -> bool:
        return self.testable_shadow is None


@dataclass(frozen=True)
class Verdict:
    """What the declared diagram licenses for one query."""

    status: str
    criterion: str
    assumptions: Tuple[str, ...] = ()
    adjustment_set: Tuple[str, ...] = ()
    # D-B's q2 point-ID rests on a derivation under review; it ships OFF so the
    # conservative bounds reading holds unless explicitly enabled (C4).
    gated_off_by_default: bool = False
    note: str = ""

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"unknown status {self.status!r}")
        if self.criterion not in CRITERIA:
            raise ValueError(f"unknown criterion {self.criterion!r}")

    @property
    def effective_status(self) -> str:
        """What v2 ACTUALLY serves. A gated point-ID degrades to bounds_only
        unless the gate is explicitly opened."""
        if self.gated_off_by_default and self.status == "point_id":
            return "bounds_only"
        return self.status

    def label(self, enabled: bool = False) -> str:
        """The assumption label that must travel with every number produced
        under this verdict (C3) — attached to the estimate object, never left
        to prose."""
        status = (
            self.status if (enabled or not self.gated_off_by_default) else "bounds_only"
        )
        if not self.assumptions:
            return f"{status} ({self.criterion})"
        return f"{status} ({self.criterion}; assumes {', '.join(self.assumptions)})"


@dataclass(frozen=True)
class CellGraph:
    """A declared SCM for one scenario.

    ``edges`` is the OBSERVATIONAL-channel edge list; the mutilated channel is
    DERIVED (never re-declared) as the same list minus in-edges of the
    intervention target.
    """

    id: str
    name: str
    nodes: Tuple[GraphNode, ...]
    edges: Tuple[Tuple[str, str], ...]
    q1: Verdict
    q2: Verdict
    intervention_target: str = "A"
    proxy_nodes: Tuple[str, ...] = ()
    instrument_nodes: Tuple[str, ...] = ()
    persistent_latent: bool = False
    assumptions: Tuple[Assumption, ...] = ()
    testable_implications: Tuple[str, ...] = ()
    asserted_by: Tuple[str, ...] = ()
    # Filled from the taxonomy manuscript, which is not in this repository.
    paper_figure: str = "TODO-verify"

    # ------------------------------------------------------------- accessors
    def node(self, name: str) -> GraphNode:
        for n in self.nodes:
            if n.name == name:
                return n
        raise KeyError(f"{self.id}: no node {name!r}")

    def has_node(self, name: str) -> bool:
        return any(n.name == name for n in self.nodes)

    def has_edge(self, src: str, dst: str) -> bool:
        return (src, dst) in self.edges

    def in_edges_of(self, name: str) -> Tuple[Tuple[str, str], ...]:
        return tuple(e for e in self.edges if e[1] == name)

    def mutilated_edges(self) -> Tuple[Tuple[str, str], ...]:
        """The do()-surgery edge list: every edge into the intervention target
        removed. THE ONLY definition of the mutilation."""
        return tuple(e for e in self.edges if e[1] != self.intervention_target)

    @property
    def observational_only(self) -> FrozenSet[Tuple[str, str]]:
        return frozenset(self.in_edges_of(self.intervention_target))

    @property
    def latents(self) -> Tuple[GraphNode, ...]:
        return tuple(n for n in self.nodes if not n.observed)

    @property
    def confounded(self) -> bool:
        """Does a latent confounder point into the intervention target?"""
        return any(
            self.has_edge(n.name, self.intervention_target)
            for n in self.nodes
            if n.kind == "latent_confounder"
        )

    @property
    def dynamics_confounded(self) -> bool:
        """Does a latent point into the next state? This is the guard that
        decides whether q2 can decompose at all: with unconfounded dynamics
        the occupancy under do(pi) is latent-independent, and V^pi factors
        into an identified occupancy times a per-(s,a) reward. With
        U -> S_next it does not, and no per-step result composes."""
        return any(
            self.has_edge(n.name, "S_next") for n in self.nodes if not n.observed
        )

    def assumption(self, name: str) -> Assumption:
        for a in self.assumptions:
            if a.name == name:
                return a
        raise KeyError(f"{self.id}: no assumption {name!r}")

    @property
    def untestable_assumptions(self) -> Tuple[str, ...]:
        return tuple(a.name for a in self.assumptions if a.untestable)

    # ------------------------------------------------------------ validation
    def validate(self) -> None:
        """Structural checks: valid DAG, no bidirectional edges, declared nodes
        only, an emission for every latent state, and declared proxy/instrument
        names that actually exist with the right kind."""
        names = {n.name for n in self.nodes}
        if self.intervention_target not in names:
            raise ValueError(
                f"{self.id}: intervention target {self.intervention_target!r} "
                "is not a declared node"
            )
        for src, dst in self.edges:
            if src not in names or dst not in names:
                raise ValueError(
                    f"{self.id}: edge ({src},{dst}) uses an undeclared node"
                )
            if (dst, src) in self.edges:
                raise ValueError(f"{self.id}: bidirectional edge ({src},{dst})")
        # Acyclicity (Kahn).
        indeg = {n: 0 for n in names}
        for _, dst in self.edges:
            indeg[dst] += 1
        frontier = [n for n, d in indeg.items() if d == 0]
        seen = 0
        while frontier:
            cur = frontier.pop()
            seen += 1
            for src, dst in self.edges:
                if src == cur:
                    indeg[dst] -= 1
                    if indeg[dst] == 0:
                        frontier.append(dst)
        if seen != len(names):
            raise ValueError(f"{self.id}: declared edges contain a cycle")
        # Latent state requires an observed emission.
        for n in self.nodes:
            if n.kind == "state" and not n.observed and n.name == "S":
                if not self.has_node("O") or not self.has_edge("S", "O"):
                    raise ValueError(
                        f"{self.id}: latent S requires an observed O with S->O"
                    )
        for p in self.proxy_nodes:
            if p not in names or self.node(p).kind != "proxy":
                raise ValueError(f"{self.id}: {p!r} is not a declared proxy node")
        for i in self.instrument_nodes:
            if i not in names or self.node(i).kind != "instrument":
                raise ValueError(f"{self.id}: {i!r} is not a declared instrument node")
        # Every assumption a verdict cites must be declared.
        declared = {a.name for a in self.assumptions}
        for v in (self.q1, self.q2):
            missing = set(v.assumptions) - declared
            if missing:
                raise ValueError(
                    f"{self.id}: verdict cites undeclared {sorted(missing)}"
                )
        # The q2 decomposition guard (see dynamics_confounded).
        if self.dynamics_confounded and self.q2.status == "point_id":
            raise ValueError(
                f"{self.id}: q2 cannot be point_id with confounded dynamics — the "
                "occupancy under do(pi) is not latent-independent, so no per-step "
                "result composes"
            )


# --------------------------------------------------------------------------- #
# Shared pieces                                                                #
# --------------------------------------------------------------------------- #
_CORE_EDGES: Tuple[Tuple[str, str], ...] = (
    ("S", "A"),
    ("S", "R"),
    ("A", "R"),
    ("S", "S_next"),
    ("A", "S_next"),
)


def _mdp_nodes(*extra: GraphNode) -> Tuple[GraphNode, ...]:
    return (
        GraphNode("S", "state", observed=True),
        GraphNode("A", "action", observed=True),
        GraphNode("R", "reward", observed=True),
        GraphNode("S_next", "state", observed=True, lag=1),
    ) + extra


_U = GraphNode("U", "latent_confounder", observed=False)

# Shared implication strings. D-A and D-C assert the SAME memorylessness
# constraint, so it is written once: the entries then differ by exactly the
# cross-step reward constraint, which is what makes M1 detectable.
_IMPL_POLICY_MEMORYLESS = "A_t indep (R_{t-1}, A_{t-1}, S_{t-1}, ...) | S_t"
_IMPL_NO_CROSS_STEP_REWARD = (
    "R_t indep R_t' | (S_t, A_t, S_t', A_t')  [no cross-step reward dependence]"
)

# Assumptions reused across entries.
_A_COMPLETENESS = Assumption(
    name="completeness",
    statement=(
        "The proxy-to-latent map is injective (the measurement matrix is "
        "non-degenerate), so the bridge equation is well posed."
    ),
    testable_shadow=None,  # irreducibly untestable
)
_A_FINITE_K = Assumption(
    name="finite_K_latent_class",
    statement=(
        "The latent takes finitely many values (|U| = K) and the latent-class "
        "model is identifiable from repeated within-episode measurements. This "
        "is what lets the sequential value re-marginalise U from the "
        "behaviour-tilted P(U|X) to the exogenous P(U)."
    ),
    testable_shadow="rank <= K on cross-time / proxy moment matrices",
)
_A_STATIC_U = Assumption(
    name="static_U",
    statement="U is constant within an episode (no U_{t-1} -> U_t edge).",
    testable_shadow="proxy informativeness does not decay with lag",
)
_A_EP_LEN = Assumption(
    name="episode_length_ge_3",
    statement=(
        "Episodes are long enough to supply three conditionally independent "
        "measurements of U (t-2, t-1, t) — the condition the finite-mixture "
        "identifiability result needs."
    ),
    testable_shadow="episode-length distribution is directly observable",
)
_A_CROSS_STRATUM_LINKING = Assumption(
    name="cross_stratum_label_linking",
    statement=(
        "The latent's LABELLING is consistent across covariate configurations. "
        "The lagged-proxy views are conditionally independent given U only "
        "TOGETHER WITH the (S, A) at the measurement times, so Kruskal's "
        "condition applies per configuration and identifies the latent "
        "structure only up to a relabelling AT EACH (s, a). Nothing in the "
        "diagram forces 'class 1 at (s,a)' to be 'class 1 at (s',a')': the "
        "obvious linking assumption -- U independent of the covariates -- is "
        "FALSE here, since S_t is a descendant of U through past actions "
        "(the same P(U|X) != P(U) asymmetry that drives the q2 derivation). "
        "The linking is supplied instead by the SHARED MECHANISM FAMILY: one "
        "P(R | S, A, U) fitted across all configurations forces a consistent "
        "labelling. That is an assumption of the MODEL CLASS, not of the "
        "graph. Note D-D does NOT need it: its declared proxies are "
        "covariate-free (parents = {U}), so P(Z|U) and P(W|U) are global and "
        "pin the labelling globally."
    ),
    # The canonicalization convention detects permutation WITHIN a fit, but
    # consistency of the latent's MEANING across covariate configurations has
    # at best a weak observable shadow. None is the truthful entry.
    testable_shadow=None,
)
_A_PROXY_SIGNAL = Assumption(
    name="proxy_informativeness",
    statement=(
        "Both proxies carry signal about U: sigma > 0 (so the action proxy is "
        "U-dependent) AND P(A = a_bad) is bounded away from 0 (so the gated "
        "reward proxy is U-dependent). Because the reward shift is ACTION-GATED, "
        "this decays as the behaviour policy improves."
    ),
    testable_shadow="estimated rank gap / proxy-U mutual information",
)


# --------------------------------------------------------------------------- #
# The catalogue                                                                #
# --------------------------------------------------------------------------- #
def _build() -> Dict[str, CellGraph]:
    entries: list[CellGraph] = []

    # ---- D-A: MDP, no latent. Asserted by the BIASED arm only. -------------
    entries.append(
        CellGraph(
            id="D-A",
            name="MDP, no latent",
            nodes=_mdp_nodes(),
            edges=_CORE_EDGES,
            q1=Verdict("point_id", "backdoor", adjustment_set=("S",)),
            q2=Verdict("point_id", "sequential_composition", adjustment_set=("S",)),
            testable_implications=(_IMPL_POLICY_MEMORYLESS, _IMPL_NO_CROSS_STEP_REWARD),
            asserted_by=("offline_mdp/biased", "offline_pomdp/biased"),
        )
    )

    # ---- D-A-null: the clean null (c_r = 0). -------------------------------
    entries.append(
        CellGraph(
            id="D-A-null",
            name="MDP, no latent, no coverage defect (reference null)",
            nodes=_mdp_nodes(),
            edges=_CORE_EDGES,
            q1=Verdict("point_id", "backdoor", adjustment_set=("S",)),
            q2=Verdict("point_id", "sequential_composition", adjustment_set=("S",)),
            testable_implications=(_IMPL_POLICY_MEMORYLESS, _IMPL_NO_CROSS_STEP_REWARD),
            asserted_by=("offline_mdp/null",),
        )
    )

    # ---- D-C: MDP, action-INDEPENDENT latent. The BASIC arm. ---------------
    entries.append(
        CellGraph(
            id="D-C",
            name="MDP, action-independent latent (reward-only)",
            nodes=_mdp_nodes(_U),
            edges=_CORE_EDGES + (("U", "R"),),
            q1=Verdict("point_id", "backdoor", adjustment_set=("S",)),
            q2=Verdict("point_id", "sequential_composition", adjustment_set=("S",)),
            # Deliberately NOT cross-step reward independence: that is exactly
            # the constraint D-C drops relative to D-A, and the one that makes
            # the two empirically distinguishable. The shared constraint is
            # stated with the SAME string as D-A's so the difference between
            # the two entries is exactly one implication (asserted in tests).
            testable_implications=(_IMPL_POLICY_MEMORYLESS,),
            asserted_by=("offline_mdp/basic",),
        )
    )

    # ---- D-B: MDP, episode-static U, action-gated reward. CONFOUNDED. ------
    entries.append(
        CellGraph(
            id="D-B",
            name="MDP, episode-static latent, action-gated reward",
            nodes=_mdp_nodes(_U),
            edges=_CORE_EDGES + (("U", "A"), ("U", "R")),
            q1=Verdict(
                "point_id",
                "proximal_lagged",
                assumptions=(
                    "static_U",
                    "episode_length_ge_3",
                    "proxy_informativeness",
                    "completeness",
                    "cross_stratum_label_linking",
                ),
                gated_off_by_default=True,
                note=(
                    "Lagged proxies W = R_{t-1}, Z = A_{t-2}, X = (S_{t-2}, S_{t-1}, S_t). "
                    "The NAIVE pairing (Z = A_{t-1}, W = R_{t-1}) is INVALID: the direct "
                    "edge A_{t-1} -> R_{t-1} violates W indep Z | U."
                ),
            ),
            q2=Verdict(
                "point_id",
                "proximal_lagged",
                assumptions=(
                    "static_U",
                    "episode_length_ge_3",
                    "proxy_informativeness",
                    "completeness",
                    "cross_stratum_label_linking",
                    "finite_K_latent_class",
                ),
                gated_off_by_default=True,
                note=(
                    "STRICTLY STRONGER than q1. Per-step proximal gives the "
                    "X-conditional effect under P(U|X); V^pi needs the exogenous "
                    "P(U). Closing that gap requires recovering the latent-class "
                    "model itself, not merely the bridge."
                ),
            ),
            assumptions=(
                _A_STATIC_U,
                _A_EP_LEN,
                _A_PROXY_SIGNAL,
                _A_COMPLETENESS,
                _A_CROSS_STRATUM_LINKING,
                _A_FINITE_K,
            ),
            testable_implications=("rank <= K on cross-time moment matrices",),
            asserted_by=("offline_mdp/confounded",),
        )
    )

    # ---- D-B': persistent U. ----------------------------------------------
    entries.append(
        CellGraph(
            id="D-B-prime",
            name="MDP, PERSISTENT latent (rho > 0), action-gated reward",
            nodes=_mdp_nodes(
                _U, GraphNode("U_next", "latent_confounder", observed=False, lag=1)
            ),
            edges=_CORE_EDGES + (("U", "A"), ("U", "R"), ("U", "U_next")),
            persistent_latent=True,
            q1=Verdict(
                "bounds_only",
                "proximal_lagged",
                assumptions=(
                    "episode_length_ge_3",
                    "proxy_informativeness",
                    "completeness",
                    "cross_stratum_label_linking",
                ),
                note=(
                    "The EXCLUSIONS survive drift (conditioning on U_t blocks the "
                    "paths through U_{t-1}); COMPLETENESS is what degrades, because "
                    "W measures U_{t-1} and is only correlated with the U_t that "
                    "confounds step t. Point-ID in principle while the latent "
                    "transition is invertible, dying by ill-conditioning at full "
                    "refresh — so the shipped verdict is the conservative one."
                ),
            ),
            q2=Verdict(
                "bounds_only",
                "proximal_lagged",
                assumptions=(
                    "episode_length_ge_3",
                    "proxy_informativeness",
                    "completeness",
                    "cross_stratum_label_linking",
                ),
            ),
            assumptions=(
                _A_EP_LEN,
                _A_PROXY_SIGNAL,
                _A_COMPLETENESS,
                _A_CROSS_STRATUM_LINKING,
            ),
            testable_implications=(
                "rank <= K on cross-time moment matrices (ill-conditioned with drift)",
            ),
            asserted_by=("offline_mdp/persistent",),
        )
    )

    # ---- D-D: explicit negative-control proxies. --------------------------
    entries.append(
        CellGraph(
            id="D-D",
            name="MDP + explicit negative-control proxies (proximal cell)",
            nodes=_mdp_nodes(
                _U,
                GraphNode("Z", "proxy", observed=True),
                GraphNode("W", "proxy", observed=True),
            ),
            edges=_CORE_EDGES + (("U", "A"), ("U", "R"), ("U", "Z"), ("U", "W")),
            proxy_nodes=("Z", "W"),
            q1=Verdict("point_id", "proximal", assumptions=("completeness",)),
            q2=Verdict(
                "point_id",
                "proximal",
                assumptions=("completeness", "finite_K_latent_class"),
            ),
            assumptions=(_A_COMPLETENESS, _A_FINITE_K),
            testable_implications=("rank <= |U| on P(Z, W | A, S)",),
            asserted_by=("offline_mdp/proximal",),
        )
    )

    # ---- D-E: instrument. Bounds, NOT point-ID. ---------------------------
    entries.append(
        CellGraph(
            id="D-E",
            name="MDP + instrument (IV cell)",
            nodes=_mdp_nodes(_U, GraphNode("I", "instrument", observed=True)),
            edges=_CORE_EDGES + (("U", "A"), ("U", "R"), ("I", "A")),
            instrument_nodes=("I",),
            q1=Verdict(
                "bounds_only",
                "iv",
                note="Balke-Pearl. A valid instrument does NOT point-identify "
                "E[R|do(a)] without monotonicity/homogeneity.",
            ),
            q2=Verdict("bounds_only", "iv"),
            testable_implications=("instrumental inequalities (refutation-only)",),
            asserted_by=("offline_mdp/iv",),
        )
    )

    # ---- D-F: POMDP, latent state, no U. q1 != q2. ------------------------
    entries.append(
        CellGraph(
            id="D-F",
            name="POMDP, latent state, no confounder",
            nodes=(
                GraphNode("S", "state", observed=False),
                GraphNode("O", "observation", observed=True),
                GraphNode("A", "action", observed=True),
                GraphNode("R", "reward", observed=True),
                GraphNode("S_next", "state", observed=False, lag=1),
            ),
            edges=(
                ("S", "O"),
                ("O", "A"),
                ("S", "R"),
                ("A", "R"),
                ("S", "S_next"),
                ("A", "S_next"),
            ),
            q1=Verdict("point_id", "backdoor", adjustment_set=("O",)),
            q2=Verdict(
                "non_id",
                "none",
                note=(
                    "Per-step adjustment does NOT compose: a history-dependent "
                    "target policy's value needs the joint law of latent-state "
                    "trajectories under do(pi), which observational data does not "
                    "pin down without proxy structure."
                ),
            ),
            testable_implications=(
                "HMM-style rank constraints on observation matrices",
            ),
            asserted_by=("offline_pomdp/basic", "offline_pomdp/biased"),
        )
    )

    # ---- D-G: POMDP + U. --------------------------------------------------
    entries.append(
        CellGraph(
            id="D-G",
            name="POMDP + confounder",
            nodes=(
                GraphNode("S", "state", observed=False),
                GraphNode("O", "observation", observed=True),
                GraphNode("A", "action", observed=True),
                GraphNode("R", "reward", observed=True),
                GraphNode("S_next", "state", observed=False, lag=1),
                _U,
            ),
            edges=(
                ("S", "O"),
                ("O", "A"),
                ("S", "R"),
                ("A", "R"),
                ("S", "S_next"),
                ("A", "S_next"),
                ("U", "A"),
                ("U", "R"),
            ),
            q1=Verdict("bounds_only", "none"),
            q2=Verdict("non_id", "none"),
            testable_implications=(
                "HMM-style rank constraints on observation matrices",
            ),
            asserted_by=("offline_pomdp/confounded",),
        )
    )

    return {e.id: e for e in entries}


CATALOGUE: Dict[str, CellGraph] = _build()
for _e in CATALOGUE.values():
    _e.validate()


def catalogue_entry(entry_id: str) -> CellGraph:
    try:
        return CATALOGUE[entry_id]
    except KeyError:
        raise KeyError(
            f"unknown diagram {entry_id!r}; declared: {sorted(CATALOGUE)}"
        ) from None
