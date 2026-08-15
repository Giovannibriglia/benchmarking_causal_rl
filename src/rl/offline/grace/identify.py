"""L2 — graphical identification. Deterministic, no statistics.

Given a declared ``CellGraph`` and a query, decide **point-ID / bounds-only /
not identified** and return the estimand form. This is not a boolean gate: it
selects what gets estimated, and it never returns point-ID by default —
anything outside the decidable subset returns ``unknown`` and is treated as
bounds-only.

Everything here is derived from the graph's NODES, EDGES and declared
proxy/instrument roles. It deliberately does **not** read the catalogue's
``Verdict`` fields, so that comparing this module's output against those fields
(gate V2) is a real test rather than a tautology.

Two queries, because they differ (see ``cell_graph``'s module docstring):

* **q1** — the per-step estimand ``E[R_t | do(a), s]``.
* **q2** — the sequential value ``V^pi``.

The q2 guard is the load-bearing one: ``V^pi`` decomposes into an identified
occupancy times a per-``(s,a)`` reward **only if the dynamics are unconfounded**,
because that is what makes the state-action trajectory under ``do(pi)``
independent of the latent. With a latent pointing into the next state, no
per-step result composes and q2 is not identified regardless of q1.

Decidable subset, in precedence order: back-door, front-door, declared
proximal, declared instrument, implicit lagged-proxy (gated off), else unknown.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Set, Tuple

from src.rl.offline.grace.cell_graph import CellGraph, STATUSES

__all__ = ["IdentificationResult", "identify", "identify_catalogue"]

_MAX_ADJUSTMENT_SET = 3  # these graphs are tiny; bound the subset search anyway


@dataclass(frozen=True)
class IdentificationResult:
    """What L2 concluded, and on what."""

    status: str  # point_id | bounds_only | non_id
    criterion: str
    assumptions: Tuple[str, ...] = ()
    adjustment_set: Tuple[str, ...] = ()
    gated: bool = False
    reason: str = ""

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"unknown status {self.status!r}")

    @property
    def effective_status(self) -> str:
        """What may actually be served: a gated point-ID degrades to bounds."""
        if self.gated and self.status == "point_id":
            return "bounds_only"
        return self.status


# --------------------------------------------------------------------------- #
# Graph primitives (d-separation on the declared DAG)                          #
# --------------------------------------------------------------------------- #
def _parents(g: CellGraph, n: str) -> Set[str]:
    return {s for s, d in g.edges if d == n}


def _children(g: CellGraph, n: str) -> Set[str]:
    return {d for s, d in g.edges if s == n}


def _descendants(g: CellGraph, n: str) -> Set[str]:
    seen: Set[str] = set()
    frontier = [n]
    while frontier:
        cur = frontier.pop()
        for c in _children(g, cur):
            if c not in seen:
                seen.add(c)
                frontier.append(c)
    return seen


def _d_separated(g: CellGraph, x: str, y: str, z: Set[str]) -> bool:
    """Is ``x`` d-separated from ``y`` given ``z``? Bayes-ball reachability."""
    z_desc: Set[str] = set()
    for n in z:
        z_desc |= {n} | _descendants(g, n)
    # (node, arrived-from-child?) reachability
    visited: Set[Tuple[str, bool]] = set()
    frontier: List[Tuple[str, bool]] = [(x, False)]
    while frontier:
        node, from_child = frontier.pop()
        if (node, from_child) in visited:
            continue
        visited.add((node, from_child))
        if node == y:
            return False
        if not from_child:
            # arrived along an edge into a child (or start): can go both ways
            if node not in z:
                for p in _parents(g, node):
                    frontier.append((p, False))
                for c in _children(g, node):
                    frontier.append((c, True))
        else:
            # arrived at a collider from below
            if node not in z:
                for c in _children(g, node):
                    frontier.append((c, True))
            if node in z_desc:  # collider opened by conditioning
                for p in _parents(g, node):
                    frontier.append((p, False))
    return True


def _graph_without_out_edges(g: CellGraph, x: str) -> CellGraph:
    """G with edges OUT of ``x`` removed — the back-door test graph."""
    return CellGraph(
        id=g.id + "_underline",
        name=g.name,
        nodes=g.nodes,
        edges=tuple(e for e in g.edges if e[0] != x),
        q1=g.q1,
        q2=g.q2,
        intervention_target=g.intervention_target,
        proxy_nodes=g.proxy_nodes,
        instrument_nodes=g.instrument_nodes,
    )


def _backdoor_sets(g: CellGraph, x: str, y: str) -> List[Tuple[str, ...]]:
    """Observed non-descendant sets satisfying the back-door criterion."""
    gx = _graph_without_out_edges(g, x)
    desc = _descendants(g, x)
    candidates = sorted(
        n.name
        for n in g.nodes
        if n.observed and n.name not in (x, y) and n.name not in desc
    )
    found: List[Tuple[str, ...]] = []
    for size in range(0, min(_MAX_ADJUSTMENT_SET, len(candidates)) + 1):
        for combo in combinations(candidates, size):
            if _d_separated(gx, x, y, set(combo)):
                found.append(combo)
        if found:  # prefer the smallest sufficient set
            break
    return found


# --------------------------------------------------------------------------- #
# Structural predicates for the non-back-door criteria                        #
# --------------------------------------------------------------------------- #
def _latent_confounders(g: CellGraph, x: str, y: str) -> List[str]:
    return [
        n.name
        for n in g.nodes
        if not n.observed
        and n.kind == "latent_confounder"
        and g.has_edge(n.name, x)
        and g.has_edge(n.name, y)
    ]


def _proximal_ok(g: CellGraph, x: str, y: str) -> bool:
    """Two declared proxies of a latent confounder, with the exclusions the
    declaration implies: neither proxy causes or is caused by anything but the
    latent."""
    if len(g.proxy_nodes) < 2 or not _latent_confounders(g, x, y):
        return False
    for p in g.proxy_nodes:
        pa = _parents(g, p)
        if not pa or any(g.node(a).observed for a in pa):
            return False  # a proxy must be driven by the latent alone
        if _children(g, p):
            return False  # ...and must cause nothing (negative control)
    return True


def _instrument_ok(g: CellGraph, x: str, y: str) -> bool:
    """A declared instrument: into X, independent of the latent, no direct
    path to Y."""
    for i in g.instrument_nodes:
        if not g.has_edge(i, x):
            continue
        if _parents(g, i):
            continue  # exogenous
        if g.has_edge(i, y):
            continue  # exclusion restriction
        return True
    return False


def _lagged_proxies_available(g: CellGraph, x: str, y: str) -> bool:
    """The IMPLICIT temporal proxies of an episode-static latent.

    Requires: a latent confounder on both X and Y; the latent NOT persistent
    (a drifting latent is measured by lagged views only through an
    increasingly ill-conditioned channel); the reward a sink, so lagged rewards
    are valid negative-control outcomes; and every state observed, since the
    conditioning set for the lagged argument is made of states.
    """
    if not _latent_confounders(g, x, y):
        return False
    if g.persistent_latent:
        return False
    if _children(g, y):  # reward must be a sink
        return False
    return all(n.observed for n in g.nodes if n.kind == "state")


# --------------------------------------------------------------------------- #
# The decision procedure                                                       #
# --------------------------------------------------------------------------- #
def identify(g: CellGraph, query: str = "q1") -> IdentificationResult:
    """Decide identification for ``query`` in ``{q1, q2}`` from the graph."""
    if query not in ("q1", "q2"):
        raise ValueError(f"query must be 'q1' or 'q2', got {query!r}")
    g.validate()
    x, y = g.intervention_target, "R"

    per_step = _identify_per_step(g, x, y)
    if query == "q1":
        return per_step

    # --- q2: the sequential value.
    if g.dynamics_confounded:
        return IdentificationResult(
            "non_id",
            "none",
            reason=(
                "a latent points into the next state, so the trajectory under "
                "do(pi) is not latent-independent and no per-step result "
                "composes into V^pi"
            ),
        )
    if per_step.status != "point_id":
        return IdentificationResult(
            per_step.status,
            per_step.criterion,
            assumptions=per_step.assumptions,
            gated=per_step.gated,
            reason="the per-step estimand is not point-identified",
        )
    if not _latent_confounders(g, x, y):
        return IdentificationResult(
            "point_id",
            "sequential_composition",
            adjustment_set=per_step.adjustment_set,
            reason=(
                "unconfounded dynamics and no latent on the reward channel: the "
                "occupancy under do(pi) is identified and composes"
            ),
        )
    # A latent survives: the sequential value integrates the EXOGENOUS P(U),
    # while the per-step bridge delivers the behaviour-tilted P(U|X). Closing
    # that gap needs the latent-class model itself -> finite-K.
    return IdentificationResult(
        "point_id",
        per_step.criterion,
        assumptions=tuple(per_step.assumptions) + ("finite_K_latent_class",),
        gated=per_step.gated,
        reason=(
            "sequential value needs the exogenous P(U), not the X-conditional "
            "P(U|X); recovering it requires the finite latent-class model"
        ),
    )


def _identify_per_step(g: CellGraph, x: str, y: str) -> IdentificationResult:
    # 1. back-door (includes the no-back-door-path case, an empty set)
    sets = _backdoor_sets(g, x, y)
    if sets:
        return IdentificationResult(
            "point_id",
            "backdoor",
            adjustment_set=sets[0],
            reason=(
                "no back-door path"
                if sets[0] == ()
                else f"back-door blocked by {list(sets[0])}"
            ),
        )
    # 2. front-door — not exercised by the current catalogue, implemented for
    #    completeness of the decidable subset.
    fd = _frontdoor_set(g, x, y)
    if fd is not None:
        return IdentificationResult(
            "point_id", "frontdoor", adjustment_set=fd, reason="front-door mediator"
        )
    # 3. declared proximal
    if _proximal_ok(g, x, y):
        return IdentificationResult(
            "point_id",
            "proximal",
            assumptions=("completeness",),
            reason="two declared negative-control proxies of the latent",
        )
    # 4. declared instrument -> BOUNDS, never point-ID
    if _instrument_ok(g, x, y):
        return IdentificationResult(
            "bounds_only",
            "iv",
            reason=(
                "a valid instrument bounds the interventional value "
                "(Balke-Pearl) but does not point-identify it"
            ),
        )
    # 5. implicit lagged proxies — GATED OFF: ships as bounds until the
    #    derivation is settled in writing.
    if _lagged_proxies_available(g, x, y):
        return IdentificationResult(
            "point_id",
            "proximal_lagged",
            assumptions=(
                "static_U",
                "episode_length_ge_3",
                "proxy_informativeness",
                "completeness",
                # The lagged views are covariate-CONDITIONAL, so Kruskal applies
                # per (s,a) and the labelling is linked only by the shared
                # mechanism family -- a model-class assumption, not a graphical
                # one. Declared proxies (D-D) are covariate-free and need no
                # such linking.
                "cross_stratum_label_linking",
            ),
            gated=True,
            reason=(
                "episode-static latent with a sink reward and observed states "
                "admits lagged proxies (W=R_{t-1}, Z=A_{t-2}); gated off"
            ),
        )
    # 6. outside the decidable subset -> NEVER point-ID by default
    return IdentificationResult(
        "bounds_only",
        "none",
        reason="unknown: no criterion in the decidable subset applies",
    )


def _frontdoor_set(g: CellGraph, x: str, y: str) -> Tuple[str, ...] | None:
    """A single observed mediator intercepting every directed X->Y path, with
    no unblocked back-door into it."""
    for n in g.nodes:
        m = n.name
        if not n.observed or m in (x, y):
            continue
        if not (g.has_edge(x, m) and g.has_edge(m, y)):
            continue
        if g.has_edge(x, y):  # a direct edge is not intercepted
            continue
        if not _backdoor_sets(g, m, y):
            continue
        return (m,)
    return None


def identify_catalogue(
    catalogue: Dict[str, CellGraph],
) -> Dict[str, Dict[str, IdentificationResult]]:
    """Run L2 over every declared diagram — the V-A product."""
    return {
        eid: {"q1": identify(g, "q1"), "q2": identify(g, "q2")}
        for eid, g in catalogue.items()
    }
