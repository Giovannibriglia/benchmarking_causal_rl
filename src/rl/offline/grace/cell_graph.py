"""Declared causal graphs for the GRACE template CBN (amendment A9).

One ``CellGraph`` per (cell, arm) family, encoding exactly the per-cell SCMs
formalized in the paper and enforced by the generation-time gates
(``compute_confounding_signature`` / ``_action_dependent_signature``):

  * ``U -> A`` — OBSERVATIONAL channel only (the behavior policy's
    confounded action choice; removed by graph surgery in the mutilated
    channel). Declared on the confounded arm and on the fitting TEMPLATE;
    absent on the basic arm (the marginally-matched sigma=0 construction makes
    A independent of U *exactly*, and the generation gate asserts it) and on the
    biased arm (no U at all).
  * ``U -> R`` — the action-gated reward channel ``r += c_r * U * 1[a=a_bad]``
    (``action_gated_reward=True``). Present wherever U is (including the basic
    arm: ``c_r_for`` keeps c_r > 0 at sigma=0, where the noise is
    action-independent and therefore NOT confounding).
  * ``S -> O`` — the emission. Identity in MDP cells (O observed = S); a real
    emission in POMDP cells, where S is latent and O is the masked observation.
  * ``S, A -> S'`` dynamics; ``S, A[, U] -> R``; optional ``U -> U'``
    persistence behind the rho switch (declared, default off).
  * PROXY NODES: declared EXPLICITLY ABSENT in every current cell
    (``proxy_nodes=()``): the benchmark constructs no confounder-correlated
    proxies (Phase-1 audit), so proximal-style point-identification claims are
    never made and the router must reflect it (brief section B1's conditional).

``TemplateCBN`` consumes a ``CellGraph`` and builds BOTH channels from it —
the observational net from the full edge list, the mutilated net from the same
list minus edges into A. No edge is introduced anywhere else (A9.2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Tuple

__all__ = [
    "GraphNode",
    "CellGraph",
    "cell_graph",
    "identification_report",
    "CELL_GRAPHS",
]

ARMS: Tuple[str, ...] = ("template", "basic", "biased", "confounded")
OBSERVABILITIES: Tuple[str, ...] = ("mdp", "pomdp")


@dataclass(frozen=True)
class GraphNode:
    name: str
    # "state" | "latent_confounder" | "action" | "reward" | "observation"
    # | "proxy" — the proxy kind exists so absence is a declaration, not an
    # omitted concept (A9.1).
    kind: str
    observed: bool


@dataclass(frozen=True)
class CellGraph:
    """A declared per-(cell, arm) SCM. ``edges`` is the OBSERVATIONAL-channel
    edge list; the mutilated channel is derived (never re-declared) as the
    same list minus in-edges of A."""

    name: str  # e.g. "offline_mdp/confounded"
    nodes: Tuple[GraphNode, ...]
    edges: Tuple[Tuple[str, str], ...]
    action_gated_reward: bool  # U->R is r += c_r*U*1[a=a_bad]
    rho_persistence: bool  # U->U' declared (fit behind opts.rho)
    identity_emission: bool  # MDP: O = S exactly
    proxy_nodes: Tuple[str, ...] = ()  # EXPLICITLY absent in all current cells

    # ------------------------------------------------------------- accessors
    def node(self, name: str) -> GraphNode:
        for n in self.nodes:
            if n.name == name:
                return n
        raise KeyError(f"{self.name}: no node '{name}'")

    def has_node(self, name: str) -> bool:
        return any(n.name == name for n in self.nodes)

    def has_edge(self, src: str, dst: str) -> bool:
        return (src, dst) in self.edges

    def in_edges_of(self, name: str) -> Tuple[Tuple[str, str], ...]:
        return tuple(e for e in self.edges if e[1] == name)

    def mutilated_edges(self) -> Tuple[Tuple[str, str], ...]:
        """The do(A)-surgery edge list: every edge INTO A removed. This is the
        ONLY place the mutilation is defined (A9.2)."""
        return tuple(e for e in self.edges if e[1] != "A")

    @property
    def observational_only(self) -> FrozenSet[Tuple[str, str]]:
        """Edges present in the observational channel but not the mutilated
        one — by construction, exactly the in-edges of A."""
        return frozenset(self.in_edges_of("A"))

    @property
    def confounded(self) -> bool:
        """Does the DECLARED structure contain the U -> A backdoor edge?"""
        return self.has_edge("U", "A")

    @property
    def pomdp(self) -> bool:
        return self.has_node("S") and not self.node("S").observed

    # ------------------------------------------------------------ validation
    def validate(self) -> None:
        """The repo's standing diagram-validity checks (A9.4): valid DAG, no
        bidirectional action edges, state nodes present (with an emission path)
        in partially observed cells, edges only between declared nodes."""
        names = {n.name for n in self.nodes}
        for src, dst in self.edges:
            if src not in names or dst not in names:
                raise ValueError(
                    f"{self.name}: edge ({src},{dst}) uses undeclared node"
                )
            if (dst, src) in self.edges:
                raise ValueError(f"{self.name}: bidirectional edge ({src},{dst})")
        # Acyclicity (Kahn over the tiny declared graph).
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
            raise ValueError(f"{self.name}: declared edges contain a cycle")
        if self.pomdp:
            if not self.has_node("O") or not self.has_edge("S", "O"):
                raise ValueError(
                    f"{self.name}: latent S requires an observed O with an S->O "
                    "emission edge"
                )
        for p in self.proxy_nodes:
            if p not in names:
                raise ValueError(f"{self.name}: proxy '{p}' not declared as a node")


def cell_graph(
    observability: str, arm: str = "template", rho: bool = False
) -> CellGraph:
    """Build the declared SCM for one (observability, arm) family.

    ``arm="template"`` is the MODEL CLASS GRACE fits — structurally the
    confounded arm (U -> A declared, observational channel only), so the fitted
    latent can EXPRESS confounding; whether the data evidences it is the
    router's question, never the graph's. The per-arm instances are the
    ground-truth SCMs used by tests, docs and the G4 labels.
    """
    if observability not in OBSERVABILITIES:
        raise ValueError(f"unknown observability '{observability}'")
    if arm not in ARMS:
        raise ValueError(f"unknown arm '{arm}'")
    pomdp = observability == "pomdp"
    has_u = arm != "biased"
    u_to_a = arm in ("template", "confounded")

    nodes = [
        GraphNode("S", "state", observed=not pomdp),
        GraphNode("O", "observation", observed=True),
        GraphNode("A", "action", observed=True),
        GraphNode("R", "reward", observed=True),
        GraphNode("S_next", "state", observed=not pomdp),
    ]
    edges: list[Tuple[str, str]] = [
        ("S", "O"),
        ("S", "A"),
        ("S", "R"),
        ("A", "R"),
        ("S", "S_next"),
        ("A", "S_next"),
    ]
    if pomdp:
        # The behavior policy sees the OBSERVATION, not the latent state.
        edges[edges.index(("S", "A"))] = ("O", "A")
    if has_u:
        nodes.append(GraphNode("U", "latent_confounder", observed=False))
        edges.append(("U", "R"))
        if u_to_a:
            edges.append(("U", "A"))
        if rho:
            nodes.append(GraphNode("U_next", "latent_confounder", observed=False))
            edges.append(("U", "U_next"))
    g = CellGraph(
        name=f"{observability}/{arm}",
        nodes=tuple(nodes),
        edges=tuple(edges),
        action_gated_reward=has_u,
        rho_persistence=bool(rho and has_u),
        identity_emission=not pomdp,
        proxy_nodes=(),  # explicitly absent in every current cell (A9.1)
    )
    g.validate()
    return g


def identification_report(graph: CellGraph) -> Dict[str, object]:
    """The A9.3 graphical precondition, evaluated against the DECLARED graph.

    * ``confounded_serving_ok`` — the U -> A edge is declared; without it the
      confounded serving mode is structurally unjustified and the router must
      not route to Q_do on confounding grounds, whatever the statistics say.
    * ``proxies_present`` — governs whether proximal-style point-
      identification claims are ever made (currently never: every declared
      cell has ``proxy_nodes=()``).
    * ``adjustment_ok`` — every common cause of (A, R) in the declared
      structure is either observed or a MODELED latent: U is the enumerated
      discrete latent block; a latent S is admissible only with a declared
      S -> O emission (checked by ``validate``).

    Returns the flags plus human-readable ``reasons`` — logged per cell.
    """
    graph.validate()
    reasons: list[str] = []
    confounded_ok = graph.has_edge("U", "A")
    if not confounded_ok:
        reasons.append(
            f"{graph.name}: no declared U->A edge — confounded serving "
            "structurally unjustified (router must never route to Q_do on "
            "confounding grounds)"
        )
    proxies = bool(graph.proxy_nodes)
    if not proxies:
        reasons.append(
            f"{graph.name}: proxy nodes declared ABSENT — no proximal-style "
            "point-identification claim is ever made for this cell"
        )

    def _reaches(src: str, dst: str) -> bool:
        frontier = [src]
        seen = set()
        while frontier:
            cur = frontier.pop()
            if cur == dst:
                return True
            if cur in seen:
                continue
            seen.add(cur)
            frontier.extend(d for s, d in graph.edges if s == cur)
        return False

    adjustment_ok = True
    for node in graph.nodes:
        if node.name in ("A", "R"):
            continue
        common_cause = _reaches(node.name, "A") and _reaches(node.name, "R")
        if not common_cause:
            continue
        if node.observed:
            continue
        if node.kind == "latent_confounder":
            continue  # the enumerated discrete latent block models it
        if node.kind == "state" and graph.has_edge(node.name, "O"):
            continue  # latent state with a declared emission
        adjustment_ok = False
        reasons.append(
            f"{graph.name}: unmodeled latent common cause '{node.name}' of "
            "(A, R) — no adjustment path for the declared structure"
        )
    return {
        "confounded_serving_ok": confounded_ok,
        "proxies_present": proxies,
        "adjustment_ok": adjustment_ok,
        "reasons": tuple(reasons),
    }


def _build_registry() -> Dict[str, Dict[str, CellGraph]]:
    reg: Dict[str, Dict[str, CellGraph]] = {}
    for regime, obs in (
        ("offline_mdp", "mdp"),
        ("offline_pomdp", "pomdp"),
        ("online_mdp", "mdp"),
        ("online_pomdp", "pomdp"),
    ):
        reg[regime] = {arm: cell_graph(obs, arm) for arm in ARMS}
        for arm, g in reg[regime].items():
            object.__setattr__(g, "name", f"{regime}/{arm}")
    return reg


# One declaration per (cell, arm); ``[cell]["template"]`` is what GRACE fits.
# docs/grace.md cross-references each entry to the paper's formalization
# figures (A9.4).
CELL_GRAPHS: Dict[str, Dict[str, CellGraph]] = _build_registry()
