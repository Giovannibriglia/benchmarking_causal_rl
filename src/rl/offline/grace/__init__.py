"""GRACE v2 — the diagram-assumption critic.

v2 asserts exactly ONE thing per scenario: the declared causal diagram
(``cell_graph``). Everything else is derived from it, learned from data, or
selected by a held-out criterion.

Currently exported: L1, the declaration layer. The v2 estimator (L2-L5) lands
in subsequent commits; the v1 tabular estimator it replaces was deleted
wholesale rather than ported, and is preserved on ``feat/grace-critic``.
"""

from src.rl.offline.grace.cell_graph import (
    Assumption,
    CATALOGUE,
    catalogue_entry,
    CellGraph,
    GraphNode,
    NODE_KINDS,
    STATUSES,
    Verdict,
)

__all__ = [
    "Assumption",
    "CATALOGUE",
    "CellGraph",
    "GraphNode",
    "NODE_KINDS",
    "STATUSES",
    "Verdict",
    "catalogue_entry",
]
