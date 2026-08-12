"""GRACE — General Regime-Adaptive Causal Estimator (feat/grace-critic).

Tabular-first slice: a declared-graph template CBN (``cell_graph``/``cbn``)
with two likelihood channels, an exact belief filter over the discrete latent
block (``filter``), a null-calibrated regime router with channel-split
detection statistics (``router``), bootstrap-ensemble interval heads, and the
routed serving surface (``heads``/``machinery``), plugged into the benchmark
through the base learners' strategy seam (``builders``).
"""

from src.rl.offline.grace.builders import (
    build_grace_bcq,
    build_grace_cql,
    build_grace_dqn,
    build_grace_dqn_recurrent,
    build_grace_iql,
    build_grace_online_dqn,
)
from src.rl.offline.grace.cbn import EpisodeData, GraceOptions, TemplateCBN
from src.rl.offline.grace.cell_graph import (
    cell_graph,
    CELL_GRAPHS,
    CellGraph,
    identification_report,
)
from src.rl.offline.grace.discretizer import Discretizer, RewardCodec
from src.rl.offline.grace.filter import BeliefFilter
from src.rl.offline.grace.heads import GraceQNetwork, GraceRecurrentQNetwork
from src.rl.offline.grace.machinery import GraceMachinery
from src.rl.offline.grace.router import RegimeRouter, RouterVerdict

__all__ = [
    "BeliefFilter",
    "CELL_GRAPHS",
    "CellGraph",
    "cell_graph",
    "Discretizer",
    "EpisodeData",
    "GraceMachinery",
    "GraceOptions",
    "GraceQNetwork",
    "GraceRecurrentQNetwork",
    "identification_report",
    "RegimeRouter",
    "RewardCodec",
    "RouterVerdict",
    "TemplateCBN",
    "build_grace_bcq",
    "build_grace_cql",
    "build_grace_dqn",
    "build_grace_dqn_recurrent",
    "build_grace_iql",
    "build_grace_online_dqn",
]
