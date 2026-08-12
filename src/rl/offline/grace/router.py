"""GRACE regime router (B5, amendment A3).

Consumes ONLY data-derivable statistics (leakage rule R5: the generation-time
gate metadata is computed WITH the recorded U and is ground truth for scoring,
never router input):

  * ``delta_a`` / ``delta_r`` — the channel-split held-out latent-evidence
    components (``TemplateCBN.router_components``): ``delta_a`` (latent
    improves P(A|.)) is the CONFOUNDING-specific signal; ``delta_r`` (latent
    improves P(R|.)) is heterogeneity and legitimately fires on the basic arm
    (c_r > 0 at sigma=0 is action-independent U reward noise).
  * ``coverage`` — the state-conditional action-coverage statistic (low
    quantile over visited state bins of ``min_a pi_b_hat(a | s)``).
  * ``width`` / ``separability`` / ``belief_entropy`` — identification
    health: ensemble interval width vs its calibrated null width, EM stratum
    separability, filter entropy.

Serving rule (B5): no defect -> Q_obs; coverage defect only -> Q_lower;
confounding detected and identification healthy -> Q_do; confounding detected
and identification unhealthy -> Q_lower. A router with NO calibrated
thresholds NEVER routes causally: verdict "uncalibrated", serve Q_obs —
mirroring the null-calibration gate's missing-reference convention.

Thresholds come from NULL CALIBRATION ON BASIC-CELL RUNS with the established
``k * noise`` convention (k = ``NULL_CALIBRATION_K`` = 1.5), stored per env in
``reproducibility/rl_regimes/_base/grace_router_reference.yaml`` (populated by
the E0 calibration block; missing entry -> uncalibrated, never a silent pass).

Public signatures exchange plain stats dicts and verdicts — nothing tabular
leaks through (A10 seam).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

_REFERENCE_PATH = (
    Path(__file__).resolve().parents[4]
    / "reproducibility"
    / "rl_regimes"
    / "_base"
    / "grace_router_reference.yaml"
)

# The repo's pinned null-calibration multiplier (regime_report.py). Imported
# lazily in calibrate() to avoid a benchmarking->rl import cycle at module
# import time; this constant is the documented fallback.
_K_FALLBACK = 1.5

ARM_LABELS = ("basic", "biased", "confounded")


@dataclass(frozen=True)
class RouterVerdict:
    label: str  # basic | biased | confounded | uncalibrated
    serve: str  # obs | do | lower
    calibrated: bool
    reasons: tuple = ()
    stats: dict = field(default_factory=dict)


class RegimeRouter:
    """Threshold container + the B5/A3 decision rule.

    ``thresholds`` keys: ``delta_a_thr``, ``delta_r_thr`` (fire above),
    ``coverage_thr`` (defect below), ``width_thr`` (unhealthy above).
    ``graph_ok`` is the A9.3 graphical precondition for the confounded serving
    mode: without a declared U -> A edge the router must not route to Q_do on
    confounding grounds, whatever the statistics say.
    """

    def __init__(
        self,
        thresholds: Optional[Dict[str, float]] = None,
        graph_ok: bool = True,
    ) -> None:
        self.thresholds = dict(thresholds) if thresholds else None
        self.graph_ok = bool(graph_ok)

    # ------------------------------------------------------------ resolution
    @classmethod
    def from_reference(cls, env_id: str | None, graph_ok: bool = True):
        """Load per-env thresholds from the E0-populated reference yaml.
        Missing file/entry -> an UNCALIBRATED router (serves Q_obs)."""
        if env_id is None or not _REFERENCE_PATH.exists():
            return cls(None, graph_ok)
        import yaml

        with open(_REFERENCE_PATH) as fh:
            data = yaml.safe_load(fh) or {}
        entry = (data.get("reference") or {}).get(env_id)
        if not entry:
            return cls(None, graph_ok)
        return cls({k: float(v) for k, v in entry.items()}, graph_ok)

    @staticmethod
    def calibrate(null_stats: List[Dict[str, float]], k: float | None = None):
        """Thresholds from basic-run null statistics (one dict per seed/run):
        fire-above stats get ``mean + k * sd``; the coverage defect (fire-
        below) gets ``mean - k * sd``. k defaults to the repo's pinned
        ``NULL_CALIBRATION_K``."""
        if k is None:
            try:
                from src.benchmarking.regime_report import NULL_CALIBRATION_K

                k = float(NULL_CALIBRATION_K)
            except Exception:  # reporting layer absent (unit tests)
                k = _K_FALLBACK
        if not null_stats:
            raise ValueError("router calibration needs at least one null run")

        def _agg(key: str):
            vals = [float(s[key]) for s in null_stats if key in s]
            if not vals:
                raise ValueError(f"null stats missing '{key}'")
            n = len(vals)
            mean = sum(vals) / n
            sd = (sum((v - mean) ** 2 for v in vals) / max(n - 1, 1)) ** 0.5
            return mean, sd

        thr: Dict[str, float] = {}
        for key in ("delta_a", "delta_r", "width"):
            mean, sd = _agg(key)
            thr[f"{key}_thr"] = mean + k * sd
        mean, sd = _agg("coverage")
        thr["coverage_thr"] = mean - k * sd
        return thr

    # -------------------------------------------------------------- decision
    def verdict(self, stats: Dict[str, float]) -> RouterVerdict:
        if self.thresholds is None:
            return RouterVerdict(
                label="uncalibrated",
                serve="obs",
                calibrated=False,
                reasons=("no calibrated thresholds — never route causally blind",),
                stats=dict(stats),
            )
        t = self.thresholds
        reasons: list[str] = []
        confounding = float(stats.get("delta_a", 0.0)) > t["delta_a_thr"]
        if confounding and not self.graph_ok:
            confounding = False
            reasons.append(
                "delta_a fired but the declared graph has no U->A edge "
                "(A9.3): confounded serving structurally unjustified"
            )
        heterogeneity = float(stats.get("delta_r", 0.0)) > t["delta_r_thr"]
        coverage_defect = float(stats.get("coverage", 1.0)) < t["coverage_thr"]
        unhealthy = float(stats.get("width", 0.0)) > t.get("width_thr", float("inf"))

        if confounding:
            label = "confounded"
            if unhealthy:
                serve = "lower"
                reasons.append(
                    "confounding detected but identification unhealthy "
                    "(ensemble spread beyond calibrated width) -> pessimism"
                )
            else:
                serve = "do"
                reasons.append("confounding detected, identification healthy")
        elif coverage_defect:
            label = "biased"
            serve = "lower"
            reasons.append("coverage defect only -> pessimistic lower bound")
        else:
            label = "basic"
            serve = "obs"
            if heterogeneity:
                reasons.append(
                    "delta_r fired without delta_a: action-independent U "
                    "heterogeneity (the basic arm's c_r noise), not confounding"
                )
        return RouterVerdict(
            label=label,
            serve=serve,
            calibrated=True,
            reasons=tuple(reasons),
            stats=dict(stats),
        )
