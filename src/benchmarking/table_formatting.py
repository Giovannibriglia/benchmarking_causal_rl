"""Table formatting utilities: per-column best-value bolding, metric direction,
sweep-family detection, and human-readable family labels.

These helpers are shared by the standard per-run table renderer and the new
sweep-family aggregation renderer in ``plotting.py``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

# Direction map: per metric, "max" means higher-is-better; "min" means
# lower-is-better. Used to decide which value in a column gets bolded.
# Unrecognized metrics default to "max" (return-style assumption). When a base
# metric name (e.g. ``eval_return``) is looked up, ``metric_direction`` also
# tries the ``<name>_mean`` form so the keys below need only list the canonical
# column names. Add new metrics here.
_METRIC_DIRECTION: Dict[str, str] = {
    "eval_return_mean": "max",  # episodic return — higher is better
    "train_return_mean": "max",  # training return — higher is better
    "eval_return_std": "min",  # tighter bands are usually better
    "train_return_std": "min",  # tighter bands are usually better
    "td_error": "min",  # temporal-difference error — lower is better
    "policy_loss": "min",  # optimization losses — lower is better
    "critic_loss": "min",  # critic regression loss — lower is better
    "actor_loss": "min",  # actor objective loss — lower is better
    "q_loss": "min",  # Q-function regression loss — lower is better
    "loss": "min",  # generic loss — lower is better
    "entropy": "max",  # higher entropy = more exploration
    "alpha": "max",  # SAC entropy temperature — higher = more stochastic
}


def metric_direction(metric_name: str) -> str:
    """Look up the optimization direction for ``metric_name``.

    Tries the exact name, then the ``<name>_mean`` form (so logical metric
    names like ``eval_return`` resolve via ``eval_return_mean``), then defaults
    to ``"max"`` for unknown metrics.
    """
    if metric_name in _METRIC_DIRECTION:
        return _METRIC_DIRECTION[metric_name]
    mean_form = f"{metric_name}_mean"
    if mean_form in _METRIC_DIRECTION:
        return _METRIC_DIRECTION[mean_form]
    return "max"


def best_indices_per_column(
    values: np.ndarray, direction: str = "max", tol: float = 1e-9
) -> List[Set[int]]:
    """For each column of ``values`` (an ``(n_rows, n_cols)`` array of means),
    return a set of row indices that achieve the best value in that column.

    ``direction`` is ``"max"`` or ``"min"`` per the metric.

    Tied bests (within ``tol``) all get returned. NaN values are excluded from
    the best computation; if all values in a column are NaN, an empty set is
    returned.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {values.shape}")
    n_rows, n_cols = values.shape
    best_indices: List[Set[int]] = []
    for col_idx in range(n_cols):
        col = values[:, col_idx]
        valid_mask = ~np.isnan(col)
        if not valid_mask.any():
            best_indices.append(set())
            continue
        if direction == "max":
            target = np.nanmax(col)
        else:
            target = np.nanmin(col)
        # All rows within tol of the target get the bold.
        best_rows = {
            i for i in range(n_rows) if valid_mask[i] and abs(col[i] - target) < tol
        }
        best_indices.append(best_rows)
    return best_indices


def format_cell(mean: float, std: float, is_best: bool, precision: int = 1) -> str:
    """Format a ``mean ± std`` cell with optional ``\\textbf{...}``.

    Returns an em-dash for missing (NaN) means.
    """
    if np.isnan(mean):
        return "—"
    cell = f"{mean:.{precision}f} ± {std:.{precision}f}"
    if is_best:
        cell = f"\\textbf{{{cell}}}"
    return cell


# Strength suffix pattern. Matches a 3-digit strength group either at the end of
# the basename (``..._050``) or mid-string with an arbitrary tail
# (``..._050_discrete_gated``). The non-greedy prefix ensures the FIRST 3-digit
# group is treated as the strength dial; the optional tail captures the
# remaining descriptor (``gated``, ``discrete_gated``, ``masked_discrete``...).
_STRENGTH_SUFFIX_RE = re.compile(r"^(.+?)_(\d{3})(?:_(.+))?$")


def detect_sweep_families(cell_dir: Path) -> Dict[str, List[Tuple[str, str]]]:
    """Scan a cell's YAML directory for sweep families.

    A sweep family is a set of >=3 YAML basenames sharing a common stem and
    differing only in a 3-digit strength suffix (``_000``, ``_025``, ``_050``,
    ``_075``, ``_100`` by convention, but any 3-digit value is accepted). The
    strength group may sit at the end of the basename or mid-string with a
    descriptor tail (the Cells 7/8 σ-sweep convention,
    ``confounded_sigma_050_discrete_gated``).

    Returns a dict ``{family_stem: [(strength, full_basename), ...]}`` sorted by
    strength ascending. Singletons and families with <3 members are excluded.
    """
    families: Dict[str, List[Tuple[str, str]]] = {}
    for yaml_path in sorted(Path(cell_dir).glob("*.yaml")):
        basename = yaml_path.stem
        match = _STRENGTH_SUFFIX_RE.match(basename)
        if not match:
            continue
        prefix, strength, tail = match.group(1), match.group(2), match.group(3)
        stem = f"{prefix}_{tail}" if tail else prefix
        families.setdefault(stem, []).append((strength, basename))
    return {
        stem: sorted(members, key=lambda x: x[0])
        for stem, members in families.items()
        if len(members) >= 3
    }


# Tokens stripped from a family stem when building a human-readable label.
# These are regime/observability descriptors, not the benchmark dial name.
_LABEL_DROP_TOKENS = {"discrete", "continuous", "gated"}


def family_label(family_stem: str) -> str:
    """Convert a family stem to a human-readable benchmark label.

    Drops regime/observability descriptors (``discrete``, ``continuous``,
    ``gated``), folds ``masked`` into a trailing modifier, and keeps the dial
    name. ``online`` is dropped when it merely pairs with a regime descriptor
    (e.g. ``online_discrete_curiosity`` → "Curiosity") but kept when it
    distinguishes an online variant from an offline sibling
    (e.g. ``online_confounded_sigma_discrete_gated`` → "Online confounded
    sigma", vs. the offline ``confounded_sigma_discrete_gated`` →
    "Confounded sigma").
    """
    tokens = family_stem.split("_")
    masked = False
    kept: List[str] = []
    for tok in tokens:
        if tok == "masked":
            masked = True
            continue
        if tok in _LABEL_DROP_TOKENS:
            continue
        kept.append(tok)
    # Drop a leading "online" only when there is no other meaningful dial token
    # that needs it for disambiguation (i.e. no "confounded" sibling concept).
    if "online" in kept and "confounded" not in kept:
        kept = [t for t in kept if t != "online"]
    label = " ".join(kept)
    if masked:
        label = f"{label} masked".strip()
    return label.strip().capitalize()


def metric_label(metric: str) -> str:
    """Human-readable metric label: strip ``_mean``/``_std`` and underscores.

    ``eval_return_mean`` → "eval return"; ``td_error`` → "td error".
    """
    base = metric
    if base.endswith("_mean"):
        base = base[: -len("_mean")]
    elif base.endswith("_std"):
        base = base[: -len("_std")]
    return base.replace("_", " ")


def strength_to_float_label(strength: str) -> str:
    """Render a 3-digit strength code as a decimal: ``050`` → "0.50"."""
    return f"{int(strength) / 100:.2f}"
