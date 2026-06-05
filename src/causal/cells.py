"""Eight-cell taxonomy registry + Gymnasium registration of causal env ids.

Cell semantics follow the paper table (Phase-0 audit §1; NOT the recovered
``_causal_cell.CELL_CONFIGS``, which contradicts it — see
docs/recovered/REVIEW.md):

| Cell | Name              | Z      | Mode / pi_b        | U confounds |
|------|-------------------|--------|--------------------|-------------|
| 1    | Crystal Clear     | obs    | online             | no          |
| 2    | Invisible Gene    | hidden | online             | no          |
| 3    | Perfect Archive   | obs    | offline, known     | no          |
| 4    | Burned Files      | hidden | offline, known     | no          |
| 5    | Doctor's Intuition| obs    | offline, unknown   | no          |
| 6    | Fog of History    | hidden | offline, unknown   | no          |
| 7    | Shadowed Vitals   | obs    | offline            | yes         |
| 8    | Dark Ages         | hidden | offline, unknown   | yes         |

Per the Phase-0 gate (decision 6), the ONLINE cells are materialized as real
Gymnasium ids under the ``causal/`` namespace, so cells 1–2 run through plain
benchmark mode, compose with vector wrappers, and are collectable with
``minari.DataCollector``. ``--mode causal_cells`` (Phase 3) orchestrates only
datasets / OPE / regret for the offline cells.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Union

import gymnasium as gym

from .masking import ObservationMasking


@dataclass(frozen=True)
class CellSpec:
    cell: int
    name: str
    z_observed: bool
    online: bool
    behavior_policy: Optional[Literal["known", "unknown"]]  # None for online cells
    confounded: bool


CELLS: Dict[int, CellSpec] = {
    1: CellSpec(1, "Crystal Clear", True, True, None, False),
    2: CellSpec(2, "Invisible Gene", False, True, None, False),
    3: CellSpec(3, "Perfect Archive", True, False, "known", False),
    4: CellSpec(4, "Burned Files", False, False, "known", False),
    5: CellSpec(5, "Doctor's Intuition", True, False, "unknown", False),
    6: CellSpec(6, "Fog of History", False, False, "unknown", False),
    7: CellSpec(7, "Shadowed Vitals", True, False, "known", True),
    8: CellSpec(8, "Dark Ages", False, False, "unknown", True),
}


def get_cell(cell: int) -> CellSpec:
    if cell not in CELLS:
        raise ValueError(f"cell must be in 1..8, got {cell}")
    return CELLS[cell]


def parse_cell_from_env_id(env_id: str, default_cell: int = 1) -> int:
    """Parse the ``cell<N>`` token out of a causal env id (id grammar adapted
    from the recovered implementation)."""
    token = "cell"
    lowered = env_id.lower()
    if token not in lowered:
        return default_cell
    suffix = lowered.split(token, 1)[1]
    digits = []
    for ch in suffix:
        if ch.isdigit():
            digits.append(ch)
        else:
            break
    if not digits:
        return default_cell
    return int("".join(digits))


def make_causal_env(
    base_id: str,
    mask_indices: Optional[Union[Sequence[int], str]] = None,
    frame_stack: Optional[int] = None,
    **kwargs,
) -> gym.Env:
    """Entry point for ``causal/`` env ids: base env, optionally masked
    (partial observability axis), optionally frame-stacked (Cell-2 variant)."""
    env = gym.make(base_id, **kwargs)
    if mask_indices is not None:
        env = ObservationMasking(env, mask_indices=mask_indices)
    if frame_stack is not None and int(frame_stack) > 1:
        from gymnasium.wrappers import FrameStackObservation

        env = FrameStackObservation(env, stack_size=int(frame_stack))
    return env


# anchor tag -> base Gymnasium id
ANCHORS: Dict[str, str] = {
    "cartpole": "CartPole-v1",
    "halfcheetah": "HalfCheetah-v5",
}

FRAME_STACK_SIZE = 4

_REGISTERED = False


def register_causal_envs() -> None:
    """Register the online-cell env ids under the ``causal/`` namespace.

    Ids (per anchor): ``causal/<anchor>-cell1`` (alias of the base env, the
    Cell-1 reference task), ``causal/<anchor>-cell2`` (velocity-masked,
    Cell-2 basic), ``causal/<anchor>-cell2fs`` (velocity-masked +
    ``FrameStackObservation`` — the Cell-2 variant; frame-stacking restores
    velocity information from position history, the canonical cheap response
    to the epistemic POMDP, Ghosh et al. 2021).
    """
    global _REGISTERED
    if _REGISTERED:
        return
    existing = set(gym.registry.keys())
    for anchor, base_id in ANCHORS.items():
        variants = {
            f"causal/{anchor}-cell1": {"base_id": base_id},
            f"causal/{anchor}-cell2": {
                "base_id": base_id,
                "mask_indices": "velocities",
            },
            f"causal/{anchor}-cell2fs": {
                "base_id": base_id,
                "mask_indices": "velocities",
                "frame_stack": FRAME_STACK_SIZE,
            },
        }
        for env_id, kwargs in variants.items():
            if env_id in existing:
                continue
            gym.register(
                id=env_id,
                entry_point="src.causal.cells:make_causal_env",
                kwargs=kwargs,
            )
    _REGISTERED = True


def causal_online_env_ids() -> List[str]:
    """All registered online-cell env ids (for ENV_SETS)."""
    return [
        f"causal/{anchor}-{tag}"
        for anchor in ANCHORS
        for tag in ("cell1", "cell2", "cell2fs")
    ]
