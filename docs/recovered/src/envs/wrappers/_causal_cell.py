"""Cell-mapping helpers for the causal eight-cell taxonomy."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CellConfig:
    cell: int
    z_exposed: bool
    u_exposed: bool
    pi_b_known: bool
    offline: bool


CELL_CONFIGS: dict[int, CellConfig] = {
    1: CellConfig(1, z_exposed=True, u_exposed=True, pi_b_known=True, offline=False),
    2: CellConfig(2, z_exposed=False, u_exposed=False, pi_b_known=False, offline=False),
    3: CellConfig(3, z_exposed=True, u_exposed=True, pi_b_known=True, offline=True),
    4: CellConfig(4, z_exposed=True, u_exposed=False, pi_b_known=True, offline=True),
    5: CellConfig(5, z_exposed=True, u_exposed=True, pi_b_known=False, offline=True),
    6: CellConfig(6, z_exposed=True, u_exposed=False, pi_b_known=False, offline=True),
    7: CellConfig(7, z_exposed=True, u_exposed=False, pi_b_known=True, offline=True),
    8: CellConfig(8, z_exposed=False, u_exposed=False, pi_b_known=False, offline=True),
}


def get_cell_config(cell: int) -> CellConfig:
    if cell not in CELL_CONFIGS:
        raise ValueError(f"cell must be in 1..8, got {cell}")
    return CELL_CONFIGS[cell]


def parse_cell_from_env_id(env_id: str, default_cell: int = 1) -> int:
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
