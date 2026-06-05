"""Dispatcher for `causal-*` environment ids.

The dispatcher keeps the registry API stable while routing each causal env id
to the proper CausalEnv wrapper implementation and cell configuration.
"""

from __future__ import annotations

from typing import Any

from src.envs.base import BaseEnv

from ._causal_cell import parse_cell_from_env_id
from .block_mdp import BlockMDPEnv
from .sepsis import SepsisCausalEnv


def build_causal_env(**kwargs: Any) -> BaseEnv:
    env_id = str(kwargs["env_id"]).lower()
    env_kwargs = dict(kwargs.pop("env_kwargs", {}) or {})
    kwargs.pop("env_entry_point", None)
    cell = int(env_kwargs.pop("cell", parse_cell_from_env_id(env_id, default_cell=1)))

    if env_id.startswith("causal-sepsis"):
        return SepsisCausalEnv(cell=cell, **kwargs, **env_kwargs)
    if env_id.startswith("causal-block-mdp"):
        return BlockMDPEnv(cell=cell, **kwargs, **env_kwargs)
    raise KeyError(
        f"Unknown causal env id '{kwargs['env_id']}'. "
        "Expected one of: causal-sepsis, causal-block-mdp."
    )
