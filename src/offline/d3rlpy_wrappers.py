"""Thin Algorithm adapters over d3rlpy 2.x (pinned 2.8.1) — CQL and IQL.

Why a wrapper: the cells pipeline talks to ``Algorithm`` (``act`` returning
``ActionOutput``) and to ``fit_source(source, n_steps)``; d3rlpy keeps its own
training loop, so ``learn`` delegates to ``fit`` on the source's MDPDataset
rather than stepping per-batch.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch

from src.data.experience_source import OfflineDatasetSource
from src.rl.base import ActionOutput, Algorithm


class D3rlpyAlgorithm(Algorithm):
    paradigm = "offline"

    def __init__(self, d3_algo, device: torch.device, action_type: str) -> None:
        super().__init__()
        self.algo = d3_algo
        self.device = device
        self.action_type = action_type

    def act(
        self,
        obs: torch.Tensor,
        state: Optional[Any] = None,
        *,
        deterministic: bool = False,
    ) -> ActionOutput:
        # d3rlpy predict() is deterministic (greedy/mean) by design.
        _ = deterministic
        arr = obs.detach().cpu().numpy().astype(np.float32)
        single = arr.ndim == 1
        if single:
            arr = arr[None, :]
        actions = self.algo.predict(arr)
        out = torch.as_tensor(actions, device=obs.device)
        if single:
            out = out.squeeze(0)
        return ActionOutput(action=out, state=state)

    def learn(self, batch: Any) -> Dict[str, float]:
        raise NotImplementedError(
            "d3rlpy algorithms train via fit_source(source, n_steps); "
            "per-batch learn() is not exposed."
        )

    def fit_source(
        self,
        source: OfflineDatasetSource,
        n_steps: int,
        batch_size: int = 256,  # noqa: ARG002 - d3rlpy configs own batch size
    ) -> Dict[str, float]:
        dataset = source.as_mdpdataset()
        history = self.algo.fit(
            dataset,
            n_steps=int(n_steps),
            n_steps_per_epoch=max(1, int(n_steps) // 2),
            show_progress=False,
            save_interval=10**9,  # no intermediate model dumps
        )
        # history: list of (epoch, metrics-dict)
        return {k: float(v) for k, v in (history[-1][1] if history else {}).items()}


def make_cql(device: torch.device, action_type: str, **overrides) -> D3rlpyAlgorithm:
    if action_type == "discrete":
        from d3rlpy.algos import DiscreteCQLConfig

        cfg = DiscreteCQLConfig(**overrides)
    else:
        from d3rlpy.algos import CQLConfig

        cfg = CQLConfig(**overrides)
    dev = "cuda:0" if device.type == "cuda" else "cpu:0"
    return D3rlpyAlgorithm(cfg.create(device=dev), device, action_type)


def make_iql(device: torch.device, action_type: str, **overrides) -> D3rlpyAlgorithm:
    if action_type == "discrete":
        # d3rlpy has no discrete IQL; CQL is the canonical discrete variant.
        raise ValueError("IQL is continuous-only in d3rlpy; use cql for discrete.")
    from d3rlpy.algos import IQLConfig

    dev = "cuda:0" if device.type == "cuda" else "cpu:0"
    return D3rlpyAlgorithm(
        IQLConfig(**overrides).create(device=dev), device, "continuous"
    )
