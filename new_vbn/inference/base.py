from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import networkx as nx
import torch


class IncompatibilityError(RuntimeError):
    pass


def _as_B1(x: torch.Tensor, device, dtype=None) -> torch.Tensor:
    """Ensure shape [B,1] (or [B] accepted) on device."""
    t = x.to(device)
    if dtype is not None:
        t = t.to(dtype)
    if t.dim() == 0:
        t = t.view(1, 1)
    elif t.dim() == 1:
        t = t.view(-1, 1)
    elif t.dim() == 2:
        if t.shape[1] != 1:
            raise RuntimeError(f"Expected shape [B,1], got {tuple(t.shape)}")
    else:
        raise RuntimeError(f"Expected <=2D tensor, got {tuple(t.shape)}")
    return t


class BaseInferencer:
    """
    Batch-Query API (no time axis):
      - evidence: Dict[str, Tensor], each [B,1] (discrete=Long, continuous=Float)
      - do      : Dict[str, Tensor], each [B,1]
      - returns:
          pdf     : dict with tensors shaped [B, N] (N=num_samples/kept/S)
          samples : {query: Tensor} with shape [B, N]
    """

    def __init__(
        self,
        dag: nx.DiGraph,
        parents: Dict[str, List[str]],
        nodes: Dict[str, Any],
        device: torch.device,
    ):
        self.dag = dag
        self.parents = parents
        self.nodes = nodes
        self.device = device
        self.topo: List[str] = list(nx.topological_sort(self.dag))

    def infer_posterior(
        self,
        query: str,
        evidence: Optional[Dict[str, torch.Tensor]] = None,
        do: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        raise NotImplementedError

    def _check_caps(self, need: Dict[str, Iterable[str]]):
        missing: List[str] = []
        for n, reqs in need.items():
            caps = getattr(self.nodes[n], "capabilities", None)
            for r in reqs:
                if not caps or not getattr(caps, r, False):
                    missing.append(f"{n}.{r}")
        if missing:
            raise IncompatibilityError(
                "Node capability check failed:\n"
                + "\n".join(f" - missing {m}" for m in missing)
            )


# three abstract “families” you asked for (for clarity / future typing)
class ExactInferencerBase(BaseInferencer):
    pass


class MonteCarloInferencerBase(BaseInferencer):
    pass


class VariationalInferencerBase(BaseInferencer):
    pass
