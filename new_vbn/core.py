from __future__ import annotations

import random

from typing import Any, Callable, Dict, List, Optional

import networkx as nx
import numpy as np
import pandas as pd
import torch

from .inference import INFERENCE_METHODS
from .learning import LEARNING_METHODS
from .learning.base import BaseNodeCPD


def _parents_map_from_dag(dag: nx.DiGraph) -> Dict[str, List[str]]:
    return {n: list(dag.predecessors(n)) for n in dag.nodes()}


class VBN:
    def __init__(self, dag: nx.DiGraph, device=None, seed: Optional[int] = None):
        self.device = (
            torch.device(device)
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dag = dag
        if seed is not None:
            self.set_seed(seed)
        else:
            self.seed = None

        self._parents = _parents_map_from_dag(self.dag)
        self._nodes: Dict[str, BaseNodeCPD] = {}

        self._learning_factory: Optional[Callable[[str, int, int], BaseNodeCPD]] = None
        self._learning_kwargs: Dict[str, Any] = {}

        self._inference_factory = None
        self._inference_kwargs: Dict[str, Any] = {}
        self._inference = None

    def set_seed(self, seed: int):
        self.seed = seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def set_learning_method(self, method, **kwargs):
        if isinstance(method, str):
            key = method.lower().strip()
            if key not in LEARNING_METHODS:
                raise ValueError(
                    f"Unknown learning method '{method}'. "
                    f"Available: {list(LEARNING_METHODS.keys())}"
                )
            self._learning_factory = LEARNING_METHODS[key]
        elif callable(method):
            self._learning_factory = method
        else:
            raise TypeError("method must be a string or a callable factory")
        self._learning_kwargs = kwargs

    def _prepare_data(
        self, data: Dict[str, torch.Tensor] | pd.DataFrame
    ) -> Dict[str, torch.Tensor]:
        if isinstance(data, pd.DataFrame):
            tensor_data = {
                c: torch.tensor(
                    data[c].values, dtype=torch.float32, device=self.device
                ).unsqueeze(-1)
                for c in data.columns
            }
        elif isinstance(data, dict):
            tensor_data = {}
            for k, v in data.items():
                t = v.to(self.device).float()
                if t.ndim == 1:
                    t = t.unsqueeze(-1)
                tensor_data[k] = t
        else:
            raise TypeError("data must be a pandas DataFrame or dict[str, Tensor]")

        missing = [n for n in self.dag.nodes() if n not in tensor_data]
        if missing:
            raise ValueError(f"Missing data for DAG nodes: {missing}")
        return tensor_data

    def _concat_parents(
        self, tensor_data: Dict[str, torch.Tensor], node: str
    ) -> torch.Tensor:
        ps = self._parents[node]
        if len(ps) == 0:
            return torch.empty((tensor_data[node].shape[0], 0), device=self.device)
        xs = [tensor_data[p].squeeze(-1) for p in ps]  # each [N]
        return torch.stack(xs, dim=-1)  # [N, |Pa|]

    # ---------- SEQUENTIAL FIT ----------
    def fit(self, data: Dict[str, torch.Tensor] | pd.DataFrame, **kwargs) -> None:
        if self._learning_factory is None:
            raise RuntimeError("Call set_learning_method(...) before fit().")

        tensor_data = self._prepare_data(data)
        topo = list(
            nx.topological_sort(self.dag)
        )  # not strictly necessary for these CPDs

        self._nodes = {}
        for n in topo:
            X = self._concat_parents(tensor_data, n)
            y = tensor_data[n]
            in_dim = X.shape[1]
            out_dim = y.shape[1]
            head = self._learning_factory(
                name=n, in_dim=in_dim, out_dim=out_dim, **self._learning_kwargs
            )
            head = head.to(self.device)
            head.fit(X, y)
            self._nodes[n] = head

    # ---------- SEQUENTIAL UPDATE ----------
    def update(self, data: Dict[str, torch.Tensor] | pd.DataFrame, **kwargs) -> None:
        if not self._nodes:
            raise RuntimeError("Call fit(...) before update(...).")
        tensor_data = self._prepare_data(data)
        for n, head in self._nodes.items():
            X = self._concat_parents(tensor_data, n)
            y = tensor_data[n]
            head.update(X, y)

    def set_inference_method(self, method, **kwargs):
        if isinstance(method, str):
            key = method.lower().strip()
            if key not in INFERENCE_METHODS:
                raise ValueError(
                    f"Unknown inference method '{method}'. Available: {list(INFERENCE_METHODS.keys())}"
                )
            self._inference_factory = INFERENCE_METHODS[key]
        elif callable(method):
            self._inference_factory = method
        else:
            raise TypeError("method must be a string or a callable factory")
        self._inference_kwargs = kwargs  # defaults (e.g., num_samples)
        self._inferencer = self._inference_factory(
            dag=self.dag,
            parents=self._parents,
            nodes=self._nodes,
            device=self.device,
        )

    def infer_posterior(
        self,
        query: str,
        evidence: Optional[Dict[str, torch.Tensor]] = None,
        do: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ):
        if self._inference_factory is None:
            raise RuntimeError(
                "Call set_inference_method(...) before infer_posterior()."
            )
        call_kwargs = {**getattr(self, "_inference_kwargs", {}), **kwargs}
        return self._inferencer.infer_posterior(
            query, evidence=evidence, do=do, **call_kwargs
        )

    @torch.no_grad()
    def counterfactual(
        self,
        query: str,
        evidence: Optional[Dict[str, torch.Tensor]] = None,
        do: Optional[Dict[str, torch.Tensor]] = None,
        *,
        base_infer: str = "montecarlo.lw",
        num_samples: int = 2048,
        **infer_kwargs,
    ):
        """
        Abduction–Action–Prediction counterfactual:
          1) Abduction: sample a posterior world consistent with evidence
          2) Action: set do(·) interventions (override mechanisms/values)
          3) Prediction: simulate forward to get query under the counterfactual action

        Returns:
          pdf:     {"weights": [B, N]}
          samples: {query:    [B, N]}
        """
        device = self.device
        ev = {k: v.to(device) for k, v in (evidence or {}).items()}
        do_ = {k: v.to(device) for k, v in (do or {}).items()}
        B = max([v.shape[0] for v in list(ev.values()) + list(do_.values())] + [1])

        # 1) Abduction: posterior samples of all nodes given E (using chosen inferencer)
        self.set_inference_method(base_infer, num_samples=num_samples, **infer_kwargs)
        # sample each non-evidence node once to get a consistent ancestral sample set
        posterior_samples: Dict[str, torch.Tensor] = {}
        for n in self.dag.nodes():
            if n in ev:
                posterior_samples[n] = ev[n].expand(B, num_samples)  # [B,N]
                continue
            pdf_n, s_n = self._inferencer.infer_posterior(
                n, evidence=ev, do=None, num_samples=num_samples
            )
            posterior_samples[n] = s_n[n]  # [B,N]

        # 2) Action: override do-nodes with fixed values across N
        for n, v in do_.items():
            posterior_samples[n] = v.expand(B, 1).repeat(1, num_samples)  # [B,N]

        # 3) Prediction: forward-sample query with mechanisms under 'do'
        #    Use CPDs but with parents taken from posterior_samples (abduced noises/parents).
        #    If query is intervened, just return its clamped value.
        if query in do_:
            q = do_[query].expand(B, 1).repeat(1, num_samples)
            pdf = {"weights": torch.ones(B, num_samples, device=device) / num_samples}
            return pdf, {query: q}

        # Otherwise sample using CPD of 'query' with parent assignments from posterior_samples
        ps = list(self.dag.predecessors(query))
        if len(ps) == 0:
            X = torch.zeros(B * num_samples, 0, device=device)
        else:
            X_b_n_p = torch.stack(
                [posterior_samples[p] for p in ps], dim=-1
            )  # [B,N,|Pa|]
            X = X_b_n_p.reshape(B * num_samples, -1).float()

        if hasattr(self._nodes[query], "sample"):
            y = (
                self._nodes[query].sample(X, n=1).squeeze(0).view(B, num_samples)
            )  # [B,N]
        else:
            # fallback: argmax from log_prob grid (rare)
            raise RuntimeError(
                f"Node '{query}' lacks sample(); cannot predict counterfactual."
            )

        pdf = {"weights": torch.ones(B, num_samples, device=device) / num_samples}
        return pdf, {query: y}
