from __future__ import annotations

from typing import Dict, Optional

import torch

from ...inference.base import _as_B1, BaseInferencer


class LikelihoodWeightingInferencer(BaseInferencer):
    @torch.no_grad()
    def infer_posterior(
        self,
        query: str,
        evidence: Optional[Dict[str, torch.Tensor]] = None,
        do: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ):
        ev = {k: _as_B1(v, self.device) for k, v in (evidence or {}).items()}
        do_ = {k: _as_B1(v, self.device) for k, v in (do or {}).items()}

        # caps
        need = {}
        for n in self.topo:
            wants = []
            if n not in do_:
                wants.append("has_sample")
            if n in ev:
                wants.append("has_log_prob")
            need[n] = wants
        self._check_caps(need)

        B = 1
        for d in list(ev.values()) + list(do_.values()):
            B = max(B, d.shape[0])

        N = kwargs.get("num_samples", 512)
        logw = torch.zeros(B, N, device=self.device)
        assign: Dict[str, torch.Tensor] = {}  # node -> [B,N]

        def parents_flat(node: str) -> torch.Tensor:
            ps = self.parents[node]
            if not ps:
                return torch.empty((B * N, 0), device=self.device)
            xs = [assign[p].float().unsqueeze(-1) for p in ps]  # [B,N,1]
            X = torch.cat(xs, dim=-1)  # [B,N,|Pa|]
            return X.reshape(B * N, -1)  # [B*N, |Pa|]

        for n in self.topo:
            if n in do_:
                assign[n] = do_[n].expand(B, N)  # [B,N]
                continue

            if n in ev:
                y_B1 = ev[n]  # [B,1]
                y_BN = y_B1.expand(B, N)  # [B,N]
                assign[n] = y_BN
                X_flat = parents_flat(n)  # [B*N, |Pa|]
                y_flat = y_BN.reshape(B * N, 1)  # [B*N,1]
                lp = self.nodes[n].log_prob(X_flat, y_flat)  # [B*N]
                logw = logw + lp.reshape(B, N)
            else:
                X_flat = parents_flat(n)  # [B*N, |Pa|]
                y = self.nodes[n].sample(X_flat, n=1).squeeze(0)  # [B*N,1]
                assign[n] = y.reshape(B, N)  # [B,N]

        w = torch.softmax(logw, dim=-1)  # [B,N]
        pdf = {"weights": w.detach()}
        samples = {query: assign[query].detach()}  # [B,N]
        return pdf, samples
