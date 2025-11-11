from __future__ import annotations

from typing import Dict, List, Optional

import torch

from ...inference.base import _as_B1, BaseInferencer, IncompatibilityError


def _num_classes(node) -> int:
    cfg = getattr(node, "cfg", {})
    n = int(cfg.get("num_classes", cfg.get("out_dim", 1)))
    if n < 2:
        raise IncompatibilityError(
            f"exact.ve needs discrete num_classes>=2 for node '{node.name}'"
        )
    return n


class VariableEliminationInferencer(BaseInferencer):
    @torch.no_grad()
    def infer_posterior(
        self,
        query: str,
        evidence: Optional[Dict[str, torch.Tensor]] = None,
        do: Optional[Dict[str, torch.Tensor]] = None,
        max_states_total: int = 10240,
        **kwargs,
    ):
        ev = {
            k: _as_B1(v, self.device, torch.long) for k, v in (evidence or {}).items()
        }
        do_ = {k: _as_B1(v, self.device, torch.long) for k, v in (do or {}).items()}

        # caps
        need = {n: ("has_log_prob",) for n in self.topo if n not in do_}
        self._check_caps(need)

        cards = {n: _num_classes(self.nodes[n]) for n in self.topo}
        B = 1
        for d in list(ev.values()) + list(do_.values()):
            B = max(B, d.shape[0])

        W_list: List[torch.Tensor] = []  # [S] per b
        Q_list: List[torch.Tensor] = []  # [S] per b (Long)

        for b in range(B):
            ev_b = {k: v[b : b + 1] for k, v in ev.items()}
            do_b = {k: v[b : b + 1] for k, v in do_.items()}
            free = [n for n in self.topo if n not in ev_b and n not in do_b]
            total = 1
            for n in free:
                total *= cards[n]
                if total > max_states_total:
                    raise IncompatibilityError("exact.ve: state space too large.")

            weights, qvals = [], []
            for flat in range(total):
                assign = {}
                tf = flat
                for n in free:
                    k = cards[n]
                    val = tf % k
                    tf //= k
                    assign[n] = torch.tensor(
                        [[val]], device=self.device, dtype=torch.long
                    )
                assign.update(ev_b)
                assign.update(do_b)

                logp = 0.0
                for n in self.topo:
                    if n in do_b:
                        continue
                    ps = self.parents[n]
                    if ps:
                        X = torch.stack(
                            [assign[p].float().view(1) for p in ps], dim=-1
                        )  # [1, |Pa|]
                    else:
                        X = torch.empty((1, 0), device=self.device)
                    y = assign[n]  # [1,1]
                    logp = logp + self.nodes[n].log_prob(X, y).view(())

                w = torch.exp(torch.as_tensor(logp, device=self.device))
                weights.append(w)
                qvals.append(assign[query])

            W_b = torch.stack(weights)  # [S]
            Q_b = torch.cat(qvals, dim=0).reshape(-1)  # [S]
            W_b = W_b / (W_b.sum() + 1e-12)
            W_list.append(W_b)
            Q_list.append(Q_b)

        # S = W_list[0].shape[0] if W_list else 0
        W = torch.stack(W_list, dim=0)  # [B,S]
        Q = torch.stack(Q_list, dim=0)  # [B,S]

        pdf = {"weights": W.detach()}
        samples = {query: Q.detach()}
        return pdf, samples
