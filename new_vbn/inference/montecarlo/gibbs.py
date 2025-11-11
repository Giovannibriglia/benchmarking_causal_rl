from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from ...inference.base import _as_B1, BaseInferencer, IncompatibilityError

"""
Discrete Gibbs sampling via Markov-blanket updates.
Assumptions:
  - Discrete nodes with num_classes >= 2 (categorical state space)
  - CPDs expose log_prob(X,y)
Evidence and do are clamped (never resampled).
"""


def _num_classes(node) -> int:
    cfg = getattr(node, "cfg", {})
    return int(cfg.get("num_classes", cfg.get("out_dim", 1)))


class GibbsSamplingInferencer(BaseInferencer):
    @torch.no_grad()
    def infer_posterior(
        self,
        query: str,
        evidence: Optional[Dict[str, torch.Tensor]] = None,
        do: Optional[Dict[str, torch.Tensor]] = None,
        num_samples: int = 4096,
        burn_in: int = 256,
        thinning: int = 1,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:

        ev = {
            k: _as_B1(v, self.device, torch.long) for k, v in (evidence or {}).items()
        }
        do_ = {k: _as_B1(v, self.device, torch.long) for k, v in (do or {}).items()}

        # Capabilities: need log_prob (for local conditionals)
        self._check_caps({n: ("has_log_prob",) for n in self.topo})

        # Check discrete-only
        K = {n: _num_classes(self.nodes[n]) for n in self.topo}
        if any(k < 2 for n, k in K.items()):
            raise IncompatibilityError(
                "gibbs: only discrete nodes are supported (num_classes>=2)."
            )

        # Batch
        B = 1
        for t in list(ev.values()) + list(do_.values()):
            B = max(B, t.shape[0])

        # Children map for Markov blanket
        children = {n: [] for n in self.topo}
        for u, v in self.dag.edges():
            children[u].append(v)

        # State init
        state: Dict[str, torch.Tensor] = {}
        for n in self.topo:
            if n in ev:
                state[n] = ev[n].clone().squeeze(-1)  # [B]
            elif n in do_:
                state[n] = do_[n].clone().squeeze(-1)
            else:
                # random init uniform
                state[n] = torch.randint(0, K[n], (B,), device=self.device)

        def parents_X(node: str, st: Dict[str, torch.Tensor]) -> torch.Tensor:
            ps = self.parents[node]
            if not ps:
                return torch.empty((B, 0), device=self.device)
            xs = [st[p].float().unsqueeze(-1) for p in ps]  # [B,1]
            X = torch.cat(xs, dim=-1)  # [B, |Pa|]
            return X

        # sweep order
        sweep_vars = [n for n in self.topo if n not in ev and n not in do_]
        if len(sweep_vars) == 0:
            # trivial case
            q = state[query].unsqueeze(-1).expand(B, num_samples)
            return {
                "weights": torch.ones(B, num_samples, device=self.device) / num_samples
            }, {query: q}

        kept = []
        total_iters = burn_in + num_samples * thinning
        for it in range(total_iters):
            for n in sweep_vars:
                # compute log p(n=v | blanket)
                k = K[n]
                logp = torch.zeros(B, k, device=self.device)
                for v_idx in range(k):
                    # temporarily set n to v_idx
                    old = state[n].clone()
                    state[n] = torch.full_like(old, v_idx)

                    # local: p(n | parents)
                    Xn = parents_X(n, state)  # [B, |Pa|]
                    yn = state[n].unsqueeze(-1)  # [B,1]
                    lp = self.nodes[n].log_prob(Xn, yn)  # [B]

                    # children factors ∏ p(child | parents)
                    for c in children[n]:
                        Xc = parents_X(c, state)  # parents from current state
                        yc = state[c].unsqueeze(-1)  # [B,1]
                        lp = lp + self.nodes[c].log_prob(Xc, yc)  # [B]

                    logp[:, v_idx] = lp
                    state[n] = old

                probs = torch.softmax(logp, dim=-1)  # [B,k]
                new_val = torch.distributions.Categorical(probs).sample()  # [B]
                state[n] = new_val

            if it >= burn_in and ((it - burn_in) % thinning == 0):
                kept.append(state[query].clone())  # [B]

        if len(kept) == 0:
            kept_q = state[query].unsqueeze(-1).expand(B, num_samples)  # fallback
        else:
            kept_q = torch.stack(kept, dim=-1)  # [B, N]
        pdf = {
            "weights": torch.ones(B, kept_q.shape[1], device=self.device)
            / kept_q.shape[1]
        }
        samples = {query: kept_q.detach()}
        return pdf, samples
