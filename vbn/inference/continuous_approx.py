from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .. import utils as U
from ..core import BNMeta, LearnParams

from .base import BaseInference


class ContinuousApproxInference(BaseInference):
    def __init__(self, meta: BNMeta, device=None, dtype=torch.float32, **kwargs):
        super().__init__(meta, device, dtype)
        self.meta = meta
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dtype = dtype

    @torch.no_grad()
    def posterior(
        self,
        lp: LearnParams,
        evidence: Dict[str, torch.Tensor],
        query: List[str],
        do: Optional[Dict[str, torch.Tensor]] = None,
        return_samples: bool = False,
        **kwargs,
    ):
        """
        Hybrid/Hierarchical Self-Normalized Importance (HSI) posterior.

        Returns:
            out: dict mapping q ->
                 - continuous: {"mean": [B], "var": [B]}
                 - discrete:   {"probs": [B, card]}
            If return_samples is True:
                 (out, samples, weights)
                 where:
                   samples[q]: [B, S] (float for continuous, long for discrete)
                   weights:    [B, S] (unnormalized importance weights)

        Notes:
          • Safeguarded against D=0 (no parents) -> uses empty [B*S, 0] feature tensors (no reshape(-1, 0)).
          • If proposal networks are available in lp.* (see below), we sample from q and weight by p/q.
            Otherwise we sample from the model CPDs p and weight only by evidence-likelihoods.
        """

        device, dtype = self.device, self.dtype
        num_samples = int(kwargs.get("num_samples", 512) or 512)
        do = do or {}

        cont_nodes = [n for n in self.meta.order if self.meta.types[n] == "continuous"]
        disc_nodes = [n for n in self.meta.order if self.meta.types[n] == "discrete"]

        # Evidence excluding vars under intervention
        e_disc = {
            k: U.to_long(v).to(device)
            for k, v in evidence.items()
            if k in disc_nodes and k not in do
        }
        e_cont = {
            k: v.to(device=device, dtype=dtype)
            for k, v in evidence.items()
            if k in cont_nodes and k not in do
        }

        # ---- batch size detection
        def _first_dim(x: torch.Tensor):
            return x.shape[0] if x.ndim >= 1 else None

        sizes = []
        sizes += [_first_dim(v) for v in e_disc.values()]
        sizes += [_first_dim(v) for v in e_cont.values()]
        sizes += [_first_dim(v) for v in do.values()]
        sizes = [s for s in sizes if s is not None]
        B = sizes[0] if sizes else None
        if B is not None:
            assert all(b == B for b in sizes), f"Inconsistent batch sizes: {sizes}"
        for v in do.values():
            if isinstance(v, torch.Tensor) and v.ndim == 1:
                B = v.shape[0]
        B_ = 1 if B is None else B

        # Storage
        samples_disc: Dict[str, torch.Tensor] = {}
        samples_cont: Dict[str, torch.Tensor] = {}
        weights = torch.ones((B_, num_samples), device=device, dtype=dtype)

        # Availability flags
        use_cont_mlp = lp.cont_mlps is not None and lp.cont_mlp_meta is not None
        use_disc_tables = lp.discrete_tables is not None
        use_disc_mlp = lp.discrete_mlps is not None

        # Optional proposal nets (if you have them; otherwise None)
        prop_cont = getattr(lp, "hsi_cont_mlps", None)
        prop_cont_meta = getattr(lp, "hsi_cont_mlp_meta", None)
        prop_disc = getattr(lp, "hsi_discrete_mlps", None)

        # ---------- model CPDs ----------
        def cont_cpd(
            child: str, pa_assign: Dict[str, torch.Tensor]
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Return (mean, var) under the model p(x|pa)."""
            if use_cont_mlp:
                info = lp.cont_mlp_meta[child]
                disc_pa, cont_pa = info["disc_pa"], info["cont_pa"]
                xs = []
                if disc_pa:
                    ohs = []
                    for p, k in zip(disc_pa, info["disc_cards"]):
                        xp = pa_assign[p]  # [B_, S]
                        ohs.append(
                            F.one_hot(xp.to(torch.long), num_classes=int(k)).to(dtype)
                        )
                    xs.append(torch.cat(ohs, dim=-1))
                if cont_pa:
                    xc = torch.stack(
                        [pa_assign[p] for p in cont_pa], dim=-1
                    )  # [B_, S, P]
                    xs.append(xc)
                if xs:
                    X = torch.cat(xs, dim=-1)  # [B_, S, D], D>0
                    D = X.shape[-1]
                    flat = X.view(B_ * num_samples, D)
                else:
                    flat = torch.empty(
                        (B_ * num_samples, 0), device=device, dtype=dtype
                    )  # D=0 safe
                mean, logvar = lp.cont_mlps[child](flat)
                return mean.view(B_, num_samples), torch.exp(logvar).view(
                    B_, num_samples
                )
            elif lp.lg is not None:
                i = lp.lg.name2idx[child]
                par_idx = (
                    torch.nonzero(lp.lg.W[i] != 0.0, as_tuple=False).flatten().tolist()
                )
                if len(par_idx) == 0:
                    mean = lp.lg.b[i].expand(B_, num_samples)
                else:
                    par_names = [lp.lg.order[j] for j in par_idx]
                    xc = torch.stack(
                        [pa_assign[p] for p in par_names], dim=-1
                    )  # [B_, S, P]
                    w = lp.lg.W[i, par_idx].to(device).reshape(1, 1, -1)
                    mean = (xc * w).sum(dim=-1) + lp.lg.b[i]
                var = lp.lg.sigma2[i].expand_as(mean)
                return mean, var
            else:
                raise RuntimeError("No continuous CPD: need cont_mlps or lg.")

        def disc_probs(node: str) -> torch.Tensor:
            """Return categorical probs under the model p(x|pa): [B_, S, card]."""
            card = int(self.meta.cards[node])
            if use_disc_tables and node in lp.discrete_tables:
                tbl = lp.discrete_tables[node]
                if not tbl.parent_names:
                    return tbl.probs[0].expand(B_, num_samples, card)
                strides = tbl.strides
                pa_idx = 0
                for j, p in enumerate(tbl.parent_names):
                    pa_idx = pa_idx + (samples_disc[p].to(torch.long) * int(strides[j]))
                return tbl.probs.index_select(0, pa_idx.view(-1)).view(
                    B_, num_samples, card
                )
            if use_disc_mlp and node in lp.discrete_mlps:
                pa = self.meta.parents[node]
                onehots, in_dim = [], 0
                for p in pa:
                    k = int(self.meta.cards[p])
                    onehots.append(
                        F.one_hot(samples_disc[p].to(torch.long), num_classes=k).to(
                            dtype
                        )
                    )
                    in_dim += k
                if onehots:
                    X = torch.cat(onehots, dim=-1)  # [B_, S, in_dim]
                    flat = X.view(B_ * num_samples, in_dim)
                else:
                    flat = torch.empty(
                        (B_ * num_samples, 0), device=device, dtype=dtype
                    )
                logits = lp.discrete_mlps[node](flat)
                return torch.softmax(logits, dim=-1).view(B_, num_samples, card)
            raise RuntimeError(f"No CPD for discrete node '{node}'.")

        # ---------- proposal (optional) ----------
        def cont_proposal(
            child: str, pa_assign: Dict[str, torch.Tensor]
        ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
            if prop_cont is None or prop_cont_meta is None or child not in prop_cont:
                return None
            info = prop_cont_meta[child]
            disc_pa, cont_pa = info["disc_pa"], info["cont_pa"]
            xs = []
            if disc_pa:
                ohs = []
                for p, k in zip(disc_pa, info["disc_cards"]):
                    xp = pa_assign[p]
                    ohs.append(
                        F.one_hot(xp.to(torch.long), num_classes=int(k)).to(dtype)
                    )
                xs.append(torch.cat(ohs, dim=-1))
            if cont_pa:
                xc = torch.stack([pa_assign[p] for p in cont_pa], dim=-1)
                xs.append(xc)
            if xs:
                X = torch.cat(xs, dim=-1)
                D = X.shape[-1]
                flat = X.view(B_ * num_samples, D)
            else:
                flat = torch.empty((B_ * num_samples, 0), device=device, dtype=dtype)
            mean, logvar = prop_cont[child](flat)
            return mean.view(B_, num_samples), torch.exp(logvar).view(B_, num_samples)

        def disc_proposal(node: str) -> Optional[torch.Tensor]:
            if prop_disc is None or node not in prop_disc:
                return None
            # Build input as in disc_probs
            pa = self.meta.parents[node]
            onehots, in_dim = [], 0
            for p in pa:
                k = int(self.meta.cards[p])
                onehots.append(
                    F.one_hot(samples_disc[p].to(torch.long), num_classes=k).to(dtype)
                )
                in_dim += k
            if onehots:
                X = torch.cat(onehots, dim=-1)
                flat = X.view(B_ * num_samples, in_dim)
            else:
                flat = torch.empty((B_ * num_samples, 0), device=device, dtype=dtype)
            logits = prop_disc[node](flat)
            # [B_, S, card]
            return torch.softmax(logits, dim=-1).view(
                B_, num_samples, int(self.meta.cards[node])
            )

        # ---------- ancestral pass with importance weights ----------
        for node in self.meta.order:
            t = self.meta.types[node]

            if t == "discrete":
                if node in do:
                    val = U.to_long(do[node]).to(device)
                    x = (
                        val.view(-1, 1)
                        if (B is not None or val.ndim == 1)
                        else val.view(1, 1)
                    )
                    samples_disc[node] = x.expand(B_, num_samples)
                    continue

                p_probs = disc_probs(node)  # model probs
                q_probs = disc_proposal(node)  # proposal probs or None

                if node in e_disc:
                    # evidence: set value and weight by p(evidence|pa). (no q correction since we didn't sample)
                    x = (
                        e_disc[node].view(-1, 1)
                        if B is not None
                        else e_disc[node].view(1, 1)
                    )
                    samples_disc[node] = x.expand(B_, num_samples)
                    onehot = F.one_hot(
                        samples_disc[node], num_classes=p_probs.shape[-1]
                    )
                    weights = weights * (p_probs * onehot).sum(dim=-1).clamp_min(1e-32)
                else:
                    if q_probs is None:
                        # sample from p
                        u = torch.rand((B_, num_samples), device=device)
                        cdf = p_probs.cumsum(dim=-1)
                        samples_disc[node] = (u.unsqueeze(-1) > cdf).sum(dim=-1)
                        # no p/q correction
                    else:
                        # sample from q and reweight by p/q
                        u = torch.rand((B_, num_samples), device=device)
                        cdf = q_probs.cumsum(dim=-1)
                        x = (u.unsqueeze(-1) > cdf).sum(dim=-1)  # [B_, S]
                        samples_disc[node] = x
                        onehot = F.one_hot(x, num_classes=q_probs.shape[-1])
                        num = (p_probs * onehot).sum(dim=-1).clamp_min(1e-38)
                        den = (q_probs * onehot).sum(dim=-1).clamp_min(1e-38)
                        weights = weights * (num / den)

            else:  # continuous
                # parent assignment (use sampled values so far)
                pa_assign = {}
                for p in self.meta.parents[node]:
                    if self.meta.types[p] == "discrete":
                        pa_assign[p] = samples_disc[p].to(dtype)  # [B_, S]
                    else:
                        pa_assign[p] = samples_cont[p]  # [B_, S]

                if node in do:
                    xv = do[node].to(device=device, dtype=dtype)
                    xv = (
                        xv.view(-1, 1)
                        if (B is not None or xv.ndim == 1)
                        else xv.view(1, 1)
                    )
                    samples_cont[node] = xv.expand(B_, num_samples)
                    continue

                # model parameters
                mu_p, var_p = cont_cpd(node, pa_assign)

                # evidence observed?
                if node in e_cont:
                    xv = (
                        e_cont[node].view(-1, 1)
                        if B is not None
                        else e_cont[node].view(1, 1)
                    )
                    xv = xv.expand(B_, num_samples)
                    # likelihood under model
                    weights = weights * U.normal_pdf(xv, mu_p, var_p).clamp_min(1e-38)
                    samples_cont[node] = xv
                else:
                    # proposal?
                    pq = cont_proposal(node, pa_assign)
                    if pq is None:
                        # sample from model p
                        eps = torch.randn_like(mu_p)
                        samples_cont[node] = (
                            mu_p + torch.sqrt(var_p.clamp_min(1e-12)) * eps
                        )
                    else:
                        mu_q, var_q = pq
                        eps = torch.randn_like(mu_q)
                        x = mu_q + torch.sqrt(var_q.clamp_min(1e-12)) * eps
                        samples_cont[node] = x
                        # importance ratio p(x)/q(x)
                        num = U.normal_pdf(x, mu_p, var_p).clamp_min(1e-38)
                        den = U.normal_pdf(x, mu_q, var_q).clamp_min(1e-38)
                        weights = weights * (num / den)

        # ---------- outputs ----------
        out: Dict[str, dict] = {}
        for q in query:
            if self.meta.types[q] == "continuous":
                xq = samples_cont[q]  # [B_, S]
                norm = weights.sum(dim=1, keepdim=True).clamp_min(1e-32)
                mean = (weights * xq).sum(dim=1, keepdim=True) / norm
                var = (weights * (xq - mean) ** 2).sum(dim=1, keepdim=True) / norm
                out[q] = {"mean": mean.squeeze(1), "var": var.squeeze(1)}
            else:
                card = int(self.meta.cards[q])
                xq = samples_disc[q]  # [B_, S]
                hist = torch.zeros((B_, card), device=device, dtype=dtype)
                for c in range(card):
                    hist[:, c] = (weights * (xq == c)).sum(dim=1)
                probs = hist / hist.sum(dim=1, keepdim=True).clamp_min(1e-32)
                out[q] = {"probs": probs}

        if not return_samples:
            return out

        ret_samples: Dict[str, torch.Tensor] = {}
        for q in query:
            ret_samples[q] = (
                samples_cont[q]
                if self.meta.types[q] == "continuous"
                else samples_disc[q]
            )

        return out, ret_samples, weights
