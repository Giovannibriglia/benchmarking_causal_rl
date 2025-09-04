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
        Returns:
            out: dict mapping each q in query ->
                 - continuous: {"mean": [B?], "var": [B?]}
                 - discrete:   {"probs": [B?, card]}
            If return_samples (or self.return_samples) is True:
                 (out, samples, weights)
                 where:
                   samples[q]: [B?, S] (float for continuous, long for discrete)
                   weights:    [B?, S] (unnormalized importance weights)
        """

        num_samples = kwargs.get("num_samples", 512)

        do = do or {}
        cont_nodes = [n for n in self.meta.order if self.meta.types[n] == "continuous"]
        disc_nodes = [n for n in self.meta.order if self.meta.types[n] == "discrete"]

        # evidence, excluding vars under do (intervention overrides evidence)
        e_disc = {
            k: U.to_long(v).to(self.device)
            for k, v in evidence.items()
            if k in disc_nodes and k not in do
        }
        e_cont = {
            k: v.to(device=self.device, dtype=self.dtype)
            for k, v in evidence.items()
            if k in cont_nodes and k not in do
        }

        # --- robust batch-size detection ---
        def _first_dim(x: torch.Tensor) -> Optional[int]:
            return x.shape[0] if x.ndim >= 1 else None

        cands = []
        cands += [_first_dim(v) for v in e_disc.values()]
        cands += [_first_dim(v) for v in e_cont.values()]
        cands += [_first_dim(v) for v in do.values()]

        B_list = [b for b in cands if b is not None]
        B = B_list[0] if B_list else None

        if B is not None:
            assert all(b == B for b in B_list), f"Inconsistent batch sizes: {B_list}"

        for k, v in do.items():
            if v.ndim == 1:
                B = v.shape[0]

        samples_disc: Dict[str, torch.Tensor] = {}
        samples_cont: Dict[str, torch.Tensor] = {}
        weights = torch.ones(
            (1 if B is None else B, num_samples), device=self.device, dtype=self.dtype
        )

        use_mlp = lp.cont_mlps is not None and lp.cont_mlp_meta is not None
        have_lg = lp.lg is not None

        def cont_cpd(
            child: str, parent_assign: Dict[str, torch.Tensor]
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            if use_mlp:
                info = lp.cont_mlp_meta[child]
                disc_pa = info["disc_pa"]
                cont_pa = info["cont_pa"]
                xs = []
                if disc_pa:
                    ohs = []
                    for p, k in zip(disc_pa, info["disc_cards"]):
                        xp = parent_assign[p]  # [B?, S]
                        ohs.append(
                            F.one_hot(xp.to(torch.long), num_classes=k).to(self.dtype)
                        )
                    xs.append(torch.cat(ohs, dim=-1))
                if cont_pa:
                    xc = torch.stack([parent_assign[p] for p in cont_pa], dim=-1)
                    xs.append(xc)
                X = (
                    torch.cat(xs, dim=-1)
                    if xs
                    else torch.zeros(
                        (1 if B is None else B, num_samples, 0),
                        device=self.device,
                        dtype=self.dtype,
                    )
                )
                flat = X.reshape(-1, X.shape[-1])
                mean, logvar = lp.cont_mlps[child](flat)
                mean = mean.reshape((1 if B is None else B), num_samples)
                var = torch.exp(logvar).reshape((1 if B is None else B), num_samples)
                return mean, var
            elif have_lg:
                i = lp.lg.name2idx[child]
                par_idx = (
                    torch.nonzero(lp.lg.W[i] != 0.0, as_tuple=False).flatten().tolist()
                )
                if len(par_idx) == 0:
                    mean = lp.lg.b[i].expand((1 if B is None else B), num_samples)
                else:
                    par_names = [lp.lg.order[j] for j in par_idx]
                    xc = torch.stack([parent_assign[p] for p in par_names], dim=-1)
                    w = lp.lg.W[i, par_idx].to(self.device).reshape(1, 1, -1)
                    mean = (xc * w).sum(dim=-1) + lp.lg.b[i]
                var = lp.lg.sigma2[i].expand_as(mean)
                return mean, var
            else:
                raise RuntimeError(
                    "No continuous CPD available (need cont_mlps or lg)."
                )

        # Helper for discrete probs (tables or MLPs)
        def disc_probs(node: str) -> torch.Tensor:
            card = int(self.meta.cards[node])
            pa = self.meta.parents[node]
            if lp.discrete_tables is not None and node in lp.discrete_tables:
                tbl = lp.discrete_tables[node]
                if not tbl.parent_names:
                    return tbl.probs[0].expand(
                        (1 if B is None else B), num_samples, card
                    )
                strides = tbl.strides
                pa_idx = sum(
                    samples_disc[p].to(torch.long) * strides[j]
                    for j, p in enumerate(tbl.parent_names)
                )
                return tbl.probs.index_select(0, pa_idx.view(-1)).reshape(
                    (1 if B is None else B), num_samples, card
                )
            if lp.discrete_mlps is not None and node in lp.discrete_mlps:
                onehots = []
                in_dim = 0
                for p in pa:
                    k = int(self.meta.cards[p])
                    onehots.append(
                        F.one_hot(samples_disc[p].to(torch.long), num_classes=k).to(
                            self.dtype
                        )
                    )
                    in_dim += k
                X = (
                    torch.cat(onehots, dim=-1)
                    if onehots
                    else torch.zeros(
                        (1 if B is None else B),
                        num_samples,
                        0,
                        device=self.device,
                        dtype=self.dtype,
                    )
                )
                flat = X.reshape(-1, in_dim)
                logits = lp.discrete_mlps[node](flat)
                probs = torch.softmax(logits, dim=-1).reshape(
                    (1 if B is None else B), num_samples, card
                )
                return probs
            raise RuntimeError(f"No CPD for discrete node '{node}'.")

        # ancestral pass
        for node in self.meta.order:
            t = self.meta.types[node]
            if t == "discrete":
                if node in do:
                    val = U.to_long(do[node]).to(self.device)
                    x = (
                        val.view((-1, 1))
                        if (B is not None or val.ndim == 1)
                        else val.view(1, 1)
                    )
                    samples_disc[node] = x.expand(
                        -1 if B is not None else 1, num_samples
                    )
                    continue
                if node in e_disc:
                    x = (
                        e_disc[node].view((-1, 1))
                        if B is not None
                        else e_disc[node].view(1, 1)
                    )
                    samples_disc[node] = x.expand(
                        -1 if B is not None else 1, num_samples
                    )
                    probs = disc_probs(node)
                    onehot = F.one_hot(
                        samples_disc[node], num_classes=int(self.meta.cards[node])
                    )
                    weights = weights * (probs * onehot).sum(dim=-1).clamp_min(1e-32)
                else:
                    probs = disc_probs(node)
                    u = torch.rand(
                        (1 if B is None else B), num_samples, device=self.device
                    )
                    cdf = probs.cumsum(dim=-1)
                    samples_disc[node] = (u.unsqueeze(-1) > cdf).sum(dim=-1)
            else:
                # build parent assignment
                pa_assign = {}
                for p in self.meta.parents[node]:
                    if self.meta.types[p] == "discrete":
                        pa_assign[p] = samples_disc[p].to(self.dtype)
                    else:
                        pa_assign[p] = samples_cont[p]
                if node in do:
                    xv = do[node].to(device=self.device, dtype=self.dtype)
                    xv = (
                        xv.view((-1, 1))
                        if (B is not None or xv.ndim == 1)
                        else xv.view(1, 1)
                    )
                    samples_cont[node] = xv.expand(
                        -1 if B is not None else 1, num_samples
                    )
                elif node in e_cont:
                    xv = (
                        e_cont[node].view((-1, 1))
                        if B is not None
                        else e_cont[node].view(1, 1)
                    )
                    xv = xv.expand(-1 if B is not None else 1, num_samples)
                    mean, var = cont_cpd(node, pa_assign)
                    weights = weights * U.normal_pdf(xv, mean, var).clamp_min(1e-38)
                    samples_cont[node] = xv
                else:
                    mean, var = cont_cpd(node, pa_assign)
                    eps = torch.randn_like(mean)
                    samples_cont[node] = mean + torch.sqrt(var.clamp_min(1e-12)) * eps

        # outputs for queries (continuous: moments; discrete: hist)
        out: Dict[str, dict] = {}
        for q in query:
            if self.meta.types[q] == "continuous":
                xq = samples_cont[q]
                if B is None:
                    w = weights.view(1, -1)
                    norm = w.sum(dim=1, keepdim=True).clamp_min(1e-32)
                    mean = (w @ xq.view(1, -1).t()).squeeze() / norm.squeeze()
                    var = (
                        w @ ((xq - mean) ** 2).view(1, -1).t()
                    ).squeeze() / norm.squeeze()
                    out[q] = {"mean": mean.reshape(1), "var": var.reshape(1)}
                else:
                    norm = weights.sum(dim=1, keepdim=True).clamp_min(1e-32)
                    mean = (weights * xq).sum(dim=1, keepdim=True) / norm
                    var = (weights * (xq - mean) ** 2).sum(dim=1, keepdim=True) / norm
                    out[q] = {"mean": mean.squeeze(1), "var": var.squeeze(1)}
            else:
                card = int(self.meta.cards[q])
                xq = samples_disc[q]
                if B is None:
                    flat = xq.view(-1)
                    w = weights.view(-1)
                    hist = torch.zeros(card, device=self.device).scatter_add_(
                        0, flat, w
                    )
                    out[q] = {
                        "probs": (hist / hist.sum().clamp_min(1e-32)).unsqueeze(0)
                    }
                else:
                    hist = torch.zeros((B, card), device=self.device)
                    for c in range(card):
                        hist[:, c] = (weights * (xq == c)).sum(dim=1)
                    out[q] = {
                        "probs": hist / hist.sum(dim=1, keepdim=True).clamp_min(1e-32)
                    }

        # ---- Optional: return particles + weights for queried vars ----
        if not return_samples:
            return out

        ret_samples: Dict[str, torch.Tensor] = {}
        for q in query:
            if self.meta.types[q] == "continuous":
                ret_samples[q] = samples_cont[q]  # [B?, S] float
            else:
                ret_samples[q] = samples_disc[q]  # [B?, S] long

        return out, ret_samples, weights
