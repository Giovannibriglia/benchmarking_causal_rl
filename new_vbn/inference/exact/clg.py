# vbn/inference/exact/clg.py
from __future__ import annotations

from typing import Dict, List, Optional

import torch

from ...inference.base import _as_B1, BaseInferencer, IncompatibilityError

# ---- Small Gaussian potential helpers (natural form) ----
# We keep continuous blocks in standard (μ, Σ); convert on the fly when combining.


def _stack_if_any(xs: List[torch.Tensor], dim: int = 0) -> Optional[torch.Tensor]:
    return torch.stack(xs, dim=dim) if len(xs) > 0 else None


class GaussianBlock:
    """
    Represents a joint Gaussian over a set of continuous variables Z (ordered),
    parameterized by mean μ ∈ R^d and covariance Σ ∈ R^{d×d}.
    """

    __slots__ = ("names", "mu", "Sigma")

    def __init__(self, names: List[str], mu: torch.Tensor, Sigma: torch.Tensor):
        self.names = names  # list of cont var names
        self.mu = mu  # [d]
        self.Sigma = Sigma  # [d,d]

    def reorder(self, new_names: List[str]) -> "GaussianBlock":
        # Permute μ and Σ to the new order
        idx = torch.tensor(
            [new_names.index(n) for n in self.names], device=self.mu.device
        )
        mu = self.mu[idx]
        Sigma = self.Sigma.index_select(0, idx).index_select(1, idx)
        return GaussianBlock(new_names, mu, Sigma)

    def marginalize(self, keep: List[str]) -> "GaussianBlock":
        # keep subset
        idx = torch.tensor([self.names.index(n) for n in keep], device=self.mu.device)
        mu = self.mu.index_select(0, idx)
        Sigma = self.Sigma.index_select(0, idx).index_select(1, idx)
        return GaussianBlock(keep, mu, Sigma)

    def condition(self, ev: Dict[str, torch.Tensor]) -> "GaussianBlock":
        """
        Condition on a subset of variables in ev: each value is [B,1].
        Returns a per-batch block: μ_B [B,d_keep], Σ_keep [d_keep,d_keep] (same across B).
        We keep all non-evidence in order.
        """
        device = self.mu.device
        all_names = self.names
        E = [n for n in all_names if n in ev]
        U = [n for n in all_names if n not in ev]
        if len(E) == 0:
            B = next(iter(ev.values())).shape[0] if len(ev) > 0 else 1
            muB = self.mu.view(1, -1).expand(B, -1)
            return GaussianBlock(
                U,
                muB[:, : len(U)],
                self.Sigma.index_select(
                    0, torch.arange(len(U), device=device)
                ).index_select(1, torch.arange(len(U), device=device)),
            )  # not used directly; see caller
        # Index sets
        idxE = torch.tensor(
            [all_names.index(n) for n in E], device=device, dtype=torch.long
        )
        idxU = torch.tensor(
            [all_names.index(n) for n in U], device=device, dtype=torch.long
        )
        # Partitions
        mu = self.mu
        Sigma = self.Sigma
        muE = mu.index_select(0, idxE)
        muU = mu.index_select(0, idxU)
        SEE = Sigma.index_select(0, idxE).index_select(1, idxE)
        SEU = Sigma.index_select(0, idxE).index_select(1, idxU)
        SUE = SEU.transpose(0, 1)
        SUU = Sigma.index_select(0, idxU).index_select(1, idxU)
        SEE_inv = torch.linalg.pinv(SEE + 1e-8 * torch.eye(SEE.shape[0], device=device))
        # Build per-batch μ_U|E
        B = next(iter(ev.values())).shape[0]
        yE = torch.stack([ev[n].squeeze(-1) for n in E], dim=1)  # [B, |E|]
        delta = (yE - muE.view(1, -1)).unsqueeze(-1)  # [B, |E|, 1]
        gain = (SUE @ SEE_inv).view(1, *SUE.shape)  # [1, |U|, |E|]
        muU_post = muU.view(1, -1, 1) + torch.bmm(gain, delta)  # [B, |U|, 1]
        muU_post = muU_post.squeeze(-1)  # [B, |U|]
        # Posterior cov does not depend on E values
        SUU_post = SUU - SUE @ SEE_inv @ SEU  # [|U|, |U|]
        out = GaussianBlock(U, muU_post, SUU_post)  # mu is [B,|U|] here
        return out


def _merge_gaussian_blocks(
    blocks: List[GaussianBlock], names_order: List[str]
) -> GaussianBlock:
    """
    Merge independent Gaussian blocks over disjoint variable sets into a joint Gaussian
    by block-diagonal Σ and concatenated μ (broadcast μ_B if any is batched).
    """
    device = blocks[0].Sigma.device
    # detect batch μ
    has_batch = any(b.mu.dim() == 2 for b in blocks)
    if has_batch:
        B = max((b.mu.shape[0] if b.mu.dim() == 2 else 1) for b in blocks)
    else:
        B = 1
    mus = []
    Sigmas = []
    names_acc: List[str] = []
    for b in blocks:
        if b.mu.dim() == 1:
            muB = b.mu.view(1, -1).expand(B, -1)
        else:
            muB = b.mu  # [B, d_i]
            if muB.shape[0] != B:
                muB = muB.expand(B, -1)
        mus.append(muB)
        Sigmas.append(b.Sigma)
        names_acc += b.names
    mu_cat = torch.cat(mus, dim=1)  # [B, sum d_i]
    Sigma_bd = torch.block_diag(*Sigmas)  # [D, D]
    # reorder if user provided order differs
    if names_order and names_acc != names_order:
        idx = torch.tensor(
            [names_acc.index(n) for n in names_order], device=device, dtype=torch.long
        )
        mu_cat = mu_cat.index_select(1, idx)
        Sigma_bd = Sigma_bd.index_select(0, idx).index_select(1, idx)
        names_acc = names_order
    return GaussianBlock(names_acc, mu_cat, Sigma_bd)


class ExactCLGInferencer(BaseInferencer):
    """
    Exact inference for Conditional Linear Gaussian BNs.

    Returns:
      pdf:     {"weights": [B, N]}
      samples: {query:    [B, N]}
    """

    @torch.no_grad()
    def _check_clg_ok(self):
        # Discrete nodes must not have continuous children? (classical CLG)
        # We allow continuous children with linear-Gaussian CPDs; discrete CPDs must be categorical.
        # Raise if node CPD lacks capabilities needed.
        for n in self.topo:
            node = self.nodes[n]
            # Discrete node: expect has_log_prob True, sample categorical
            # Continuous linear-Gaussian node: must expose beta, sigma2 buffers
            # We do a duck-typing probe:
            if hasattr(node, "beta") and hasattr(node, "sigma2"):
                continue
            # Otherwise assume discrete CPD or general diff CPD with log_prob & sample
            if not node.capabilities.has_log_prob:
                raise IncompatibilityError(
                    f"Node '{n}' lacks log_prob required for exact.clg."
                )

    @torch.no_grad()
    def infer_posterior(
        self,
        query: str,
        evidence: Optional[Dict[str, torch.Tensor]] = None,
        do: Optional[Dict[str, torch.Tensor]] = None,
        num_samples: int = 4096,
        **kwargs,
    ):
        self._check_clg_ok()
        ev = {k: _as_B1(v, self.device) for k, v in (evidence or {}).items()}
        do_ = {k: _as_B1(v, self.device) for k, v in (do or {}).items()}
        B = max([v.shape[0] for v in list(ev.values()) + list(do_.values())] + [1])

        # Split nodes
        disc = [
            n
            for n in self.topo
            if not (hasattr(self.nodes[n], "beta") and hasattr(self.nodes[n], "sigma2"))
        ]
        cont = [n for n in self.topo if n not in disc]

        # Strategy:
        # 1) Sum out continuous variables in closed form → produce a *discrete* factor over disc vars
        #    that contains log-likelihood contributions from continuous evidence.
        # 2) Exact VE over discrete graph with that augmented likelihood → posterior over discrete parents.
        # 3) For continuous query: condition Gaussian on ev + sampled disc config → draw samples.

        # ---- 1) Build continuous joint conditional on discrete assignment
        # For tractability we evaluate the continuous evidence likelihood per *local* CLG and multiply.
        # For general graphs, we assemble a joint Gaussian over all continuous nodes (linear-Gaussian),
        # then condition on continuous evidence and integrate residuals → contributes a scalar log-lik per B.
        # We adopt the joint approach.

        # Build joint Gaussian block (μ, Σ) for all continuous nodes given discrete selection (through betas)
        # We cannot enumerate disc assignments here; instead, we treat betas as functions of disc parents.
        # For exact VE we carry a factor φ_disc that, for each disc configuration, stores:
        #   • logZ_cont(config; evidence_cont, do_cont)  (continuous integrated likelihood)
        #
        # Implementation note: to keep code compact, we enumerate discrete parents *locally* when needed.
        # If disc cardinalities are large, this will be expensive—as expected for exact CLG.

        # Collect discrete domains (assume softmax heads expose num_classes)
        dom: Dict[str, int] = {}
        for n in disc:
            head = self.nodes[n]
            K = getattr(head, "num_classes", None) or getattr(head, "cfg", {}).get(
                "num_classes", None
            )
            if K is None:
                # attempt from linear layer out_features
                K = getattr(getattr(head, "linear", None), "out_features", None)
            if K is None:
                raise IncompatibilityError(f"Discrete node '{n}' missing num_classes.")
            dom[n] = int(K)

        # Build a discrete factor (table) φ(evidence) over all disc nodes that accumulates:
        #  φ_disc(config) ∝ ∏_disc p(d | Pa_d=config) × NLL_cont(config; ev_cont, do)
        #
        # For generality we’ll do VE directly with mixed factors:
        #   - Discrete factors: tables over subsets of disc vars
        #   - Continuous joint is handled by a single callback that, given a partial assignment to disc
        #     that covers all disc parents of any continuous node, yields the (μ, Σ) and evidence log-lik.
        #
        # To keep this implementable, we adopt *full enumeration over all disc variables* here.
        # (Users with big disc treewidth should use MC/VI.)
        #
        configs = []
        weights_B = []
        # Enumerate discrete assignments as Cartesian product
        grids = [torch.arange(dom[n], device=self.device) for n in disc]
        mesh = (
            torch.cartesian_prod(*grids)
            if len(grids) > 0
            else torch.empty(0, 0, device=self.device, dtype=torch.long)
        )
        if mesh.numel() == 0:
            mesh = mesh.view(1, 0)  # single empty assignment

        for row in mesh:  # each row is one assignment over disc in topo order 'disc'
            assign_d = {
                n: row[i].view(1, 1).expand(B, 1).float() for i, n in enumerate(disc)
            }
            # Compute discrete prior log-prob under CPDs
            logp_disc = 0.0
            for n in disc:
                if n in do_:
                    # do on discrete: clamp, zero out if mismatch
                    v = do_[n]  # [B,1]
                    ok = (assign_d[n] == v).squeeze(-1)  # [B,]
                    # if mismatch -> -inf weight; we handle by adding very negative logp
                    logp_disc = (
                        logp_disc
                        + torch.where(
                            ok,
                            torch.zeros_like(
                                ok, dtype=torch.float32, device=self.device
                            ),
                            torch.full_like(
                                ok, -1e9, dtype=torch.float32, device=self.device
                            ),
                        ).float()
                    )
                    continue
                # parents (discrete only in CLG)
                ps = [p for p in self.parents[n] if p in disc]
                if len(ps) == 0:
                    X = torch.zeros(B, 0, device=self.device)
                else:
                    X = torch.stack([assign_d[p].squeeze(-1) for p in ps], dim=-1)
                y = assign_d[n]
                logp_disc = logp_disc + self.nodes[n].log_prob(
                    X.float(), y.long()
                )  # [B]

            # Assemble joint Gaussian (μ, Σ) over all continuous nodes under this disc config + do
            names_c = []
            # mu_c_terms = []
            # Sigma_blocks = []
            name_to_idx = {}
            # We do a single pass building μ and Σ via topological order
            for n in cont:
                head = self.nodes[n]
                names_c.append(n)
                name_to_idx[n] = len(name_to_idx)
            d = len(names_c)
            mu = torch.zeros(d, device=self.device)
            Sigma = torch.zeros(d, d, device=self.device)

            # Fill using linear-Gaussian equations with discrete-conditional betas
            for n in cont:
                i = name_to_idx[n]
                node = self.nodes[n]
                # pick beta, sigma2 for current disc config by "pretending" X_d are given:
                # ps_d = [p for p in self.parents[n] if p in disc]
                ps_c = [p for p in self.parents[n] if p in cont]
                # For beta: call node.beta; we assume node.beta is already fitted for single regime; for CLG
                # we approximate by using the learned beta ignoring discrete switching OR expose a callback in node.
                # Minimal viable: assume linear-Gaussian CPDs do not switch with disc parents (common in many CLGs).
                beta = node.beta.squeeze(-1)  # [1 + |Pa|]
                sigma2 = node.sigma2.squeeze()  # scalar

                # Mean contribution
                if len(ps_c) == 0:
                    mu[i] = beta[0]
                    Sigma[i, i] = sigma2
                else:
                    J = torch.tensor(
                        [name_to_idx[p] for p in ps_c],
                        device=self.device,
                        dtype=torch.long,
                    )
                    mu[i] = (
                        beta[0]
                        + (mu[J] * beta[1 : 1 + len(ps_c)].to(self.device)).sum()
                    )
                    Sigma_pp = Sigma.index_select(0, J).index_select(1, J)
                    var_y = (
                        beta[1 : 1 + len(ps_c)].to(self.device).unsqueeze(0)
                        @ Sigma_pp
                        @ beta[1 : 1 + len(ps_c)].to(self.device).unsqueeze(-1)
                    ).squeeze()
                    Sigma[i, i] = var_y + sigma2
                    cov_y_p = Sigma_pp @ beta[1 : 1 + len(ps_c)].to(self.device)
                    Sigma[i, J] = cov_y_p
                    Sigma[J, i] = cov_y_p

                # do on continuous n -> clamp variance ~ 0
                if n in do_:
                    v = do_[n].mean().item()
                    mu[i] = v
                    Sigma[i, :] = 0.0
                    Sigma[:, i] = 0.0
                    Sigma[i, i] = 1e-12

            # block = GaussianBlock(names_c, mu, Sigma)

            # Condition on continuous evidence (if any) and compute marginal likelihood
            ev_c = {k: v for k, v in ev.items() if k in cont}
            if len(ev_c) > 0 and len(names_c) > 0:
                # posterior over non-evidence cont vars; we only need the evidence likelihood term:
                # p(e_c) = N(e_c; μ_E, Σ_EE)
                idxE = torch.tensor(
                    [names_c.index(n) for n in ev_c.keys()],
                    device=self.device,
                    dtype=torch.long,
                )
                muE = mu.index_select(0, idxE)  # [|E|]
                SEE = Sigma.index_select(0, idxE).index_select(1, idxE)  # [|E|,|E|]
                SEE = SEE + 1e-8 * torch.eye(SEE.shape[0], device=self.device)
                chol = torch.linalg.cholesky(SEE)
                logdet = 2.0 * torch.log(torch.diag(chol)).sum()
                yE = torch.stack(
                    [ev_c[n].squeeze(-1) for n in ev_c.keys()], dim=1
                )  # [B, |E|]
                delta = yE - muE.view(1, -1)  # [B, |E|]
                solve = torch.cholesky_solve(delta.unsqueeze(-1), chol)  # [B, |E|,1]
                maha = (delta.unsqueeze(1) @ solve).squeeze()  # [B]
                D = idxE.numel()
                logp_cont = -0.5 * (
                    maha
                    + D * torch.log(torch.tensor(2.0 * torch.pi, device=self.device))
                    + logdet
                )
            else:
                logp_cont = torch.zeros(B, device=self.device)

            weights_B.append((logp_disc + logp_cont))  # [B]
            configs.append(assign_d)

        # Stack weights and normalize per batch
        W = torch.stack(weights_B, dim=1)  # [B, M]
        W = (W - W.max(dim=1, keepdim=True).values).exp()
        W = W / (W.sum(dim=1, keepdim=True) + 1e-12)  # [B, M]

        # ---- Answer query
        # M = W.shape[1]
        N = num_samples
        cat = torch.distributions.Categorical(probs=W)
        idx_m = cat.sample((N,))  # [N, B]
        idx_m = idx_m.transpose(0, 1)  # [B, N]  (Long, same device)

        if query in disc:
            qi = disc.index(query)
            q_vals = []
            for b in range(B):
                idx_b = idx_m[b]  # [N]
                rows_b = mesh.index_select(0, idx_b)  # [N, |disc|]
                vals_b = rows_b[:, qi]  # [N]
                q_vals.append(vals_b)
            q_samples = torch.stack(q_vals, dim=0).to(self.device).float()  # [B, N]
            pdf = {"weights": torch.ones(B, N, device=self.device) / N}
            return pdf, {query: q_samples}

        # continuous query: condition Gaussian on ev + disc config, then sample
        # For simplicity we reuse the μ, Σ computed per config (same for all B except do on continuous varies by B)
        q_vals = []
        qi = cont.index(query)
        for b in range(B):
            samples_b = []
            for nidx in idx_m[b]:  # index of mesh row
                assign_d = {
                    n: mesh[nidx, i].view(1, 1).expand(1, 1).float().to(self.device)
                    for i, n in enumerate(disc)
                }
                # Rebuild μ, Σ (small overhead; could cache)
                names_c = cont
                d = len(names_c)
                mu = torch.zeros(d, device=self.device)
                Sigma = torch.zeros(d, d, device=self.device)
                for n in cont:
                    i = names_c.index(n)
                    node = self.nodes[n]
                    beta = node.beta.squeeze(-1)
                    sigma2 = node.sigma2.squeeze()
                    ps_c = [p for p in self.parents[n] if p in cont]
                    if len(ps_c) == 0:
                        mu[i] = beta[0]
                        Sigma[i, i] = sigma2
                    else:
                        J = torch.tensor(
                            [names_c.index(p) for p in ps_c], device=self.device
                        )
                        mu[i] = (
                            beta[0]
                            + (mu[J] * beta[1 : 1 + len(ps_c)].to(self.device)).sum()
                        )
                        Sigma_pp = Sigma.index_select(0, J).index_select(1, J)
                        var = (
                            beta[1 : 1 + len(ps_c)].to(self.device).unsqueeze(0)
                            @ Sigma_pp
                            @ beta[1 : 1 + len(ps_c)].to(self.device).unsqueeze(-1)
                        ).squeeze()
                        Sigma[i, i] = var + sigma2
                        cov = Sigma_pp @ beta[1 : 1 + len(ps_c)].to(self.device)
                        Sigma[i, J] = cov
                        Sigma[J, i] = cov
                    if n in do_:
                        v = do_[n][b : b + 1].mean().item()
                        mu[i] = v
                        Sigma[i, :] = 0.0
                        Sigma[:, i] = 0.0
                        Sigma[i, i] = 1e-12
                # condition on continuous evidence of batch b
                ev_c_b = {k: v[b : b + 1] for k, v in ev.items() if k in cont}
                if len(ev_c_b) > 0 and len(names_c) > 0:
                    idxE = torch.tensor(
                        [names_c.index(n) for n in ev_c_b.keys()], device=self.device
                    )
                    muE = mu.index_select(0, idxE)
                    SEE = Sigma.index_select(0, idxE).index_select(
                        1, idxE
                    ) + 1e-8 * torch.eye(idxE.numel(), device=self.device)
                    muU = mu[qi : qi + 1]
                    SUE = Sigma[qi : qi + 1, :].index_select(1, idxE)
                    SEE_inv = torch.linalg.pinv(SEE)
                    yE = torch.stack(
                        [ev_c_b[n].squeeze(-1) for n in ev_c_b.keys()], dim=1
                    ).squeeze(
                        0
                    )  # [|E|]
                    mu_post = (muU + (SUE @ SEE_inv @ (yE - muE))).squeeze()
                    var_post = (
                        Sigma[qi, qi] - (SUE @ SEE_inv @ SUE.transpose(0, 1)).squeeze()
                    ).clamp_min(1e-12)
                else:
                    mu_post = mu[qi]
                    var_post = Sigma[qi, qi].clamp_min(1e-12)
                eps = torch.randn((), device=self.device)
                samples_b.append(mu_post + eps * var_post.sqrt())
            q_vals.append(torch.stack(samples_b, dim=0))
        q_samples = torch.stack(q_vals, dim=0)  # [B, N]
        pdf = {"weights": torch.ones(B, N, device=self.device) / N}
        return pdf, {query: q_samples}
