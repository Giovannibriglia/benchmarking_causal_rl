from typing import Dict, List, Optional, Tuple

import torch

from ...inference.base import _as_B1, BaseInferencer, IncompatibilityError


def _is_linear_gaussian(node) -> bool:
    # Duck-typing check against your ParametricLinearGaussianCPD
    return hasattr(node, "beta") and hasattr(node, "sigma2")


class ExactLinearGaussianInferencer(BaseInferencer):
    @torch.no_grad()
    def _assemble_linear_gaussian(
        self,
        ev: Dict[str, torch.Tensor],
        do_: Dict[str, torch.Tensor],
    ) -> Tuple[List[str], Dict[str, int], torch.Tensor, torch.Tensor]:
        """
        Build:
          - names: topological node order
          - idx:   name -> index
          - mu_B:  [B, d] per-batch prior mean, respecting do-values
          - Sigma: [d, d] global covariance, respecting do (incoming edges removed)
        """
        names = self.topo
        idx = {n: i for i, n in enumerate(names)}
        d = len(names)

        # batch size B comes from evidence/do (or defaults to 1)
        B = 1
        for t in list(ev.values()) + list(do_.values()):
            B = max(B, t.shape[0])

        # Initialize outputs
        mu_B = torch.zeros(B, d, device=self.device)  # per-batch mean
        Sigma = torch.zeros(d, d, device=self.device)  # global covariance

        # We build Sigma once, but treat 'do' nodes as constants (zero variance)
        # and zero their incoming edges in the recursion.
        for n in names:
            if n not in do_:
                if not _is_linear_gaussian(self.nodes[n]):
                    raise IncompatibilityError(
                        f"exact.gaussian: node '{n}' is not linear-Gaussian."
                    )

        # Recursive construction in topological order
        for n in names:
            i = idx[n]
            node = self.nodes[n]

            # If intervened, clamp to per-batch value: zero variance and zero covariances
            if n in do_:
                # mean per batch
                mu_B[:, i] = do_[n].squeeze(-1)  # [B]
                # variance ~ 0, covariances ~ 0
                Sigma[i, :] = 0.0
                Sigma[:, i] = 0.0
                Sigma[i, i] = 1e-12
                continue

            ps = self.parents[n]
            if len(ps) == 0:
                # Root: Y = beta0 + eps
                beta = node.beta.squeeze(-1)  # [1]
                sigma2 = node.sigma2.squeeze()  # scalar
                mu_B[:, i] = beta[0].expand(B)  # constant bias for all batches
                Sigma[i, i] = sigma2
            else:
                beta = node.beta.squeeze(-1)  # [1 + |Pa|]
                sigma2 = node.sigma2.squeeze()  # scalar
                # Parent indices
                J = torch.tensor(
                    [idx[p] for p in ps], device=self.device, dtype=torch.long
                )

                # Per-batch mean: beta0 + sum_j beta_j * E[P_j]
                mu_B[:, i] = beta[0] + (mu_B[:, J] * beta[1:].to(self.device)).sum(
                    dim=1
                )

                # Covariance update (global, independent of batch values):
                # Var(Y) = beta_{1:}^T Σ_pp beta_{1:} + σ^2
                Sigma_pp = Sigma.index_select(0, J).index_select(1, J)  # [|Pa|, |Pa|]
                var_y = (
                    beta[1:].to(self.device).unsqueeze(0)
                    @ Sigma_pp
                    @ beta[1:].to(self.device).unsqueeze(-1)
                ).squeeze()
                Sigma[i, i] = var_y + sigma2

                # Cov(Y, P) = Σ_pp beta_{1:}
                cov_y_p = Sigma_pp @ beta[1:].to(self.device)  # [|Pa|]
                Sigma[i, J] = cov_y_p
                Sigma[J, i] = cov_y_p

        return names, idx, mu_B, Sigma

    @torch.no_grad()
    def infer_posterior(
        self,
        query: str,
        evidence: Optional[Dict[str, torch.Tensor]] = None,
        do: Optional[Dict[str, torch.Tensor]] = None,
        num_samples: int = 4096,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Exact inference for linear-Gaussian BNs with per-batch evidence and do-values.
        Inputs:
          evidence: Dict[name] -> [B,1] float
          do      : Dict[name] -> [B,1] float (interventions)
        Outputs:
          pdf     : {"weights": [B, N]}
          samples : {query: [B, N]}
        """
        ev = {k: _as_B1(v, self.device).float() for k, v in (evidence or {}).items()}
        do_ = {k: _as_B1(v, self.device).float() for k, v in (do or {}).items()}

        names, idx, mu_B, Sigma = self._assemble_linear_gaussian(ev, do_)

        # Batch size
        B = mu_B.shape[0]
        i_q = idx[query]

        # If no evidence: sample directly from prior (with do applied)
        if len(ev) == 0:
            var_q = Sigma[i_q, i_q].clamp_min(1e-12)  # scalar
            std_q = var_q.sqrt()
            eps = torch.randn(B, num_samples, device=self.device)
            q_samples = mu_B[:, i_q : i_q + 1] + eps * std_q  # [B, N]
            return {
                "weights": torch.ones(B, num_samples, device=self.device) / num_samples
            }, {query: q_samples}

        # Evidence indices and per-batch values
        E_names = list(ev.keys())
        E_idx = torch.tensor(
            [idx[n] for n in E_names], device=self.device, dtype=torch.long
        )

        # Σ partitions
        Sigma_EE = Sigma.index_select(0, E_idx).index_select(1, E_idx)  # [|E|, |E|]
        Sigma_qE = Sigma[i_q, :].index_select(0, E_idx).unsqueeze(0)  # [1, |E|]
        var_q = Sigma[i_q, i_q].clamp_min(1e-12)  # scalar
        # precompute Σ_EE^{-1}
        Sigma_EE_inv = torch.linalg.pinv(
            Sigma_EE + 1e-8 * torch.eye(Sigma_EE.shape[0], device=self.device)
        )

        # Per-batch conditioning
        # yE_B: [B, |E|], muE_B: [B, |E|]
        yE_B = torch.stack([ev[n].squeeze(-1) for n in E_names], dim=1)  # [B, |E|]
        muE_B = mu_B.index_select(1, E_idx)  # [B, |E|]
        delta = (yE_B - muE_B).unsqueeze(-1)  # [B, |E|, 1]

        # Posterior mean of query (per batch)
        # μ_q|E(b) = μ_q + Σ_qE Σ_EE^{-1} (y_E(b) - μ_E(b))
        gain = (Sigma_qE @ Sigma_EE_inv).expand(B, -1)  # [B, |E|]
        mu_q_post = mu_B[:, i_q : i_q + 1] + (gain.unsqueeze(1) @ delta).squeeze(
            -1
        )  # [B, 1]

        # Posterior variance (same for all batches; evidence value only affects mean)
        var_q_post = (
            var_q - (Sigma_qE @ Sigma_EE_inv @ Sigma_qE.transpose(1, 0)).squeeze()
        ).clamp_min(1e-12)
        std_q_post = var_q_post.sqrt()

        # Sample [B, N]
        eps = torch.randn(B, num_samples, device=self.device)
        q_samples = mu_q_post + eps * std_q_post

        pdf = {"weights": torch.ones(B, num_samples, device=self.device) / num_samples}
        samples = {query: q_samples}
        return pdf, samples
