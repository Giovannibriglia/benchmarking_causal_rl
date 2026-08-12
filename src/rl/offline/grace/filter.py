"""GRACE belief filter — exact sequential conditioning over the DISCRETE
LATENT BLOCK (B3; A10 seam).

The latent block is (S, U) in POMDP cells and (U,) in MDP cells (S observed,
degenerate). The filter is written against that block, never against "discrete
everything": evidence enters through per-step log-likelihood lookups that the
continuous slice swaps for mechanism log-probs without touching the recursion.

Serving-path semantics (the mutilated channel): the acting policy's own
actions are do()-actions, so the behavior CPD ``P_b(A | ., U)`` is NEVER
scored during serving — evidence on U comes from the reward channel (and the
emission, in POMDP cells). The offline target path may smooth; the serving
path always filters (B3).
"""

from __future__ import annotations

from typing import Optional

import torch

from src.rl.offline.grace.cbn import _EPS, CBNParams


class BeliefFilter:
    """Exact forward filter over (S x U) [POMDP] or (U,) [MDP].

    Beliefs are ``(B, n_s * n_u)`` joint tensors (POMDP) or ``(B, n_u)``
    (MDP). ``entropy`` telemetry feeds the router's inference-health check.
    """

    def __init__(self, params: CBNParams, n_classes: int, pomdp: bool, device) -> None:
        self.p = params
        self.n_c = int(n_classes)
        self.pomdp = bool(pomdp)
        self.device = device
        self.n_u = int(params.p_u.numel())
        self.n_s = int(params.p0.numel())

    # ------------------------------------------------------------------ init
    def reset(self, batch_size: int) -> torch.Tensor:
        """Prior belief: P(U) (MDP) or P0(S) x P(U) (POMDP)."""
        if not self.pomdp:
            return self.p.p_u.view(1, -1).expand(batch_size, -1).clone()
        joint = torch.einsum("s,u->su", self.p.p0, self.p.p_u).reshape(1, -1)
        return joint.expand(batch_size, -1).clone()

    # ------------------------------------------------------------ observation
    def condition_obs(self, belief: torch.Tensor, obs_symbol: torch.Tensor):
        """POMDP correction step on the emission: b(s, u) *= P(o | s)."""
        if not self.pomdp:
            return belief
        lik = self.p.emis[:, obs_symbol].T  # (B, n_s)
        b = belief.reshape(belief.shape[0], self.n_s, self.n_u)
        b = b * lik.unsqueeze(-1)
        b = b / b.sum(dim=(1, 2), keepdim=True).clamp_min(_EPS)
        return b.reshape(belief.shape[0], -1)

    # ------------------------------------------------------------------ step
    def step(
        self,
        belief: torch.Tensor,
        s_or_obs: torch.Tensor,
        action: torch.Tensor,
        reward_class: Optional[torch.Tensor] = None,
        next_obs_symbol: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """One filter step under the MUTILATED channel: condition on the
        observed reward (U evidence), then predict S through the dynamics and
        condition on the next emission (POMDP). ``s_or_obs`` is the observed
        state bin (MDP) or the current obs symbol (POMDP, ignored — the
        emission was applied by ``condition_obs``)."""
        bsz = belief.shape[0]
        if not self.pomdp:
            if reward_class is None:
                return belief
            # b(u) *= P(rc | s, a, u) — exact discrete-latent conditioning.
            pr = self.p.p_r  # (n_u_r, n_s, n_a, n_c)
            n_u_r = pr.shape[0]
            lik = pr[:, s_or_obs, action, reward_class].T  # (B, n_u_r)
            if n_u_r != self.n_u:
                lik = lik.expand(bsz, self.n_u)
            b = belief * lik
            return b / b.sum(dim=1, keepdim=True).clamp_min(_EPS)
        b = belief.reshape(bsz, self.n_s, self.n_u)
        if reward_class is not None:
            pr = self.p.p_r  # (n_u_r, S, A, C)
            n_u_r = pr.shape[0]
            lik = pr[:, :, action, reward_class]  # (n_u_r, S, B) -> permute
            lik = lik.permute(2, 1, 0)  # (B, S, n_u_r)
            if n_u_r != self.n_u:
                lik = lik.expand(bsz, self.n_s, self.n_u)
            b = b * lik
            b = b / b.sum(dim=(1, 2), keepdim=True).clamp_min(_EPS)
        # Predict: b(s', u) = sum_s T(s' | s, a) b(s, u); U is episode-static.
        tr = self.p.trans[:, action]  # (S, B, S') -> permute
        tr = tr.permute(1, 0, 2)  # (B, S, S')
        b = torch.einsum("bsu,bsj->bju", b, tr)
        b = b / b.sum(dim=(1, 2), keepdim=True).clamp_min(_EPS)
        b = b.reshape(bsz, -1)
        if next_obs_symbol is not None:
            b = self.condition_obs(b, next_obs_symbol)
        return b

    # ------------------------------------------------------------- telemetry
    def entropy(self, belief: torch.Tensor) -> torch.Tensor:
        """Normalized belief entropy in [0, 1] — router inference-health
        telemetry (1 = uninformative)."""
        p = belief.clamp_min(_EPS)
        h = -(p * p.log()).sum(dim=1)
        import math

        return h / max(math.log(belief.shape[1]), _EPS)

    def u_marginal(self, belief: torch.Tensor) -> torch.Tensor:
        """(B, n_u) marginal over the confounder block."""
        if not self.pomdp:
            return belief
        return belief.reshape(belief.shape[0], self.n_s, self.n_u).sum(dim=1)
