from __future__ import annotations

from typing import Iterable, List

import torch
from torch import nn
from torch.distributions import Categorical, kl_divergence, Normal
from torch.optim import Adam

from src.algos.base_actor_critic import BaseActorCritic


# ────────────────────────── small utils ──────────────────────────


def flat_params(params: Iterable[torch.Tensor]) -> torch.Tensor:
    return torch.cat([p.view(-1) for p in params])


def set_params_(params: Iterable[torch.Tensor], vector: torch.Tensor) -> None:
    """In-place set of parameters from a flat vector."""
    offset = 0
    for p in params:
        n = p.numel()
        p.data.copy_(vector[offset : offset + n].view_as(p))
        offset += n


def flat_grad(
    y: torch.Tensor,
    params: Iterable[torch.Tensor],
    retain_graph=False,
    create_graph=False,
):
    g = torch.autograd.grad(
        y, params, retain_graph=retain_graph, create_graph=create_graph
    )
    return torch.cat([gi.contiguous().view(-1) for gi in g])


def conjugate_grad(f_Ax, b, iters=10, tol=1e-10):
    """CG for Ax = b using only matrix-vector product f_Ax(v)."""
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rr = r.dot(r)
    for _ in range(iters):
        Ap = f_Ax(p)
        denom = p.dot(Ap) + 1e-10
        alpha = rr / denom
        x += alpha * p
        r -= alpha * Ap
        new_rr = r.dot(r)
        if new_rr < tol:
            break
        beta = new_rr / rr
        p = r + beta * p
        rr = new_rr
    return x


class TRPO(BaseActorCritic):
    """
    Trust Region Policy Optimization (Schulman et al. 2015).
    - Policy updated by natural gradient step solved via conjugate gradient + line search.
    - Value function updated by standard MSE regression (Adam).
    - Supports discrete (Categorical) and continuous (diag Normal) action spaces.
    """

    def __init__(
        self,
        *args,
        max_kl: float = 1e-2,  # trust region radius
        cg_iters: int = 10,  # conjugate gradient iterations
        backtrack_coeff: float = 0.8,  # line search shrink
        backtrack_iters: int = 10,  # line search steps
        damping: float = 1e-2,  # Hessian damping
        vf_lr: float = 3e-4,  # critic LR
        ent_coeff: float = 0.0,  # optional entropy bonus
        adv_norm: bool = True,  # normalize advantages
        **kw,
    ):
        super().__init__(*args, **kw)

        self.max_kl = max_kl
        self.cg_iters = cg_iters
        self.backtrack_coeff = backtrack_coeff
        self.backtrack_iters = backtrack_iters
        self.damping = damping
        self.vf_lr = vf_lr
        self.ent_coeff = ent_coeff
        self.adv_norm = adv_norm

        # Use a **separate** optimizer for the value function (critic)
        # while policy is updated manually via natural gradient
        self.vf_optim = Adam(
            list(self.encoder.parameters()) + list(self.critic.parameters()),
            lr=self.vf_lr,
        )

    # ---------- helpers ----------
    def _policy_params(self) -> List[torch.nn.Parameter]:
        # policy params include encoder + actor head + (log_std if continuous)
        if self.is_discrete:
            actor_params = list(self.actor.parameters())
        else:
            actor_params = list(self.actor_mu.parameters()) + [self.log_std]
        return list(self.encoder.parameters()) + actor_params

    def _distribution(self, latent):
        if self.is_discrete:
            logits = self.actor(latent)
            return Categorical(logits=logits)
        else:
            mu = self.actor_mu(latent)
            std = torch.exp(self.log_std)  # [act_dim]
            # broadcast std to batch
            std = std.expand_as(mu)
            return Normal(mu, std)

    def _surrogate(self, obs, act, old_logp, adv):
        """
        Computes the TRPO surrogate objective J(θ) = E[(πθ/πold) * A] (+ optional entropy bonus).
        Shapes are normalized so that:
          - old_logp: [B]
          - adv:      [B]
          - act:      [B] (discrete) or [B, act_dim] (continuous)
        """
        # --- normalize shapes ---
        if old_logp.ndim > 1:
            old_logp = old_logp.squeeze(-1)
        if adv.ndim > 1:
            adv = adv.squeeze(-1)

        if self.is_discrete:
            # actions must be 1D Long [B]
            if act.ndim > 1:
                act = act.squeeze(-1)
            act = act.long()
        else:
            # actions must be [B, act_dim]
            act_dim = self.log_std.numel()
            if act.ndim == 1:
                act = act.view(-1, act_dim)
            elif act.ndim == 2 and act.shape[-1] != act_dim:
                act = act.reshape(-1, act_dim)

        # --- current policy ---
        latent = self._encode(obs)
        dist = self._distribution(latent)

        # log prob of executed actions
        if self.is_discrete:
            logp = dist.log_prob(act)  # [B]
            ent = dist.entropy().mean()
        else:
            logp = dist.log_prob(act).sum(-1)  # [B]
            ent = dist.entropy().sum(-1).mean()

        # --- surrogate ---
        ratio = torch.exp(logp - old_logp)  # [B]
        surr = (ratio * adv).mean()
        if self.ent_coeff != 0.0:
            surr = surr + self.ent_coeff * ent

        return surr, dist

    def _mean_kl(self, dist_old, dist_new):
        # Use analytical KL; average over batch
        kl = kl_divergence(dist_old, dist_new)
        # for Normal, kl has shape [B, act_dim] -> sum over dims; for Categorical -> [B]
        if kl.dim() > 1:
            kl = kl.sum(-1)
        return kl.mean()

    def _fvp(self, obs, dist_old, v):
        """
        Fisher-vector product via Pearlmutter trick:
        F v ≈ ∇²_θ KL(π_θ_old || π_θ) v + damping * v
        """
        # Build current dist (depends on current θ)
        latent = self._encode(obs)
        dist_new = self._distribution(latent)

        mean_kl = self._mean_kl(dist_old, dist_new)
        grads = flat_grad(
            mean_kl, self._policy_params(), retain_graph=True, create_graph=True
        )
        g_v = (grads * v).sum()
        hvp = flat_grad(g_v, self._policy_params(), retain_graph=True)
        return hvp + self.damping * v

    def _line_search(
        self,
        obs,
        act,
        adv,
        old_logp,
        dist_old,
        params_flat,
        full_step,
        expected_improve_rate,
    ):
        """Backtracking line search on the surrogate objective with KL constraint."""
        prev_params = params_flat
        for i in range(self.backtrack_iters):
            step_frac = self.backtrack_coeff**i
            new_params = prev_params + step_frac * full_step
            set_params_(self._policy_params(), new_params)

            # compute new surrogate and KL
            surr, dist_new = self._surrogate(obs, act, old_logp, adv)
            improve = surr.item()
            kl = self._mean_kl(dist_old, dist_new).item()

            if kl <= self.max_kl and improve > 0:
                return True, new_params
        # revert if failed
        set_params_(self._policy_params(), prev_params)
        return False, prev_params

    # ---------- core update ----------
    def _algo_update(self, mem):
        """
        TRPO step: (1) fit value function by regression, (2) compute natural gradient step for policy.
        Uses mem["old_logp"] if present (e.g., in causal variants); otherwise falls back to mem["logp"].
        """
        # ----- flatten rollout buffers -----
        obs = self.flat(mem["obs"]).detach()  # [B, ...]
        act = self.flat(mem["actions"])  # [B] or [B,Ad] or [B,1]
        old_logp = self.flat(mem.get("old_logp", mem["logp"])).detach()  # [B] or [B,1]
        returns = self.flat(mem["returns"]).detach()  # [B] or [B,1]
        adv = self.flat(mem["advantages"]).detach()  # [B] or [B,1]

        # ----- normalize shapes -----
        if old_logp.ndim > 1:
            old_logp = old_logp.squeeze(-1)
        if adv.ndim > 1:
            adv = adv.squeeze(-1)
        if returns.ndim > 1:
            returns = returns.squeeze(-1)

        if self.is_discrete:
            if act.ndim > 1:
                act = act.squeeze(-1)
            act = act.long()  # [B]
        else:
            act_dim = self.log_std.numel()
            if act.ndim == 1:
                act = act.view(-1, act_dim)
            elif act.ndim == 2 and act.shape[-1] != act_dim:
                act = act.reshape(-1, act_dim)  # [B, Ad]

        # sanity checks
        B = old_logp.shape[0]
        assert adv.shape == (B,), f"adv {adv.shape}"
        assert returns.shape == (B,), f"returns {returns.shape}"
        if self.is_discrete:
            assert act.shape == (B,), f"act {act.shape}"
        else:
            assert act.shape == (B, self.log_std.numel()), f"act {act.shape}"

        # Advantage normalization (recommended)
        if self.adv_norm:
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        # ------------------ 1) Critic (value) regression ------------------
        latent = self._encode(obs)
        value = self.critic(latent).squeeze(-1)  # [B]
        value_loss = 0.5 * (returns - value).pow(2).mean()

        self.vf_optim.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.critic.parameters()), 0.5
        )
        self.vf_optim.step()

        # ------------------ 2) Policy natural gradient --------------------
        with torch.no_grad():
            latent_old = self._encode(obs)
            dist_old = self._distribution(latent_old)

        # Surrogate gradient (maximize)
        surr, _ = self._surrogate(obs, act, old_logp, adv)
        policy_params = self._policy_params()
        g = flat_grad(surr, policy_params, retain_graph=True).detach()  # ∇θ J

        # Fisher-vector product operator
        def f_Ax(v):
            return self._fvp(obs, dist_old, v)

        # Conjugate gradient to solve A x = g
        step_dir = conjugate_grad(f_Ax, g, iters=self.cg_iters)

        # Scale to satisfy KL constraint: x^T A x = 2 ε
        shs = (step_dir * f_Ax(step_dir)).sum()
        step_scale = torch.sqrt(2.0 * self.max_kl / (shs + 1e-10))
        full_step = step_scale * step_dir

        expected_improve = (g * full_step).sum().item()

        # Line search
        params_flat = flat_params(policy_params).detach()
        ok, _ = self._line_search(
            obs, act, adv, old_logp, dist_old, params_flat, full_step, expected_improve
        )

        # Optional: entropy logging
        entropy_mean = (
            self._policy_entropy(obs).item()
            if hasattr(self, "_policy_entropy")
            else 0.0
        )

        self._log_update_metrics(
            entropy=entropy_mean,
            adv_var=adv.var(unbiased=False).item(),
            value_mse=((returns - value.detach()) ** 2).mean().item(),
            trpo_line_ok=float(ok),
            trpo_expected_improve=float(expected_improve),
            natgrad_norm=step_dir.norm().item(),
        )

    def _policy_entropy(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Mean entropy of the current policy πθ over a batch of obs.
        For Normal, we sum entropies over action dims then average over batch.
        """
        with torch.no_grad():
            latent = self._encode(obs)
            dist = self._distribution(latent)
            ent = dist.entropy()
            if ent.dim() > 1:  # Normal: [B, act_dim] -> sum over dims
                ent = ent.sum(-1)
            return ent.mean()  # scalar tensor

    def _get_action(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            latent = self._encode(obs)
            dist = self._distribution(latent)
            return dist.sample()

    # ---------- persistence ----------
    def save_policy(self, path):
        torch.save(
            {
                "state_dict": self.state_dict(),
                "max_kl": self.max_kl,
                "cg_iters": self.cg_iters,
                "backtrack_iters": self.backtrack_iters,
                "backtrack_coeff": self.backtrack_coeff,
                "damping": self.damping,
                "vf_lr": self.vf_lr,
                "ent_coeff": self.ent_coeff,
                "adv_norm": self.adv_norm,
                "is_discrete": self.is_discrete,
            },
            self._ensure_pt_path(path),
        )

    def load_policy(self, path):
        ckpt = torch.load(self._ensure_pt_path(path), map_location=self.device)
        self.load_state_dict(ckpt["state_dict"])
        if "cfg" in ckpt:
            self.max_kl = ckpt["max_kl"]
            self.cg_iters = ckpt["cg_iters"]
            self.backtrack_coeff = ckpt["backtrack_coeff"]
            self.backtrack_iters = ckpt["backtrack_iters"]
            self.damping = ckpt["damping"]
            self.vf_lr = ckpt["vf_lr"]
            self.ent_coeff = ckpt["ent_coeff"]
            self.adv_norm = ckpt["adv_norm"]
