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
        latent = self.encoder(obs)
        dist = self._distribution(latent)
        if self.is_discrete:
            logp = dist.log_prob(act)
            ent = dist.entropy().mean()
        else:
            # actions shape [B, act_dim]
            logp = dist.log_prob(act).sum(-1)
            ent = dist.entropy().sum(-1).mean()
        ratio = torch.exp(logp - old_logp)
        surr = (ratio * adv).mean()
        if self.ent_coeff != 0.0:
            surr += self.ent_coeff * ent
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
        latent = self.encoder(obs)
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
        """
        # Flatten rollout
        obs = self.flat(mem["obs"]).detach()
        act = self.flat(mem["actions"])
        old_logp = self.flat(mem["logp"]).detach()
        returns = self.flat(mem["returns"]).detach()
        adv = self.flat(mem["advantages"]).detach()

        # Advantage normalization (recommended)
        if self.adv_norm:
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        # ------------------ 1) Critic (value) regression ------------------
        latent = self.encoder(obs)
        value = self.critic(latent).squeeze(-1)
        value_loss = 0.5 * (returns - value).pow(2).mean()

        self.vf_optim.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.critic.parameters()), 0.5
        )
        self.vf_optim.step()

        # ------------------ 2) Policy natural gradient --------------------
        # Build old distribution snapshot (no grad)
        with torch.no_grad():
            latent_old = self.encoder(obs)
            dist_old = self._distribution(latent_old)

        # Surrogate gradient (note: maximize surrogate)
        surr, _ = self._surrogate(obs, act, old_logp, adv)
        policy_params = self._policy_params()
        grad = flat_grad(surr, policy_params, retain_graph=True)  # grad of J(θ)
        g = grad.detach()

        # Fisher-vector product operator
        def f_Ax(v):
            return self._fvp(obs, dist_old, v)

        # Solve A x = g for x (natural gradient direction)
        step_dir = conjugate_grad(f_Ax, g, iters=self.cg_iters)

        # Compute step size to satisfy KL constraint: x^T A x = 2 ε  -> step = sqrt(2 ε / (x^T A x)) * x
        shs = (step_dir * f_Ax(step_dir)).sum()  # x^T A x
        step_scale = torch.sqrt(2.0 * self.max_kl / (shs + 1e-10))
        full_step = step_scale * step_dir

        # Expected improvement rate ≈ g^T step
        expected_improve = (g * full_step).sum().item()

        # Line search
        params_flat = flat_params(policy_params).detach()
        ok, new_params = self._line_search(
            obs, act, adv, old_logp, dist_old, params_flat, full_step, expected_improve
        )

        # Metrics
        self.train_metrics.add(
            trpo_line_ok=float(ok),
            trpo_expected_improve=float(expected_improve),
            value_mse=float(((returns - value.detach()) ** 2).mean().item()),
        )

    def _get_action(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            latent = self.encoder(obs)
            dist = self._distribution(latent)
            if self.is_discrete:
                return dist.sample()
            else:
                return dist.sample()  # already [N, act_dim]

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
