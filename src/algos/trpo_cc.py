from __future__ import annotations

from pathlib import Path
from typing import Union

import torch

from src.algos.cbn_critic import VBNCritic
from src.algos.trpo import conjugate_grad, flat_grad, flat_params, TRPO


class TRPO_CC(TRPO, VBNCritic):
    """
    TRPO variant with a Causal Bayesian Network critic:
      * advantages: A = R̂ - V_causal(x)  (no neural value loss)
      * policy step: standard TRPO (CG + line search under KL)
      * optional KL(pi || pi_causal) regularization can be added via the mixin (for discrete)
    """

    def __init__(self, *args, cbn_kwargs: dict | None = None, **trpo_kwargs):
        # init base TRPO (builds nets & sets hyperparams)
        TRPO.__init__(self, *args, **trpo_kwargs)
        # init CBN mixin (builds/loads BN, inference backends, knobs)
        VBNCritic.__init__(self, **(cbn_kwargs or {}))

    # after rollout: compute causal V, advantages, and old_logp under current policy
    def _post_update(self, mem):
        self._post_update_fill_adv_and_logp(mem)

    # main TRPO step BUT: no neural value regression (critic is the BN)
    def _algo_update(self, mem):
        obs = self.flat(mem["obs"]).detach()
        act = self.flat(mem["actions"])
        old_logp = mem["old_logp"].detach()
        adv = self.flat(mem["advantages"]).detach()

        if self.adv_norm:
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        # snapshot old dist (no grad)
        with torch.no_grad():
            latent_old = self.encoder(obs)
            dist_old = self._distribution(latent_old)

        # surrogate (maximize)
        surr, _ = self._surrogate(obs, act, old_logp, adv)
        policy_params = self._policy_params()
        g = flat_grad(surr, policy_params, retain_graph=True).detach()  # ∇_θ J

        # Fisher-vector product
        f_Ax = lambda v: self._fvp(obs, dist_old, v)

        # natural gradient direction via CG
        step_dir = conjugate_grad(f_Ax, g, iters=self.cg_iters)

        # step size for KL constraint: x^T A x = 2*max_kl
        shs = (step_dir * f_Ax(step_dir)).sum()
        step_scale = torch.sqrt(2.0 * self.max_kl / (shs + 1e-10))
        full_step = step_scale * step_dir
        expected_improve = (g * full_step).sum().item()

        # line search under KL
        params_flat = flat_params(policy_params).detach()
        ok, _ = self._line_search(
            obs, act, adv, old_logp, dist_old, params_flat, full_step, expected_improve
        )

        # metrics (no value MSE; critic is causal BN)
        self.train_metrics.add(
            trpo_line_ok=float(ok),
            trpo_expected_improve=float(expected_improve),
            value_mse=0.0,
        )

    # ---------- persistence (include BN params) ----------
    def save_policy(self, path: Union[str, Path]) -> None:
        torch.save(
            {
                "state_dict": self.state_dict(),
                # TRPO hypers
                "max_kl": self.max_kl,
                "cg_iters": self.cg_iters,
                "backtrack_iters": self.backtrack_iters,
                "backtrack_coeff": self.backtrack_coeff,
                "damping": self.damping,
                "vf_lr": self.vf_lr,
                "ent_coeff": self.ent_coeff,
                "adv_norm": self.adv_norm,
                "is_discrete": self.is_discrete,
                # CBN knobs
                "pi_samples": getattr(self, "pi_samples", 32),
                "kl_coeff": getattr(self, "kl_coeff", 0.0),
                "kl_beta": getattr(self, "kl_beta", 5.0),
            },
            self._ensure_pt_path(path),
        )
        self.save_bn_params(path)  # writes <checkpoint>.td next to .pt

    def load_policy(self, path: Union[str, Path]) -> None:
        ckpt = torch.load(self._ensure_pt_path(path), map_location=self.device)
        self.load_state_dict(ckpt["state_dict"])

        # restore TRPO hypers (match your current TRPO class)
        self.max_kl = ckpt.get("max_kl", self.max_kl)
        self.cg_iters = ckpt.get("cg_iters", self.cg_iters)
        self.backtrack_iters = ckpt.get("backtrack_iters", self.backtrack_iters)
        self.backtrack_coeff = ckpt.get("backtrack_coeff", self.backtrack_coeff)
        self.damping = ckpt.get("damping", self.damping)
        self.vf_lr = ckpt.get("vf_lr", self.vf_lr)
        self.ent_coeff = ckpt.get("ent_coeff", self.ent_coeff)
        self.adv_norm = ckpt.get("adv_norm", self.adv_norm)
        self.is_discrete = ckpt.get("is_discrete", self.is_discrete)

        # restore CBN knobs
        self.pi_samples = ckpt.get("pi_samples", getattr(self, "pi_samples", 32))
        self.kl_coeff = ckpt.get("kl_coeff", getattr(self, "kl_coeff", 0.0))
        self.kl_beta = ckpt.get("kl_beta", getattr(self, "kl_beta", 5.0))

        # load BN params (.td) and reinit inference
        self.load_bn_params(path)
