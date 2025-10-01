from __future__ import annotations

from pathlib import Path
from typing import Union

import torch

from src.algos.trpo import conjugate_grad, flat_grad, flat_params, TRPO

from src.algos.vbn_critic import VBNCritic


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

    # ── hook: after rollout collection, fill advantages & old_logp via CBN ──
    def _post_update(self, mem):
        """
        After a rollout, (1) fit/update the causal BN on this batch,
        (2) compute V_c(s) and overwrite mem['advantages'] = returns - V_c,
        (3) fill mem['old_logp'] for algorithms that need it (kept for consistency).
        """
        # 1) update/fit the BN on this rollout
        self._causal_update(mem)

        # 2) fill *both* advantages and old_logp in one shot
        self._post_update_fill_adv_and_logp(mem)

        # 3) (optional) log summary
        self._log_adv_summary(mem)

    # main TRPO step BUT: no neural value regression (critic is the BN)
    # in src/algos/trpo_cc.py
    def _algo_update(self, mem):
        obs = self.flat(mem["obs"]).detach()  # [B, ...]
        act = self.flat(mem["actions"])  # [B] / [B,1] or [B,Ad]
        old_logp = self.flat(mem["old_logp"]).detach()  # [B] expected; may arrive wrong
        adv = self.flat(mem["advantages"]).detach()  # [B] or [B,1]

        # ---- normalize shapes ----
        if adv.ndim > 1:
            adv = adv.squeeze(-1)
        if old_logp.ndim > 1:
            old_logp = old_logp.squeeze(-1)

        B = adv.shape[0]

        # If old_logp was produced as a pairwise matrix [B,B] and then flattened -> [B*B], fix by taking the diagonal
        if old_logp.numel() == B * B and old_logp.ndim == 1:
            old_logp = old_logp.view(B, B).diag()

        # Actions cleanup (avoid pairwise [B,B])
        if self.is_discrete:
            if act.ndim == 2 and act.shape[0] == act.shape[1] == B:
                act = act.diag()
            if act.ndim > 1:
                act = act.squeeze(-1)
            act = act.long()
            assert act.shape == (B,)
        else:
            act_dim = self.log_std.numel()
            if act.ndim == 1:
                act = act.view(-1, act_dim)
            elif act.ndim == 2 and act.shape[-1] != act_dim:
                act = act.reshape(-1, act_dim)
            assert act.shape == (B, act_dim)

        assert old_logp.shape == (
            B,
        ), f"old_logp shape {old_logp.shape}, expected {(B,)}"
        assert adv.shape == (B,), f"adv shape {adv.shape}, expected {(B,)}"

        if self.adv_norm:
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        with torch.no_grad():
            latent_old = self._encode(obs)
            dist_old = self._distribution(latent_old)

        surr, _ = self._surrogate(obs, act, old_logp, adv)
        policy_params = self._policy_params()
        g = flat_grad(surr, policy_params, retain_graph=True).detach()

        f_Ax = lambda v: self._fvp(obs, dist_old, v)
        step_dir = conjugate_grad(f_Ax, g, iters=self.cg_iters)
        shs = (step_dir * f_Ax(step_dir)).sum()
        step_scale = torch.sqrt(2.0 * self.max_kl / (shs + 1e-10))
        full_step = step_scale * step_dir
        expected_improve = (g * full_step).sum().item()

        params_flat = flat_params(policy_params).detach()
        ok, _ = self._line_search(
            obs, act, adv, old_logp, dist_old, params_flat, full_step, expected_improve
        )

        entropy_mean = (
            self._policy_entropy(obs).item()
            if hasattr(self, "_policy_entropy")
            else 0.0
        )
        self._log_update_metrics(
            entropy=entropy_mean,
            adv_var=adv.var(unbiased=False).item(),
            value_mse=0.0,
            trpo_line_ok=float(ok),
            trpo_expected_improve=float(expected_improve),
            natgrad_norm=step_dir.norm().item(),
            uses_causal_critic=1.0,
        )

    # ---------- persistence (include BN params) ----------
    # ---------- persistence (single-file bundle: policy + VBN) ----------
    def save_policy(self, path: Union[str, Path]) -> None:
        """
        Save everything needed to restore TRPO_CC in one file:
          • torch state_dict (encoder/actor[/log_std]/critic, etc.)
          • TRPO hyperparams (max_kl, cg/backtrack, damping, vf_lr, ent_coeff, adv_norm)
          • discrete/continuous flag + causal knobs (pi_samples, kl_coeff, kl_beta)
          • VBN bundle (DAG/types/cards + learned params + backend choices)
        """
        payload = {
            "state_dict": self.state_dict(),
            # ---- TRPO hyperparams ----
            "max_kl": float(self.max_kl),
            "cg_iters": int(self.cg_iters),
            "backtrack_iters": int(self.backtrack_iters),
            "backtrack_coeff": float(self.backtrack_coeff),
            "damping": float(self.damping),
            "vf_lr": float(self.vf_lr),
            "ent_coeff": float(self.ent_coeff),
            "adv_norm": bool(self.adv_norm),
            # ---- policy / causal flags ----
            "is_discrete": bool(self.is_discrete),
            "pi_samples": int(getattr(self, "pi_samples", 32)),
            "kl_coeff": float(getattr(self, "kl_coeff", 0.0)),
            "kl_beta": float(getattr(self, "kl_beta", 5.0)),
            # ---- pack the whole causal BN into the same file ----
            "vbn_bundle": self._bn_to_bundle(),
            "format": "trpo_cc+vbn@1",  # simple version tag
        }
        torch.save(payload, self._ensure_pt_path(path))

    def load_policy(
        self,
        path: Union[str, Path],
        *,
        ensure_vbn: bool = True,
        warm_start: bool = False,
    ) -> None:
        """
        Load from a single-file checkpoint created by save_policy().
        - Restores weights (strict by default; set warm_start=True to allow missing keys).
        - Restores TRPO hyperparameters and the full causal BN (if present).
        - If no VBN bundle is present, falls back to legacy sidecars next to the .pt.
        """
        ckpt = torch.load(
            self._ensure_pt_path(path),
            map_location=self.device,
            weights_only=False,  # required for custom VBN objects on PyTorch 2.6+
        )

        # 1) weights
        self.load_state_dict(ckpt["state_dict"], strict=not warm_start)

        # 2) TRPO hyperparams
        self.max_kl = ckpt.get("max_kl", self.max_kl)
        self.cg_iters = ckpt.get("cg_iters", self.cg_iters)
        self.backtrack_iters = ckpt.get("backtrack_iters", self.backtrack_iters)
        self.backtrack_coeff = ckpt.get("backtrack_coeff", self.backtrack_coeff)
        self.damping = ckpt.get("damping", self.damping)
        self.vf_lr = ckpt.get("vf_lr", self.vf_lr)
        self.ent_coeff = ckpt.get("ent_coeff", self.ent_coeff)
        self.adv_norm = ckpt.get("adv_norm", self.adv_norm)

        # 3) policy/causal flags
        self.is_discrete = ckpt.get("is_discrete", self.is_discrete)
        self.pi_samples = ckpt.get("pi_samples", getattr(self, "pi_samples", 32))
        self.kl_coeff = ckpt.get("kl_coeff", getattr(self, "kl_coeff", 0.0))
        self.kl_beta = ckpt.get("kl_beta", getattr(self, "kl_beta", 5.0))

        # 4) VBN restore
        bundle = ckpt.get("vbn_bundle", None)
        if bundle is not None:
            self._bn_from_bundle(bundle)
        else:
            # legacy fallback: <checkpoint>.td + <checkpoint>.bnmeta.json
            self.load_bn_params(path)

        if ensure_vbn:
            self._ensure_vbn_ready()
