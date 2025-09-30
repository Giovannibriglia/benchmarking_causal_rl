from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch
from torch import nn

from src.algos.vanilla_ac import VanillaAC  # <- the class we just wrote
from src.algos.vbn_critic import VBNCritic


class VanillaAC_CC(VanillaAC, VBNCritic):
    """
    Vanilla Actor–Critic with a Causal Bayesian Critic (V^c).
    - Replaces neural critic with CBN: advantages = returns - V_causal(s)
    - No critic MSE term (we ignore vf_coeff entirely)
    - Optional KL(pi || pi_causal_prior) using Q_c (discrete single-head)
    - Keeps extra_* hooks and unified metrics
    """

    def __init__(self, *args, cbn_kwargs: dict | None = None, **vac_kwargs):
        # 1) init base algo (nets/device/optim exist)
        VanillaAC.__init__(self, *args, **vac_kwargs)
        # 2) init VBN mixin (sets BN, inference, knobs, etc.)
        VBNCritic.__init__(self, **(cbn_kwargs or {}))

    # ── after rollout: fit/update BN, compute causal adv, fill old_logp ──
    def _post_update(self, mem):
        """
        (1) update/fit BN on this rollout
        (2) compute V_c(s) and overwrite mem['advantages'] = returns - V_c
        (3) fill mem['old_logp'] = log πθ(a|s) for on-policy bookkeeping
        """
        self._causal_update(mem)
        self._post_update_fill_adv_and_logp(mem)
        self._log_adv_summary(mem)  # optional metrics: mean/std/min/max

    # ── main optimization step: actor-only (no neural critic term) ──
    def _algo_update(self, mem):
        obs = self.flat(mem["obs"])
        actions = self.flat(mem["actions"])
        # Vanilla AC style: do NOT normalize advantages
        adv = self.flat(mem["advantages"]).detach()

        # encode once
        latent = self._encode(obs)

        if self.is_discrete:
            logits = self.actor(latent)
            dist = self.dist_fn(logits)
            logp = dist.log_prob(actions.view(-1).long())  # [B]
            entropy = dist.entropy().mean()
            # keep your hook signature consistent with A2C_CC
            extra_a = self.extra_actor_loss(latent, logits)
        else:
            mu = self.actor_mu(latent)
            dist = self.dist_fn(mu)
            act = (
                actions if actions.ndim == 2 else actions.view(-1, self.log_std.numel())
            )
            logp = dist.log_prob(act).sum(-1)  # [B]
            entropy = dist.entropy().sum(-1).mean()
            extra_a = self.extra_actor_loss(latent, mu)

        base_actor_loss = -(logp * adv).mean()

        # Optional KL(π || π_causal_prior) for discrete single-head policies
        kl_loss = torch.tensor(0.0, device=self.device)
        if (
            getattr(self, "kl_coeff", 0.0) > 0.0
            and self.is_discrete
            and getattr(self, "_cached_q_table", None) is not None
        ):
            pi_c = self._pi_causal_prior(self._cached_q_table.detach())
            probs = dist.probs.clamp_min(1e-8)
            log_probs = probs.log()
            log_probs_c = pi_c.probs.clamp_min(1e-8).log()
            kl = (probs * (log_probs - log_probs_c)).sum(-1).mean()
            kl_loss = self.kl_coeff * kl

        # No critic regression here; still allow extra_critic_loss hook (zero target)
        extra_c = self.extra_critic_loss(latent, torch.zeros_like(logp))

        loss = base_actor_loss + extra_a + kl_loss - self.ent_coeff * entropy

        self.optim.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.parameters(), 0.5).item()
        self.optim.step()

        # metrics
        self._log_update_metrics(
            total_loss=float(loss),
            actor_loss=float(base_actor_loss),
            critic_loss=0.0,  # no neural critic
            entropy=float(entropy),
            adv_var=float(adv.var(unbiased=False).item()),
            value_mse=0.0,  # no critic MSE
            extra_actor_loss=float(extra_a),
            extra_critic_loss=float(extra_c),
            grad_norm=grad_norm,
            kl_loss=float(kl_loss),
            uses_causal_critic=1.0,
        )

    # ---------- persistence ----------
    def save_policy(self, path: Union[str, Path]) -> None:
        path = self._ensure_pt_path(path)
        ckpt = {
            "state_dict": self.state_dict(),
            "is_discrete": self.is_discrete,
            "ent_coeff": self.ent_coeff,
            "pi_samples": getattr(self, "pi_samples", 32),
            "kl_coeff": getattr(self, "kl_coeff", 0.0),
            "kl_beta": getattr(self, "kl_beta", 5.0),
            # NEW: causal bundle packed inline (or None if BN not initialized)
            "causal": self._bn_to_bundle(),
            "_schema_version": 1,
        }
        torch.save(ckpt, path)

    def load_policy(
        self,
        path: Union[str, Path],
        *,
        map_location: Optional[torch.device] = None,
        strict: bool = True,
        ensure_vbn: bool = True,
        warm_start: bool = False,
        warm_steps: int = 1024,
        mem: Optional[dict] = None,
    ):
        """
        Load actor/encoder + causal BN (bundle first, legacy fallback).
        Matches A2C_CC semantics for consistency.
        """
        path = self._ensure_pt_path(path)
        map_location = map_location or self.device

        ckpt = torch.load(path, map_location=map_location)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt

        self.load_state_dict(state, strict=strict)

        self.is_discrete = ckpt.get("is_discrete", getattr(self, "is_discrete", True))
        self.ent_coeff = ckpt.get("ent_coeff", getattr(self, "ent_coeff", 0.01))
        self.pi_samples = ckpt.get("pi_samples", getattr(self, "pi_samples", 32))
        self.kl_coeff = ckpt.get("kl_coeff", getattr(self, "kl_coeff", 0.0))
        self.kl_beta = ckpt.get("kl_beta", getattr(self, "kl_beta", 5.0))

        causal_bundle = ckpt.get("causal", None) if isinstance(ckpt, dict) else None
        if causal_bundle is not None:
            self._bn_from_bundle(causal_bundle)
        else:
            try:
                self.load_bn_params(path)
            except Exception:
                pass

        if ensure_vbn:
            self.ensure_bn(env=getattr(self, "env", None))
            if warm_start and getattr(self, "lp", None) is None:
                self.fit_bn_from_rollout(mem=mem, steps=warm_steps)

        return self
