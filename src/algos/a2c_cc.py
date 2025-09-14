from __future__ import annotations

from pathlib import Path
from typing import Union

import torch
from torch import nn

from src.algos.a2c import A2C
from src.algos.vbn_critic import VBNCritic


class A2C_CC(A2C, VBNCritic):
    """
    A2C variant that replaces the neural critic with a Causal Bayesian Network (CBN).
    - Advantages come from CBN:  adv = returns - V_causal(x)
    - No critic MSE term (set vf_coeff=0.0; we ignore it anyway)
    - Optional KL(pi || pi_causal_prior) (discrete single-head only):
        pi_c(a|x) ∝ exp(beta * Q_c(x,a))
        weighted by kl_coeff
    """

    def __init__(self, *args, cbn_kwargs: dict | None = None, **a2c_kwargs):
        # initialize the base algo first (so nets/device exist)
        A2C.__init__(self, *args, **a2c_kwargs)
        # then initialize the CBN mixin
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

    # ── main optimization step (actor only; no neural critic) ──
    def _algo_update(self, mem):
        obs = self.flat(mem["obs"])
        actions = self.flat(mem["actions"])
        adv = self.flat(mem["advantages"]).detach()

        # normalize advantages
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        # encode once
        latent = self._encode(obs)

        if self.is_discrete:
            logits = self.actor(latent)
            dist = self.dist_fn(logits)
            logp = dist.log_prob(actions.view(-1).long())  # [B]
            entropy = dist.entropy().mean()
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

        # base actor loss (no critic mse in CC)
        base_actor_loss = -(logp * adv).mean()

        # optional KL(pi || pi_causal_prior) for discrete
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

        extra_c = self.extra_critic_loss(latent, torch.zeros_like(logp))
        loss = base_actor_loss + extra_a + kl_loss - self.ent_coeff * entropy

        self.optim.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.parameters(), 0.5).item()
        self.optim.step()

        self._log_update_metrics(
            total_loss=float(loss),
            actor_loss=float(base_actor_loss),
            critic_loss=0.0,
            entropy=float(entropy),
            adv_var=adv.var(unbiased=False).item(),
            value_mse=0.0,
            extra_actor_loss=float(extra_a),
            extra_critic_loss=float(extra_c),
            grad_norm=grad_norm,
            kl_loss=float(kl_loss),
            uses_causal_critic=1.0,
        )

    # ── persistence ──
    def save_policy(self, path: Union[str, Path]) -> None:
        torch.save(
            {
                "state_dict": self.state_dict(),
                "is_discrete": self.is_discrete,
                "ent_coeff": self.ent_coeff,
                "pi_samples": getattr(self, "pi_samples", 32),
                "kl_coeff": getattr(self, "kl_coeff", 0.0),
                "kl_beta": getattr(self, "kl_beta", 5.0),
            },
            self._ensure_pt_path(path),
        )
        self.save_bn_params(path)  # <— save BN

    def load_policy(self, path: Union[str, Path]) -> None:
        ckpt = torch.load(self._ensure_pt_path(path), map_location=self.device)
        self.load_state_dict(ckpt["state_dict"])
        self.is_discrete = ckpt.get("is_discrete", self.is_discrete)
        self.ent_coeff = ckpt.get("ent_coeff", self.ent_coeff)
        self.pi_samples = ckpt.get("pi_samples", getattr(self, "pi_samples", 32))
        self.kl_coeff = ckpt.get("kl_coeff", getattr(self, "kl_coeff", 0.0))
        self.kl_beta = ckpt.get("kl_beta", getattr(self, "kl_beta", 5.0))
        self.load_bn_params(path)  # <— load BN
