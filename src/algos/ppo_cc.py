from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np, torch
from torch import nn

from src.algos.ppo import PPO

from src.algos.vbn_critic import VBNCritic


class PPO_CC(PPO, VBNCritic):
    def __init__(self, *args, cbn_kwargs: dict | None = None, **ppo_kwargs):
        PPO.__init__(self, *args, **ppo_kwargs)
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

    def _algo_update(self, mem):
        """
        PPO with causal critic variant (current minimal fixes):
          • Uses mem["logp"] (not "old_logp")
          • Removes value loss entirely (no critic forward/MSE)
        """
        T, N = mem["actions"].shape[:2]

        obs = self.flat(mem["obs"])
        act = self.flat(mem["actions"])
        old_logp = self.flat(mem["logp"])  # ← FIX 1: consistent key
        adv = self.flat(mem["advantages"])
        # returns = self.flat(mem["returns"])  # kept only for logging if you wish
        entropy_all = self.flat(mem["entropy"])

        # advantage normalisation
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        idxs = np.arange(T * N)
        for _ in range(self.n_epochs):
            np.random.shuffle(idxs)
            for start in range(0, T * N, self.batch_size):
                b = torch.as_tensor(
                    idxs[start : start + self.batch_size], device=self.device
                )
                states = self._encode(obs[b])

                if self.is_discrete:
                    logits = self.actor(states)
                    dist = self.dist_fn(logits)
                    new_logp = dist.log_prob(act[b])
                    batch_entropy = dist.entropy().mean()
                    extra_a = self.extra_actor_loss(states, logits)
                else:
                    mu = self.actor_mu(states)
                    dist = self.dist_fn(mu)
                    new_logp = dist.log_prob(act[b]).sum(-1)
                    batch_entropy = dist.entropy().sum(-1).mean()
                    extra_a = self.extra_actor_loss(states, mu)

                ratio = torch.exp(new_logp - old_logp[b])
                surr1 = ratio * adv[b]
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv[b]
                )
                base_actor_loss = -torch.min(surr1, surr2).mean()

                approx_kl = (old_logp[b] - new_logp).mean().item()
                clip_frac = ((ratio - 1.0).abs() > self.clip_eps).float().mean().item()

                # No critic forward / no value loss here  ← FIX 2
                actor_loss = base_actor_loss + extra_a
                loss = actor_loss - self.ent_coeff * batch_entropy

                self.optim.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.parameters(), 0.5).item()
                self.optim.step()

                self._log_update_metrics(
                    total_loss=loss.item(),
                    actor_loss=base_actor_loss.item(),
                    critic_loss=0.0,  # logged as zero
                    entropy=batch_entropy.item(),
                    adv_var=adv[b].var(unbiased=False).item(),
                    value_mse=0.0,
                    extra_actor_loss=extra_a.item(),
                    extra_critic_loss=0.0,  # no extra critic terms
                    grad_norm=grad_norm,
                    ppo_clip_frac=clip_frac,
                    ppo_approx_kl=approx_kl,
                    uses_causal_critic=1.0,
                )

        # epoch-level logs
        self._log_ac_metrics(
            mse=0.0,
            adv_var=adv.var(unbiased=False).item(),
            entropy=entropy_all.mean().item(),
        )

    # ---------- persistence ----------
    # ---------- persistence (single-file bundle: policy + VBN) ----------
    def save_policy(self, path: Union[str, Path]) -> None:
        """
        Save everything needed to restore PPO_CC in one file:
          • torch state_dict (actor/critic/encoder/etc.)
          • PPO hyperparams (clip_eps, vf_coeff, ent_coeff, n_epochs, batch_size)
          • discrete/continuous flags + causal knobs (pi_samples, kl_coeff, kl_beta)
          • VBN bundle (DAG/types/cards + learned params + backend choices)
        """
        payload = {
            "state_dict": self.state_dict(),
            "clip_eps": float(self.clip_eps),
            "vf_coeff": float(self.vf_coeff),
            "ent_coeff": float(self.ent_coeff),
            "n_epochs": int(self.n_epochs),
            "batch_size": int(self.batch_size),
            "is_discrete": bool(self.is_discrete),
            "pi_samples": int(getattr(self, "pi_samples", 32)),
            "kl_coeff": float(getattr(self, "kl_coeff", 0.0)),
            "kl_beta": float(getattr(self, "kl_beta", 5.0)),
            # pack the whole causal BN into the same file
            "vbn_bundle": self._bn_to_bundle(),
            "format": "ppo_cc+vbn@1",  # simple version tag
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
        - By default, strictly restores weights; set warm_start=True to allow missing keys.
        - Restores PPO hparams and the full causal BN (if present).
        - If no VBN bundle is present, falls back to legacy sidecars next to the .pt.
        """
        ckpt = torch.load(
            self._ensure_pt_path(path),
            map_location=self.device,
            weights_only=False,  # needed for custom VBN objects (PyTorch 2.6+)
        )

        # 1) weights
        self.load_state_dict(ckpt["state_dict"], strict=not warm_start)

        # 2) hparams
        self.clip_eps = ckpt.get("clip_eps", self.clip_eps)
        self.vf_coeff = ckpt.get("vf_coeff", self.vf_coeff)
        self.ent_coeff = ckpt.get("ent_coeff", self.ent_coeff)
        self.n_epochs = ckpt.get("n_epochs", getattr(self, "n_epochs", 4))
        self.batch_size = ckpt.get("batch_size", getattr(self, "batch_size", 64))
        self.is_discrete = ckpt.get("is_discrete", self.is_discrete)
        self.pi_samples = ckpt.get("pi_samples", getattr(self, "pi_samples", 32))
        self.kl_coeff = ckpt.get("kl_coeff", getattr(self, "kl_coeff", 0.0))
        self.kl_beta = ckpt.get("kl_beta", getattr(self, "kl_beta", 5.0))

        # 3) VBN
        bundle = ckpt.get("vbn_bundle", None)
        if bundle is not None:
            self._bn_from_bundle(bundle)
        else:
            # legacy fallback: .td + .bnmeta.json next to .pt (if you still have them)
            self.load_bn_params(path)

        if ensure_vbn:
            self._ensure_vbn_ready()
