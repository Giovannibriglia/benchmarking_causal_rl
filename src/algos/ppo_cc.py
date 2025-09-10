from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np, torch
from torch import nn

from src.algos.cbn_critic import VBNCritic
from src.algos.ppo import PPO


class PPO_CC(PPO, VBNCritic):
    def __init__(self, *args, cbn_kwargs: dict | None = None, **ppo_kwargs):
        PPO.__init__(self, *args, **ppo_kwargs)
        VBNCritic.__init__(self, **(cbn_kwargs or {}))

    def _post_update(self, mem):
        self._post_update_fill_adv_and_logp(mem)

    def _algo_update(self, mem):
        # (unchanged from the previous answer, but uses mem["advantages"] and no critic loss)
        T, N = mem["actions"].shape[:2]
        obs = self.flat(mem["obs"])
        act = self.flat(mem["actions"])
        old_logp = mem["old_logp"]
        adv = self.flat(mem["advantages"])
        entropy_all = self.flat(mem["entropy"])

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

                extra_c = self.extra_critic_loss(states, torch.zeros_like(new_logp))
                loss = base_actor_loss + extra_a - self.ent_coeff * batch_entropy

                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                self.optim.step()

                self.train_metrics.add(
                    total_loss=float(loss),
                    actor_loss=float(base_actor_loss),
                    extra_actor_loss=float(extra_a),
                    critic_loss=0.0,
                    extra_critic_loss=float(extra_c),
                )

        self._log_ac_metrics(
            mse=0.0,
            adv_var=adv.var(unbiased=False).item(),
            entropy=entropy_all.mean().item(),
        )

    def save_policy(self, path: Union[str, Path]) -> None:
        torch.save(
            {
                "state_dict": self.state_dict(),
                "clip_eps": self.clip_eps,
                "vf_coeff": self.vf_coeff,
                "ent_coeff": self.ent_coeff,
                "is_discrete": self.is_discrete,
                "pi_samples": getattr(self, "pi_samples", 32),
                "kl_coeff": getattr(self, "kl_coeff", 0.0),
                "kl_beta": getattr(self, "kl_beta", 5.0),
            },
            self._ensure_pt_path(path),
        )
        self.save_bn_params(path)

    def load_policy(self, path: Union[str, Path]) -> None:
        ckpt = torch.load(self._ensure_pt_path(path), map_location=self.device)
        self.load_state_dict(ckpt["state_dict"])
        self.clip_eps = ckpt.get("clip_eps", self.clip_eps)
        self.vf_coeff = ckpt.get("vf_coeff", self.vf_coeff)
        self.ent_coeff = ckpt.get("ent_coeff", self.ent_coeff)
        self.is_discrete = ckpt.get("is_discrete", self.is_discrete)
        self.pi_samples = ckpt.get("pi_samples", getattr(self, "pi_samples", 32))
        self.kl_coeff = ckpt.get("kl_coeff", getattr(self, "kl_coeff", 0.0))
        self.kl_beta = ckpt.get("kl_beta", getattr(self, "kl_beta", 5.0))
        self.load_bn_params(path)
