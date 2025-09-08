from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch import nn

from src.algos.base_actor_critic import BaseActorCritic


class PPO(BaseActorCritic):
    def __init__(
        self,
        *args,
        clip_eps=0.2,
        vf_coeff=0.5,
        ent_coeff=0.01,
        n_epochs=4,
        batch_size=64,
        **kw,
    ):
        super().__init__(*args, **kw)
        self.clip_eps, self.vf_coeff, self.ent_coeff = clip_eps, vf_coeff, ent_coeff
        self.n_epochs, self.batch_size = n_epochs, batch_size

    def _algo_update(self, mem):
        T, N = mem["actions"].shape[:2]

        obs = self.flat(mem["obs"])
        act = self.flat(mem["actions"])
        old_logp = self.flat(mem["logp"])
        old_value = self.flat(mem["values"])
        returns = self.flat(mem["returns"])
        adv = self.flat(mem["advantages"])
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

                value = self.critic(states).squeeze(-1)
                extra_c = self.extra_critic_loss(states, value)

                ratio = torch.exp(new_logp - old_logp[b])
                surr1 = ratio * adv[b]
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv[b]
                )

                base_actor_loss = -torch.min(surr1, surr2).mean()
                base_critic_loss = 0.5 * (returns[b] - value).pow(2).mean()

                actor_loss = base_actor_loss + extra_a
                critic_loss = base_critic_loss + extra_c
                loss = (
                    actor_loss
                    + self.vf_coeff * critic_loss
                    - self.ent_coeff * batch_entropy
                )

                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                self.optim.step()

                self.train_metrics.add(
                    total_loss=float(loss),
                    actor_loss=float(base_actor_loss),
                    extra_actor_loss=float(extra_a),
                    critic_loss=float(base_critic_loss),
                    extra_critic_loss=float(extra_c),
                )

        # ---------- log extra metrics ----------
        value_mse = (returns - old_value).pow(2).mean().item()
        self._log_ac_metrics(
            value_mse, adv.var(unbiased=False).item(), entropy_all.mean().item()
        )

    def _get_action(self, obs):
        with torch.no_grad():
            latent = self._encode(obs)
            dist = self.dist_fn(
                self.actor(latent) if self.is_discrete else self.actor_mu(latent)
            )
            return dist.sample()

    # ---------- persistence ----------
    def save_policy(self, path: Union[str, Path]) -> None:
        torch.save(
            {
                "state_dict": self.state_dict(),
                "clip_eps": self.clip_eps,
                "vf_coeff": self.vf_coeff,
                "ent_coeff": self.ent_coeff,
                "is_discrete": self.is_discrete,
            },
            self._ensure_pt_path(path),
        )

    def load_policy(self, path: Union[str, Path]) -> None:
        ckpt = torch.load(self._ensure_pt_path(path), map_location=self.device)
        self.load_state_dict(ckpt["state_dict"])
        # hyper‑params useful if you want to inspect them later
        self.clip_eps = ckpt.get("clip_eps", 0.2)
        self.vf_coeff = ckpt.get("vf_coeff", 0.5)
        self.ent_coeff = ckpt.get("ent_coeff", 0.01)
        self.is_discrete = ckpt.get("is_discrete", False)
