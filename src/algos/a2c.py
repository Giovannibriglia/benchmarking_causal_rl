from pathlib import Path
from typing import Union

import torch
from torch import nn

from src.algos.base_actor_critic import BaseActorCritic


class A2C(BaseActorCritic):
    """Synchronous Advantage Actor‑Critic built on BaseActorCritic."""

    def __init__(
        self,
        *args,
        vf_coeff: float = 0.5,
        ent_coeff: float = 0.01,
        lr: float = 7e-4,
        **kw,
    ):
        super().__init__(*args, lr=lr, **kw)
        self.vf_coeff, self.ent_coeff = vf_coeff, ent_coeff

    def _algo_update(self, mem):
        obs = self.flat(mem["obs"])
        actions = (
            self.flat(mem["actions"]).long()
            if self.is_discrete
            else self.flat(mem["actions"])
        )
        returns = self.flat(mem["returns"]).detach()
        adv = self.flat(mem["advantages"]).detach()
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        latent = self._encode(obs)

        if self.is_discrete:
            logits = self.actor(latent)
            dist = self.dist_fn(logits)
            logp = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            extra_a = self.extra_actor_loss(latent, dist)
        else:
            mu = self.actor_mu(latent)
            dist = self.dist_fn(mu)
            logp = dist.log_prob(actions).sum(-1)
            entropy = dist.entropy().sum(-1).mean()
            extra_a = self.extra_actor_loss(latent, dist)

        value = self.critic(latent).squeeze(-1)  # [B]
        extra_c = self.extra_critic_loss(latent, value)

        base_actor_loss = -(logp * adv).mean()
        base_critic_loss = 0.5 * (returns - value).pow(2).mean()

        actor_loss = base_actor_loss + extra_a
        critic_loss = base_critic_loss + extra_c
        loss = actor_loss + self.vf_coeff * critic_loss - self.ent_coeff * entropy

        # ---------- step ----------
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        self.optim.step()

        # ---------- metrics ----------
        self.train_metrics.add(
            total_loss=float(loss.item()),
            actor_loss=float(base_actor_loss),
            extra_actor_loss=float(extra_a),
            critic_loss=float(base_critic_loss),
            extra_critic_loss=float(extra_c),
        )
        self._log_ac_metrics(
            mse=base_critic_loss.item(),
            adv_var=adv.var(unbiased=False).item(),
            entropy=entropy.item(),
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
                "vf_coeff": self.vf_coeff,
                "ent_coeff": self.ent_coeff,
                "is_discrete": self.is_discrete,
            },
            self._ensure_pt_path(path),
        )

    def load_policy(self, path: Union[str, Path]) -> None:
        ckpt = torch.load(self._ensure_pt_path(path), map_location=self.device)
        self.load_state_dict(ckpt["state_dict"])
        self.vf_coeff = ckpt.get("vf_coeff", 0.5)
        self.ent_coeff = ckpt.get("ent_coeff", 0.01)
