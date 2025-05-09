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

    # collect rollout is inherited
    def train(self):
        mem, L, R = self._collect_rollout()
        self.train_metrics.add(training_length=L, training_return=R)
        self._a2c_update(mem)

    def _a2c_update(self, mem):
        def flat(x):
            return x.reshape(-1, *x.shape[2:])

        # ---------- flatten T×N to B ----------
        # T, N = mem["actions"].shape[:2]

        obs = flat(mem["obs"])  # [B, obs_dim]
        actions = flat(mem["actions"])  # [B, ...]
        returns = flat(mem["returns"]).detach()  # target; no grad
        adv = flat(mem["advantages"]).detach()  # advantage; no grad

        # ---------- forward pass WITH gradient ----------
        latent = self.encoder(obs)
        if self.is_discrete:
            logits = self.actor(latent)
            dist = self.dist_fn(logits)
            logp = dist.log_prob(actions)  # [B]
            entropy = dist.entropy().mean()
        else:
            mu = self.actor_mu(latent)
            dist = self.dist_fn(mu)
            logp = dist.log_prob(actions).sum(-1)
            entropy = dist.entropy().sum(-1).mean()

        values = self.critic(latent).squeeze(-1)  # [B]

        base_actor_loss = -(logp * adv).mean()
        base_critic_loss = 0.5 * (returns - values).pow(2).mean()
        extra_a = self.extra_actor_loss()
        extra_c = self.extra_critic_loss()

        actor_loss = base_actor_loss + extra_a
        critic_loss = base_critic_loss + extra_c
        loss = actor_loss + self.vf_coeff * critic_loss - self.ent_coeff * entropy

        # ---------- optimiser step ----------
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        self.optim.step()

        # ---------- logging ----------
        self.train_metrics.add(
            total_loss=float(loss),
            actor_loss=float(base_actor_loss),
            extra_actor_loss=float(extra_a),
            critic_loss=float(base_critic_loss),
            extra_critic_loss=float(extra_c),
        )

        self._log_ac_metrics(
            mse=critic_loss.item(),
            adv_var=adv.var(unbiased=False).item(),
            entropy=entropy.item(),
        )

    def _get_action(self, obs):
        with torch.no_grad():
            dist = self.dist_fn(
                self.actor(self.encoder(obs))
                if self.is_discrete
                else self.actor_mu(self.encoder(obs))
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
