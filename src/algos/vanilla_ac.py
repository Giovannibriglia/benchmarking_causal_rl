from pathlib import Path
from typing import Optional, Union

import torch
from torch import nn

from src.algos.base_actor_critic import BaseActorCritic


class VanillaAC(BaseActorCritic):
    """Vanilla Actor–Critic (TD(0) baseline), synchronous, built on BaseActorCritic.

    Differences vs your A2C:
      • Uses one-step TD target:  y_t = r_t + γ (1 - done_t) V(s_{t+1})
      • Actor uses advantage = (y_t - V(s_t))  (no GAE, no normalization)
      • Keeps entropy bonus and extra_* hooks for compatibility
    """

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
        # ---- fetch & flatten rollout tensors ----
        obs = self.flat(mem["obs"])

        actions = (
            self.flat(mem["actions"]).long()
            if self.is_discrete
            else self.flat(mem["actions"])
        )

        rewards = self.flat(mem["rewards"]).detach()

        # Support multiple done conventions: dones | (terminated or truncated)
        if "dones" in mem:
            dones = self.flat(mem["dones"]).to(rewards.dtype)
        else:
            term = self.flat(mem.get("terminated", torch.zeros_like(rewards))).to(
                rewards.dtype
            )
            trunc = self.flat(mem.get("truncated", torch.zeros_like(rewards))).to(
                rewards.dtype
            )
            dones = torch.clamp(term + trunc, 0, 1)

        # Prefer precomputed next values if provided; otherwise use next_obs if available
        if "next_values" in mem:
            next_values = self.flat(mem["next_values"]).detach()
        else:
            if "next_obs" in mem:
                with torch.no_grad():
                    next_latent = self._encode(self.flat(mem["next_obs"]))
                    next_values = self.critic(next_latent).squeeze(-1)
            else:
                # Fallback: no bootstrap (treat as terminal)
                next_values = torch.zeros_like(rewards)

        gamma = getattr(self, "gamma", 0.99)

        # ---- critic: TD(0) target & loss ----
        # y_t = r_t + γ (1 - done_t) V(s_{t+1})
        td_target = rewards + gamma * (1.0 - dones) * next_values
        td_target = td_target.detach()  # stop gradient through target

        latent = self._encode(obs)

        value = self.critic(latent).squeeze(-1)  # [B]
        td_error = td_target - value
        base_critic_loss = 0.5 * td_error.pow(2).mean()

        # Extra critic loss hook (kept for compatibility)
        extra_c = self.extra_critic_loss(latent, value)

        # ---- actor: policy gradient with baseline V(s) ----
        # advantage = y_t - V(s_t) (detach so actor doesn't backprop into critic)
        adv = td_error.detach()

        if self.is_discrete:
            logits = self.actor(latent)
            dist = self.dist_fn(logits)
            logp = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            extra_a = self.extra_actor_loss(latent, dist)
        else:
            mu = self.actor_mu(latent)
            dist = self.dist_fn(mu)
            # Sum log-probs across action dims
            logp = dist.log_prob(actions).sum(-1)
            # Sum entropies across dims, then mean over batch
            entropy = dist.entropy().sum(-1).mean()
            extra_a = self.extra_actor_loss(latent, dist)

        base_actor_loss = -(logp * adv).mean()

        # ---- total loss & update ----
        actor_loss = base_actor_loss + extra_a
        critic_loss = base_critic_loss + extra_c
        loss = actor_loss + self.vf_coeff * critic_loss - self.ent_coeff * entropy

        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        self.optim.step()

        # ---- metrics ----
        self._log_update_metrics(
            total_loss=float(loss.item()),
            actor_loss=float(base_actor_loss.item()),
            critic_loss=float(base_critic_loss.item()),
            entropy=float(entropy.item()),
            adv_var=float(adv.var(unbiased=False).item()),
            td_error_mean=float(td_error.mean().item()),
            td_error_std=float(td_error.std(unbiased=False).item()),
            value_mse=float(base_critic_loss.item()),
            extra_actor_loss=float(extra_a.item()),
            extra_critic_loss=float(extra_c.item()),
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

    def load_policy(
        self,
        path: Union[str, Path],
        *,
        map_location: Optional[torch.device] = None,
        strict: bool = True,
    ):
        path = self._ensure_pt_path(path)
        map_location = map_location or self.device

        ckpt = torch.load(path, map_location=map_location)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt

        self.load_state_dict(state, strict=strict)
        self.ent_coeff = ckpt.get("ent_coeff", getattr(self, "ent_coeff", 0.01))
        self.is_discrete = ckpt.get("is_discrete", getattr(self, "is_discrete", True))
        return self
