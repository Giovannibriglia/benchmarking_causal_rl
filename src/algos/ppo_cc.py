from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch import nn

from src.algos import PPO


class CausalCriticPPO(PPO):
    def __init__(
        self,
        *args,
        new_param=1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.new_param = new_param
        # You can add custom initialization logic here

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

    def causal_prior_probs(self, states: torch.Tensor) -> torch.Tensor:
        """
        Dummy prior – replace with your own CPD.
        Returns categorical probabilities over actions for each state.
          states : [B, feat]
          out    : [B, n_actions]  (rows sum to 1)
        """

        """
        B = states.size(0)
        nA = self.env.action_space.n  # use inside the class, just shown here
        probs = torch.rand(B, nA, device=states.device)
        return probs
        """

        angle = states[:, 2]
        p_left = torch.where(angle < 0, 0.8, 0.2).to(states)
        return torch.stack((p_left, 1 - p_left), dim=-1)

    def _post_update(self, mem):
        """
        Compute causal-baseline advantages for PPO.
        Expects mem['obs'], ['actions'], ['returns'] shaped [T, N, ...].
        """
        if not self.is_discrete:  # only for categorical actions
            return

        # ---------- flatten rollout tensors ----------
        states = mem["obs"].reshape(-1, *mem["obs"].shape[2:])  # [B, obs_dim]
        actions = mem["actions"].reshape(-1).long()  # [B]
        returns = mem["returns"].reshape(-1).detach()  # [B]

        # ---------- prepare encoder input ----------
        # if encoder is an Embedding (discrete obs), keep long indices
        if isinstance(self.encoder, nn.Embedding):
            enc_in = states.long()
        else:
            enc_in = states.float()
        latent = self.encoder(enc_in)  # [B, hidden]

        # ---------- get V(s) (collapse Q-vector if needed) ----------
        crit_out = self.critic(latent)  # [B,1] or [B,nA]
        if crit_out.dim() == 2 and crit_out.size(1) > 1:  # Q-vector → scalar
            state_values = crit_out.mean(dim=1)
        else:
            state_values = crit_out.squeeze(-1)  # [B]

        # ---------- causal prior log-probs ----------
        with torch.no_grad():
            q_probs = self.causal_prior_probs(states)  # [B, nA]
            old_logp = torch.log(q_probs[torch.arange(len(actions)), actions])
        old_logp = old_logp.detach()

        # ---------- causal advantages ----------
        adv = returns - state_values.detach()
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        # ---------- stash for PPO update ----------
        mem["causal_advantages"] = adv
        mem["old_logp"] = old_logp

    def _ppo_update(self, mem):
        # T, N = mem["actions"].shape[:2]
        obs = self.flat(mem["obs"])
        act = self.flat(mem["actions"])
        old_value = self.flat(mem["values"])
        returns = self.flat(mem["returns"])
        adv = mem["causal_advantages"]  # computed in post_update
        old_logp = mem["old_logp"]
        entropy_all = self.flat(mem["entropy"])

        # normalise again just in case
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        idxs = np.arange(len(act))
        for _ in range(self.n_epochs):
            np.random.shuffle(idxs)
            for st in range(0, len(act), self.batch_size):
                b = torch.as_tensor(idxs[st : st + self.batch_size], device=self.device)

                lat = self.encoder(obs[b])
                logits = self.actor(lat)
                dist = self.dist_fn(logits)
                logp = dist.log_prob(act[b])  # [batch]
                entropy = dist.entropy().mean()

                value = self.critic(lat).squeeze(-1)
                ratio = torch.exp(logp - old_logp[b])
                s1, s2 = (
                    ratio * adv[b],
                    torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv[b],
                )
                base_actor_loss = -torch.min(s1, s2).mean()
                base_critic_loss = 0.5 * (returns[b] - value).pow(2).mean()

                extra_a = self.extra_actor_loss(lat, logits)
                extra_c = self.extra_critic_loss(lat, value)

                loss = (
                    (base_actor_loss + extra_a)
                    + self.vf_coeff * (base_critic_loss + extra_c)
                    - self.ent_coeff * entropy
                )

                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                self.optim.step()

                # metric logging
                self.train_metrics.add(
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
