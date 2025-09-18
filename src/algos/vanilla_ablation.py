from __future__ import annotations

from pathlib import Path
from typing import Union

import torch
from torch import nn

from src.algos.utils import (
    explained_variance,
    js_divergence_hist,
    kl_divergence_hist,
    mse,
    mutual_information,
    pearson_corr,
    spearman_corr,
    wasserstein_1d,
)

from src.algos.vanilla_ac import VanillaAC
from src.algos.vbn_critic import VBNCritic


class VanillaAC_Ablation(VanillaAC, VBNCritic):
    """
    Train EXACTLY like VanillaAC (actor uses TD(0) advantage, NN-critic trains with TD loss).
    In parallel, fit a VBN on the same rollout and log 'probe_*' metrics:

      • probe_adv_var           – Var(G - V^c)   (here we use TD target y_t as in Vanilla AC)
      • probe_value_mse         – MSE(y_t, V^c)
      • probe_explained_variance– EV(y_t, V^c)
      • probe_td_mean/std       – TD residual under V^c (r + γ(1-d)V^c(s') - V^c(s))
      • probe_v_corr/spearman   – correlation of V^c with y_t (and rank corr)

    Notes:
    - We DO NOT overwrite mem['advantages'] (VanillaAC uses its own TD(0) advantage).
    - We reconstruct next_obs flat via a shift (since rollout stores only obs). Good enough for diagnostics.
    """

    def __init__(self, *args, cbn_kwargs: dict | None = None, **vac_kwargs):
        VanillaAC.__init__(self, *args, **vac_kwargs)  # standard VAC
        VBNCritic.__init__(self, **(cbn_kwargs or {}))  # causal tools (probe)

    # ---- helpers to build BN states and policy dist without touching training path ----
    def _probe_states_and_dist(self, obs_flat: torch.Tensor):
        """
        Returns:
            states: [B, feat] for BN (long if Embedding case, else float)
            dist  : current policy distribution (Categorical/Normal)
        """
        if isinstance(self.encoder, nn.Embedding):
            if obs_flat.ndim == 2 and obs_flat.shape[1] == 1:
                obs_idx = obs_flat.squeeze(-1)  # [B]
            else:
                obs_idx = obs_flat[..., 0]
            states = obs_idx.view(-1, 1).long()  # BN expects integer feat
            latent = self._encode(obs_idx.view(-1))  # policy latent
        else:
            states = obs_flat.view(obs_flat.shape[0], -1).float()
            latent = self._encode(states)

        dist = self._policy_dist_from_latent(latent)  # provided by VBNCritic
        return states, dist

    def _post_update(self, mem):
        """
        1) Update VBN on this rollout (probe only).
        2) Compute NN and causal TD(0) targets and log diagnostics.
        """

        # Update BN (probe only, no grads)
        with torch.no_grad():
            self._causal_update(mem)

        device = self.device
        obs = self.flat(mem["obs"]).to(device)  # [B, feat]
        rewards = self.flat(mem["rewards"]).to(device)  # [B]

        if "dones" in mem:
            dones = self.flat(mem["dones"]).to(device).float()
        else:
            term = (
                self.flat(mem.get("terminated", torch.zeros_like(rewards)))
                .to(device)
                .float()
            )
            trunc = (
                self.flat(mem.get("truncated", torch.zeros_like(rewards)))
                .to(device)
                .float()
            )
            dones = torch.clamp(term + trunc, 0, 1)

        # reconstruct next_obs (diagnostic only)
        nxt_obs = obs.clone()
        nxt_obs[:-1] = obs[1:]
        nxt_obs[-1] = obs[-1]

        gamma = getattr(self, "gamma", 0.99)

        with torch.no_grad():
            # ---- NN critic and NN TD target y_t^nn ----
            lat = self._encode(obs)
            lat_next = self._encode(nxt_obs)
            v_nn = self.critic(lat).squeeze(-1)
            v_nn_next = self.critic(lat_next).squeeze(-1)
            y_t_nn = rewards + gamma * (1.0 - dones) * v_nn_next
            delta_nn = y_t_nn - v_nn

            # ---- Policy distributions ----
            dist = self._policy_dist_from_latent(lat)
            dist_next = self._policy_dist_from_latent(lat_next)

            # ---- Build BN states ----
            if isinstance(self.encoder, nn.Embedding):
                obs_idx = (
                    obs.squeeze(-1)
                    if (obs.ndim == 2 and obs.shape[1] == 1)
                    else obs[..., 0]
                )
                nxt_obs_idx = (
                    nxt_obs.squeeze(-1)
                    if (nxt_obs.ndim == 2 and nxt_obs.shape[1] == 1)
                    else nxt_obs[..., 0]
                )
                states = obs_idx.view(-1, 1).long()
                states_next = nxt_obs_idx.view(-1, 1).long()
            else:
                states = obs.view(obs.shape[0], -1).float()
                states_next = nxt_obs.view(nxt_obs.shape[0], -1).float()

            # ---- Causal values and causal TD target y_t^c ----
            v_c, _ = self._v_causal(states, dist)
            v_c_next, _ = self._v_causal(states_next, dist_next)
            v_c = v_c.view(-1)
            v_c_next = v_c_next.view(-1)

            y_t_c = rewards + gamma * (1.0 - dones) * v_c_next
            a_probe = y_t_c - v_c
            delta_c = y_t_c - v_c  # same as a_probe

        # ---- Log: NN critic ----
        self.train_metrics.add(
            adv_var=float(delta_nn.var(unbiased=False).item()),
            value_mse=float(((y_t_nn - v_nn) ** 2).mean().item()),
            v_explained_variance=explained_variance(y_t_nn, v_nn),
            td_mean=float(delta_nn.mean().item()),
            td_std=float(delta_nn.std(unbiased=False).item()),
            v_corr=pearson_corr(y_t_nn, v_nn),
            v_spearman=spearman_corr(y_t_nn, v_nn),
            v_mi=mutual_information(
                y_t_nn, v_nn, n_bins=20, strategy="quantile", normalized=True
            ),
            v_wass=wasserstein_1d(y_t_nn, v_nn),
            v_kl=kl_divergence_hist(y_t_nn, v_nn, n_bins=20, strategy="quantile"),
            v_js=js_divergence_hist(y_t_nn, v_nn, n_bins=20, strategy="quantile"),
        )

        # ---- Log: causal critic (causal-consistent) ----
        self.train_metrics.add(
            causal_adv_var=float(a_probe.var(unbiased=False).item()),
            causal_value_mse=mse(y_t_c, v_c),
            causal_v_explained_variance=explained_variance(y_t_c, v_c),
            causal_td_mean=float(delta_c.mean().item()),
            causal_td_std=float(delta_c.std(unbiased=False).item()),
            causal_v_corr=pearson_corr(y_t_c, v_c),
            causal_v_spearman=spearman_corr(y_t_c, v_c),
            causal_adv_mean=float(a_probe.mean().item()),
            causal_adv_std=float(a_probe.std(unbiased=False).item()),
            causal_adv_min=float(a_probe.min().item()),
            causal_adv_max=float(a_probe.max().item()),
            causal_v_mi=mutual_information(
                y_t_c, v_c, n_bins=20, strategy="quantile", normalized=True
            ),
            causal_v_wass=wasserstein_1d(y_t_c, v_c),
            causal_v_kl=kl_divergence_hist(y_t_c, v_c, n_bins=20, strategy="quantile"),
            causal_v_js=js_divergence_hist(y_t_c, v_c, n_bins=20, strategy="quantile"),
        )

    # ---------- persistence (same as VanillaAC; probe has no extra state here) ----------
    def save_policy(self, path: Union[str, Path]) -> None:
        torch.save(
            {
                "state_dict": self.state_dict(),
                "is_discrete": self.is_discrete,
                "ent_coeff": self.ent_coeff,
            },
            self._ensure_pt_path(path),
        )

    def load_policy(self, path: Union[str, Path]) -> None:
        ckpt = torch.load(self._ensure_pt_path(path), map_location=self.device)
        self.load_state_dict(ckpt["state_dict"])
        self.is_discrete = ckpt.get("is_discrete", self.is_discrete)
        self.ent_coeff = ckpt.get("ent_coeff", self.ent_coeff)
