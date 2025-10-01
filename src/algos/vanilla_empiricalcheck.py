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


class VanillaAC_EmpiricalCheck(VanillaAC, VBNCritic):
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
        1) Fit/update the VBN on this rollout (probe only).
        2) Compute V^c and probe advantages; DO NOT overwrite mem['advantages'].
        3) Log probe metrics side-by-side with normal A2C metrics.
        """
        # 1) Fit/update BN on collected batch
        self._causal_update(mem)

        device = self.device
        obs = self.flat(mem["obs"]).to(device)  # [B, feat]
        nxt_obs = obs.new_empty(obs.shape)  # will rebuild below if needed
        # actions = self.flat(mem["actions"])
        returns = self.flat(mem["returns"]).to(device)  # [B]
        dones = self.flat(mem["dones"]).to(device).float()  # [B]
        rewards = self.flat(mem["rewards"]).to(device)  # [B]

        # Rebuild next_obs flat from rollout tail (we can approximate with shift)
        # (Because mem doesn't store "next_obs" once flattened; approximate by shifting obs)
        # This is fine for TD-like diagnostics.
        nxt_obs[:-1] = obs[1:]
        nxt_obs[-1] = obs[-1]

        # 2) VBN value and advantages (probe)
        with torch.no_grad():
            states, dist = self._probe_states_and_dist(obs)
            Vc, _ = self._v_causal(states, dist)  # [B]
            Vc = Vc.view(-1)

            # next-step for TD residual using VBN
            nxt_states, _ = self._probe_states_and_dist(nxt_obs)
            Vc_next, _ = self._v_causal(
                nxt_states, dist
            )  # reuse current dist for simplicity
            Vc_next = Vc_next.view(-1)

            A_probe = returns - Vc  # causal advantages (probe)
            delta_c = rewards + self.gamma * (1.0 - dones) * Vc_next - Vc

            # Baselines from NN critic for side-by-side (already used for training)
            # Recompute quickly here to keep code local:
            latent = self._encode(obs)
            """if self.is_discrete:
                logits = self.actor(latent)
                dist_nn = self.dist_fn(logits)
            else:
                mu = self.actor_mu(latent)
                dist_nn = self.dist_fn(mu)"""
            V_nn = self.critic(latent).squeeze(-1)  # [B]
            A_nn = returns - V_nn

            # Next V_nn for TD residuals
            latent_next = self._encode(nxt_obs)
            V_nn_next = self.critic(latent_next).squeeze(-1)
            delta_nn = rewards + self.gamma * (1.0 - dones) * V_nn_next - V_nn

        # 3) Log diagnostics (both sides)
        # — NN critic (training one)
        self.train_metrics.add(
            adv_var=float(A_nn.var(unbiased=False).item()),
            value_mse=mse(returns, V_nn),
            v_explained_variance=explained_variance(returns, V_nn),
            td_mean=float(delta_nn.mean().item()),
            td_std=float(delta_nn.std(unbiased=False).item()),
            v_corr=pearson_corr(returns, V_nn),
            v_spearman=spearman_corr(returns, V_nn),
            v_mi=mutual_information(
                returns, V_nn, n_bins=20, strategy="quantile", normalized=True
            ),
            v_wass=wasserstein_1d(returns, V_nn),
            v_kl=kl_divergence_hist(returns, V_nn, n_bins=20, strategy="quantile"),
            v_js=js_divergence_hist(returns, V_nn, n_bins=20, strategy="quantile"),
        )

        # — VBN probe (prefixed)
        self.train_metrics.add(
            causal_adv_var=float(A_probe.var(unbiased=False).item()),
            causal_value_mse=mse(returns, Vc),
            causal_v_explained_variance=explained_variance(returns, Vc),
            causal_td_mean=float(delta_c.mean().item()),
            causal_td_std=float(delta_c.std(unbiased=False).item()),
            causal_v_corr=pearson_corr(returns, Vc),
            causal_v_spearman=spearman_corr(returns, Vc),
            causal_v_mi=mutual_information(
                returns, Vc, n_bins=20, strategy="quantile", normalized=True
            ),
            causal_v_wass=wasserstein_1d(returns, Vc),
            causal_v_kl=kl_divergence_hist(returns, Vc, n_bins=20, strategy="quantile"),
            causal_v_js=js_divergence_hist(returns, Vc, n_bins=20, strategy="quantile"),
        )

        # Optional: compact summary of advantage distribution for both
        self.train_metrics.add(
            causal_adv_mean=float(A_probe.mean().item()),
            causal_adv_std=float(A_probe.std(unbiased=False).item()),
            causal_adv_min=float(A_probe.min().item()),
            causal_adv_max=float(A_probe.max().item()),
        )

    def save_policy(self, path: Union[str, Path]) -> None:
        """
        Save two artifacts:
          1) VanillaAC policy weights at `path`
          2) Causal critic (VBN) as a single-file bundle named `causal_critic.pt`
             in the same directory.

        The critic bundle contains DAG/types/cards, learned params, and backend
        config produced by `_bn_to_bundle()`; it can be restored via `_bn_from_bundle`.
        """
        base_path = self._ensure_pt_path(path)
        critic_path = self._ensure_pt_path(
            str(path).replace("episode", "causal_critic_episode")
        )

        # --- (1) save the VanillaAC policy (unchanged format) ---
        torch.save(
            {
                "state_dict": self.state_dict(),
                "vf_coeff": self.vf_coeff,
                "ent_coeff": self.ent_coeff,
                "is_discrete": self.is_discrete,
            },
            base_path,
        )

        # --- (2) save the causal critic as a single-file bundle ---
        bundle = self._bn_to_bundle()
        assert bundle is not None, "VBN critic not initialized/fitted; nothing to save."
        torch.save(bundle, critic_path)
