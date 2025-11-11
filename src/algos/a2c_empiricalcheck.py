from __future__ import annotations

from pathlib import Path
from typing import Union

import torch
from torch import nn

from src.algos.a2c import A2C
from src.algos.new_vbn_critic import VBNCritic
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


class A2C_EmpiricalCheck(A2C, VBNCritic):
    """
    Train exactly like A2C (actor uses NN-critic advantages).
    In parallel, fit a VBN on the *same* rollout and log probe metrics:
      - probe_adv_var, probe_ev, probe_mse, probe_td_mean/std, probe_corr, probe_spearman
    The rollout/actor/critic updates are *unchanged*.
    """

    def __init__(self, *args, cbn_kwargs: dict | None = None, **a2c_kwargs):
        A2C.__init__(self, *args, **a2c_kwargs)  # usual A2C
        VBNCritic.__init__(self, **(cbn_kwargs or {}))  # causal tools (probe only)

    # === helper mirrors VBNCritic's internal logic but WITHOUT touching mem ===
    def _probe_states_and_dist(self, obs_flat: torch.Tensor):
        """
        Returns:
            states: [B, feat_for_bn] for BN
            dist  : current policy distribution (Categorical/Normal)
        """
        if isinstance(self.encoder, nn.Embedding):
            # Discrete observation index case
            if obs_flat.ndim == 2 and obs_flat.shape[1] == 1:
                obs_idx = obs_flat.squeeze(-1)  # [B]
            else:
                obs_idx = obs_flat[..., 0]
            states = obs_idx.view(-1, 1).long()  # BN expects [B,1]
            latent = self._encode(obs_idx.view(-1))  # policy latent
        else:
            states = obs_flat.view(obs_flat.shape[0], -1).float()
            latent = self._encode(states)

        dist = self._policy_dist_from_latent(latent)  # from VBNCritic
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
        Empirical check:
          (1) save base A2C policy to `path`
          (2) save VBN critic-only bundle as `<same_dir>/causal_critic.pt`
        """
        critic_path = self._ensure_pt_path(
            str(path).replace("episode", "causal_critic_episode")
        )
        base_path = self._ensure_pt_path(path)

        # (1) base A2C payload
        torch.save(
            {
                "state_dict": self.state_dict(),
                "vf_coeff": float(self.vf_coeff),
                "ent_coeff": float(self.ent_coeff),
                "is_discrete": bool(self.is_discrete),
                "format": "a2c_empirical@1",
            },
            base_path,
        )

        # (2) critic-only bundle
        bundle = self._bn_to_bundle()
        assert bundle is not None, "VBN critic not initialized/fitted; nothing to save."
        torch.save(bundle, critic_path)
