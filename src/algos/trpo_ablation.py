from __future__ import annotations

import torch
from torch import nn

from src.algos.trpo import TRPO
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
from src.algos.vbn_critic import VBNCritic


class TRPO_Ablation(TRPO, VBNCritic):
    """
    Train EXACTLY like TRPO (policy via natgrad step, critic regresses to returns).
    In parallel, fit a VBN on the same rollout and log 'probe_*' diagnostics:
      • probe_adv_var              Var(G - V^c)
      • probe_value_mse            MSE(G, V^c)
      • probe_explained_variance   EV(G, V^c)
      • probe_td_mean/std          TD residual using V^c: r + γ(1-d)V^c(s') - V^c(s)
      • probe_v_corr/spearman      Corrs between V^c and G
      • probe_v_mi                 Mutual information I(G; V^c) (normalized)
    NOTE: We do NOT modify advantages the policy uses; this is a passive ablation.
    """

    def __init__(self, *args, cbn_kwargs: dict | None = None, **trpo_kwargs):
        TRPO.__init__(self, *args, **trpo_kwargs)
        VBNCritic.__init__(self, **(cbn_kwargs or {}))

    # --- helper: build BN states and current policy dist without touching training path
    def _probe_states_and_dist(self, obs_flat: torch.Tensor):
        """
        Returns:
            states: [B, feat] for BN (long if Embedding case, else float)
            dist  : current policy distribution (Categorical/Normal)
        """
        if isinstance(self.encoder, nn.Embedding):
            # discrete observation index
            if obs_flat.ndim == 2 and obs_flat.shape[1] == 1:
                obs_idx = obs_flat.squeeze(-1)  # [B]
            else:
                obs_idx = obs_flat[..., 0]
            states = obs_idx.view(-1, 1).long()  # BN expects integer feat
            latent = self._encode(obs_idx.view(-1))  # policy latent
        else:
            states = obs_flat.view(obs_flat.shape[0], -1).float()
            latent = self._encode(states)

        # VBNCritic provides a policy-dist shim that calls TRPO._distribution(latent)
        dist = self._policy_dist_from_latent(latent)
        return states, dist

    def _post_update(self, mem):
        """
        1) Fit/update BN on this rollout (probe only).
        2) Compute V^c(s) and diagnostics vs TRPO targets (returns).
        """
        # 1) Update BN with this batch (uses obs/actions/returns)
        self._causal_update(mem)

        device = self.device
        obs = self.flat(mem["obs"]).to(device)  # [B, feat]
        rewards = self.flat(mem["rewards"]).to(device)  # [B]
        returns = self.flat(mem["returns"]).to(device)  # [B]  (TRPO target)
        if "dones" in mem:
            dones = self.flat(mem["dones"]).to(device).float()
        else:
            # robust fallback
            dones = torch.zeros_like(rewards)

        # reconstruct next_obs by shift (diagnostic only)
        nxt_obs = obs.clone()
        nxt_obs[:-1] = obs[1:]
        nxt_obs[-1] = obs[-1]

        gamma = getattr(self, "gamma", 0.99)

        # --- Baseline (neural critic) for side-by-side
        with torch.no_grad():
            lat = self._encode(obs)
            v_nn = self.critic(lat).squeeze(-1)  # [B]
            lat_next = self._encode(nxt_obs)
            v_nn_next = self.critic(lat_next).squeeze(-1)
            # TD residual using NN critic (diagnostic)
            delta_nn = rewards + gamma * (1.0 - dones) * v_nn_next - v_nn

        # --- Probe: compute V^c, TD-like residuals, MI, etc.
        with torch.no_grad():
            states, dist = self._probe_states_and_dist(obs)
            v_c, _ = self._v_causal(states, dist)  # [B]
            v_c = v_c.view(-1)

            states_next, _ = self._probe_states_and_dist(nxt_obs)
            v_c_next, _ = self._v_causal(states_next, dist)
            v_c_next = v_c_next.view(-1)

            a_probe = returns - v_c
            delta_c = rewards + gamma * (1.0 - dones) * v_c_next - v_c

            # ---- Log: NN critic (training one)
            self.train_metrics.add(
                adv_var=float((returns - v_nn).var(unbiased=False).item()),
                value_mse=mse(returns, v_nn),
                v_explained_variance=explained_variance(returns, v_nn),
                td_mean=float(delta_nn.mean().item()),
                td_std=float(delta_nn.std(unbiased=False).item()),
                v_corr=pearson_corr(returns, v_nn),
                v_spearman=spearman_corr(returns, v_nn),
                v_mi=mutual_information(
                    returns, v_nn, n_bins=20, strategy="quantile", normalized=True
                ),
                v_wass=wasserstein_1d(returns, v_nn),
                v_kl=kl_divergence_hist(returns, v_nn, n_bins=20, strategy="quantile"),
                v_js=js_divergence_hist(returns, v_nn, n_bins=20, strategy="quantile"),
            )

            # ---- Log: VBN probe (prefixed)
            self.train_metrics.add(
                causal_adv_var=float(a_probe.var(unbiased=False).item()),
                causal_value_mse=mse(returns, v_c),
                causal_v_explained_variance=explained_variance(returns, v_c),
                causal_td_mean=float(delta_c.mean().item()),
                causal_td_std=float(delta_c.std(unbiased=False).item()),
                causal_v_corr=pearson_corr(returns, v_c),
                causal_v_spearman=spearman_corr(returns, v_c),
                causal_v_mi=mutual_information(
                    returns, v_c, n_bins=20, strategy="quantile", normalized=True
                ),
                causal_adv_mean=float(a_probe.mean().item()),
                causal_adv_std=float(a_probe.std(unbiased=False).item()),
                causal_adv_min=float(a_probe.min().item()),
                causal_adv_max=float(a_probe.max().item()),
                causal_v_wass=wasserstein_1d(returns, v_c),
                causal_v_kl=kl_divergence_hist(
                    returns, v_c, n_bins=20, strategy="quantile"
                ),
                causal_v_js_base=js_divergence_hist(
                    returns, v_c, n_bins=20, strategy="quantile"
                ),
            )
