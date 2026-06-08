"""Confounding axis (§6.2): ConfoundedEnv, biased logging policy, and the
``assert_confounded`` dataset gate.

Data-generating process for Cells 7–8:

* A per-episode latent ``U`` is sampled at ``reset()`` (default Bernoulli
  over {−1,+1}, p=0.5), fixed for the episode, NEVER written into
  observations, and logged to ``info["confounder_u"]``.
* **U → reward** (environment side): ``r' = r + delta * U * psi(s, a)`` with
  ``psi = 1`` initially.
* **U → action** (LOGGING POLICY side, not the env): the behavior policy's
  preferred-action logit is biased, ``logits' = logits + beta * U * e_{a*}``,
  and the EXACT propensity of the BIASED policy is logged.

The gate ``assert_confounded`` accepts a dataset only if confounding is
functionally present (all three conditions); on failure it RAISES — never
warns. It validates the data-generating process, so it may read
``confounder_u`` from the dataset infos; estimators in ``src/eval/ope.py``
must never do that (§8).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
import torch

from src.data.behavior_policies import BehaviorPolicy
from src.data.experience_source import OfflineDatasetSource
from src.eval.ope import IPWEstimator, NaiveEstimator, StochasticPolicyAdapter


class ConfoundedEnv(gym.Wrapper):
    """Adds the per-episode confounder U and its reward pathway.

    ``u_dist="bernoulli_pm1"`` (default): U ∈ {−1,+1} w.p. 1/2.
    ``u_dist="normal"``: U ~ N(0, sigma_u^2).
    """

    def __init__(
        self,
        env: gym.Env,
        delta: float = 0.5,
        u_dist: str = "bernoulli_pm1",
        sigma_u: float = 1.0,
        seed: int = 0,
    ) -> None:
        super().__init__(env)
        self.delta = float(delta)
        self.u_dist = str(u_dist)
        self.sigma_u = float(sigma_u)
        self._rng = np.random.default_rng(seed)
        self._u: float = 0.0

    @property
    def current_u(self) -> float:
        return self._u

    def _sample_u(self) -> float:
        if self.u_dist == "bernoulli_pm1":
            return float(self._rng.choice([-1.0, 1.0]))
        if self.u_dist == "normal":
            return float(self._rng.normal(0.0, self.sigma_u))
        raise ValueError(f"Unknown u_dist '{self.u_dist}'")

    def _psi(self, obs, action) -> float:
        """Reward-shift gain psi(s, a); start with the constant 1 (§6.2)."""
        _ = obs, action
        return 1.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._u = self._sample_u()
        info = dict(info)
        info["confounder_u"] = np.float64(self._u)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = float(reward) + self.delta * self._u * self._psi(obs, action)
        info = dict(info)
        info["confounder_u"] = np.float64(self._u)
        return obs, reward, terminated, truncated, info


class ConfoundedExplorer(BehaviorPolicy):
    """U-biased discrete logging policy (adapted from the recovered
    ``biased_explorer.ConfoundedExplorer`` per the Phase-1.5 verdict (c):
    interface kept, coupling changed to the §6.2 form).

    ``logits' = logits + beta * U * e_{a*}`` where ``a*`` is the base
    policy's preferred (greedy) action at ``s``. Exact propensities are
    returned under the BIASED distribution.
    """

    def __init__(self, base_logits_fn, beta: float = 1.0) -> None:
        """``base_logits_fn(obs) -> [B, A]`` logits of the unbiased policy."""
        self.base_logits_fn = base_logits_fn
        self.beta = float(beta)

    def _biased_log_probs(
        self, obs: torch.Tensor, latent: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            logits = self.base_logits_fn(obs)
            if logits.ndim != 2:
                raise ValueError("base_logits_fn(obs) must return [B, A].")
            a_star = logits.argmax(dim=-1, keepdim=True)
            u = latent.reshape(-1, 1).to(logits.device).float()
            biased = logits.scatter_add(1, a_star, self.beta * u.expand(-1, 1))
            return torch.log_softmax(biased, dim=-1)

    def select_action(
        self, obs: torch.Tensor, latent: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if latent is None:
            raise ValueError(
                "ConfoundedExplorer requires the latent U at action-selection "
                "time (U -> action pathway)."
            )
        log_probs = self._biased_log_probs(obs, latent)
        action = torch.multinomial(log_probs.exp(), num_samples=1).squeeze(-1)
        logp = log_probs.gather(1, action.unsqueeze(-1)).squeeze(-1)
        return action, logp


class GaussianConfoundedExplorer(BehaviorPolicy):
    """Continuous U-biased logging policy (§6.2 continuous coupling).

    Behavior: ``a ~ N(mu_b(s) + gamma*U*v, sigma^2 I)`` where ``mu_b`` is a
    base deterministic policy (e.g. the cached Cell-1 SAC actor mean) and
    ``v`` a fixed unit direction. The EXACT Gaussian log-density of the
    SAMPLED action is logged. The sampled action is passed to the env
    unclipped (MuJoCo clamps ctrl internally); the biased mean is kept
    inside (−0.95, 0.95) so boundary effects are negligible — documented
    rather than hidden in the propensities.
    """

    def __init__(
        self,
        base_mean_fn,
        gamma: float = 1.0,
        sigma: float = 0.3,
        v: Optional[torch.Tensor] = None,
        action_dim: int = 6,
    ) -> None:
        self.base_mean_fn = base_mean_fn
        self.gamma = float(gamma)
        self.sigma = float(sigma)
        if v is None:
            v = torch.ones(action_dim) / float(action_dim) ** 0.5
        self.v = v

    def _biased_mean(self, obs: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            mu = self.base_mean_fn(obs)
        u = latent.reshape(-1, 1).to(mu.device).float()
        v = self.v.to(mu.device)
        return (mu + self.gamma * u * v).clamp(-0.95, 0.95)

    def select_action(
        self, obs: torch.Tensor, latent: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if latent is None:
            raise ValueError(
                "GaussianConfoundedExplorer requires the latent U at "
                "action-selection time (U -> action pathway)."
            )
        mean = self._biased_mean(obs, latent)
        dist = torch.distributions.Normal(mean, self.sigma)
        action = dist.sample()
        logp = dist.log_prob(action).sum(-1)
        return action, logp


# ---------------------------------------------------------------------------
# The confounding gate
# ---------------------------------------------------------------------------


class ConfoundingGateError(AssertionError):
    """Raised when a dataset labeled confounded is functionally unconfounded."""


@dataclass
class GateReport:
    naive_value: float
    ipw_value: float
    naive_ipw_gap: float
    action_u_dependence: float  # TV distance between P(a|U=+1) and P(a|U=-1)
    action_u_zscore: float
    reward_u_effect: float  # mean reward difference between U signs
    reward_u_zscore: float
    passed: bool


def _episode_u(source: OfflineDatasetSource) -> torch.Tensor:
    us = []
    for ep in source.episodes:
        if "confounder_u" not in ep:
            raise ConfoundingGateError(
                "dataset has no confounder_u infos - was it collected with "
                "ConfoundedEnv?"
            )
        us.append(ep["confounder_u"].reshape(-1)[0])
    return torch.stack(us).float()


def assert_confounded(
    source: OfflineDatasetSource,
    tau: float = 0.05,
    z_crit: float = 3.0,
    clone_seed: int = 0,
) -> GateReport:
    """Accept a dataset only if ALL hold (§6.2): (i) |J_naive − J_IPW| > tau
    (relative to |J_naive|, IPW of the U-blind behavior clone with LOGGED
    propensities); (ii) A depends on U significantly; (iii) R depends on U
    significantly. Raises :class:`ConfoundingGateError` otherwise.

    Failure mode this prevents: buffers labeled confounded but functionally
    unconfounded (e.g. beta = delta = 0).
    """
    if source.behavior_logprob is None:
        raise ConfoundingGateError(
            "the gate needs LOGGED propensities; run it on the known-pi_b "
            "view of the dataset before any cell switches discard them."
        )
    continuous = source.actions.dtype not in (torch.int64, torch.int32)

    # --- U-blind behavior clone (shared by conditions i and ii) ------------
    ep_u = _episode_u(source).cpu()
    lengths = [int(ep["rewards"].shape[0]) for ep in source.episodes]
    step_u = torch.repeat_interleave(ep_u, torch.tensor(lengths))
    tv_a = 0.0

    if continuous:
        # Gaussian MLE clone of the marginal behavior policy
        from src.eval.ope import clone_behavior_policy

        torch.manual_seed(clone_seed)
        clone_logp_fn = clone_behavior_policy(source, seed=clone_seed)
        clone_logp = clone_logp_fn(source.obs.float(), source.actions).cpu()

        # --- (i) truncated-horizon SN-IPW vs naive ------------------------
        # Full-episode importance products degenerate at MuJoCo horizons
        # (curse of horizon), so the gate uses a TRUNCATED-horizon (K-step)
        # self-normalized IPW as a DETECTION STATISTIC (not an estimator):
        # partial products over the first K steps against K-step returns.
        K = 50
        log_ratio = (-(source.behavior_logprob.cpu() - clone_logp)).cpu()
        w_list, g_list, naive_list = [], [], []
        start = 0
        for T in lengths:
            k = min(K, T)
            lr = log_ratio[start : start + k].sum().clamp(min=-30.0, max=30.0)
            w_list.append(torch.exp(lr))
            g = source.rewards[start : start + k].sum().cpu()
            g_list.append(g)
            naive_list.append(g)
            start += T
        w = torch.stack(w_list)
        g = torch.stack(g_list).float()
        naive = float(torch.stack(naive_list).float().mean())
        ipw = float((w * g).sum() / w.sum().clamp_min(1e-12))
        gap = abs(naive - ipw)
        cond_i = gap > tau * max(abs(naive), 1e-8)
    else:
        from src.offline.bc import BehaviorCloning
        from src.rl.on_policy.policy import ActorCriticMLP

        torch.manual_seed(clone_seed)
        n_actions = int(source.actions.max().item()) + 1
        clone = BehaviorCloning(
            ActorCriticMLP(source.obs.shape[-1], n_actions, "discrete", source.device),
            source.device,
        )
        clone.fit_source(source, n_steps=1500)
        target = StochasticPolicyAdapter(clone.policy)
        naive = NaiveEstimator().estimate(source, target).value
        ipw = (
            IPWEstimator(behavior="known", self_normalized=True)
            .estimate(source, target)
            .value
        )
        gap = abs(naive - ipw)
        cond_i = gap > tau * max(abs(naive), 1e-8)
        clone_logp = target.log_prob(source.obs.float(), source.actions).cpu()
        actions = source.actions.long().cpu()
        # marginal TV kept as a descriptive diagnostic only (discrete)
        pos, neg = actions[step_u > 0], actions[step_u < 0]
        p_pos = torch.bincount(pos, minlength=n_actions).float() / max(len(pos), 1)
        p_neg = torch.bincount(neg, minlength=n_actions).float() / max(len(neg), 1)
        tv_a = 0.5 * (p_pos - p_neg).abs().sum().item()

    # --- (ii) A depends on U GIVEN s ---------------------------------------
    # Conditional dependence via the propensity residual: logged
    # log pi_b(a|s,U) minus the U-blind clone's log pi_bar(a|s).
    residual = source.behavior_logprob.cpu() - clone_logp
    res_pos, res_neg = residual[step_u > 0], residual[step_u < 0]
    effect_a = float(res_pos.mean() - res_neg.mean())
    se_a = float(
        np.sqrt(
            res_pos.var().item() / max(len(res_pos), 1)
            + res_neg.var().item() / max(len(res_neg), 1)
        )
    )
    z_a = abs(effect_a) / max(se_a, 1e-12)
    cond_ii = z_a > z_crit

    # --- (iii) R depends on U ----------------------------------------------
    rewards = source.rewards.cpu()
    r_pos, r_neg = rewards[step_u > 0], rewards[step_u < 0]
    effect = float(r_pos.mean() - r_neg.mean())
    se_r = float(
        np.sqrt(
            r_pos.var().item() / max(len(r_pos), 1)
            + r_neg.var().item() / max(len(r_neg), 1)
        )
    )
    z_r = abs(effect) / max(se_r, 1e-12)
    cond_iii = z_r > z_crit

    if continuous:
        # Option A (Phase-6C gate ruling): on the long-horizon continuous
        # anchor, condition (i)'s naive-vs-IPW gap is non-discriminative
        # (H~1000 IPW degeneracy = curse of horizon, independent of
        # confounding). It is COMPUTED and REPORTED as a diagnostic only;
        # acceptance rests on the horizon-independent (ii) A-U|s and (iii)
        # R-U. The gate's causal meaning is unchanged across anchors.
        passed = bool(cond_ii and cond_iii)
        cond_i_label = cond_i  # reported, not required
    else:
        passed = bool(cond_i and cond_ii and cond_iii)
        cond_i_label = cond_i

    report = GateReport(
        naive_value=float(naive),
        ipw_value=float(ipw),
        naive_ipw_gap=float(gap),
        action_u_dependence=float(tv_a),
        action_u_zscore=z_a,
        reward_u_effect=effect,
        reward_u_zscore=float(z_r),
        passed=passed,
    )
    if not report.passed:
        i_clause = (
            f"(i) |naive-ipw|={gap:.3f} vs tau*|naive|={tau * abs(naive):.3f} "
            f"-> {cond_i_label}"
            + (" [diagnostic only on continuous]" if continuous else "")
        )
        raise ConfoundingGateError(
            "dataset labeled confounded is functionally unconfounded: "
            f"{i_clause}; (ii) A-U z={z_a:.2f} (TV={tv_a:.3f}) -> {cond_ii}; "
            f"(iii) R-U z={z_r:.2f} (effect={effect:.4f}) -> {cond_iii}"
        )
    return report
