"""Frozen SCALAR ProximalEM.run_em (pre-vectorization, master f1e0676) — the parity
reference for the vectorized port. Byte-faithful copy of the old per-episode Python
loops (M-step NLL, E-step LLR/residual, guarded canonicalization) + warm-start, so
the vectorized version can be held to scalar-vs-vectorized parity forever.

Operates on a live ProximalEM `em` (reads em.rm / em.opt / em.seq_buffer / em.prior_p
/ em.delta_l2 / em._canon_eps / em.reward_steps) exactly as the old code did.
"""

from __future__ import annotations

import math

import torch


def _ep_tensors(em, ep):
    obs = torch.stack([tr["obs"] for tr in ep]).to(em.device).float()
    actions = torch.stack([tr["actions"] for tr in ep]).to(em.device)
    rewards = torch.stack([tr["rewards"] for tr in ep]).to(em.device).float()
    r_tau = float(ep[0].get("r_tau", torch.tensor(em.prior_p)))
    return obs, actions, rewards, r_tau


def scalar_warm_start(em) -> None:
    """Frozen v2 warm-start: per-episode MEAN reward + no-spread->prior fallback."""
    episodes = list(em.seq_buffer.iter_episodes())
    if episodes:
        means = torch.tensor(
            [float(torch.stack([tr["rewards"] for tr in ep]).mean()) for ep in episodes]
        )
        if float(means.std()) < em._canon_eps:
            for ep in episodes:
                for tr in ep:
                    tr["r_tau"] = torch.tensor(em.prior_p)
        else:
            med = float(means.median())
            for ep, m in zip(episodes, means.tolist()):
                init = torch.tensor(1.0 if m > med else 0.0)
                for tr in ep:
                    tr["r_tau"] = init


def scalar_run_em(em) -> None:
    """Frozen scalar run_em (the pre-vectorization per-episode loops)."""
    if em.seq_buffer is None:
        return
    episodes = list(em.seq_buffer.iter_episodes())
    if not episodes:
        return

    for _ in range(em.reward_steps):
        em.opt.zero_grad(set_to_none=True)
        loss = torch.zeros((), device=em.device)
        total = 0
        for ep in episodes:
            obs, actions, rewards, r_tau = _ep_tensors(em, ep)
            r_hat = em.rm.reward_at(obs, actions)
            sigma = em.rm.log_sigma.exp().clamp_min(1e-3)
            ll0 = -0.5 * ((rewards - r_hat) / sigma) ** 2 - torch.log(sigma)
            ll1 = -0.5 * ((rewards - r_hat - em.rm.delta) / sigma) ** 2 - torch.log(
                sigma
            )
            loss = loss - ((1.0 - r_tau) * ll0 + r_tau * ll1).sum()
            total += len(ep)
        loss = loss / max(total, 1) + em.delta_l2 * em.rm.delta**2
        loss.backward()
        em.opt.step()

    logit_prior = math.log(em.prior_p / (1.0 - em.prior_p))
    r_taus, residuals = [], []
    with torch.no_grad():
        sigma = em.rm.log_sigma.exp().clamp_min(1e-3)
        for ep in episodes:
            obs, actions, rewards, _ = _ep_tensors(em, ep)
            r_hat = em.rm.reward_at(obs, actions)
            ll0 = -0.5 * ((rewards - r_hat) / sigma) ** 2
            ll1 = -0.5 * ((rewards - r_hat - em.rm.delta) / sigma) ** 2
            llr = (ll1 - ll0).sum() + logit_prior
            r_taus.append(float(torch.sigmoid(llr)))
            residuals.append(float((rewards - r_hat).mean()))

    rt = torch.tensor(r_taus)
    res = torch.tensor(residuals)
    if float(res.std()) < em._canon_eps:
        r_taus = [em.prior_p] * len(r_taus)
    else:
        non_degenerate = (
            float(em.rm.delta) > em._canon_eps and float(rt.std()) > em._canon_eps
        )
        if non_degenerate:
            corr = float(torch.corrcoef(torch.stack([rt, res]))[0, 1])
            if corr < 0.0:
                r_taus = [1.0 - x for x in r_taus]

    for ep, val in zip(episodes, r_taus):
        t = torch.tensor(val)
        for tr in ep:
            tr["r_tau"] = t
