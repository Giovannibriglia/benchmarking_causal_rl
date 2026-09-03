"""The POMDP branch: window selection -> augmented state -> the MDP branch.

Declared-POMDP is declared-MDP with the window k free (the contract's
symmetry): ``select_window`` picks the smallest k not falsified at the
material scale, the observation is augmented with the last k (O, A) pairs —
the "past observations as proxies for the latent state" structure the
catalogue names — and the EXISTING reward-transform machinery runs on the
augmented view. Nothing else is new: the fit, L4, the pessimism rule, the
cache and the C3 labels are the MDP branch's, applied to a different state.

**``dr2_cut`` is REQUIRED and comes from the calibration report** — the
equivalence cut that stops the selector chasing floor-level rejections to
``k_max`` on true MDPs (measured; contract row 2's mechanism). The branch is
finished code waiting on that one number; it refuses to run without it
rather than defaulting to the statistical-only selector.

**Selector exhaustion (k is None) ABSTAINS** — the finite-memory machinery
cannot honour the declaration within its budget: an L4-family fit-mechanism
condition (``BUDGET-BOUND``), NOT a declaration override; L5 falsification
itself never stops serving (the 2026-09-03 ruling, module docstring of l5).
Scope, stated: on the velocity-masked grid k=1 is expected by construction
and the exhaustion path is exercised by fixtures only.

**Row alignment.** Early transitions (t < j) EDGE-PAD their lag features with
the episode's first row, so the augmented view has exactly one row per buffer
transition and the substituted reward column covers the buffer completely
(the coverage check in ``apply_reward_transform`` stays exact).
"""

from __future__ import annotations

from typing import List

import torch

from src.rl.offline.grace import l5
from src.rl.offline.grace.serving import (
    _episode_data_from_buffer,
    apply_reward_transform,
    GraceServing,
    transform_offline_rewards,
)


def _episodes_from_data(data, nxt) -> List[l5.Episode]:
    """Rebuild per-episode (obs [T+1, D], act [T], rew [T]) arrays for l5."""
    out = []
    ep_ids = data.episode_ids
    for e in torch.unique_consecutive(ep_ids):
        m = ep_ids == e
        obs_t = data.state[m]
        last_next = nxt[m][-1:] if nxt is not None else obs_t[-1:]
        out.append(
            l5.Episode(
                obs=torch.cat([obs_t, last_next]).cpu().numpy(),
                act=data.action[m].cpu().numpy(),
                rew=data.reward[m].cpu().numpy(),
            )
        )
    return out


class _DictBuffer(dict):
    """A dict of columns that also satisfies ``apply_reward_transform``:
    ``_data`` aliases self and ``len()`` is the ROW count (the fill, which is
    what the coverage check must compare against), not the key count. The
    inner apply writes into these column copies — harmless; the REAL buffer
    is substituted by the caller afterwards."""

    def __init__(self, cols):
        super().__init__(cols)
        self._data = self

    def __len__(self):
        return int(self["rewards"].shape[0])


def _augmented_cols(data, k: int) -> dict:
    """The lag-k augmented view as a dict buffer (one row per transition,
    edge-padded), consumable by ``_episode_data_from_buffer``'s dict path —
    which is what lets the MDP fit run on it unmodified."""
    ep_ids = data.episode_ids
    blocks = [data.state]
    act_col = data.action.reshape(-1, 1).to(data.state.dtype)
    for j in range(1, k + 1):
        lag_s = torch.empty_like(data.state)
        lag_a = torch.empty_like(act_col)
        for e in torch.unique_consecutive(ep_ids):
            m = (ep_ids == e).nonzero(as_tuple=True)[0]
            src = torch.clamp(torch.arange(len(m), device=m.device) - j, min=0)
            lag_s[m] = data.state[m][src]
            lag_a[m] = act_col[m][src]
        blocks.extend([lag_a, lag_s])
    cols = dict(
        obs=torch.cat(blocks, dim=1),
        actions=data.action,
        rewards=data.reward,
        episode_ids=ep_ids,
    )
    for name, v in data.proxy.items():
        cols[f"proxy_{name}"] = v
    return cols


def transform_offline_rewards_pomdp(
    buffer,
    *,
    dr2_cut: float,
    alpha: float = 0.05,
    k_max: int = 2,
    cache_dir=None,
    dataset_id: str = "",
    **options,
) -> GraceServing:
    """The declared-POMDP entry point (mirrors ``transform_offline_rewards``).

    Selection evidence travels in ``meta`` (C3): the chosen ``window_k``, each
    stage's p and Delta-R^2, and the cut applied. ``dr2_cut`` has no default
    on purpose — it is the calibration report's number, and running without
    one would re-create the measured k_max-chasing selector.
    """
    if dr2_cut is None:
        raise ValueError(
            "the POMDP branch requires dr2_cut from the L5 calibration report "
            "(results/l5_calibration) — refusing the statistical-only selector"
        )
    pn = tuple(options.get("proxy_names", ()) or ())
    try:
        data, nxt, _dn = _episode_data_from_buffer(
            buffer, proxy_names=pn, device=options.get("device")
        )
    except Exception as exc:
        return GraceServing(reason=f"buffer not episode-grouped: {exc}")
    if data is None:
        return GraceServing(reason="buffer carried no usable episodes")

    k, verdicts = l5.select_window(
        _episodes_from_data(data, nxt), alpha=alpha, k_max=k_max, dr2_cut=dr2_cut
    )
    evidence = dict(
        l5_alpha=float(alpha),
        l5_dr2_cut=float(dr2_cut),
        l5_k_max=int(k_max),
        l5_stage_p=" ".join(f"{v.p_value:.4f}" for v in verdicts),
        l5_stage_dr2=" ".join(f"{v.statistic:.3e}" for v in verdicts),
    )
    if k is None:
        s = GraceServing(
            reason=f"window-exhausted: not Markov at any lag <= {k_max} (BUDGET-BOUND)",
            fit_label=verdicts[-1].label(alpha),
        )
        s.meta.update(
            evidence, window_k=None, transform_applied=False, n_rewards_written=0
        )
        return s

    if k == 0:
        # The declared-POMDP branch on Markov-sufficient data IS the MDP
        # branch — the collapse that makes over-assumption cheap (row 2).
        serving = transform_offline_rewards(
            buffer, cache_dir=cache_dir, dataset_id=dataset_id, **options
        )
        serving.meta.update(evidence, window_k=0)
        return serving

    # k >= 1: fit on the augmented view; substitute into the REAL buffer.
    # The '#k=' suffix keys the cache audit trail; content-addressing already
    # separates the entries (the augmented state IS different data).
    aug = _DictBuffer(_augmented_cols(data, k))
    serving = transform_offline_rewards(
        aug, cache_dir=cache_dir, dataset_id=f"{dataset_id}#k={k}", **options
    )
    serving.meta.update(evidence, window_k=int(k))
    serving.fit_label = f"pomdp[window={k}] {serving.fit_label}".strip()
    if not serving.abstained:
        apply_reward_transform(buffer, serving)
    return serving
