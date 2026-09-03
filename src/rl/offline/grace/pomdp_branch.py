"""The declared-observability path: ``(observability, optionally k)`` -> the
lag-k augmented state -> the MDP branch. ONE code path (ruled 2026-09-03):

    declared              GRACE uses
    MDP                   k = 0
    POMDP with k          that k
    POMDP without k       selects k by MATERIALITY-BY-REFIT

Declared-MDP IS k = 0 — the contract's symmetry — and the observation is
augmented with the last k (O, A) pairs (the "past observations as proxies for
the latent state" structure the catalogue names). The EXISTING
reward-transform machinery runs on the augmented view; the fit, L4, the
pessimism rule, the cache and the C3 labels are the MDP branch's, applied to a
different state. The selector's features are EXACTLY these (O, A) lags (S19:
a feature set used to certify sufficiency must be the feature set the model
receives); lagged R is deliberately NOT in the state — at deployment U -> R is
intact, so an R-carrying state is U-informative and the prior-marginal served
estimand's warrant fails on it (l5 docstring; the belief-state alternative is
the D-F/D-G shape).

**Selection: materiality-by-refit against L4's own interval, no constant.**
    k* = min { k : |contrast(k+1) - contrast(k)| <= w_k }
where ``contrast(k)`` is the served action contrast (``meta.contrast_point``)
of the fit on the lag-k view and ``w_k`` is L4's half-width there. Every term
is measured per fit; nothing is calibrated per environment (A2). The
predictive Markov statistic (l5) does NOT select — it asks "exactly Markov?",
whose answer is always no (S18) — it is REPORTED at the served lag as a C3
record. Cost on a true MDP: fits at k = 0 and k = 1 (measured ratio 1.71 at
k = 1), replacing the former ~900 s selection pass; under the transform cache
the k = 0 fit is a hit whenever the MDP-declared arm ran on the same data.

**A user-supplied k is an INPUT, never a hypothesis.** GRACE uses it and
reports two diagnostics, both report-only, neither overriding:
  * ``window_sufficient`` — does lag k+1 move the served contrast by more
    than w_k? If so the window is too short: WARN, serve anyway.
  * ``window_necessary`` — does k-1 already suffice? If so the window is
    longer than needed: costs compute and estimator variance, no correctness
    harm — contract row 2 ("over-assumption is cheap") in its exact,
    measurable form.
``k_max`` applies only when selection is delegated; a supplied k is not
subject to it. Diagnostics cost one (sufficient) or two (necessary) extra
fits and are a BUDGET switch (``k_diagnostics``), disclosed when off.

**Selector exhaustion (no k <= k_max passes) ABSTAINS** — the finite-memory
machinery cannot honour the declaration within its budget: an L4-family
fit-mechanism condition (``BUDGET-BOUND``), NOT a declaration override. So
does a fit that abstains at a window the selection needs (the comparison is
undefined without an interval). L5's record never stops serving.

**Row alignment.** Early transitions (t < j) EDGE-PAD their lag features with
the episode's first row, so the augmented view has exactly one row per buffer
transition and the substituted reward column covers the buffer completely.
The augmented view also carries ``next_obs`` (the NEXT row's augmented state,
exact: the lag blocks shift by one) and ``dones`` — an earlier version handed
the fit a view without them, so the extractor rolled next-obs ACROSS episode
boundaries and saw no terminations (fixed 2026-09-03).
"""

from __future__ import annotations

from typing import Dict, List, Optional

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
    what the coverage check must compare against), not the key count."""

    def __init__(self, cols):
        super().__init__(cols)
        self._data = self

    def __len__(self):
        return int(self["rewards"].shape[0])


def _lag_blocks(data, k: int):
    """Edge-padded lagged (action, state) blocks j = 1..k, per episode."""
    ep_ids = data.episode_ids
    act_col = data.action.reshape(-1, 1).to(data.state.dtype)
    lag_a, lag_s = [], []
    for j in range(1, k + 1):
        a_j = torch.empty_like(act_col)
        s_j = torch.empty_like(data.state)
        for e in torch.unique_consecutive(ep_ids):
            m = (ep_ids == e).nonzero(as_tuple=True)[0]
            src = torch.clamp(torch.arange(len(m), device=m.device) - j, min=0)
            s_j[m] = data.state[m][src]
            a_j[m] = act_col[m][src]
        lag_a.append(a_j)
        lag_s.append(s_j)
    return act_col, lag_a, lag_s


def _augmented_cols(data, k: int, nxt=None, dones=None) -> dict:
    """The lag-k augmented view as a dict buffer (one row per transition,
    edge-padded), consumable by ``_episode_data_from_buffer``'s dict path —
    which is what lets the MDP fit run on it unmodified. ``obs`` is
    ``[s_t, a_{t-1}, s_{t-1}, ..., a_{t-k}, s_{t-k}]``; ``next_obs`` is the
    same construction one step later, ``[s_{t+1}, a_t, s_t, ..., a_{t-k+1},
    s_{t-k+1}]`` (exact, not a roll)."""
    act_col, lag_a, lag_s = _lag_blocks(data, k)
    blocks = [data.state]
    for j in range(k):
        blocks.extend([lag_a[j], lag_s[j]])
    cols = dict(
        obs=torch.cat(blocks, dim=1),
        actions=data.action,
        rewards=data.reward,
        episode_ids=data.episode_ids,
    )
    if nxt is not None:
        nblocks = [nxt]
        if k >= 1:
            nblocks.extend([act_col, data.state])
        for j in range(k - 1):
            nblocks.extend([lag_a[j], lag_s[j]])
        cols["next_obs"] = torch.cat(nblocks, dim=1)
    if dones is not None:
        cols["dones"] = dones
    for name, v in data.proxy.items():
        cols[f"proxy_{name}"] = v
    return cols


def _contrast(s: GraceServing):
    """(contrast, half-width) of a served fit; None when it abstained."""
    if s.abstained or "contrast_point" not in s.meta:
        return None
    return float(s.meta["contrast_point"]), 0.5 * (float(s.hi) - float(s.lo))


def transform_offline_rewards_declared(
    buffer,
    *,
    observability: str = "mdp",
    k: Optional[int] = None,
    k_max: int = 2,
    k_diagnostics: bool = True,
    l5_report: bool = True,
    l5_alpha: float = 0.05,
    l5_b: int = l5._B_DRAWS,
    cache_dir=None,
    dataset_id: str = "",
    **options,
) -> GraceServing:
    """THE entry point for a GRACE arm: honour ``(observability, k)``, select
    k by materiality when delegated, write the REAL buffer exactly once,
    report everything (C3). See the module docstring for the rules."""
    obs_decl = str(observability).lower()
    if obs_decl not in ("mdp", "pomdp"):
        raise ValueError(
            f"declared observability must be mdp|pomdp, got {observability!r}"
        )
    if obs_decl == "mdp":
        if k not in (None, 0):
            raise ValueError(
                "declared MDP IS k = 0; a window k > 0 is a POMDP declaration"
            )
        k_decl: Optional[int] = 0
        source = "declared-mdp"
    else:
        k_decl = None if k is None else int(k)
        source = "selected" if k_decl is None else "declared"
        if k_decl is not None and k_decl < 0:
            raise ValueError("a declared window k must be >= 0")

    pn = tuple(options.get("proxy_names", ()) or ())
    try:
        data, nxt, dn = _episode_data_from_buffer(
            buffer, proxy_names=pn, device=options.get("device")
        )
    except Exception as exc:
        return GraceServing(reason=f"buffer not episode-grouped: {exc}")
    if data is None:
        return GraceServing(reason="buffer carried no usable episodes")

    fits: Dict[int, GraceServing] = {}

    def fit_at(kk: int) -> GraceServing:
        if kk in fits:
            return fits[kk]
        if kk == 0:
            # The REAL buffer, unwritten (apply=False): its content address
            # equals the MDP-declared arm's, so this is a cache HIT whenever
            # that arm ran on the same data — the k = 0 collapse.
            view, did = buffer, dataset_id
        else:
            view = _DictBuffer(_augmented_cols(data, kk, nxt, dn))
            did = f"{dataset_id}#k={kk}"
        s = transform_offline_rewards(
            view, cache_dir=cache_dir, dataset_id=did, apply=False, **options
        )
        fits[kk] = s
        return s

    def material(kk: int):
        """Does lag kk+1 move the served contrast beyond w_kk? Returns
        (material | None, delta, w, note)."""
        c0, c1 = _contrast(fit_at(kk)), _contrast(fit_at(kk + 1))
        if c0 is None or c1 is None:
            which = kk if c0 is None else kk + 1
            return None, float("nan"), float("nan"), f"fit at k={which} abstained"
        delta = abs(c1[0] - c0[0])
        return bool(delta > c0[1]), delta, c0[1], ""

    evidence: dict = dict(
        window_source=source, window_k_diagnostics=bool(k_diagnostics)
    )

    # ---- the window ------------------------------------------------------
    if k_decl is None:
        evidence["window_k_max"] = int(k_max)
        k_served: Optional[int] = None
        for kk in range(0, k_max + 1):
            m, delta, w, note = material(kk)
            evidence[f"window_stage{kk}_delta"] = delta
            evidence[f"window_stage{kk}_w"] = w
            if m is None:
                s = GraceServing(
                    reason=f"window selection undefined at k={kk}: {note}",
                    fit_label=fit_at(kk).fit_label,
                )
                s.meta.update(
                    evidence,
                    window_k=None,
                    transform_applied=False,
                    n_rewards_written=0,
                )
                return s
            if not m:
                k_served = kk
                break
        if k_served is None:
            s = GraceServing(
                reason=(
                    f"window-exhausted: another lag still moves the served contrast "
                    f"beyond L4's half-width at every k <= {k_max} (BUDGET-BOUND)"
                ),
                fit_label=fit_at(k_max).fit_label,
            )
            s.meta.update(
                evidence, window_k=None, transform_applied=False, n_rewards_written=0
            )
            return s
    else:
        k_served = k_decl
        if k_diagnostics:
            m, delta, w, note = material(k_served)
            evidence.update(
                window_sufficient=(None if m is None else (not m)),
                window_sufficient_delta=delta,
                window_sufficient_w=w,
                window_sufficient_note=note,
            )
            if k_served >= 1:
                m2, delta2, w2, note2 = material(k_served - 1)
                evidence.update(
                    window_necessary=(None if m2 is None else bool(m2)),
                    window_necessary_delta=delta2,
                    window_necessary_w=w2,
                    window_necessary_note=note2,
                )
            else:
                evidence.update(
                    window_necessary=None,
                    window_necessary_note="k=0 has no shorter window",
                )

    serving = fit_at(k_served)
    evidence["window_k"] = int(k_served)
    evidence["window_stages"] = " ".join(
        f"k={kk}:"
        + (
            "abstained"
            if _contrast(s_) is None
            else f"contrast={_contrast(s_)[0]:+.4f}:w={_contrast(s_)[1]:.4f}"
        )
        for kk, s_ in sorted(fits.items())
    )
    warn = ""
    if evidence.get("window_sufficient") is False:
        warn += " WINDOW-TOO-SHORT(warn)"
    if evidence.get("window_necessary") is False:
        warn += " WINDOW-LONGER-THAN-NEEDED(info)"

    # ---- L5's record at the served lag: report-only ------------------------
    if l5_report:
        try:
            v = l5.markov_test(
                _episodes_from_data(data, nxt), lag=k_served, b=l5_b, seed=k_served
            )
            evidence.update(v.record(l5_alpha))
            c = _contrast(serving)
            if c is not None:
                sm = l5.serving_material(v, w=c[1])
                evidence.update({f"l5_material_{kk}": vv for kk, vv in sm.items()})
            if v.rejected(l5_alpha):
                warn += " L5-CONTRADICTS-DECLARATION(report)"
        except ValueError as exc:  # episodes too short for this lag
            evidence["l5_note"] = str(exc)

    serving.meta.update(evidence)
    serving.fit_label = (
        f"window[k={k_served}|{source}]{warn} {serving.fit_label}".strip()
    )
    if serving.abstained:
        serving.meta.update(transform_applied=False, n_rewards_written=0)
        return serving
    apply_reward_transform(buffer, serving)  # the ONE write
    return serving
