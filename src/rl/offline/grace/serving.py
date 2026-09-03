"""The GRACE critic seam — fit once at the buffer handoff, then serve.

**Base-parity by construction.** ``build_grace_<base>`` builds the base
algorithm with its OWN builder and its own strategy (the observational floor),
then wraps only the SERVING surface. Training is therefore byte-identical to
the base run; the only difference is which Q the agent acts on. That is what
makes the experiment's contrast attributable to the identification method
rather than to a different learner — and it is the v1 arrangement, restated
(``CRITIC_LIBRARY``'s note: "training is exactly the observational floor's;
only the SERVING surface routes").

**Fit-once-then-serve** (ruled 2026-08-30). ``set_sequence_buffer`` is the
handoff: the runner calls it after the Minari fill and BEFORE any gradient
step (``runner.py:1661``). The L3 fit happens there, once. The fork verdict
permits cadence refit; the first experiment does not need it, so it is not
built.

**The serving rule is L4's**, and it is the whole point of the seam:

===============  =====================================================
L4 verdict       served
===============  =====================================================
``interval``     ``Q⁻`` — the pessimistic end
``bounds``       ``Q⁻`` — pessimism over the identified set
``abstain``      the base algorithm's own critic, run labelled
                 ``GRACE-ABSTAINED``
===============  =====================================================

A silent fallback would make every comparison meaningless, so the label is
load-bearing: an abstained run is evidence the cell was out of scope, never
evidence about the method, and is reported separately rather than pooled.

**C3 travels.** ``GraceServing.label()`` carries the fit's conditions and the
serving mode so every run artifact records what produced its numbers.

**No per-environment parameters.** Nothing here reads an env id; the binding
audit is the enforcement.

Deferred, noted not built (the merged architectural item): L2's verdict does
not select the estimator or its channels, so the D-D path is hand-wired here
exactly as D-E's bounds were.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from .estimator import EpisodeData, LatentClassEstimator
from .l4 import point_id_interval

# The serving modes, as strings so they can travel into run artifacts.
SERVE_PESSIMISTIC = "Q-minus"
SERVE_ABSTAINED = "GRACE-ABSTAINED"

# The fit's effective defaults, ONE construction site: the signature below and
# the transform cache's key builder both read these — a default that drifted
# between them would silently key two different fits identically.
DEFAULT_ALPHA = 0.1
DEFAULT_B = 19
DEFAULT_FIT_SEED = 0
DEFAULT_INIT_SEEDS = (1, 2)
DEFAULT_FIT_KWARGS = dict(max_iter=30, m_step_budget=400, batch_size=4096)


@dataclass
class GraceServing:
    """What the fit produced, and what the head serves.

    ``reward_lo`` is the PESSIMISTIC end of L4's interval on
    ``E[R | do(a), s]`` per action — the object the serving rule names ``Q⁻``
    at the reward level. It is ``None`` when the fit abstained, and the head
    then passes through to the base critic.
    """

    mode: str = SERVE_ABSTAINED
    reason: str = ""
    fit_label: str = ""
    l4_kind: str = ""
    lo: float = float("nan")
    hi: float = float("nan")
    # THE SERVED OBJECT: interventional rewards, one per buffer transition,
    # with the pessimistic contrast already applied on a_bad. None when the
    # fit abstained, and the buffer is then left untouched.
    rewards: Optional[torch.Tensor] = None
    meta: dict = field(default_factory=dict)

    @property
    def abstained(self) -> bool:
        return self.mode == SERVE_ABSTAINED

    def label(self) -> str:
        """C3: the conditions travel with every number the head serves."""
        bits = [f"serving={self.mode}"]
        if self.abstained:
            bits.append(f"reason={self.reason!r}")
        else:
            bits.append(f"l4={self.l4_kind}[{self.lo:+.4f},{self.hi:+.4f}]")
        if self.fit_label:
            bits.append(self.fit_label)
        return " ".join(bits)


def fit_reward_transform(
    buffer,
    *,
    proxy_names: tuple = (),
    alpha: float = DEFAULT_ALPHA,
    b: int = DEFAULT_B,
    fit_seed: int = DEFAULT_FIT_SEED,
    init_seeds: tuple = DEFAULT_INIT_SEEDS,
    fit_kwargs: Optional[dict] = None,
    device=None,
) -> GraceServing:
    """THE handoff: fit L3 + L4 once, then hand back the INTERVENTIONAL rewards.

    **Why a reward transform and not a served Q** (ruled 2026-08-31). Catalogue
    fact 3: no wired cell has a ``U -> S_next`` edge, so on these cells the
    dynamics are UNCONFOUNDED and the confounding enters through the reward
    channel alone. Then ``Q_do`` is exactly the value the base algorithm
    computes when trained on interventional rewards -- the transition model was
    the only thing distinguishing "causal critic" from "reward-corrected base",
    and the diagram says it contributes nothing here. The model-based path was
    also measured DIVERGING (V 1.11 -> 251 over 60 sweeps on d_a_null, where
    truth is ~9: LinearGaussian samples s' off the manifold, the Q-net
    extrapolates, the max selects the extrapolation, and it compounds).

    So the served object is built ONLY from components that have been measured:
    ``r_hat`` via ``interventional_sweep`` (V-C1: 80-97% of the correctable
    bias removed), L4's interval (V4: coverage measured), and the recorded
    transitions and dones (exact by construction). Nothing new is served.

    It also leaves the base algorithm's own conservatism intact: handing a
    plain fitted-Q to CQL would bypass the very regulariser CQL exists for and
    turn the comparison into "regularised vs unregularised". Substituting one
    column does not.
    """
    fk = dict(fit_kwargs or DEFAULT_FIT_KWARGS)
    try:
        data, _nxt, _dn = _episode_data_from_buffer(
            buffer, proxy_names=proxy_names, device=device
        )
    except Exception as exc:
        return GraceServing(reason=f"buffer not episode-grouped: {exc}")
    if data is None:
        return GraceServing(reason="buffer carried no usable episodes")
    missing = [p_ for p_ in proxy_names if p_ not in data.proxy]
    if missing:
        return GraceServing(
            reason=(
                f"declared proxy channels {missing} absent from the buffer -- "
                "the loader needs load_proxies=True for a proximal cell"
            )
        )

    dev = data.state.device
    n_actions = int(data.action.max().item()) + 1
    a_bad = min(1, n_actions - 1)
    others = [j for j in range(n_actions) if j != a_bad]

    def make_estimator(seed: int) -> LatentClassEstimator:
        return LatentClassEstimator(
            state_dim=int(data.state.shape[1]),
            n_actions=n_actions,
            proxy_names=tuple(proxy_names),
            device=dev,
            seed=int(seed),
        )

    def _sweep(est, fit, action, states):
        parts = []
        for k in range(0, states.shape[0], 4096):
            chunk = states[k : k + 4096]
            v = est.interventional_sweep(
                chunk, [action] * chunk.shape[0], fit
            ).value.reshape(-1)
            parts.append(v.detach().cpu().numpy())  # query_batch: inference mode
        return np.concatenate(parts)

    # ONE interval, on the CONTRAST. Per-action intervals come from different
    # bootstrap draws, so their difference is not the difference's interval --
    # and the contrast is what the pessimism rule acts on.
    def contrast_target(est_r, fit_r):
        bad = _sweep(est_r, fit_r, a_bad, data.state).mean()
        oth = float(
            np.mean([_sweep(est_r, fit_r, j, data.state).mean() for j in others])
        )
        return float(bad - oth)

    res = point_id_interval(
        make_estimator=make_estimator,
        data=data,
        target=contrast_target,
        fit_kwargs=dict(fk, init="proxy" if proxy_names else "random"),
        alpha=alpha,
        b=b,
        fit_seed=fit_seed,
        init_seeds=init_seeds,
    )
    if res.kind == "abstain":
        # An abstention carries diagnostics too, and an abstained run is
        # reported SEPARATELY rather than pooled -- so why it abstained is the
        # whole content of that row.
        return GraceServing(
            reason=res.reason, fit_label=res.label, meta=_l4_diagnostics(res)
        )

    # r_hat per transition, from the observed fit (deterministic, so this is
    # the same fit the interval was built around).
    est0 = make_estimator(fit_seed)
    fit0 = est0.fit(data, **dict(fk, init="proxy" if proxy_names else "random"))
    r_hat = np.stack([_sweep(est0, fit0, a, data.state) for a in range(n_actions)], 1)
    contrast_hat = float((r_hat[:, a_bad] - r_hat[:, others].mean(axis=1)).mean())
    # PESSIMISM, clamped: it may only ever REDUCE a_bad's advantage. The
    # bootstrap's low end can sit above the point estimate, and applying that
    # unclamped would raise a_bad -- inverting the correction.
    pess = max(0.0, contrast_hat - float(res.lo))
    # Scale for the determinism sentinel below. Normalising by the CONTRAST
    # alone is ill-conditioned exactly where the contrast is ~0 -- on d_a_null
    # both quantities are float noise near zero and the ratio came out 1.3,
    # 10^5 past the flag line, on a perfectly healthy fit. The summation error
    # the sentinel watches scales with the SUMMANDS (r_hat, order of the reward)
    # and not with their difference, so the reward scale is the honest floor.
    _r_scale = float(np.mean(np.abs(r_hat))) or 1.0
    _sc_abs = abs(contrast_hat - float(res.observed))
    acts = data.action.reshape(-1).detach().cpu().numpy()
    new_r = r_hat[np.arange(acts.size), acts].astype(np.float32)
    new_r[acts == a_bad] -= pess
    return GraceServing(
        mode=SERVE_PESSIMISTIC,
        fit_label=res.label,
        l4_kind=res.kind,
        lo=res.lo,
        hi=res.hi,
        rewards=torch.as_tensor(new_r, device=dev),
        meta={
            "contrast_point": contrast_hat,
            "contrast_observed_l4": res.observed,
            # THE SELF-CHECK, as a number rather than an eyeball. These two are
            # the same quantity from the same deterministic fit, summed in two
            # different orders -- mean-of-differences here, difference-of-means
            # inside ``contrast_target`` -- so over ~5e4 float32 rows they agree
            # to ~1e-7 relative and to nothing tighter. That makes this a
            # DETERMINISM sentinel, not a drift detector between two estimates:
            # it can only fire if the observed fit stopped being reproducible,
            # which is the CUDA nondeterminism this module carried until
            # deterministic kernels became the default. Flag past ~1e-5; that is
            # far too wide to be summation order.
            "contrast_selfcheck_abs": _sc_abs,
            "contrast_selfcheck_rel": _sc_abs / max(abs(float(res.observed)), _r_scale),
            "pessimism_applied": pess,
            "n_transitions": int(data.n),
            "n_a_bad": int((acts == a_bad).sum()),
            "procedural_share": res.procedural_share,
            "failure_rate": res.failure_rate,
            **_l4_diagnostics(res),
        },
    )


def _l4_diagnostics(res) -> dict:
    """L4's variance decomposition and bootstrap health, flattened to scalars.

    **The founding rule this restores** (ruled 2026-08-23, stated in
    ``l4.py``): "failures may correlate with the statistic, so their REASONS
    travel with every interval -- a rate without reasons is uninterpretable."
    ``point_id_interval`` obeys it and puts them in ``res.meta``. This seam
    dropped ``res.meta`` wholesale, so every run artifact recorded a bare
    ``failure_rate`` with no reasons attached -- precisely the shape the rule
    forbids, and the reason a 3/19 rate could not be read for structure.

    Flattened because the artifact writers keep only ``(int, float, str,
    bool)``: a nested dict was discarded a second time downstream, so
    propagating ``res.meta`` unflattened would have fixed nothing.
    """
    m = dict(getattr(res, "meta", None) or {})
    out = {
        "optimiser_var": float(m.get("optimiser_var", float("nan"))),
        "replicate_var": float(m.get("replicate_var", float("nan"))),
        "n_init_fits": int(m.get("n_init_fits", 0)),
    }
    for k, v in (m.get("bootstrap_diagnostics") or {}).items():
        if k == "reasons":
            # The REASONS themselves, not just how many. Joined rather than
            # counted: two replicates failing for different causes is a
            # different fact than two failing for one.
            out["boot_reasons"] = " | ".join(v) if v else ""
        elif isinstance(v, (int, float, bool)):
            out[f"boot_{k}"] = v
    return out


def apply_reward_transform(buffer, serving: GraceServing) -> bool:
    """Overwrite the buffer's reward column in place. Returns whether it fired.

    On abstention the rewards are left UNTOUCHED, so an abstained run is
    byte-identical to its base -- which is what makes ``GRACE-ABSTAINED`` a
    safe fallback rather than a silent third behaviour.
    """
    serving.meta["transform_applied"] = False
    serving.meta["n_rewards_written"] = 0
    if serving.abstained or serving.rewards is None:
        return False
    eps = getattr(buffer, "episodes", None)
    if eps is not None and len(eps) > 0:  # sequence layout: write per transition
        vals = serving.rewards.reshape(-1)
        i = 0
        for e in eps:
            for tr in e.transitions:
                tr["rewards"] = vals[i].to(tr["rewards"].dtype).to(tr["rewards"].device)
                i += 1
        _record_write(serving, written=i, rows=i, spare=int(vals.numel()) - i)
        return True
    col = getattr(buffer, "_data", {}).get("rewards")
    if col is None:
        raise TypeError(f"{type(buffer).__name__} exposes no reward column to rewrite")
    n = serving.rewards.shape[0]
    col[:n] = serving.rewards.reshape(col[:n].shape).to(col.dtype).to(col.device)
    # ``col[:n]`` is a PREFIX write: any row past n keeps its OBSERVATIONAL
    # reward, and nothing about the run would look wrong if that happened.
    # Compare against the buffer's FILL, never its capacity -- a ReplayBuffer
    # allocates its column at capacity, so ``col.shape[0]`` reported coverage
    # 300/310 = 0.97 on a complete transform and would have flagged a partial
    # write on every healthy run.
    rows = len(buffer) if hasattr(buffer, "__len__") else int(col.shape[0])
    _record_write(serving, written=n, rows=int(rows), spare=0)
    return True


def _record_write(serving: GraceServing, *, written: int, rows: int, spare: int):
    """Evidence that the substitution ACTUALLY HAPPENED, on the artifact.

    The sixth silent failure of this campaign was a GRACE arm that ran the
    whole way through and produced correct-looking CSVs while the transform sat
    on a path nothing called -- byte-identical to its own baseline, no error
    anywhere. The rule adopted from it: **a component that can no-op silently
    must record what it DID, not only what it produced**, because output that
    looks right is compatible with nothing having happened.

    So the count of rewards actually overwritten travels into the provenance.
    ``coverage < 1`` means part of the buffer trained on observational rewards
    -- a partial transform, which is a silent half-no-op of the same species.
    """
    serving.meta["transform_applied"] = written > 0
    serving.meta["n_rewards_written"] = int(written)
    serving.meta["n_buffer_rows"] = int(rows)
    serving.meta["n_rewards_unused"] = int(spare)
    serving.meta["rewards_coverage"] = float(written) / float(rows) if rows else 0.0


def _episode_data_from_buffer(buffer, *, proxy_names=(), device=None):
    """Returns ``(EpisodeData, next_obs, dones)`` -- the fitted iteration needs
    the transition targets and the termination labels, not just the episode
    blocks."""
    """Rebuild ``EpisodeData`` from whatever the runner handed over.

    Accepts the sequence buffer's padded ``(B, T, ...)`` layout and the flat
    replay layout with an episode id column. Returns ``None`` when there is
    nothing to fit rather than raising, so the caller can abstain with a
    reason.
    """
    # ``ReplayBuffer`` (the runner's offline path) exposes its columns only
    # through ``gather``/``_data``, never as attributes -- reading attributes
    # alone would make GRACE abstain on EVERY real run while looking like a
    # scope decision. Take the dict form first, then fall back to attributes
    # (the sequence buffer and the test fixtures).
    cols = None
    # SequenceReplayBuffer: episodes -> transitions. This is the buffer the
    # GROUPED offline loop uses, which is the path taken whenever critic
    # ablation is configured -- so a transform that only understood the flat
    # ReplayBuffer would never fire on a run that has an ablation critic, and
    # the arm would be byte-identical to its baseline with no error anywhere.
    eps = getattr(buffer, "episodes", None)
    if eps is not None and len(eps) > 0:
        keys = list(eps[0].transitions[0].keys())
        cols = {
            k: torch.stack([tr[k] for e in eps for tr in e.transitions]) for k in keys
        }
        cols["episode_ids"] = torch.cat(
            [
                torch.full((len(e.transitions),), i, dtype=torch.long)
                for i, e in enumerate(eps)
            ]
        )
    elif hasattr(buffer, "gather") and len(buffer) > 0:
        cols = buffer.gather(range(len(buffer)))
    elif isinstance(buffer, dict):
        cols = buffer
    if cols is not None:
        obs, act = cols.get("obs"), cols.get("actions")
        rew, nxt = cols.get("rewards"), cols.get("next_obs")
        dn, ep = cols.get("dones"), cols.get("episode_ids")
        # The DECLARED proxy channels, if the loader was asked for them
        # (load_proxies). Their absence is not an error here -- a cell may
        # declare none -- but a cell that DOES declare them and does not carry
        # them would silently fit the "without" arm, so the caller checks.
        found = {
            k[len("proxy_") :]: v for k, v in cols.items() if k.startswith("proxy_")
        }
    else:
        found = {}
        obs = getattr(buffer, "obs", None)
        if obs is None:
            obs = getattr(buffer, "observations", None)
        act = getattr(buffer, "actions", None)
        rew = getattr(buffer, "rewards", None)
        nxt = getattr(buffer, "next_obs", None)
        dn = getattr(buffer, "dones", None)
        ep = getattr(buffer, "episode_ids", None)
    if obs is None or act is None or rew is None:
        raise TypeError(
            f"{type(buffer).__name__} exposes no obs/actions/rewards to fit on"
        )
    if ep is None and getattr(obs, "ndim", 0) == 3:  # padded (B, T, *) layout
        bsz, t = obs.shape[0], obs.shape[1]
        ep = (
            torch.arange(bsz, device=obs.device).reshape(-1, 1).expand(bsz, t)
        ).reshape(-1)
        obs = obs.reshape(bsz * t, -1)
        act = act.reshape(-1)
        rew = rew.reshape(-1)
        nxt = nxt.reshape(bsz * t, -1) if nxt is not None else None
        dn = dn.reshape(-1) if dn is not None else None
    if ep is None and dn is not None:
        # Flat layout WITH done flags: episode ids are the running count of
        # completed episodes, which is what makes the blocks recoverable.
        d = torch.as_tensor(dn).reshape(-1)
        ep = torch.cat([torch.zeros(1, device=d.device), d[:-1].cumsum(0)]).long()
    if ep is None:
        raise TypeError("flat buffer without episode ids or dones: GRACE needs blocks")
    dev = device or getattr(obs, "device", "cpu")
    t_ = lambda x, dt=torch.float32: torch.as_tensor(x, dtype=dt, device=dev)
    obs, act, rew, ep = t_(obs), t_(act, torch.long), t_(rew), t_(ep, torch.long)
    if obs.ndim == 1:
        obs = obs.reshape(-1, 1)
    nxt = t_(nxt) if nxt is not None else obs.roll(-1, dims=0)
    if nxt.ndim == 1:
        nxt = nxt.reshape(-1, 1)
    dn = t_(dn) if dn is not None else torch.zeros(obs.shape[0], device=dev)
    if int(torch.unique(ep).numel()) < 2:
        return None, None, None
    proxy = {k: t_(v).reshape(-1) for k, v in (found or {}).items() if k in proxy_names}
    return (
        EpisodeData(state=obs, action=act, reward=rew, episode_ids=ep, proxy=proxy),
        nxt,
        dn.reshape(-1).float(),
    )


def transform_offline_rewards(
    buffer, *, cache_dir=None, dataset_id: str = "", **options
) -> GraceServing:
    """The runner's entry point: fit (or load the cached fit), substitute,
    report.

    Called once after the offline fill and BEFORE any gradient step, so the
    base algorithm trains on interventional rewards from its first step. The
    returned ``GraceServing`` carries the C3 label into the run artifacts.

    ``cache_dir`` enables the transform cache (transform_cache.py): the fit is
    a measured pure function of (data, options) — 10/10 bitwise-identical
    production pairs, equal reward hashes under perturbed global RNG — so a
    hit substitutes the cached reward column and skips the ~1h fit. The key is
    content-addressed over the EXACT fit inputs; hit/store is recorded in
    ``meta`` (S15: the artifact says what the run DID). Cached ABSTENTIONS are
    honoured too, and stay visibly abstentions.
    """
    key = None
    if cache_dir:
        from src.rl.offline.grace import transform_cache as tc

        try:
            data, nxt, dn = _episode_data_from_buffer(
                buffer,
                proxy_names=tuple(options.get("proxy_names", ()) or ()),
                device=options.get("device"),
            )
        except Exception:
            data = None
        if data is not None:
            key = tc.build_key(
                dataset_id=dataset_id,
                data_sha256=tc.data_fingerprint(data, nxt, dn),
                proxy_names=tuple(options.get("proxy_names", ()) or ()),
                alpha=options.get("alpha", DEFAULT_ALPHA),
                b=options.get("b", DEFAULT_B),
                fit_seed=options.get("fit_seed", DEFAULT_FIT_SEED),
                init_seeds=options.get("init_seeds", DEFAULT_INIT_SEEDS),
                fit_kwargs=dict(options.get("fit_kwargs") or DEFAULT_FIT_KWARGS),
                device_kind=str(data.state.device.type),
            )
            hit = tc.load(cache_dir, key)
            if hit is not None:
                if not hit.abstained:
                    apply_reward_transform(buffer, hit)
                else:
                    hit.meta["transform_applied"] = False
                    hit.meta["n_rewards_written"] = 0
                return hit

    serving = fit_reward_transform(buffer, **options)
    if key is not None:
        from src.rl.offline.grace import transform_cache as tc

        entry = tc.store(cache_dir, key, serving)
        serving.meta["transform_cache_stored"] = str(entry)
    if not serving.abstained:
        apply_reward_transform(buffer, serving)
    else:
        # EVERY run records what it did, abstentions included -- an absent key
        # and a False one are not the same evidence, and "no key" is exactly
        # how a silent no-op looks.
        serving.meta["transform_applied"] = False
        serving.meta["n_rewards_written"] = 0
    return serving
