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

import torch
import torch.nn as nn

from .estimator import EpisodeData, LatentClassEstimator
from .l4 import point_id_interval

# The serving modes, as strings so they can travel into run artifacts.
SERVE_PESSIMISTIC = "Q-minus"
SERVE_ABSTAINED = "GRACE-ABSTAINED"


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
    # per-action offsets applied to the base critic's Q, shape (A,)
    action_offset: Optional[torch.Tensor] = None
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


class GraceQNetwork(nn.Module):
    """The serving head: the base critic, plus GRACE's per-action correction.

    Before a fit lands (and whenever the fit abstains) this is a LITERAL
    pass-through, so a GRACE run that never fits is byte-identical to its base
    — which is what makes ``GRACE-ABSTAINED`` a safe fallback rather than a
    silent third behaviour.

    The correction is an additive per-action offset rather than a wholesale Q
    replacement, deliberately: the base learner's Q carries the sequential
    value the fitted iteration would otherwise have to reproduce, while the
    offset carries the identified do-effect the base learner cannot see. A
    wholesale replacement would discard the learner and make the comparison a
    comparison of two different algorithms.
    """

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base
        self.serving: GraceServing = GraceServing()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.base(x)
        off = self.serving.action_offset
        if off is None:  # not fitted, or abstained -> pass through
            return q
        return q + off.to(q.device, q.dtype).reshape(1, -1)

    # The base critic's other surfaces must keep working: an agent may call
    # ``q_su``/``q_at`` on its network (the OracleU/Proximal hooks). Delegate
    # rather than reimplement, so a missing attribute fails loudly upstream.
    def __getattr__(self, name):
        # nn.Module owns parameter/buffer/submodule lookup (``base`` included);
        # only genuinely-missing names delegate onward. Overriding this without
        # deferring to super() first makes ``self.base`` itself unreachable.
        try:
            return super().__getattr__(name)
        except AttributeError:
            base = self._modules.get("base")
            if base is None:
                raise
            return getattr(base, name)


def fit_from_buffer(
    agent,
    buffer,
    *,
    proxy_names: tuple = (),
    alpha: float = 0.1,
    b: int = 19,
    fit_seed: int = 0,
    # (1, 2) not (1, 2, 3, 4): the init-perturbation arm is DIAGNOSTIC (it
    # gives the procedural share), so two perturbations buy the diagnostic at
    # half the fits. Matches V4's production setting.
    init_seeds: tuple = (1, 2),
    fit_kwargs: Optional[dict] = None,
    device=None,
) -> GraceServing:
    """THE handoff: fit L3 once on the episode-grouped buffer, serve thereafter.

    Returns the ``GraceServing`` that the head will use. Every failure path
    returns an ABSTENTION carrying its reason — never a silent fallback, and
    never an exception that would take the run down: a cell GRACE cannot serve
    is a cell the base algorithm still trains on, and the label says so.
    """
    fk = dict(fit_kwargs or dict(max_iter=30, m_step_budget=400, batch_size=4096))
    try:
        data = _episode_data_from_buffer(buffer, proxy_names=proxy_names, device=device)
    except Exception as exc:  # buffer shape the seam does not understand
        return GraceServing(reason=f"buffer not episode-grouped: {exc}")
    if data is None:
        return GraceServing(reason="buffer carried no usable episodes")

    n_actions = int(data.action.max().item()) + 1
    state_dim = int(data.state.shape[1])

    def make_estimator(seed: int) -> LatentClassEstimator:
        return LatentClassEstimator(
            state_dim=state_dim,
            n_actions=n_actions,
            proxy_names=tuple(proxy_names),
            device=data.state.device,
            seed=int(seed),
        )

    # The estimand served per action: E[R | do(a), s] averaged over the
    # buffer's own states. ``interventional_sweep`` is the READ-ONLY path
    # (query_batch, N1a) -- nothing here feeds a loss, so the non-
    # differentiable API is the right one.
    ev = data.state

    # ONE interval, on the CONTRAST -- not one per action. Two reasons, and
    # the second is the load-bearing one:
    #   * cost: a per-action loop pays the whole bootstrap twice (2 x (1
    #     observed + inits + B replicates)) for a quantity that is a difference;
    #   * COHERENCE: per-action intervals come from DIFFERENT bootstrap draws,
    #     so their difference is not the difference's interval. Only the
    #     contrast can move an argmax (the offsets are centred below), so the
    #     contrast is the estimand whose uncertainty the serving rule needs.
    a_bad = min(1, n_actions - 1)
    others = [j for j in range(n_actions) if j != a_bad]

    def contrast_target(est, fit):
        bad = est.interventional_sweep(ev, [a_bad] * ev.shape[0], fit).value.mean()
        oth = sum(
            est.interventional_sweep(ev, [j] * ev.shape[0], fit).value.mean()
            for j in others
        ) / max(len(others), 1)
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
    label = res.label
    if res.kind == "abstain":
        return GraceServing(reason=res.reason, fit_label=res.label)
    # THE SERVING RULE: the pessimistic end, for both `interval` and `bounds`.
    # For a CONTRAST, pessimism is its LOW end -- confounding inflates a_bad's
    # apparent value, so the conservative statement is the smallest advantage
    # the identified set allows it. No compatible model contradicts it.
    los, his, kinds = [res.lo], [res.hi], [res.kind]
    off = torch.zeros(n_actions, dtype=torch.float32, device=data.state.device)
    off[a_bad] = float(res.lo)
    # Only the CONTRAST between actions can matter to a policy, so centre the
    # offset: a constant shift on every action changes no argmax and would
    # otherwise leak the reward scale into the base critic's magnitudes.
    off = off - off.mean()
    return GraceServing(
        mode=SERVE_PESSIMISTIC,
        fit_label=label,
        l4_kind=kinds[0] if kinds else "",
        lo=min(los) if los else float("nan"),
        hi=max(his) if his else float("nan"),
        action_offset=off,
        meta={
            "per_action_lo": los,
            "per_action_hi": his,
            "n_episodes": int(torch.unique(data.episode_ids).numel()),
            "n_transitions": int(data.n),
        },
    )


def _episode_data_from_buffer(buffer, *, proxy_names=(), device=None):
    """Rebuild ``EpisodeData`` from whatever the runner handed over.

    Accepts the sequence buffer's padded ``(B, T, ...)`` layout and the flat
    replay layout with an episode id column. Returns ``None`` when there is
    nothing to fit rather than raising, so the caller can abstain with a
    reason.
    """
    get = getattr(buffer, "as_episode_arrays", None)
    if callable(get):  # explicit seam if a buffer offers one
        obs, act, rew, ep = get()
    else:
        obs = getattr(buffer, "observations", None)
        act = getattr(buffer, "actions", None)
        rew = getattr(buffer, "rewards", None)
        ep = getattr(buffer, "episode_ids", None)
        if obs is None or act is None or rew is None:
            raise TypeError(
                f"{type(buffer).__name__} exposes neither as_episode_arrays() nor "
                "observations/actions/rewards"
            )
        if ep is None:  # padded (B, T, *) sequence layout
            if getattr(obs, "ndim", 0) != 3:
                raise TypeError(
                    "flat buffer without episode ids: GRACE needs episode blocks"
                )
            bsz, t = obs.shape[0], obs.shape[1]
            ep = (
                torch.arange(bsz, device=obs.device).reshape(-1, 1).expand(bsz, t)
            ).reshape(-1)
            obs = obs.reshape(bsz * t, -1)
            act = act.reshape(-1)
            rew = rew.reshape(-1)
    dev = device or getattr(obs, "device", "cpu")
    t_ = lambda x, dt=torch.float32: torch.as_tensor(x, dtype=dt, device=dev)
    obs, act, rew, ep = t_(obs), t_(act, torch.long), t_(rew), t_(ep, torch.long)
    if obs.ndim == 1:
        obs = obs.reshape(-1, 1)
    if int(torch.unique(ep).numel()) < 2:
        return None
    return EpisodeData(state=obs, action=act, reward=rew, episode_ids=ep)


def install_grace(agent, net: GraceQNetwork, **fit_options) -> None:
    """Bind the fit to the buffer handoff. Idempotent per agent."""

    def _set_sequence_buffer(buffer):
        serving = fit_from_buffer(agent, buffer, **fit_options)
        net.serving = serving
        tgt = getattr(agent, "target_network", None)
        if isinstance(tgt, GraceQNetwork):
            tgt.serving = serving
        agent.grace_serving = serving  # so the runner can record the label
        return serving

    agent.set_sequence_buffer = _set_sequence_buffer


def _wrap(agent, net, **fit_options):
    wrapped = GraceQNetwork(net)
    agent.q_network = wrapped
    tgt = getattr(agent, "target_network", None)
    if tgt is not None:
        agent.target_network = GraceQNetwork(tgt)
    install_grace(agent, wrapped, **fit_options)
    return wrapped, agent


def _grace_options(kwargs) -> dict:
    """Method options only — never an environment id (the A1/binding rule)."""
    opts = dict(kwargs.pop("grace_options", None) or {})
    allowed = {
        "proxy_names",
        "alpha",
        "b",
        "fit_seed",
        "init_seeds",
        "fit_kwargs",
        "device",
    }
    unknown = set(opts) - allowed - {"router", "interval", "deploy"}
    if unknown:
        raise ValueError(f"unknown grace option(s): {sorted(unknown)}")
    # v1's router/deploy switches are not part of the v2 serving rule; accept
    # and ignore them so the frozen CRITIC_LIBRARY specs stay loadable, but do
    # not silently pretend to honour them.
    for legacy in ("router", "interval", "deploy"):
        opts.pop(legacy, None)
    return opts


def build_grace_cql(**kwargs):
    from src.rl.offline.cql import build_cql

    opts = _grace_options(kwargs)
    net, agent = build_cql(**kwargs)
    return _wrap(agent, net, **opts)


def build_grace_iql(**kwargs):
    from src.rl.offline.iql import build_iql

    opts = _grace_options(kwargs)
    net, agent = build_iql(**kwargs)
    return _wrap(agent, net, **opts)


def build_grace_dqn(**kwargs):
    from src.rl.offline.dqn import build_offline_dqn

    opts = _grace_options(kwargs)
    net, agent = build_offline_dqn(**kwargs)
    return _wrap(agent, net, **opts)
