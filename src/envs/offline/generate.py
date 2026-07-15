"""B2 offline GENERATE pipeline: train -> snapshot-by-return -> rollout -> write.

Produces tiered (random/medium/expert), optionally provenance-varied, Minari
datasets for unhosted domains by reusing the online training loop
(``BenchmarkRunner``), the checkpoint machinery, the A-series collection-policy
seam (so a dataset is characterized by ``tier x behavior_policy``), and the
``make_*_offline`` Minari-write path. Consumption stays through B1's load path
(``--offline-dataset`` -> ``load_minari_dataset``).

Purely additive: no edits to the online/offline/load paths. The generator is an
OFF-POLICY online algo (dqn/sac/ddpg) — the rollout drives ``agent.act`` and the
provenance policies reach into the critic/buffer; on-policy generators (a
``policy.act`` adapter) are deferred.
"""

from __future__ import annotations

import csv
import os
import sys

import numpy as np
import torch

_DISCRETE_ONLY = {"dqn"}
_CONTINUOUS_ONLY = {"sac", "ddpg"}


# --------------------------------------------------------------------------
# Tier selection (pure / deterministic) — performance-defined, sign-robust
# --------------------------------------------------------------------------
def select_tier_episode(
    returns: dict[int, float], tier: str, fraction: float = 1.0 / 3.0
):
    """Select the checkpoint episode for ``tier`` from ``{episode: eval_return}``.

    * ``expert`` -> the argmax-return checkpoint (earliest, on ties).
    * ``medium`` -> the FIRST checkpoint reaching
      ``R_random + fraction*(R_expert - R_random)`` (the range-based
      generalization of D4RL's "1/3 of expert return"; sign-robust for
      negative-return envs like Pendulum). ``R_random`` is the lowest
      checkpoint return.
    * ``random`` -> ``None`` (signals a fresh untrained agent, no checkpoint).
    """
    if tier == "random":
        return None
    if not returns:
        raise ValueError("no eval returns recorded; cannot select a tier")
    items = sorted(returns.items())  # by episode
    r_expert = max(returns.values())
    if tier == "expert":
        return min(ep for ep, r in items if r == r_expert)
    if tier == "medium":
        r_random = min(returns.values())
        target = r_random + fraction * (r_expert - r_random)
        for ep, r in items:
            if r >= target:
                return ep
        return items[-1][0]
    raise ValueError(f"unknown tier '{tier}' (expected random/medium/expert)")


# --------------------------------------------------------------------------
# Guards (both fire BEFORE any training)
# --------------------------------------------------------------------------
def assert_online_generator(algo: str) -> None:
    """Reject generating WITH an offline algo (the category error)."""
    from src.benchmarking.registry import registry

    if registry.get(algo).data_regime != "online":
        raise ValueError(
            f"cannot generate with offline algo '{algo}'; the generator must be "
            "an online algo (dqn/sac/ddpg)."
        )


def assert_action_space_match(algo: str, env_action_type: str) -> None:
    """Reject a generator whose action type can't match the env's."""
    if algo in _DISCRETE_ONLY and env_action_type != "discrete":
        raise ValueError(
            f"generator '{algo}' is discrete-only but the env action space is "
            f"{env_action_type}; use sac/ddpg for continuous envs."
        )
    if algo in _CONTINUOUS_ONLY and env_action_type != "continuous":
        raise ValueError(
            f"generator '{algo}' is continuous-only but the env action space is "
            f"{env_action_type}; use dqn for discrete envs."
        )


# --------------------------------------------------------------------------
# Naming + rollout env (provenance: confounded wraps the rollout env)
# --------------------------------------------------------------------------
def _sigma_suffix(behavior_strength: float) -> str:
    """``-sigma{NNN}`` where NNN = round(sigma * 100) zero-padded to 3 digits.
    ``round`` (not truncation) so 0.3 * 100 = 29.999... -> 030, not 029."""
    return f"-sigma{int(round(behavior_strength * 100)):03d}"


def dataset_name(
    env_id: str,
    tier: str,
    behavior_policy: str = "agent",
    behavior_strength: float | None = None,
) -> str:
    """``generated/{env_slug}/{tier}[-{behavior}][-sigma{NNN}]-v0``.

    The behavior suffix is omitted for the clean 'agent' rollout. For
    ``bias_confounded`` WITH a set ``behavior_strength`` the rollout strength
    sigma is encoded as ``-sigma{NNN}`` (sigma x 100, 3-digit zero-padded) so
    different sigma produce DISTINCT dataset ids — required for Cell 7's sigma
    sweep. ``bias_confounded`` with ``behavior_strength=None`` falls back to the
    bare ``-bias_confounded-v0`` form (the pre-PR8 placeholder, which no Cell
    uses) so the existing convention is preserved.
    """
    slug = env_id.split("-v")[0].lower().replace("/", "-")
    if behavior_policy == "agent":
        suffix = ""
    elif (
        behavior_policy in ("bias_confounded", "bias_confounded_action")
        and behavior_strength is not None
    ):
        # action-gated carries its own name so its datasets never collide with the
        # additive confounder's; both encode sigma for the strength sweep.
        suffix = f"-{behavior_policy}{_sigma_suffix(float(behavior_strength))}"
    else:
        suffix = f"-{behavior_policy}"
    return f"generated/{slug}/{tier}{suffix}-v0"


def build_rollout_env(
    env_id,
    n_envs,
    device,
    seed,
    behavior_policy="agent",
    strength=None,
    c_r=None,
    a_bad=1,
):
    """Build the rollout env, wrapped in the confounder iff bias_confounded[_action].

    ``c_r`` (action-dependent path only) is the FIXED U->R reward-shift magnitude,
    decoupled from ``strength`` (sigma): sigma scales the U->A edge via the behavior
    policy, while c_r on U->R is invariant across the sigma sweep (default 1.0). The
    additive path ignores c_r and keeps ``c_r = c_a = sigma`` (byte-frozen)."""
    from src.envs.registry import build_env

    env = build_env(env_id=env_id, n_envs=n_envs, device=device, seed=seed)
    if behavior_policy in ("bias_confounded", "bias_confounded_action"):
        from src.envs.wrappers.confounded import ConfoundedCollectionWrapper

        sig = 1.0 if strength is None else float(strength)
        # action_gated (action-dependent cell) gates the reward shift on a==a_bad;
        # additive (default) is the byte-frozen cells-7/8 path. Thread the generation
        # seed into the wrapper's isolated U RNG (issue #36) so the confounding latent
        # — and thus the gate-test signature — is reproducible regardless of cumulative
        # process RNG state.
        kind = (
            "action_gated"
            if behavior_policy == "bias_confounded_action"
            else "additive"
        )
        c_r_val = (
            (1.0 if c_r is None else float(c_r)) if kind == "action_gated" else sig
        )
        env = ConfoundedCollectionWrapper(
            env, c_a=sig, c_r=c_r_val, seed=seed, confounder_kind=kind, a_bad=int(a_bad)
        )
    return env


def _env_dims(env):
    if len(env.obs_space.shape) == 0:
        obs_dim = 1
    else:
        obs_dim = int(torch.tensor(env.obs_space.shape).prod().item())
    obs_shape = tuple(env.obs_space.shape)
    act_space = env.act_space
    if hasattr(act_space, "n"):
        return obs_dim, obs_shape, "discrete", int(act_space.n), act_space
    return obs_dim, obs_shape, "continuous", int(act_space.shape[0]), act_space


def _to_np(obs):
    return obs.reshape(obs.shape[0], -1)[0].detach().cpu().numpy()


def _rollout(env, collection_policy, n_episodes, seed, action_type, max_steps=1000):
    """Roll out ``n_episodes`` (n_envs=1) into Minari EpisodeBuffers. Explicit
    per-episode reset + break-on-done keeps clean episode boundaries (and the
    confounder's per-episode U resamples at each reset).

    Returns ``(buffers, signature_samples)`` where ``signature_samples`` is a
    dict of flat float arrays ``{a, r, u}`` over all transitions — the scalar
    action (L2 norm for multi-dim), reward, and the per-transition latent ``U``
    (read from the confounder via ``env.current_u`` BEFORE the step, i.e. the
    ``U`` that this transition's action and reward share). It is ``None`` when
    the env is not a confounder (``current_u`` absent), so the clean path is
    unchanged.
    """
    from minari.data_collector.episode_buffer import EpisodeBuffer

    confounded = hasattr(env, "current_u")
    sig_a, sig_r, sig_u, sig_iv, sig_ps = [], [], [], [], []
    # The action-dependent gate needs the per-transition pi_basic(a_bad|s). The
    # marginally-matched policy exposes it via ``_base_action_probs`` (a READ of
    # pi_basic — no behavior change, and DQN's argmax path draws no RNG, so the
    # dataset stays byte-identical). Additive's policy has no such method -> None.
    _ps_fn = getattr(collection_policy, "_base_action_probs", None)
    _ps_a_bad = int(getattr(collection_policy, "a_bad", 1))
    # State-conditional coverage: log per-transition min_a p_b(a|s), the U-MARGINALIZED
    # realized per-state action distribution. Exposed by the three arms' policies via
    # ``action_probs`` (pi_basic / skew-on-pi_basic / confounded=pi_basic); the clean
    # 'agent' and additive ConfoundedBehaviorPolicy have no such method -> None ->
    # those datasets stay byte-identical.
    _probs_fn = getattr(collection_policy, "action_probs", None)
    buffers = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + 1000 + ep)
        obs_list = [_to_np(obs)]
        acts, rews, terms, truncs = [], [], [], []
        ep_u: list[float] = []  # per-transition U for this episode (confounded only)
        ep_iv: list[bool] = (
            []
        )  # per-transition intervened flag (when the policy emits it)
        ep_cmin: list[float] = []  # per-transition min_a p_b(a|s) (arms only)
        done = False
        steps = 0
        while not done and steps < max_steps:
            # current_u BEFORE the step is the latent this transition shares
            # (the confounder resamples U at done, AFTER perturbing the reward).
            u_t = float(env.current_u.reshape(-1)[0].item()) if confounded else None
            # min_a p_b(a|s) BEFORE act (a read; no RNG -> act/dataset unchanged).
            cmin_t = (
                float(_probs_fn(obs)[0].min().item()) if _probs_fn is not None else None
            )
            # pi_basic(a_bad|s) BEFORE act (a read; no RNG consumed -> act unchanged).
            ps_t = (
                float(_ps_fn(obs)[0, _ps_a_bad].item())
                if (confounded and _ps_fn is not None)
                else None
            )
            act_out = collection_policy.act(obs)
            action = act_out.action
            # intervened: emitted only by the marginally-matched confounded policy
            # (None otherwise -> the additive / clean paths stay byte-identical).
            iv_t = (
                bool(act_out.intervened.reshape(-1)[0].item())
                if act_out.intervened is not None
                else None
            )
            obs, reward, term, trunc, _ = env.step(action)
            obs_list.append(_to_np(obs))
            a = action.reshape(action.shape[0], -1)[0].detach().cpu().numpy()
            acts.append(
                int(a[0]) if action_type == "discrete" else a.astype(np.float32)
            )
            r_t = float(reward.reshape(-1)[0].item())
            rews.append(r_t)
            terms.append(bool(term.reshape(-1)[0].item()))
            truncs.append(bool(trunc.reshape(-1)[0].item()))
            if confounded:
                # Scalar action: the index (discrete) or L2 norm (continuous).
                sig_a.append(
                    float(a[0])
                    if action_type == "discrete"
                    else float(np.linalg.norm(a))
                )
                sig_r.append(r_t)
                sig_u.append(u_t)
                ep_u.append(u_t)
                if iv_t is not None:  # flat intervened stream for the A5 gate check
                    sig_iv.append(iv_t)
                if ps_t is not None:  # flat pi_basic(a_bad|s) for the A2'/A3' checks
                    sig_ps.append(ps_t)
            if iv_t is not None:
                ep_iv.append(iv_t)
            if cmin_t is not None:  # state-conditional coverage, all three arms
                ep_cmin.append(cmin_t)
            done = terms[-1] or truncs[-1]
            steps += 1
        adt = np.int64 if action_type == "discrete" else np.float32
        # Persist the per-transition latent U (length T, aligned with actions/
        # rewards) into the episode infos ONLY when confounded — the oracle-U
        # ceiling reads it back. The clean path omits infos so non-confounded
        # datasets stay byte-identical to the pre-oracle generator.
        infos: dict = {}
        if confounded:
            infos["confounder_u"] = np.asarray(ep_u, dtype=np.float32)
        # Persist the per-transition intervened flag (marginally-matched confounded
        # policy only) so it survives to the dataset. Offline it is all-False by
        # construction; it is written alongside confounder_u without perturbing the
        # additive path (which emits no intervened flag -> ep_iv empty -> not added).
        if ep_iv:
            infos["intervened"] = np.asarray(ep_iv, dtype=bool)
        # State-conditional coverage min_a p_b(a|s) for the arm policies (basic / biased
        # / confounded); absent for the clean 'agent' + additive paths (byte-frozen).
        if ep_cmin:
            infos["coverage_min"] = np.asarray(ep_cmin, dtype=np.float32)
        ep_kwargs = {"infos": infos} if infos else {}
        buffers.append(
            EpisodeBuffer(
                observations=np.asarray(obs_list, dtype=np.float32),
                actions=np.asarray(acts, dtype=adt),
                rewards=np.asarray(rews, dtype=np.float32),
                terminations=np.asarray(terms, dtype=bool),
                truncations=np.asarray(truncs, dtype=bool),
                **ep_kwargs,
            )
        )
    samples = (
        {
            "a": np.asarray(sig_a, dtype=np.float64),
            "r": np.asarray(sig_r, dtype=np.float64),
            "u": np.asarray(sig_u, dtype=np.float64),
            "intervened": np.asarray(sig_iv, dtype=np.float64),  # empty for additive
            "p_s": np.asarray(sig_ps, dtype=np.float64),  # empty for additive
        }
        if confounded
        else None
    )
    return buffers, samples


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    # Guard against a constant series (zero variance -> undefined corr -> 0).
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


# Declarative gate defaults (docs/rl_regimes_restructure.md §3.4). ``additive`` keeps
# its byte-frozen thresholds; ``action_dependent`` is the PR-2 point check.
ADDITIVE_GATE: dict = {"type": "additive"}
ACTION_DEPENDENT_GATE: dict = {
    "type": "action_dependent",
    "corr_tolerance": 0.03,
    "ungated_reward_corr_max": 0.05,
    "intervened_tolerance": 0.02,
    "entropy_min": 0.05,  # min mean(p_s(1-p_s)): the confounder is inert below this
}


def default_gate_for(behavior_policy: str) -> dict:
    """The declarative gate config implied by a behavior policy when a YAML/CLI does
    not give one: ``bias_confounded_action`` -> action_dependent, else additive."""
    if behavior_policy == "bias_confounded_action":
        return dict(ACTION_DEPENDENT_GATE)
    return dict(ADDITIVE_GATE)


def compute_confounding_signature(
    samples: dict,
    sigma: float | None,
    *,
    gate: dict | None = None,
    a_bad: int = 1,
    is_online: bool = False,
) -> dict:
    """Per-dataset confounding signature, DISPATCHED on ``gate['type']`` (never on the
    behavior-policy name). See docs/rl_regimes_restructure.md §3.

    ``additive`` (byte-frozen, cells 7/8): ``corr_a_r_marginal`` = Corr(A, R),
    ``corr_a_r_partial_given_u`` = partial Corr(A, R | U); passes iff
    ``|marginal| > 0.2`` AND ``|partial| < 0.05``. Exact 4-key dict, unchanged.

    ``action_dependent`` (the action-gated confounder): a POINT check against the swap's
    closed form ``corr(1[a=a_bad], U) = sigma*sqrt(p(1-p))`` — see
    ``_action_dependent_signature``.
    """
    gate = gate or ADDITIVE_GATE
    if gate.get("type", "additive") == "additive":
        a, r, u = samples["a"], samples["r"], samples["u"]
        r_ar, r_au, r_ru = _pearson(a, r), _pearson(a, u), _pearson(r, u)
        denom = np.sqrt((1 - r_au**2) * (1 - r_ru**2))
        partial = float((r_ar - r_au * r_ru) / denom) if denom > 0 else 0.0
        gate_passed = bool(abs(r_ar) > 0.2 and abs(partial) < 0.05)
        return {
            "corr_a_r_marginal": float(r_ar),
            "corr_a_r_partial_given_u": partial,
            "gate_test_passed": gate_passed,
            "behavior_strength_sigma": float(sigma) if sigma is not None else None,
        }
    return _action_dependent_signature(
        samples, float(sigma or 0.0), gate, int(a_bad), bool(is_online)
    )


def _action_dependent_signature(
    samples: dict, sigma: float, gate: dict, a_bad: int, is_online: bool
) -> dict:
    """POINT check for the action-gated confounder. NOT a ``corr > tau`` threshold.

    A2 uses the per-transition ``p_s = pi_basic(a_bad|s)``, NOT the closed form
    ``corr(1[a=a_bad], U) = sigma*sqrt(p(1-p))`` the brief specifies. That closed form
    holds only for a STATE-INDEPENDENT pi_basic; over a rollout with state-dependent
    ``p(s)`` the aggregate corr is Jensen-deflated below the marginal-p prediction
    (empirically obs 0.386 vs pred 0.461), so the specified check REJECTS a legitimately
    confounded dataset — the opposite of this PR's goal. The exact, aggregation-
    invariant statistic (derivation in docs/report):

        mean( (1[a=a_bad] - p_s) * (2U - 1) )  ==  sigma * mean( p_s*(1-p_s) )

    A2 asserts these agree within ``corr_tolerance``. A3 (the inert-confounder catch)
    asserts ``mean(p_s(1-p_s)) > entropy_min`` — this catches a GREEDY per-state-
    degenerate pi_basic (p_s in {0,1}) that the marginal-p check MISSES (its marginal p
    can look non-degenerate, e.g. 0.49, while the confounder is fully inert).
    """
    a, r, u = samples["a"], samples["r"], samples["u"]
    p_s = samples.get("p_s")
    corr_tol = float(gate.get("corr_tolerance", 0.03))
    ungated_max = float(gate.get("ungated_reward_corr_max", 0.05))
    iv_tol = float(gate.get("intervened_tolerance", 0.02))
    entropy_min = float(gate.get("entropy_min", 0.05))

    ab = (a == a_bad).astype(np.float64)  # 1[a == a_bad]
    p_hat = float(ab.mean()) if ab.size else 0.0  # marginal (diagnostic only)
    if p_s is None or p_s.size != a.size:
        raise ValueError(
            "action_dependent gate requires per-transition p_s = pi_basic(a_bad|s); "
            "the collection policy did not expose _base_action_probs."
        )
    entropy = float(np.mean(p_s * (1.0 - p_s))) if p_s.size else 0.0
    # A2 — exact, aggregation-invariant point check (centered by p_s).
    stat = float(np.mean((ab - p_s) * (2.0 * u - 1.0)))
    target = sigma * entropy
    a2 = bool(abs(stat - target) < corr_tol)
    # A3 — the confounder is NOT inert: pi_basic has real entropy on the gated pair.
    a3 = bool(entropy > entropy_min)
    # A4 — gated U->R live within a==a_bad; dead within a!=a_bad.
    mask = a == a_bad
    corr_r_u_gated = _pearson(r[mask], u[mask]) if int(mask.sum()) > 1 else 0.0
    corr_r_u_ungated = _pearson(r[~mask], u[~mask]) if int((~mask).sum()) > 1 else 0.0
    a4 = bool(corr_r_u_gated > 0.0 and abs(corr_r_u_ungated) < ungated_max)
    # A5 — interventional fraction: ~= 1-sigma online, == 0 offline.
    iv = samples.get("intervened")
    mean_iv = float(np.mean(iv)) if iv is not None and iv.size else 0.0
    target_iv = (1.0 - sigma) if is_online else 0.0
    a5 = bool(abs(mean_iv - target_iv) < iv_tol)
    return {
        "gate_type": "action_dependent",
        "gate_test_passed": bool(a2 and a3 and a4 and a5),
        "behavior_strength_sigma": sigma,
        "p_hat": p_hat,
        "pi_basic_entropy": entropy,
        "edge_statistic_observed": stat,
        "edge_statistic_predicted": target,
        "corr_a_bad_u_marginal": float(_pearson(ab, u)),  # diagnostic (Jensen-deflated)
        "corr_r_u_gated": float(corr_r_u_gated),
        "corr_r_u_ungated": float(corr_r_u_ungated),
        "intervened_mean": mean_iv,
        "check_a2_point_corr": a2,
        "check_a3_p_nondegenerate": a3,
        "check_a4_gated_reward": a4,
        "check_a5_intervened": a5,
    }


def enforce_confounding_gate(meta: dict, dataset_id: str) -> None:
    """Single enforcement point for the confounding gate (deduped from the two verbatim
    copies in the runner). Raises on a missing signature or a failed gate.

    Declarative dispatch, keyed on the STAMPED ``gate_type`` (never the behavior-policy
    name): the ``action_dependent`` gate is computed correctly AT sigma=0 (A2 predicts
    corr ~ 0 and asserts the OBSERVED corr is ~ 0 -> no U->A edge), so it is
    authoritative with NO exemption. The byte-frozen ``additive`` gate has no
    ``gate_type`` key and CANNOT validate its sigma=0 baseline (marginal Corr(A,R) ~ 0
    by construction), so that ONE case is skipped exactly as before.
    """
    if "gate_test_passed" not in meta:
        raise ValueError(
            f"Confounded offline run on dataset '{dataset_id}' requires the "
            "confounding-signature metadata, but none is present (likely generated "
            "before this metadata existed). Regenerate with tools/generate_offline.py."
        )
    gate_type = meta.get("gate_type")  # present only for action_dependent
    if gate_type is None and meta.get("behavior_strength_sigma") == 0.0:
        print(
            "[runner] sigma=0.0 additive anchor: skipping the additive confounding gate "
            "(the dataset is the unconfounded baseline by construction).",
            file=sys.stderr,
        )
        return
    if not bool(meta["gate_test_passed"]):
        if gate_type == "action_dependent":
            failed = [
                k
                for k in (
                    "check_a2_point_corr",
                    "check_a3_p_nondegenerate",
                    "check_a4_gated_reward",
                    "check_a5_intervened",
                )
                if not meta.get(k, True)
            ]
            raise ValueError(
                f"Dataset '{dataset_id}' failed the action-dependent confounding gate "
                f"(failed checks: {', '.join(failed) or 'unknown'}; "
                f"p_hat={meta.get('p_hat')}, corr_obs={meta.get('corr_a_bad_u_observed')}"
                f", corr_pred={meta.get('corr_a_bad_u_predicted')}). "
                "Regenerate or inspect the dataset."
            )
        raise ValueError(
            f"Dataset '{dataset_id}' failed the confounding gate test "
            "(gate_test_passed=False): the confounding signature (non-zero marginal "
            "Corr(A,R), near-zero partial Corr(A,R|U)) did not hold at generation. "
            "Regenerate or inspect the dataset."
        )


def _read_eval_returns(run_dir: str) -> dict[int, float]:
    with open(os.path.join(run_dir, "eval_metrics.csv")) as f:
        rows = list(csv.DictReader(f))
    return {int(r["episode"]): float(r["eval_return_mean"]) for r in rows}


# --------------------------------------------------------------------------
# Shared-generator plumbing (PR 5, CHANGE 1)
# --------------------------------------------------------------------------
def generator_checkpoint_hash(agent) -> str:
    """A stable content hash of the generator's parameters.

    Every arm of a cell must be collected under ONE ``pi_basic``; ``pi_basic`` (and
    the biased / confounded arms) all derive from ``agent``'s policy nets, so the
    agent's ``state_dict`` fully determines it. The sweep driver stamps this into
    each dataset's metadata and refuses a cell whose arms carry different hashes.
    Deterministic: keys sorted, raw tensor bytes; identical agent -> identical hash."""
    import hashlib

    h = hashlib.sha256()
    sd = None
    fn = getattr(agent, "state_dict", None)
    if callable(fn):
        try:
            sd = fn()
        except Exception:
            sd = None
    if not sd:  # fall back to scanning nn.Module attributes (the policy nets)
        import torch.nn as nn

        sd = {}
        for attr, val in vars(agent).items():
            if isinstance(val, nn.Module):
                for pk, pv in val.state_dict().items():
                    sd[f"{attr}.{pk}"] = pv
    for k in sorted(sd.keys()):
        v = sd[k]
        h.update(str(k).encode())
        if torch.is_tensor(v):
            h.update(v.detach().cpu().contiguous().float().numpy().tobytes())
        else:
            h.update(repr(v).encode())
    return h.hexdigest()[:16]


def build_generator_agent(
    env_id: str,
    generator_algo: str,
    tier: str,
    *,
    seed: int = 0,
    train_episodes: int = 50,
    n_checkpoints: int = 10,
    fraction: float = 1.0 / 3.0,
    run_dir: str | None = None,
    device: str | None = None,
):
    """Build ONE generator agent for a cell and return ``(agent, hash)``.

    The sweep driver calls this once per (env, seed) and hands the SAME ``agent`` to
    every ``generate_offline_dataset`` call of the cell (see CHANGE 1) so the basic /
    biased / confounded arms share a single ``pi_basic``. For the ``random`` tier the
    agent is a fresh (untrained) build; otherwise it is trained and snapshotted at
    the tier's return percentile — the identical train/build/load the fresh-agent
    path performs internally, just surfaced so the object can be reused."""
    import gymnasium as gym

    from src.benchmarking.registry import register_default_algorithms, registry
    from src.config.device import detect_device
    from src.config.seeding import set_seed
    from src.envs.registry import register_default_env_wrappers

    register_default_algorithms()
    register_default_env_wrappers()
    # Seed BEFORE building the generator net so the shared checkpoint is REPRODUCIBLE
    # per (env, seed): the whole cell is collected under one deterministic pi_basic,
    # so its confounding realization (and the gate outcome) is stable across runs.
    set_seed(seed, deterministic=True)
    assert_online_generator(generator_algo)
    probe = gym.make(env_id)
    env_action_type = "discrete" if hasattr(probe.action_space, "n") else "continuous"
    probe.close()
    assert_action_space_match(generator_algo, env_action_type)

    dev = torch.device(device) if device else detect_device()
    sel_ep = None
    if tier != "random":
        if run_dir is None:
            raise ValueError("non-random tiers require run_dir for the generator")
        _train_generator(
            env_id, generator_algo, train_episodes, n_checkpoints, seed, run_dir, dev
        )
        sel_ep = select_tier_episode(_read_eval_returns(run_dir), tier, fraction)

    # A plain (behavior_policy="agent") rollout env just to read the canonical dims.
    probe_env = build_rollout_env(env_id, 1, dev, seed, "agent", None)
    obs_dim, obs_shape, action_type, action_dim, action_space = _env_dims(probe_env)
    probe_env.close()
    _, agent = registry.get(generator_algo).builder(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_type=action_type,
        device=dev,
        action_space=action_space,
        obs_shape=obs_shape,
    )
    if sel_ep is not None:
        from src.benchmarking.checkpoints import load_checkpoint

        tag = env_id.replace("/", "-")
        ckpt = load_checkpoint(
            os.path.join(
                run_dir,
                "checkpoints",
                f"{tag}_{generator_algo}_seed{seed}",
                f"ckpt_ep{sel_ep:04d}.pt",
            )
        )
        agent.load_state_dict(ckpt["agent_state"])
    return agent, generator_checkpoint_hash(agent)


# --------------------------------------------------------------------------
# The pipeline
# --------------------------------------------------------------------------
def generate_offline_dataset(
    env_id: str,
    generator_algo: str,
    tier: str,
    *,
    behavior_policy: str = "agent",
    behavior_strength: float | None = None,
    confounder_c_r: float | None = None,
    pi_basic_epsilon: float | None = None,
    a_bad: int = 1,
    gate: dict | None = None,
    fraction: float = 1.0 / 3.0,
    train_episodes: int = 50,
    n_checkpoints: int = 10,
    rollout_episodes: int = 20,
    seed: int = 0,
    dataset_id: str | None = None,
    run_dir: str | None = None,
    device: str | None = None,
    agent=None,
):
    """Train an online generator, snapshot the ``tier`` policy by return, roll it
    out (optionally via a collection policy), and write a Minari dataset to the
    local cache. Returns the created MinariDataset.

    ``agent`` (PR 5, CHANGE 1): a PRE-BUILT generator agent. When supplied, the
    per-call train/build/checkpoint-load is SKIPPED and this exact agent defines
    ``pi_basic`` for the rollout. The sweep driver builds ONE agent per (env, seed)
    and passes it to every sweep point of a cell, so the basic / biased / confounded
    arms all share a SINGLE ``pi_basic`` — otherwise each arm gets a fresh generator
    and every cross-arm comparison is confounded by generator variance. The
    generator's parameter hash is stamped into the dataset metadata
    (``generator_checkpoint_hash``) so the driver can REFUSE a cell whose arms carry
    different hashes. With ``agent=None`` the legacy fresh-agent-per-call path is
    byte-unchanged."""
    import gymnasium as gym

    from src.benchmarking.registry import register_default_algorithms, registry
    from src.config.device import detect_device
    from src.envs.registry import register_default_env_wrappers
    from src.rl.policies.behavior_policy import (
        AgentBehaviorPolicy,
        build_collection_policy,
    )

    register_default_algorithms()
    register_default_env_wrappers()

    # --- guards (before any training) ---
    assert_online_generator(generator_algo)
    probe = gym.make(env_id)
    env_action_type = "discrete" if hasattr(probe.action_space, "n") else "continuous"
    probe.close()
    assert_action_space_match(generator_algo, env_action_type)

    dev = torch.device(device) if device else detect_device()

    # --- train (skipped for the random tier, and when a pre-built agent is given) ---
    sel_ep = None
    if agent is None and tier != "random":
        if run_dir is None:
            raise ValueError("non-random tiers require run_dir for the generator")
        _train_generator(
            env_id, generator_algo, train_episodes, n_checkpoints, seed, run_dir, dev
        )
        sel_ep = select_tier_episode(_read_eval_returns(run_dir), tier, fraction)

    # --- rollout env + agent ---
    rollout_env = build_rollout_env(
        env_id,
        1,
        dev,
        seed,
        behavior_policy,
        behavior_strength,
        c_r=confounder_c_r,
        a_bad=a_bad,
    )
    obs_dim, obs_shape, action_type, action_dim, action_space = _env_dims(rollout_env)
    # CHANGE 1: a pre-built shared generator agent short-circuits the fresh build +
    # checkpoint load — the SAME pi_basic across all sweep points of the cell.
    if agent is None:
        _, agent = registry.get(generator_algo).builder(
            obs_dim=obs_dim,
            action_dim=action_dim,
            action_type=action_type,
            device=dev,
            action_space=action_space,
            obs_shape=obs_shape,
        )
    # sel_ep is set ONLY on the internal-build path (train ran); a pre-built agent
    # leaves it None, so this tier-checkpoint load is correctly skipped for it.
    if sel_ep is not None:
        from src.benchmarking.checkpoints import load_checkpoint

        tag = env_id.replace("/", "-")
        ckpt = load_checkpoint(
            os.path.join(
                run_dir,
                "checkpoints",
                f"{tag}_{generator_algo}_seed{seed}",
                f"ckpt_ep{sel_ep:04d}.pt",
            )
        )
        agent.load_state_dict(ckpt["agent_state"])

    if behavior_policy == "agent":
        collection_policy = AgentBehaviorPolicy(agent)
    else:
        collection_policy = build_collection_policy(
            behavior_policy,
            agent,
            action_type,
            action_space,
            behavior_strength,
            env=rollout_env,
            pi_basic_epsilon=pi_basic_epsilon,
        )

    buffers, sig_samples = _rollout(
        rollout_env, collection_policy, rollout_episodes, seed, action_type
    )
    rollout_env.close()

    import minari

    name = dataset_id or dataset_name(env_id, tier, behavior_policy, behavior_strength)
    ds = minari.create_dataset_from_buffers(
        dataset_id=name,
        buffer=buffers,
        env=gym.make(env_id),
        algorithm_name=f"{generator_algo}-{tier}-{behavior_policy}",
    )

    # Confounding-signature metadata: computed once per dataset and stored in the Minari
    # metadata block. The gate is DECLARATIVE — dispatched on gate['type'] (default
    # derived from behavior_policy), NOT string-matched on the policy name — so the
    # action-dependent path now gets a proper signature instead of the all-None dict the
    # runner used to reject. The four fields stay None for non-confounded datasets.
    if (
        behavior_policy in ("bias_confounded", "bias_confounded_action")
        and sig_samples is not None
    ):
        _gate = gate if gate is not None else default_gate_for(behavior_policy)
        signature = compute_confounding_signature(
            sig_samples, behavior_strength, gate=_gate, a_bad=a_bad, is_online=False
        )
    else:
        signature = {
            "corr_a_r_marginal": None,
            "corr_a_r_partial_given_u": None,
            "gate_test_passed": None,
            "behavior_strength_sigma": None,
        }
    # CHANGE 1: stamp the generator's parameter hash so a sweep driver can PROVE all
    # arms of a cell were collected under one pi_basic (refuse a cell whose arms
    # differ). Stamped on every dataset — internal-build or pre-built agent alike.
    signature["generator_checkpoint_hash"] = generator_checkpoint_hash(agent)
    ds.storage.update_metadata(signature)
    return ds


def _train_generator(env_id, algo, train_episodes, n_checkpoints, seed, run_dir, dev):
    from src.benchmarking.registry import registry
    from src.benchmarking.runner import BenchmarkRunner
    from src.config.defaults import EnvConfig, RunConfig, TrainingConfig

    env_cfg = EnvConfig(
        env_id=env_id, n_train_envs=4, n_eval_envs=4, rollout_len=64, seed=seed
    )
    train_cfg = TrainingConfig(
        n_episodes=train_episodes,
        n_checkpoints=n_checkpoints,
        device=str(dev),
        algorithm=algo,
        aggregation="mean",
    )
    BenchmarkRunner(
        env_cfg,
        train_cfg,
        RunConfig(run_dir=run_dir, timestamp="t"),
        registry.get(algo),
    ).run()
