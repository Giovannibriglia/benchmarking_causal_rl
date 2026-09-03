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
    behavior_mask_indices: tuple | None = None,
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
    if behavior_mask_indices:
        # S6: a masked-behavior dataset is a DIFFERENT identity -- same config
        # without the marker would collide with the full-view dataset.
        suffix += "-om" + "".join(str(i) for i in behavior_mask_indices)
    return f"generated/{slug}/{tier}{suffix}-v0"


class _MaskedViewPolicy:
    """Behavior-policy INFORMATION-SET restriction (Finding 1, 2026-09-02).

    Deletes the masked observation columns BEFORE delegating to the inner
    policy, so the behavior acts on the partial view (`O -> A`, the edge
    D-F/D-G declare) while the rollout loops -- untouched -- keep storing the
    FULL observation for ground truth and preflight. Masking at load time
    instead leaves the logged actions dependent on the hidden components
    (`S -> A`), a different diagram than the catalogue asserts.

    The inner policy's diagnostic prob reads (``action_probs``,
    ``_base_action_probs``) are forwarded THROUGH the same view: the inner
    network is masked-dim, and their ABSENCE is mirrored (``__getattr__``
    raises), so the rollout's ``getattr(..., None)`` probes see exactly what
    the inner policy exposes.
    """

    def __init__(self, inner, indices):
        self._inner = inner
        self._idx = tuple(int(i) for i in indices)

    def _view(self, obs):
        if isinstance(obs, torch.Tensor):
            keep = [i for i in range(obs.shape[-1]) if i not in set(self._idx)]
            return obs[..., keep]
        a = np.asarray(obs)
        return np.delete(a, list(self._idx), axis=-1)

    def act(self, obs):
        return self._inner.act(self._view(obs))

    def __getattr__(self, name):
        v = getattr(self._inner, name)  # AttributeError propagates, mirrored
        if name in ("action_probs", "_base_action_probs"):
            return lambda obs, _f=v: _f(self._view(obs))
        return v


def build_rollout_env(
    env_id,
    n_envs,
    device,
    seed,
    behavior_policy="agent",
    strength=None,
    c_r=None,
    a_bad=1,
    proxy_strength=None,
    instrument_strength=None,
    u_drift=0.0,
    gate_probs=None,
    n_proxies=2,
):
    """Build the rollout env, wrapped in the confounder iff bias_confounded[_action].

    ``c_r`` (action-dependent path only) is the FIXED U->R reward-shift magnitude,
    decoupled from ``strength`` (sigma): sigma scales the U->A edge via the behavior
    policy, while c_r on U->R is invariant across the sigma sweep (default 1.0). The
    additive path ignores c_r and keeps ``c_r = c_a = sigma`` (byte-frozen).

    ``proxy_strength`` / ``instrument_strength`` / ``u_drift`` are the GRACE v2
    diagram-arm channels (D-D / D-E / D-B'). All default off and are drawn from
    the wrapper's dedicated auxiliary generator, so an arm that leaves them off
    consumes exactly the RNG it always did."""
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
            env,
            c_a=sig,
            c_r=c_r_val,
            seed=seed,
            confounder_kind=kind,
            a_bad=int(a_bad),
            proxy_strength=proxy_strength,
            n_proxies=n_proxies,
            instrument_strength=instrument_strength,
            u_drift=u_drift,
            gate_probs=gate_probs,
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
    # GRACE v2 diagram channels, present only on the arms that enable them.
    # Read at the SAME point as U (before the step), so a row's (U, Z, W, I) is
    # the tuple that this transition's action and reward actually shared.
    _has_proxy = getattr(env, "current_z", None) is not None
    _has_v = getattr(env, "current_v", None) is not None
    _has_instr = getattr(env, "current_i", None) is not None
    sig_a, sig_r, sig_u, sig_iv, sig_ps = [], [], [], [], []
    sig_z, sig_w, sig_v, sig_i, sig_ep = [], [], [], [], []
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
    from tqdm import tqdm

    buffers = []
    # Progress for the generation phase (display-only; the RL training loops
    # carry their own tqdm) — without it a 3000-episode rollout is minutes of
    # silence in the sweep-worker logs.
    for ep in tqdm(range(n_episodes), desc="dataset generation", leave=False):
        obs, _ = env.reset(seed=seed + 1000 + ep)
        obs_list = [_to_np(obs)]
        acts, rews, terms, truncs = [], [], [], []
        ep_u: list[float] = []  # per-transition U for this episode (confounded only)
        ep_iv: list[bool] = (
            []
        )  # per-transition intervened flag (when the policy emits it)
        ep_cmin: list[float] = []  # per-transition min_a p_b(a|s) (arms only)
        ep_z: list[float] = []  # D-D proxy Z (episode-constant, logged per row)
        ep_w: list[float] = []  # D-D proxy W
        ep_v: list[float] = []  # D-D proxy V (third view, 2026-08-21 revision)
        ep_i: list[float] = []  # D-E instrument I
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
            if _has_proxy:
                ep_z.append(float(env.current_z.reshape(-1)[0].item()))
                ep_w.append(float(env.current_w.reshape(-1)[0].item()))
                if _has_v:
                    ep_v.append(float(env.current_v.reshape(-1)[0].item()))
            if _has_instr:
                ep_i.append(float(env.current_i.reshape(-1)[0].item()))
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
                sig_ep.append(ep)
                if _has_proxy:
                    sig_z.append(ep_z[-1])
                    sig_w.append(ep_w[-1])
                    if _has_v:
                        sig_v.append(ep_v[-1])
                if _has_instr:
                    sig_i.append(ep_i[-1])
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
        # D-D / D-E channels. Written only when the arm enables them, so every
        # pre-existing dataset's infos block is unchanged key-for-key.
        if ep_z:
            infos["proxy_z"] = np.asarray(ep_z, dtype=np.float32)
            infos["proxy_w"] = np.asarray(ep_w, dtype=np.float32)
            if ep_v:
                infos["proxy_v"] = np.asarray(ep_v, dtype=np.float32)
        if ep_i:
            infos["instrument_i"] = np.asarray(ep_i, dtype=np.float32)
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
            # Diagram-arm channels + the episode index, flat over transitions.
            # ``episode`` is what makes the preflight's permutation null
            # EPISODE-level: U and the proxies are episode-constant, so a
            # step-level shuffle shatters the blocks and the null comes out far
            # too tight (a zero-signal view was called rank 2).
            "z": np.asarray(sig_z, dtype=np.float64),
            "w": np.asarray(sig_w, dtype=np.float64),
            "v": np.asarray(sig_v, dtype=np.float64),
            "i": np.asarray(sig_i, dtype=np.float64),
            "episode": np.asarray(sig_ep, dtype=np.int64),
        }
        if confounded
        else None
    )
    return buffers, samples


def _rollout_vectorized(
    env, collection_policy, n_episodes, seed, action_type, max_steps=1000
):
    """Vectorized twin of ``_rollout``: ``env.n_envs`` slots stepped in lockstep
    with ONE batched policy forward per step (strategy S2 in
    docs/dataset_generation_speedup.md).

    Same OUTPUT CONTRACT as ``_rollout`` — ``n_episodes`` EpisodeBuffers plus the
    flat ``signature_samples`` dict — so the gate, the metadata stamp and the
    Minari write are all unchanged. Three deliberate differences:

      * NOT byte-identical to ``_rollout``, and not meant to be: policy draws are
        batched across slots and the per-slot env streams come from a single
        vector reset (slot ``i`` gets ``seed + 1000 + i``). A vectorized dataset
        is reproducible for a fixed ``(seed, n_envs)`` pair and self-certifies
        through the confounding gate exactly as the scalar path does.
      * Deterministic OUTPUT ORDER: episodes are keyed by the index assigned when
        a slot picks them up and emitted sorted by that index, so the dataset
        bytes never depend on which slot happened to finish first.
      * Autoreset (gymnasium NEXT_STEP): the step AFTER a slot's terminal step
        returns that slot's reset observation with a dummy action/reward. It
        opens the slot's next episode and is never recorded as a transition.

    A slot with no episode left to collect keeps being stepped (the vector env is
    all-or-nothing) but its data is discarded.
    """
    from minari.data_collector.episode_buffer import EpisodeBuffer

    confounded = hasattr(env, "current_u")
    _has_proxy = getattr(env, "current_z", None) is not None
    _has_v = getattr(env, "current_v", None) is not None
    _has_instr = getattr(env, "current_i", None) is not None
    # Same optional per-transition readers as the scalar path (pure reads, no RNG).
    _ps_fn = getattr(collection_policy, "_base_action_probs", None)
    _ps_a_bad = int(getattr(collection_policy, "a_bad", 1))
    _probs_fn = getattr(collection_policy, "action_probs", None)

    n_slots = int(env.n_envs)
    obs, _ = env.reset(seed=seed + 1000)

    def _blank():
        return {
            "obs": [],
            "acts": [],
            "rews": [],
            "terms": [],
            "truncs": [],
            "u": [],
            "iv": [],
            "cmin": [],
            "z": [],
            "v": [],
            "w": [],
            "i": [],
            "sig_a": [],
            "sig_r": [],
            "sig_u": [],
            "sig_iv": [],
            "sig_ps": [],
        }

    obs_np = obs.reshape(obs.shape[0], -1).detach().cpu().numpy()
    slot_ep: list = [None] * n_slots  # episode index each slot is building
    slot_buf: list = [None] * n_slots
    awaiting = [False] * n_slots  # True => next step returns this slot's reset obs
    next_ep = 0
    for i in range(n_slots):
        if next_ep < n_episodes:
            slot_ep[i] = next_ep
            slot_buf[i] = _blank()
            slot_buf[i]["obs"].append(obs_np[i])
            next_ep += 1

    done_eps: dict = {}
    # Loop guard: every episode is TimeLimit-bounded, so this can only trip on a
    # misconfigured env. Sized generously (worst case: every episode runs the
    # full cap, one slot at a time).
    step_cap = max_steps * (n_episodes + n_slots) + 100
    steps_taken = 0

    from tqdm import tqdm

    pbar = tqdm(total=n_episodes, desc="dataset generation (vec)", leave=False)
    while len(done_eps) < n_episodes:
        if steps_taken > step_cap:
            raise RuntimeError(
                f"vectorized rollout exceeded {step_cap} steps with "
                f"{len(done_eps)}/{n_episodes} episodes collected — is the env "
                "missing a TimeLimit?"
            )
        u_t = env.current_u.reshape(-1).detach().cpu().numpy() if confounded else None
        cmin_t = (
            _probs_fn(obs).min(dim=1).values.detach().cpu().numpy()
            if _probs_fn is not None
            else None
        )
        ps_t = (
            _ps_fn(obs)[:, _ps_a_bad].detach().cpu().numpy()
            if (confounded and _ps_fn is not None)
            else None
        )
        # Diagram channels, read at the same point as U (before the step).
        z_t = env.current_z.reshape(-1).detach().cpu().numpy() if _has_proxy else None
        w_t = env.current_w.reshape(-1).detach().cpu().numpy() if _has_proxy else None
        v_t = env.current_v.reshape(-1).detach().cpu().numpy() if _has_v else None
        i_t_aux = (
            env.current_i.reshape(-1).detach().cpu().numpy() if _has_instr else None
        )
        act_out = collection_policy.act(obs)
        action = act_out.action
        iv_t = (
            act_out.intervened.reshape(-1).detach().cpu().numpy()
            if act_out.intervened is not None
            else None
        )
        obs, reward, term, trunc, _ = env.step(action)
        steps_taken += 1

        obs_np = obs.reshape(obs.shape[0], -1).detach().cpu().numpy()
        act_np = action.reshape(action.shape[0], -1).detach().cpu().numpy()
        rew_np = reward.reshape(-1).detach().cpu().numpy()
        term_np = term.reshape(-1).detach().cpu().numpy()
        trunc_np = trunc.reshape(-1).detach().cpu().numpy()

        for i in range(n_slots):
            if awaiting[i]:
                # Autoreset step: this slot's action/reward are dummies and the
                # observation is the new episode's first one.
                awaiting[i] = False
                if slot_ep[i] is not None:
                    slot_buf[i]["obs"].append(obs_np[i])
                continue
            if slot_ep[i] is None:
                continue  # idle slot (no episodes left to assign)
            b = slot_buf[i]
            a_i = act_np[i]
            b["obs"].append(obs_np[i])
            b["acts"].append(
                int(a_i[0]) if action_type == "discrete" else a_i.astype(np.float32)
            )
            r_i = float(rew_np[i])
            b["rews"].append(r_i)
            b["terms"].append(bool(term_np[i]))
            b["truncs"].append(bool(trunc_np[i]))
            if confounded:
                b["sig_a"].append(
                    float(a_i[0])
                    if action_type == "discrete"
                    else float(np.linalg.norm(a_i))
                )
                b["sig_r"].append(r_i)
                b["sig_u"].append(float(u_t[i]))
                b["u"].append(float(u_t[i]))
                if iv_t is not None:
                    b["sig_iv"].append(bool(iv_t[i]))
                if ps_t is not None:
                    b["sig_ps"].append(float(ps_t[i]))
            if iv_t is not None:
                b["iv"].append(bool(iv_t[i]))
            if cmin_t is not None:
                b["cmin"].append(float(cmin_t[i]))
            if z_t is not None:
                b["z"].append(float(z_t[i]))
                b["w"].append(float(w_t[i]))
                if v_t is not None:
                    b["v"].append(float(v_t[i]))
            if i_t_aux is not None:
                b["i"].append(float(i_t_aux[i]))

            if bool(term_np[i]) or bool(trunc_np[i]):
                done_eps[slot_ep[i]] = b
                pbar.update(1)
                awaiting[i] = True
                # Hand this slot the next unassigned episode (or idle it).
                if next_ep < n_episodes:
                    slot_ep[i] = next_ep
                    slot_buf[i] = _blank()
                    next_ep += 1
                else:
                    slot_ep[i] = None
                    slot_buf[i] = None
    pbar.close()

    adt = np.int64 if action_type == "discrete" else np.float32
    buffers = []
    sig_a: list = []
    sig_r: list = []
    sig_u: list = []
    sig_iv: list = []
    sig_ps: list = []
    sig_z: list = []
    sig_w: list = []
    sig_v: list = []
    sig_i: list = []
    sig_ep: list = []
    for ep_idx in range(n_episodes):
        b = done_eps[ep_idx]
        infos: dict = {}
        if confounded:
            infos["confounder_u"] = np.asarray(b["u"], dtype=np.float32)
        if b["iv"]:
            infos["intervened"] = np.asarray(b["iv"], dtype=bool)
        if b["cmin"]:
            infos["coverage_min"] = np.asarray(b["cmin"], dtype=np.float32)
        if b["z"]:
            infos["proxy_z"] = np.asarray(b["z"], dtype=np.float32)
            infos["proxy_w"] = np.asarray(b["w"], dtype=np.float32)
            if b["v"]:
                infos["proxy_v"] = np.asarray(b["v"], dtype=np.float32)
        if b["i"]:
            infos["instrument_i"] = np.asarray(b["i"], dtype=np.float32)
        ep_kwargs = {"infos": infos} if infos else {}
        buffers.append(
            EpisodeBuffer(
                observations=np.asarray(b["obs"], dtype=np.float32),
                actions=np.asarray(b["acts"], dtype=adt),
                rewards=np.asarray(b["rews"], dtype=np.float32),
                terminations=np.asarray(b["terms"], dtype=bool),
                truncations=np.asarray(b["truncs"], dtype=bool),
                **ep_kwargs,
            )
        )
        sig_a.extend(b["sig_a"])
        sig_r.extend(b["sig_r"])
        sig_u.extend(b["sig_u"])
        sig_iv.extend(b["sig_iv"])
        sig_ps.extend(b["sig_ps"])
        sig_z.extend(b["z"])
        sig_w.extend(b["w"])
        sig_v.extend(b["v"])
        sig_i.extend(b["i"])
        # The episode INDEX, not a running counter: the preflight's permutation
        # null must shuffle whole episodes (U and the proxies are
        # episode-constant), and it needs these blocks to do it.
        sig_ep.extend([ep_idx] * len(b["sig_a"]))

    samples = (
        {
            "a": np.asarray(sig_a, dtype=np.float64),
            "r": np.asarray(sig_r, dtype=np.float64),
            "u": np.asarray(sig_u, dtype=np.float64),
            "intervened": np.asarray(sig_iv, dtype=np.float64),
            "p_s": np.asarray(sig_ps, dtype=np.float64),
            "z": np.asarray(sig_z, dtype=np.float64),
            "w": np.asarray(sig_w, dtype=np.float64),
            "v": np.asarray(sig_v, dtype=np.float64),
            "i": np.asarray(sig_i, dtype=np.float64),
            "episode": np.asarray(sig_ep, dtype=np.int64),
        }
        if confounded
        else None
    )
    return buffers, samples


def _preflight_certification(
    samples: dict,
    buffers,
    *,
    proxy_strength,
    instrument_strength,
    u_drift,
    max_episodes: int,
    null_arm: bool = False,
    a_bad: int = 1,
) -> dict:
    """Run the ground-truth preflight and flatten it into metadata keys.

    Direction, restated because it is the whole point: this validates the
    GENERATOR against ground truth -- the logged U and the declared parameters.
    It never consults GRACE's estimator or L5. L5 is validated against the
    generator afterwards, never the reverse.
    """
    from src.envs.offline.arm_preflight import (
        check_drift,
        check_instrument,
        check_null_arm,
        check_proxies,
    )

    ep = samples.get("episode")
    if ep is None or ep.size == 0:
        return {"preflight_ran": False, "preflight_reason": "no episode index"}
    keep = ep < int(max_episodes)
    ep_k = ep[keep]
    out: dict = {
        "preflight_ran": True,
        "preflight_episodes": int(np.unique(ep_k).size),
        "preflight_transitions": int(keep.sum()),
    }
    reasons: list = []
    passed = True

    if proxy_strength is not None:
        states = np.concatenate(
            [b.observations[:-1] for b in buffers[: int(max_episodes)]], axis=0
        )
        _v = samples.get("v")
        rep = check_proxies(
            z=samples["z"][keep],
            w=samples["w"][keep],
            v=None if _v is None or _v.size == 0 else _v[keep],
            u=samples["u"][keep],
            state=states[: int(keep.sum())],
            action=samples["a"][keep],
            reward=samples["r"][keep],
            episode_ids=ep_k,
        )
        ok = rep.covariate_free and rep.exclusions_hold and rep.kruskal_ok
        passed &= ok
        reasons += list(rep.reasons)
        # Realised R-informativeness: episode-mean R AUC against logged U --
        # binning-free, and the quantity the compensated gate sweep dials.
        # Stamped per dataset so the decorative->load-bearing transition is
        # LOCATED from certification stamps, never re-derived from memory.
        _ep = samples["episode"][keep]
        _r = samples["r"][keep]
        _u = samples["u"][keep]
        _eps = np.unique(_ep)
        _rm = np.array([_r[_ep == e].mean() for e in _eps])
        _um = np.array([_u[_ep == e][0] for e in _eps])
        _pos, _neg = _rm[_um == 1], _rm[_um == 0]
        if _pos.size and _neg.size:
            _gt = (_pos[:, None] > _neg[None, :]).mean()
            _eq = (_pos[:, None] == _neg[None, :]).mean()
            out["preflight_r_auc_episode"] = float(_gt + 0.5 * _eq)
        out.update(
            {
                "preflight_proxy_corr_z_u": rep.corr_z_u,
                "preflight_proxy_corr_v_u": rep.corr_v_u,
                "preflight_proxy_k_ranks": dict(rep.k_ranks),
                # The MARGIN, not just the verdict: Kruskal is exactly tight at
                # |U| = 2, so an arm near the boundary is fragile to sample size
                # and that has to be visible without rerunning anything (R5).
                "preflight_proxy_margins": {
                    k: float(v) for k, v in rep.condition_numbers.items()
                },
                # The permutation p-values are what the VERDICTS read; the
                # z-scores are kept only as a human-readable effect size, since
                # a max-over-a-family null is right-skewed and a 3-sd rule on it
                # is not the level it appears to be.
                "preflight_proxy_null_p": {k: float(v) for k, v in rep.null_p.items()},
                "preflight_proxy_null_sds": {
                    k: float(v) for k, v in rep.null_sds.items()
                },
                "preflight_proxy_episodes": int(rep.n_episodes),
                # A collapsed quantile grid is a FAILED MEASUREMENT, not an
                # uninformative view (S3/S8), so it travels separately from the
                # k-rank verdict it would otherwise be mistaken for.
                "preflight_proxy_binning_degenerate": {
                    k: bool(v) for k, v in rep.binning_degenerate.items()
                },
                "preflight_proxy_covariate_free": bool(rep.covariate_free),
                "preflight_proxy_kruskal_ok": bool(rep.kruskal_ok),
            }
        )

    if instrument_strength is not None:
        rep = check_instrument(
            i=samples["i"][keep],
            u=samples["u"][keep],
            action=samples["a"][keep],
            reward=samples["r"][keep],
            episode_ids=ep_k,
        )
        # Untestable is credited but RECORDED AS SUCH, never conflated with
        # verified (R2/R4).
        ok = (
            rep.independent_of_u
            and rep.relevant
            and (rep.exclusion_holds or not rep.exclusion_testable)
        )
        passed &= ok
        reasons += list(rep.reasons)
        out.update(
            {
                "preflight_instrument_null_p": {
                    k: float(v) for k, v in rep.null_p.items()
                },
                "preflight_instrument_null_sds": {
                    k: float(v) for k, v in rep.null_sds.items()
                },
                "preflight_instrument_episodes": int(rep.n_episodes),
                "preflight_instrument_exogenous": bool(rep.independent_of_u),
                "preflight_instrument_relevant": bool(rep.relevant),
                "preflight_instrument_excluded": bool(rep.exclusion_holds),
                "preflight_instrument_exclusion_testable": bool(rep.exclusion_testable),
            }
        )

    if u_drift:
        by_ep = [samples["u"][keep][ep_k == e] for e in np.unique(ep_k)]
        rep = check_drift(u_by_episode=by_ep, rho=float(u_drift))
        passed &= rep.matches
        if not rep.matches:
            reasons.append(
                f"realised autocorr {rep.realised_autocorr:+.3f} != predicted "
                f"{rep.predicted_autocorr:+.3f} for rho={u_drift}"
            )
        out.update(
            {
                "preflight_drift_rho": float(u_drift),
                "preflight_drift_realised_autocorr": float(rep.realised_autocorr),
                "preflight_drift_predicted_autocorr": float(rep.predicted_autocorr),
                # D-B' is the ONE statistic left at transition level. Its S1b
                # exemption rests on rho being homogeneous across episodes, so
                # the exemption is measured (short- vs long-episode halves)
                # rather than asserted -- a gap here means a future
                # state-dependent drift variant has voided it.
                "preflight_drift_autocorr_short_episodes": float(
                    rep.autocorr_short_episodes
                ),
                "preflight_drift_autocorr_long_episodes": float(
                    rep.autocorr_long_episodes
                ),
                "preflight_drift_length_weighting_gap": float(rep.length_weighting_gap),
                "preflight_drift_length_weighting_inert": bool(
                    rep.length_weighting_inert
                ),
            }
        )

    if null_arm:
        rep = check_null_arm(
            u=samples["u"][keep],
            action=samples["a"][keep],
            reward=samples["r"][keep],
            episode_ids=ep_k,
            a_bad=float(a_bad),
        )
        passed &= rep.u_inert
        reasons += list(rep.reasons)
        out.update(
            {
                "preflight_null_arm_u_inert": bool(rep.u_inert),
                "preflight_null_arm_null_p": {
                    k: float(v) for k, v in rep.null_p.items()
                },
                "preflight_null_arm_null_sds": {
                    k: float(v) for k, v in rep.null_sds.items()
                },
                "preflight_null_arm_gated_episodes": int(rep.gated_episodes),
                # S8: an untestable channel is credited but never reported as a
                # verified pass -- this is the arm L5's false-positive rate is
                # read from, so the distinction is the whole point.
                "preflight_null_arm_reward_testable": bool(rep.reward_testable),
                "preflight_null_arm_gated_testable": bool(rep.gated_testable),
            }
        )

    out["preflight_passed"] = bool(passed)
    out["preflight_reasons"] = list(reasons)
    return out


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
    # Whether the dataset is DECLARED to carry a U->R edge at all. A null arm
    # (c_r = 0) has none, so A4 has nothing to detect and must be SKIPPED
    # rather than expected to pass on noise (see _action_dependent_signature).
    "expect_gated_reward": True,
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
    if gate.get("type") == "instrument":
        return _instrument_signature(samples, float(sigma or 0.0), gate, int(a_bad))
    return _action_dependent_signature(
        samples, float(sigma or 0.0), gate, int(a_bad), bool(is_online)
    )


def _instrument_signature(samples: dict, sigma: float, gate: dict, a_bad: int) -> dict:
    """Gate for the D-E arm: certify the IV CONDITIONS, not the U->A point check.

    The action-gated gate's A2 identity
    ``mean((1[a=a_bad] - p_s)(2U-1)) == sigma * mean(p_s(1-p_s))`` was derived for
    the un-instrumented swap policy. An exogenous action override breaks it in
    two ways at once: it removes the U-conditional draw on a lambda fraction of
    in-pair steps, AND it changes which states get visited, which the derivation
    holds fixed. A tempting (1 - lambda) correction does NOT rescue it -- measured
    against it, the realised dilution was 0.951 / 0.827 / 0.233 at
    lambda = 0.1 / 0.3 / 0.6 versus the predicted 0.9 / 0.7 / 0.4, missing by 1.0,
    2.4 and 2.6 standard errors and erring in BOTH directions. Shipping that
    would have been a fabricated closed form dressed as a point check.

    So D-E certifies what its verdict actually rests on: I independent of U,
    I relevant for A, I excluded from R given (A, U) -- the same three
    statements the preflight measures, at the same EPISODE granularity, since I
    is drawn once per episode.
    """
    from src.envs.offline.arm_preflight import check_instrument

    i = samples.get("i")
    ep = samples.get("episode")
    if i is None or i.size == 0 or ep is None or ep.size != i.size:
        raise ValueError(
            "instrument gate requires per-transition i and episode ids; the "
            "generator did not log them."
        )
    rep = check_instrument(
        i=i, u=samples["u"], action=samples["a"], reward=samples["r"], episode_ids=ep
    )
    # Exclusion is credited when it holds OR when the env makes it untestable
    # (deterministic reward given (A,U)); the metadata records which, so a reader
    # can tell "verified" from "not applicable" -- never silently conflated.
    exclusion_ok = rep.exclusion_holds or not rep.exclusion_testable
    return {
        "gate_type": "instrument",
        "gate_test_passed": bool(
            rep.independent_of_u and rep.relevant and exclusion_ok
        ),
        "behavior_strength_sigma": sigma,
        "instrument_strength": float(gate.get("instrument_strength", 0.0) or 0.0),
        "corr_i_u": rep.corr_i_u,
        "corr_i_action": rep.corr_i_action,
        "corr_i_reward_given_action_and_u": rep.corr_i_reward_given_action_and_u,
        "null_sds": dict(rep.null_sds),
        "check_i_exogenous": rep.independent_of_u,
        "check_i_relevant": rep.relevant,
        "check_i_excluded": rep.exclusion_holds,
        "exclusion_testable": rep.exclusion_testable,
    }


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
    #
    # NULL ARM: when the gate declares no U->R edge (expect_gated_reward=False,
    # set by the generator for c_r == 0), there is no gated correlation to
    # detect. corr_r_u_gated is then pure noise around 0 and is NEGATIVE about
    # half the time, so requiring `> 0.0` would reject a deliberately
    # signature-free dataset on a coin flip. The check is SKIPPED by
    # declaration and the metadata records that it was, so a later reader can
    # tell "not applicable" from "passed".
    expect_gated = bool(gate.get("expect_gated_reward", True))
    mask = a == a_bad
    corr_r_u_gated = _pearson(r[mask], u[mask]) if int(mask.sum()) > 1 else 0.0
    corr_r_u_ungated = _pearson(r[~mask], u[~mask]) if int((~mask).sum()) > 1 else 0.0
    if expect_gated:
        a4 = bool(corr_r_u_gated > 0.0 and abs(corr_r_u_ungated) < ungated_max)
    else:
        a4 = True  # vacuous: no U->R edge is claimed
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
        "gated_reward_expected": expect_gated,
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


def generation_fingerprint(
    *,
    env_id: str,
    generator_algo: str,
    tier: str,
    behavior_policy: str,
    behavior_strength,
    confounder_c_r,
    pi_basic_epsilon,
    a_bad: int,
    rollout_episodes: int,
    seed: int,
    generator_hash: str,
    rollout_device: str,
    rollout_n_envs: int,
    legacy_rollout: bool,
    proxy_strength=None,
    instrument_strength=None,
    u_drift: float = 0.0,
    gate_probs=None,
    n_proxies: int = 2,
) -> str:
    """Hash of EVERY input that determines a generated dataset's contents (S4).

    Stamped into the dataset metadata at generation and compared before a
    regeneration: an identical fingerprint means re-running the pipeline would
    reproduce the same dataset, so the existing one can be reused instead. This
    is what makes cross-simulation reuse safe — ``classical`` and
    ``critic_ablation`` cells of one regime share dataset ids (the id carries no
    simulation component), so the second cell would otherwise re-generate
    byte-equivalent data from scratch.

    Everything that moves the data is in the key, INCLUDING the rollout mode
    (device / slot count / legacy flag), because the fast rollout paths are not
    byte-identical to the legacy one. The generator's parameter hash covers
    ``pi_basic`` itself. A change to any of them misses the cache and
    regenerates — the fingerprint is never a claim about equivalence, only about
    identity of inputs.
    """
    import hashlib

    parts = [
        ("env_id", env_id),
        ("generator_algo", generator_algo),
        ("tier", tier),
        ("behavior_policy", behavior_policy),
        ("behavior_strength", behavior_strength),
        ("confounder_c_r", confounder_c_r),
        ("pi_basic_epsilon", pi_basic_epsilon),
        ("a_bad", int(a_bad)),
        ("rollout_episodes", int(rollout_episodes)),
        ("seed", int(seed)),
        ("generator_hash", generator_hash),
        # Rollout mode: only the device TYPE matters (cuda:0 vs cuda is the same
        # numerics); slot count and the legacy flag both change the trajectories.
        ("rollout_device", torch.device(rollout_device).type),
        ("rollout_n_envs", 1 if legacy_rollout else int(rollout_n_envs)),
        ("legacy_rollout", bool(legacy_rollout)),
    ]
    # The GRACE v2 diagram channels are appended ONLY when enabled. Appending
    # them unconditionally would change every existing dataset's fingerprint and
    # force a full regeneration for no change in contents; appending them never
    # would let a D-D dataset be reused to serve a D-A request, which is the
    # dangerous direction. Conditional append gives new arms a distinct key and
    # leaves the frozen ones bit-for-bit.
    for key, val, off in (
        ("proxy_strength", proxy_strength, None),
        ("instrument_strength", instrument_strength, None),
        ("u_drift", float(u_drift), 0.0),
        ("gate_probs", gate_probs, None),
        ("n_proxies", int(n_proxies), 2),
    ):
        if val != off:
            parts.append((key, val))
    h = hashlib.sha256()
    for k, v in parts:
        h.update(f"{k}={v!r};".encode())
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
    behavior_mask_indices: tuple | None = None,
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
            env_id,
            generator_algo,
            train_episodes,
            n_checkpoints,
            seed,
            run_dir,
            dev,
            mask_indices=behavior_mask_indices,
        )
        sel_ep = select_tier_episode(_read_eval_returns(run_dir), tier, fraction)

    # A plain (behavior_policy="agent") rollout env just to read the canonical dims.
    probe_env = build_rollout_env(env_id, 1, dev, seed, "agent", None)
    obs_dim, obs_shape, action_type, action_dim, action_space = _env_dims(probe_env)
    probe_env.close()
    if behavior_mask_indices:
        # The generator was TRAINED on the masked view (``_train_generator``
        # above), so the agent that loads its checkpoint must be masked-dim
        # too — the same reduction ``generate_offline_dataset`` applies. First
        # execution of this path (2026-09-03) failed here with a state-dict
        # size mismatch ([64, 2] vs [64, 4]): the build read the canonical
        # dims while the checkpoint carried the masked ones.
        obs_dim = obs_dim - len(tuple(behavior_mask_indices))
        obs_shape = (obs_dim,)
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
    rollout_device: str | None = "cpu",
    rollout_n_envs: int = 1,
    legacy_rollout: bool = False,
    proxy_strength: float | None = None,
    instrument_strength: float | None = None,
    u_drift: float = 0.0,
    gate_probs=None,
    n_proxies: int = 2,
    preflight_episodes: int = 600,
    behavior_mask_indices: tuple | None = None,
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
    byte-unchanged.

    Rollout speed (docs/dataset_generation_speedup.md). Generator TRAINING always
    runs on ``device`` (batched, GPU-friendly); the ROLLOUT is placed separately:

      ``rollout_device`` (default ``"cpu"``, strategy S1): stepping the vector env
        on CUDA costs a ~15 ms host<->device round trip PER STEP at batch 1 versus
        ~33 us on CPU — a ~70x end-to-end rollout speedup. A CPU copy of the
        generator serves the collection policy; the ORIGINAL agent is what gets
        hashed and its parameter VALUES are identical, so
        ``generator_checkpoint_hash`` is unaffected.
      ``rollout_n_envs`` (default 1, strategy S2): >1 collects through
        ``_rollout_vectorized`` — N slots stepped in lockstep with one batched
        policy forward. Episodes are emitted in assignment order, so the dataset
        never depends on completion order.
      ``legacy_rollout``: restores the pre-speedup path exactly (rollout on
        ``device``, one slot, scalar ``_rollout``) for regenerating historical
        dataset ids bit-for-bit.

    The fast paths are NOT byte-identical to the legacy one (CPU/CUDA argmax ties
    break differently; batched draws reorder the policy RNG stream). That is sound
    here because datasets SELF-CERTIFY: the confounding signature is recomputed at
    generation and enforced at load, so a regenerated dataset carries its own
    validity proof."""
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
    # Rollout placement (S1/S2). legacy_rollout pins the historical behavior:
    # rollout on the training device, a single slot, scalar _rollout.
    if legacy_rollout:
        roll_dev, n_slots = dev, 1
    else:
        roll_dev = (
            torch.device(rollout_device) if rollout_device else torch.device("cpu")
        )
        n_slots = max(1, int(rollout_n_envs))

    # --- train (skipped for the random tier, and when a pre-built agent is given) ---
    sel_ep = None
    if agent is None and tier != "random":
        if run_dir is None:
            raise ValueError("non-random tiers require run_dir for the generator")
        _train_generator(
            env_id,
            generator_algo,
            train_episodes,
            n_checkpoints,
            seed,
            run_dir,
            dev,
            mask_indices=behavior_mask_indices,
        )
        sel_ep = select_tier_episode(_read_eval_returns(run_dir), tier, fraction)

    # --- rollout env + agent ---
    rollout_env = build_rollout_env(
        env_id,
        n_slots,
        roll_dev,
        seed,
        behavior_policy,
        behavior_strength,
        c_r=confounder_c_r,
        a_bad=a_bad,
        proxy_strength=proxy_strength,
        instrument_strength=instrument_strength,
        u_drift=u_drift,
        gate_probs=gate_probs,
        n_proxies=n_proxies,
    )
    obs_dim, obs_shape, action_type, action_dim, action_space = _env_dims(rollout_env)
    if behavior_mask_indices:
        # The generator/behavior networks are MASKED-dim: they were trained on
        # the masked view and act through _MaskedViewPolicy below. The dataset
        # still stores the full observation (the loops are untouched).
        obs_dim = obs_dim - len(tuple(behavior_mask_indices))
        obs_shape = (obs_dim,)
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

    # S1: the collection policy must live on the ROLLOUT device. Parameter values
    # are preserved exactly by the copy (nn.Module .to() is a dtype-preserving
    # move), so the hash stamped below — taken from the ORIGINAL agent — is
    # unchanged. Copying only when the devices actually differ keeps the legacy
    # path allocation-for-allocation identical.
    rollout_agent = agent
    _agent_dev = getattr(agent, "device", None)
    if _agent_dev is not None and torch.device(_agent_dev).type != roll_dev.type:
        import copy as _copy

        rollout_agent = _copy.deepcopy(agent).to(roll_dev)
        rollout_agent.device = roll_dev

    if behavior_policy == "agent":
        collection_policy = AgentBehaviorPolicy(rollout_agent)
    else:
        collection_policy = build_collection_policy(
            behavior_policy,
            rollout_agent,
            action_type,
            action_space,
            behavior_strength,
            env=rollout_env,
            pi_basic_epsilon=pi_basic_epsilon,
        )

    if behavior_mask_indices:
        # A pre-built full-obs agent cannot act through the masked view -- its
        # input dim will mismatch loudly on the first act; the guard makes the
        # cause readable instead.
        collection_policy = _MaskedViewPolicy(collection_policy, behavior_mask_indices)

    # S2: >1 slot routes to the vectorized collector (same output contract).
    _collect = _rollout_vectorized if n_slots > 1 else _rollout
    buffers, sig_samples = _collect(
        rollout_env, collection_policy, rollout_episodes, seed, action_type
    )
    rollout_env.close()

    import minari

    name = dataset_id or dataset_name(
        env_id,
        tier,
        behavior_policy,
        behavior_strength,
        behavior_mask_indices=behavior_mask_indices,
    )
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
        _gate = dict(gate if gate is not None else default_gate_for(behavior_policy))
        # A null arm declares no U->R edge; tell the gate so it skips A4 rather
        # than testing for a signature the dataset deliberately lacks.
        if _gate.get("type") == "action_dependent" and not confounder_c_r:
            _gate["expect_gated_reward"] = False
        # D-E certifies through the IV conditions, NOT through the action-gated
        # closed form. Derived from the arm rather than declared in YAML, for the
        # same reason the channels are: the diagram decides.
        if instrument_strength is not None and gate is None:
            _gate = {"type": "instrument", "instrument_strength": instrument_strength}
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
    # The behavior policy's INFORMATION SET, stamped so the two constructions
    # (O->A masked-behavior vs S->A load-time mask) can never be conflated: a
    # reader of the dataset can tell which diagram the logged actions realise.
    signature["behavior_information_set"] = (
        "masked:" + ",".join(str(i) for i in behavior_mask_indices)
        if behavior_mask_indices
        else "full"
    )
    # V-B: the arm's PREFLIGHT CERTIFICATION travels with the dataset, in the same
    # metadata block as the confounding signature and for the same reason -- a
    # dataset should carry its own validity proof rather than depend on a
    # verification someone remembers having run. Capped at
    # ``preflight_episodes`` because the permutation nulls are O(n_perm * n) and
    # a production rollout is far larger than these statistics need; the cap is
    # recorded so a reader knows what was actually certified.
    if sig_samples is not None and (
        proxy_strength is not None
        or instrument_strength is not None
        or u_drift
        # The NULL arm is certified too: c_r = 0 means U must be provably inert,
        # and that is exactly the claim L5's false-positive rate rests on.
        or not confounder_c_r
    ):
        signature.update(
            _preflight_certification(
                sig_samples,
                buffers,
                proxy_strength=proxy_strength,
                instrument_strength=instrument_strength,
                u_drift=u_drift,
                max_episodes=preflight_episodes,
                null_arm=not confounder_c_r,
                a_bad=a_bad,
            )
        )
    # S4: stamp the input fingerprint so a later run can PROVE that regenerating
    # this dataset would reproduce it, and reuse it instead (see
    # generation_fingerprint + regime_sweep's reuse check).
    signature["generation_fingerprint"] = generation_fingerprint(
        env_id=env_id,
        generator_algo=generator_algo,
        tier=tier,
        behavior_policy=behavior_policy,
        behavior_strength=behavior_strength,
        confounder_c_r=confounder_c_r,
        pi_basic_epsilon=pi_basic_epsilon,
        a_bad=a_bad,
        rollout_episodes=rollout_episodes,
        seed=seed,
        generator_hash=signature["generator_checkpoint_hash"],
        rollout_device=str(roll_dev),
        rollout_n_envs=n_slots,
        legacy_rollout=legacy_rollout,
        proxy_strength=proxy_strength,
        instrument_strength=instrument_strength,
        u_drift=u_drift,
        gate_probs=gate_probs,
    )
    ds.storage.update_metadata(signature)
    return ds


def _train_generator(
    env_id,
    algo,
    train_episodes,
    n_checkpoints,
    seed,
    run_dir,
    dev,
    *,
    mask_indices=None,
):
    from src.benchmarking.registry import registry
    from src.benchmarking.runner import BenchmarkRunner
    from src.config.defaults import EnvConfig, RunConfig, TrainingConfig

    env_cfg = EnvConfig(
        env_id=env_id,
        n_train_envs=4,
        n_eval_envs=4,
        rollout_len=64,
        seed=seed,
        # Masked-behavior generation: the generator LEARNS on the partial view,
        # so the pi_basic it defines has the O->A information set by training,
        # not by projection of a full-obs policy.
        mask_indices=(tuple(mask_indices) if mask_indices else None),
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
