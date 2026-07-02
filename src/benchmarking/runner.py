from __future__ import annotations

import functools
import os
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import torch
from tqdm import tqdm

from src.benchmarking.aux_models import (
    AUX_METRICS_COLUMNS,
    AuxModelConfig,
    AuxModelManager,
)
from src.benchmarking.critic_ablation import (
    CRITIC_ABLATION_COLUMNS,
    CRITIC_LIBRARY,
    CriticAblationConfig,
    CriticAblationManager,
    STRATEGY_CRITIC_ABLATION_COLUMNS,
)
from src.config.defaults import EnvConfig, RunConfig, TrainingConfig
from src.config.seeding import set_seed
from src.data.experience_source import OnlineSource, validate_pairing
from src.envs.registry import build_env
from src.logging.logger import CSVLogger
from src.rl.on_policy.base_actor_critic import RolloutBatch

TRAIN_COLUMNS: List[str] = [
    "episode",
    "algorithm",
    "environment",
    "train_return_mean",
    "train_return_std",
    "loss",
    "policy_loss",
    "value_loss",
    "entropy",
    "kl",
    "critic_loss",
    "actor_loss",
    "q_loss",
]

EVAL_COLUMNS: List[str] = [
    "episode",
    "algorithm",
    "environment",
    "eval_return_mean",
    "eval_return_std",
]

# Per-context eval breakdown (docs/experimental_design.md §6.1), written only
# when --mask-indices is set (Cell 2). Separate file/schema; the frozen
# TRAIN_COLUMNS/EVAL_COLUMNS above are untouched.
EVAL_PER_CONTEXT_COLUMNS: List[str] = [
    "episode",
    "algorithm",
    "environment",
    "context_bin",
    "context_value_low",
    "context_value_high",
    "n_episodes_in_bin",
    "return_iqm",
    "return_iqr_std",
]

# Apparent-training-value trace (docs/experimental_design.md §6.2), written only
# for confounded offline runs (Cells 7-8). Separate file/schema; the frozen
# TRAIN_COLUMNS/EVAL_COLUMNS above are untouched.
OFFLINE_VALUE_TRACE_COLUMNS: List[str] = [
    "epoch",
    "algorithm",
    "environment",
    "apparent_value_iqm",
    "apparent_value_iqr_std",
]

# Additive u=0 anchor columns (oracle-U ceiling, *_oracle_u variants only).
# Appended to the value-trace schema ONLY when a U-variant is in the run; pure
# base runs keep OFFLINE_VALUE_TRACE_COLUMNS exactly (frozen schema preserved).
OFFLINE_VALUE_TRACE_U0_COLUMNS: List[str] = [
    "apparent_value_u0_iqm",
    "apparent_value_u0_iqr_std",
]


@functools.lru_cache(maxsize=None)
def _render_capable(env_id: str) -> bool:
    """Probe ONCE per env_id whether this machine can render the env.

    Eval renders frames into an .mp4 via ``start_video`` (which flips
    ``record_video`` on), so ``evaluate`` calls ``env.render()`` every step. On a
    headless box (no display/GL context, broken pygame-gfxdraw) that ``render()``
    raises and crashes eval. Probing a throwaway env (never the real train/eval
    env) lets eval fall back to ``render=False``, where ``render()`` returns
    ``None`` and is skipped instead of crashing. Render is side-effect-only, so
    this never changes numerics where rendering already works (golden-bitwise).
    The ``lru_cache`` makes this a one-time probe per env_id (one warning at most).
    """
    try:
        import gymnasium as gym

        e = gym.make(env_id, render_mode="rgb_array")  # match build_env path
        e.reset(seed=0)
        e.render()
        e.close()
        return True
    except Exception:
        print(
            f"[warn] Rendering unavailable for {env_id}; eval video disabled "
            "(eval metrics unaffected).",
            file=sys.stderr,
        )
        return False


@dataclass
class AlgorithmSpec:
    builder: Callable
    kind: str  # "on_policy" or "off_policy"
    # Data-source axis, orthogonal to ``kind``: "online" (live env interaction)
    # or "offline" (logged dataset; Stage B). Distinct from the agent's
    # vestigial ``Algorithm.paradigm`` (the on/off-policy learning regime).
    data_regime: str = "online"
    # True iff this algo is a deconfounding variant that reads the per-transition
    # latent U (the *_oracle_u oracle reference line). Drives the offline loader's
    # load_u and the value-trace u0-anchor schema. Default False keeps every
    # existing AlgorithmSpec(...) construction (and the base algos) unchanged.
    requires_confounder_u: bool = False
    # True iff this algo consumes episode-grouped sequences (the *_proximal
    # latent-class variants, whose per-episode posterior needs whole episodes).
    # Default False. PR-1 registers the flag + the Proximal stub; wiring the
    # offline loop to honor it (Minari -> SequenceReplayBuffer) is PR-2.
    needs_episode_grouping: bool = False


def _is_strategy_ablation(cfg: CriticAblationConfig | None) -> bool:
    """True iff the requested ablation critics are Cell-7 strategy critics (they
    fit the episode-grouped stream, not the on-policy V-head path)."""
    if cfg is None or not cfg.critics:
        return False
    return any(
        CRITIC_LIBRARY.get(str(c).lower().strip())
        and CRITIC_LIBRARY[str(c).lower().strip()].kind == "strategy"
        for c in cfg.critics
    )


def _validate_algos_against_behavior_policy(
    algos: list,
    behavior_policy: str,
    on_policy_algos=None,
) -> None:
    """Reject YAMLs where on-policy algorithms appear alongside a behavior
    policy whose mechanism doesn't touch them.

    On-policy algorithms collect their own rollouts via their internal
    actor-critic loop and ignore the configured behavior_policy. So if the
    behavior_policy's mechanism is action-bias-only (curiosity, anti_reward,
    bias_skew, bias_suboptimal), an on-policy algo in the YAML is a no-op
    across the run — it produces identical results to the same algo run with
    behavior_policy="agent". Listing it is structurally redundant and wastes
    compute.

    Only behaviors that affect on-policy algorithms (currently bias_confounded,
    via the ConfoundedCollectionWrapper's reward perturbation) may include
    on-policy algos in the same YAML.

    ``on_policy_algos`` is the set of algo names treated as on-policy. When None
    (the default), it is derived from the algorithm registry — the source of
    truth is ``AlgorithmSpec.kind == "on_policy"`` (so adding a new on-policy
    algo needs no change here). The registry must be populated first
    (``register_default_algorithms()``); main.py does this before calling.
    """
    if behavior_policy == "agent":
        return  # No active behavior mechanism; on-policy algos are fine.

    from src.rl.policies.behavior_policy import behavior_policy_class

    policy_class = behavior_policy_class(behavior_policy)
    if policy_class is None:
        return  # Unknown behavior_policy; other validation handles this.
    if policy_class.affects_on_policy():
        return  # Behavior's mechanism affects on-policy too; PPO+DQN is valid.

    # Behavior is action-bias-only. Find any on-policy algos in the list.
    if on_policy_algos is not None:
        on_policy_in_list = sorted(set(algos) & set(on_policy_algos))
    else:
        from src.benchmarking.registry import registry

        on_policy_in_list = []
        for a in algos:
            try:
                if registry.get(a).kind == "on_policy":
                    on_policy_in_list.append(a)
            except KeyError:
                pass  # Unknown algo; let registry.get raise downstream instead.
        on_policy_in_list = sorted(set(on_policy_in_list))

    if on_policy_in_list:
        raise ValueError(
            f"YAML configuration error: behavior_policy={behavior_policy!r} "
            f"is action-bias-only (it does not affect on-policy algorithms), "
            f"so listing on-policy algos {on_policy_in_list} in the same YAML "
            f"is structurally redundant — those algos see no effect of the "
            f"behavior and produce identical results to behavior_policy='agent'. "
            f"Remove the on-policy algos OR change behavior_policy to 'agent' "
            f"OR use a behavior_policy whose mechanism affects on-policy algos "
            f"(currently only bias_confounded, via its reward wrapper)."
        )


class BenchmarkRunner:
    def __init__(
        self,
        env_cfg: EnvConfig,
        train_cfg: TrainingConfig,
        run_cfg: RunConfig,
        algo_spec: AlgorithmSpec,
        critic_ablation_cfg: CriticAblationConfig | None = None,
        aux_model_cfg: "AuxModelConfig | None" = None,
        progress_label: str | None = None,
    ):
        self.env_cfg = env_cfg
        self.train_cfg = train_cfg
        self.run_cfg = run_cfg
        self.algo_spec = algo_spec
        # Seed BEFORE env/network construction so weight init is reproducible
        # (sanctioned Phase-0 gate change; run() re-seeds at the same point as
        # before, so the in-training RNG stream is unchanged).
        set_seed(env_cfg.seed, deterministic=train_cfg.deterministic)
        self.device = torch.device(train_cfg.device)
        self.progress_label = (
            progress_label or f"{self.train_cfg.algorithm} - {self.env_cfg.env_id}"
        )
        self._env_tag = self.env_cfg.env_id.replace("/", "-")
        self.aggregation = self.train_cfg.aggregation
        # Eval-per-context gate (Cell 2): non-empty iff --mask-indices was set.
        # When open, evaluate() bins eval returns by the hidden Z component and
        # writes eval_per_context.csv via the logger opened in run(). Empty =>
        # no file, no extra work (default path byte-identical, goldens intact).
        self._mask_indices: tuple[int, ...] = tuple(
            int(i) for i in (env_cfg.mask_indices or ())
        )
        self._eval_per_context_logger: CSVLogger | None = None
        # Offline-value-trace gate (Cells 7-8): open iff the collection regime is
        # confounded AND the algorithm trains offline. When open, _train_offline
        # reads the dataset's confounding-signature metadata (rejecting
        # gate-failed/missing) and writes the critic's apparent Q per epoch to
        # offline_value_trace.csv. Closed => no file, no work (goldens intact).
        self._value_trace_gate_open: bool = (
            getattr(env_cfg, "behavior_policy", "agent")
            in ("bias_confounded", "bias_confounded_action")
            and algo_spec.data_regime == "offline"
        )
        # Oracle-U ceiling: the deconfounding variant (*_oracle_u) reads the
        # latent U and records the u=0 anchor in offline_value_trace.csv.
        #   _requires_confounder_u (per-algo): drives load_u and the u0 WRITE gate
        #     (only this oracle agent computes oracle_anchor_q).
        #   _value_trace_oracle (per-algo): gate-open AND this algo needs U -> the
        #     agent emits u0 anchor rows.
        #   _value_trace_schema_oracle (run-level): gate-open AND (this algo OR any
        #     sibling algo in the run needs U, via run_cfg.value_trace_u0_schema)
        #     -> the SHARED offline_value_trace.csv uses the u0 SUPERSET header so
        #     all sibling runners agree; base runners blank-fill the u0 cells. The
        #     `requires_u or` term keeps a standalone variant runner self-sufficient.
        self._requires_confounder_u: bool = getattr(
            algo_spec, "requires_confounder_u", False
        )
        self._value_trace_oracle: bool = (
            self._value_trace_gate_open and self._requires_confounder_u
        )
        self._value_trace_schema_oracle: bool = self._value_trace_gate_open and (
            self._requires_confounder_u
            or getattr(run_cfg, "value_trace_u0_schema", False)
        )
        self._offline_value_trace_logger: CSVLogger | None = None

        self.run_dir = run_cfg.resolve_run_dir()
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "videos"), exist_ok=True)

        self.train_env = build_env(
            env_id=env_cfg.env_id,
            n_envs=env_cfg.n_train_envs,
            device=self.device,
            seed=env_cfg.seed,
            render=False,
            record_video=False,
            env_wrapper=env_cfg.env_wrapper,
            env_entry_point=env_cfg.env_entry_point,
            env_kwargs=env_cfg.env_kwargs,
        )
        # Confounded collection wraps the TRAIN env only (eval stays clean): a
        # per-episode latent U biases the action AND perturbs the reward while
        # never entering the obs -> unobserved confounding. Opt-in; default
        # leaves train_env byte-identical (off-policy golden bitwise).
        _bp = getattr(env_cfg, "behavior_policy", "agent")
        if _bp in ("bias_confounded", "bias_confounded_action"):
            from src.envs.wrappers.confounded import ConfoundedCollectionWrapper

            _sigma = getattr(env_cfg, "behavior_strength", None)
            _sigma = 1.0 if _sigma is None else float(_sigma)
            # action_gated (action-dependent cell) gates r += c_r*U on a==a_bad;
            # additive (default) is the byte-frozen cells-7/8 path.
            _kind = "action_gated" if _bp == "bias_confounded_action" else "additive"
            self.train_env = ConfoundedCollectionWrapper(
                self.train_env, c_a=_sigma, c_r=_sigma, confounder_kind=_kind
            )
        self.eval_env = build_env(
            env_id=env_cfg.env_id,
            n_envs=env_cfg.n_eval_envs,
            device=self.device,
            seed=env_cfg.seed + env_cfg.n_train_envs,
            render=_render_capable(env_cfg.env_id),
            record_video=False,
            env_wrapper=env_cfg.env_wrapper,
            env_entry_point=env_cfg.env_entry_point,
            env_kwargs=env_cfg.env_kwargs,
        )
        # Observation masking (Z-hidden axis): drop the configured indices from
        # the flat obs of BOTH train and eval, so the agent is built for the
        # reduced dim and sees the same masked obs throughout. Applied on the
        # OUTSIDE of any train-env confounding (base -> Confounded -> Masked,
        # docs/experimental_design.md §8). Opt-in; default leaves envs untouched
        # (off-policy/on-policy goldens stay bitwise). For offline runs the same
        # indices are also projected off the dataset in the loader below.
        if getattr(env_cfg, "mask_indices", None):
            from src.envs.wrappers.masked import MaskedObservationWrapper

            self.train_env = MaskedObservationWrapper(
                self.train_env, env_cfg.mask_indices
            )
            self.eval_env = MaskedObservationWrapper(
                self.eval_env, env_cfg.mask_indices
            )

        if len(self.train_env.obs_space.shape) == 0:
            self.obs_dim = 1
        else:
            self.obs_dim = int(
                torch.tensor(self.train_env.obs_space.shape).prod().item()
            )
        # Real observation shape for the backbone selector. Vector envs flatten
        # to (obs_dim,) in the env wrapper, so this is byte-identical to the
        # builders' default and keeps the vector golden bitwise; image envs
        # expose (C, H, W) and route to the CNN.
        self.obs_shape = tuple(self.train_env.obs_space.shape)
        act_space = self.train_env.act_space
        if hasattr(act_space, "n"):
            self.action_type = "discrete"
            self.action_dim = act_space.n
        else:
            self.action_type = "continuous"
            self.action_dim = act_space.shape[0]

        self.policy, self.agent = self.algo_spec.builder(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            action_type=self.action_type,
            device=self.device,
            action_space=act_space,
            obs_shape=self.obs_shape,
            # Per-component network selection (on-policy ActorCritic). Off-policy
            # builders take **kwargs and ignore these. network_kwargs carries
            # optional hidden_dim/num_layers for recurrent trunks.
            actor_network=getattr(self.train_cfg, "actor_network", "mlp"),
            critic_network=getattr(self.train_cfg, "critic_network", "mlp"),
            # Recurrent off-policy uses the episode-aware SequenceReplayBuffer
            # (constructed here, passed to the builder); flat/MLP and on-policy
            # pass None and the builder keeps its own buffer (golden bitwise).
            buffer=self._make_replay_buffer(),
            **getattr(self.train_cfg, "network_kwargs", {}),
        )
        self.replay_buffer = None
        self.collection_policy = None
        self.offpolicy_batch_size = 128
        self.offpolicy_warmup = 1000
        # Sequence length for recurrent off-policy sampling/BPTT (zero-init R2D2).
        self.offpolicy_seq_len = 8
        if self.algo_spec.kind == "off_policy":
            self.replay_buffer = self.agent.buffer  # type: ignore[attr-defined]
            # Default collection seam: delegates to agent.act (exact pre-seam
            # behavior). Opt-in A1 policies (anti_reward/bias_*) plug in here;
            # the "agent" branch is byte-identical to the pre-A1 path so the
            # off-policy golden stays bitwise.
            behavior_policy = getattr(self.env_cfg, "behavior_policy", "agent")
            if behavior_policy == "agent":
                from src.rl.policies.behavior_policy import AgentBehaviorPolicy

                self.collection_policy = AgentBehaviorPolicy(self.agent)
            else:
                from src.rl.policies.behavior_policy import build_collection_policy

                self.collection_policy = build_collection_policy(
                    behavior_policy,
                    self.agent,
                    self.action_type,
                    act_space,
                    getattr(self.env_cfg, "behavior_strength", None),
                    env=self.train_env,
                )
        self.experience_source = OnlineSource(self.train_env, self.device)
        validate_pairing(
            self.algo_spec.kind,
            self.experience_source,
            data_regime=self.algo_spec.data_regime,
        )
        self.critic_ablation = None
        if critic_ablation_cfg is not None:
            gamma = float(getattr(self.agent, "gamma", 0.99))
            if _is_strategy_ablation(critic_ablation_cfg):
                # Strategy-critic ablation (Cell-7 deconfounding): the three critics
                # {observational, proximal, oracle_u} fit a SHARED episode-grouped
                # confounded stream, scored estimation-vs-oracle. Guard swap: the
                # on-policy-only requirement becomes an episode-grouped-stream
                # requirement. Implemented for the OFFLINE regime (cell-7 datasets;
                # routed via _train_offline_grouped); the online (bias_confounded)
                # and rnn rows are deferred behind the recurrent-offline
                # prerequisite (= cell 8).
                if self.algo_spec.data_regime != "offline":
                    raise ValueError(
                        "strategy-critic ablation (observational/proximal/oracle_u) "
                        "requires an offline cell-7 dataset base (data_regime="
                        "'offline'); the online bias_confounded and rnn regimes are "
                        "deferred (recurrent-offline prerequisite = cell 8)."
                    )
                self.critic_ablation = CriticAblationManager(
                    obs_dim=self.obs_dim,
                    device=self.device,
                    config=critic_ablation_cfg,
                    gamma=gamma,
                    base_algo=self.train_cfg.algorithm,
                    action_dim=self.action_dim,
                )
            else:
                # V-head ablation (standard_mlp/residual): the frozen on-policy path.
                if self.algo_spec.kind != "on_policy":
                    raise ValueError(
                        "Critic ablation mode is supported only for on-policy "
                        "algorithms."
                    )
                self.critic_ablation = CriticAblationManager(
                    obs_dim=self.obs_dim,
                    device=self.device,
                    config=critic_ablation_cfg,
                    gamma=gamma,
                )

        # Auxiliary reward/next-state models (opt-in, off by default; orthogonal
        # to kind/data_regime, so available in all three loops). Built AFTER the
        # policy/agent; the manager snapshots/restores the global RNG around
        # construction so an aux-enabled run is bitwise-identical to aux-off.
        self.aux_models = None
        if aux_model_cfg is not None:
            self.aux_models = AuxModelManager(
                obs_dim=self.obs_dim,
                obs_shape=tuple(self.train_env.obs_space.shape) or (1,),
                action_dim=self.action_dim,
                action_type=self.action_type,
                device=self.device,
                config=aux_model_cfg,
            )

    def _collect_on_policy(self) -> tuple[RolloutBatch, float, float]:
        # Recurrent policies (any non-MLP trunk) use the separate state-threading
        # rollout; the MLP path keeps the verbatim, golden-pinned rollout (Phase 1).
        if getattr(self.policy, "is_recurrent", False):
            batch, ep_returns = self.experience_source.rollout_recurrent(
                self.policy,
                self.agent,
                n_steps=self.env_cfg.rollout_len,
                n_envs=self.env_cfg.n_train_envs,
            )
        else:
            batch, ep_returns = self.experience_source.rollout(
                self.policy,
                self.agent,
                n_steps=self.env_cfg.rollout_len,
                n_envs=self.env_cfg.n_train_envs,
            )
        train_return_mean, train_return_std = self._aggregate_returns(ep_returns)
        return batch, train_return_mean, train_return_std

    def _train_on_policy(
        self,
        train_logger: CSVLogger,
        eval_logger: CSVLogger,
        critic_logger: CSVLogger | None = None,
        aux_logger: CSVLogger | None = None,
    ) -> None:
        checkpoint_eps = self.train_cfg.checkpoint_episodes()
        for ep in tqdm(range(self.train_cfg.n_episodes), desc=self.progress_label):
            batch, train_ret_mean, train_ret_std = self._collect_on_policy()
            metrics = self.agent.update(batch)
            critic_losses = {}
            if self.critic_ablation is not None:
                critic_losses = self.critic_ablation.update(batch)
            if self.aux_models is not None:
                self.aux_models.update(batch)

            if ep in checkpoint_eps:
                self._save_checkpoint(ep)
                self._save_aux_critic_checkpoint(ep)
                eval_metrics = self.evaluate(ep)
                eval_metrics.update(
                    {
                        "algorithm": self.train_cfg.algorithm,
                        "environment": self.env_cfg.env_id,
                    }
                )
                train_metrics = metrics.copy()
                train_metrics.update(
                    {
                        "algorithm": self.train_cfg.algorithm,
                        "environment": self.env_cfg.env_id,
                        "train_return_mean": train_ret_mean,
                        "train_return_std": train_ret_std,
                    }
                )
                eval_row = self._make_row(EVAL_COLUMNS, eval_metrics, ep)
                train_row = self._make_row(TRAIN_COLUMNS, train_metrics, ep)
                eval_logger.log(eval_row)
                train_logger.log(train_row)
                if self.critic_ablation is not None and critic_logger is not None:
                    for row in self.critic_ablation.checkpoint_rows(
                        batch=batch,
                        episode=ep,
                        algorithm=self.train_cfg.algorithm,
                        environment=self.env_cfg.env_id,
                        latest_losses=critic_losses,
                    ):
                        critic_logger.log(row)
                self._log_aux_rows(aux_logger, batch, ep)

    def _log_aux_rows(self, aux_logger, batch, episode: int) -> None:
        # train_loss comes from the manager's last update (no extra batch /
        # no extra RNG draw); mse/mae are computed fresh on the given batch.
        if self.aux_models is None or aux_logger is None or batch is None:
            return
        for row in self.aux_models.checkpoint_rows(
            batch=batch,
            episode=episode,
            algorithm=self.train_cfg.algorithm,
            environment=self.env_cfg.env_id,
        ):
            aux_logger.log(row)

    def _save_checkpoint(self, episode: int):
        ckpt_dir = os.path.join(
            self.run_dir,
            "checkpoints",
            f"{self._env_tag}_{self.train_cfg.algorithm}_seed{self.env_cfg.seed}",
        )
        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, f"ckpt_ep{episode:04d}.pt")
        from .checkpoints import save_checkpoint

        save_checkpoint(
            path,
            {
                "episode": episode,
                "policy_state": self.policy.state_dict(),
                "agent_state": getattr(self.agent, "state_dict", lambda: {})(),
                "config": {
                    "env": self.env_cfg.__dict__,
                    "training": self.train_cfg.__dict__,
                },
            },
        )

    def _save_aux_critic_checkpoint(self, episode: int) -> None:
        if self.critic_ablation is None:
            return
        ckpt_dir = os.path.join(
            self.run_dir,
            "checkpoints",
            f"{self._env_tag}_{self.train_cfg.algorithm}_seed{self.env_cfg.seed}",
        )
        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, f"aux_critics_ep{episode:04d}.pt")
        from .checkpoints import save_checkpoint

        save_checkpoint(
            path,
            {"episode": episode, "aux_critics": self.critic_ablation.state_dict()},
        )

    def evaluate(self, episode: int) -> Dict[str, float]:
        env = self.eval_env
        video_path = os.path.join(
            self.run_dir,
            "videos",
            f"{self._env_tag}_{self.train_cfg.algorithm}_seed{self.env_cfg.seed}_ckpt{episode:04d}.mp4",
        )
        env.start_video(video_path)
        obs, _ = env.reset()
        total_rewards = torch.zeros(env.n_envs, device=self.device)
        # Eval-per-context (Cell 2): accumulate the PRE-mask values of the hidden
        # Z components over the rollout, per env. Gated; zero work when no mask.
        gate_open = bool(self._mask_indices)
        z_idx = (
            torch.tensor(self._mask_indices, device=self.device) if gate_open else None
        )
        z_sum = (
            torch.zeros(env.n_envs, len(self._mask_indices), device=self.device)
            if gate_open
            else None
        )
        z_steps = 0
        for _ in range(self.env_cfg.rollout_len):
            with torch.no_grad():
                if self.algo_spec.kind == "off_policy":
                    if self.action_type == "discrete":
                        action = self.agent.act(obs, epsilon=0.0).action  # type: ignore[arg-type]
                    else:
                        action = self.agent.act(obs, noise=False).action  # type: ignore[arg-type]
                else:
                    if hasattr(self.policy, "act_deterministic"):
                        action = self.policy.act_deterministic(obs)
                    else:
                        action, _ = self.policy.act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = torch.logical_or(terminated, truncated)
            total_rewards += reward * (~done)
            if gate_open:
                # env.last_unmasked_obs is the full obs vector (the mask wrapper
                # exposes it); pick out the hidden components the agent can't see.
                full_obs = env.last_unmasked_obs
                z_sum += full_obs.index_select(-1, z_idx)
                z_steps += 1
        env.stop_video()
        if gate_open:
            self._write_eval_per_context(
                episode, z_sum / max(z_steps, 1), total_rewards
            )
        mean, std = self._aggregate_returns(total_rewards)
        return {"eval_return_mean": mean, "eval_return_std": std}

    def _write_eval_per_context(
        self, episode: int, z_mean: torch.Tensor, returns: torch.Tensor
    ) -> None:
        """Bin per-env eval returns by the hidden Z component and write one row
        per non-empty bin to eval_per_context.csv (docs/experimental_design.md
        §6.1). ``z_mean`` is the per-env episode-mean of the masked components
        ([n_envs, k]); the binning scalar is the signed mean for a single hidden
        component, else the L2 norm across components.
        """
        logger = self._eval_per_context_logger
        if logger is None:
            return
        if z_mean.shape[1] == 1:
            z_scalar = z_mean[:, 0]
        else:
            z_scalar = torch.linalg.norm(z_mean, dim=1)
        z = z_scalar.detach().cpu()
        r = returns.detach().cpu()

        n_bins = 10
        zmin = float(z.min())
        zmax = float(z.max())
        if zmax <= zmin:
            # Degenerate range (all envs identical) -> widen so bins have
            # positive width and context_value_low < context_value_high holds.
            zmax = zmin + 1e-8
        edges = torch.linspace(zmin, zmax, n_bins + 1)
        # np.searchsorted(edges, z, side="right") - 1, clamped to a valid bin.
        bin_idx = torch.clamp(
            torch.bucketize(z, edges, right=True) - 1, min=0, max=n_bins - 1
        )
        for b in sorted({int(x) for x in bin_idx.tolist()}):
            in_bin = bin_idx == b
            bin_returns = r[in_bin]
            iqm, iqr_std = self._bin_stats(bin_returns)
            logger.log(
                {
                    "episode": episode,
                    "algorithm": self.train_cfg.algorithm,
                    "environment": self.env_cfg.env_id,
                    "context_bin": b,
                    "context_value_low": float(edges[b]),
                    "context_value_high": float(edges[b + 1]),
                    "n_episodes_in_bin": int(in_bin.sum()),
                    "return_iqm": iqm,
                    "return_iqr_std": iqr_std,
                }
            )

    @staticmethod
    def _bin_stats(values: torch.Tensor) -> Tuple[float, float]:
        """IQM ± IQR-STD over a bin's returns; mean ± std fallback when fewer
        than 3 episodes (IQM is ill-defined on tiny samples). Population std
        (unbiased=False), matching ``_aggregate_returns``."""
        n = values.numel()
        if n < 3:
            return float(values.mean()), float(values.std(unbiased=False))
        sorted_vals = torch.sort(values).values
        lower = int(0.25 * (n - 1))
        upper = int(0.75 * (n - 1))
        mid = sorted_vals[lower : upper + 1]
        return float(mid.mean()), float(mid.std(unbiased=False))

    def _apparent_value_unqueryable(self) -> bool:
        """True if the agent's critic can't be queried at (s, a) for the
        apparent-value trace — neither a discrete ``q_network`` nor a continuous
        twin-Q (``q1``/``q2`` + ``_q_input``). All current offline algos are
        queryable, so this only guards a future critic shape."""
        agent = self.agent
        discrete_q = hasattr(agent, "q_network")
        twin_q = (
            hasattr(agent, "q1") and hasattr(agent, "q2") and hasattr(agent, "_q_input")
        )
        return not (discrete_q or twin_q)

    def _apparent_q(self, batch: Dict[str, torch.Tensor]) -> "torch.Tensor | None":
        """Critic-predicted Q at the batch's data ``(s, a)`` pairs, shape ``[B]``.

        Discrete (offline_dqn/cql/iql/bcq): gather ``Q(s, .)`` at the data action.
        Continuous (cql/iql/bcq _continuous): ``min`` of the twin critics on
        ``cat(s, a)`` — the value the conservative algorithms treat as Q (BCQ's
        CVAE/perturbation net selects ACTIONS; Q itself is queried here directly).
        ``None`` if the critic can't be queried.
        """
        agent = self.agent
        obs = batch["obs"]
        actions = batch["actions"]
        with torch.no_grad():
            if hasattr(agent, "q_network"):
                q_all = agent.q_network(obs)
                a_idx = actions.long().view(-1, 1)
                return q_all.gather(1, a_idx).squeeze(-1)
            if (
                hasattr(agent, "q1")
                and hasattr(agent, "q2")
                and hasattr(agent, "_q_input")
            ):
                x = agent._q_input(obs, actions.float())
                return torch.min(agent.q1(x), agent.q2(x)).squeeze(-1)
        return None

    def _log_offline_value_trace(
        self, epoch: int, batch: Dict[str, torch.Tensor]
    ) -> None:
        logger = self._offline_value_trace_logger
        if logger is None:
            return
        q_vals = self._apparent_q(batch)
        if q_vals is None:
            return
        iqm, iqr_std = self._bin_stats(q_vals.reshape(-1))
        row = {
            "epoch": epoch,
            "algorithm": self.train_cfg.algorithm,
            "environment": self.env_cfg.env_id,
            "apparent_value_iqm": iqm,
            "apparent_value_iqr_std": iqr_std,
        }
        # u=0 validation anchor (AM3): Q(s, a_data, u=0) tracks the clean value;
        # the built-in check that conditioning on U works. Oracle runs only.
        if self._value_trace_oracle and hasattr(self.agent, "oracle_anchor_q"):
            q0 = self.agent.oracle_anchor_q(batch)
            u0_iqm, u0_iqr_std = self._bin_stats(q0.reshape(-1))
            row["apparent_value_u0_iqm"] = u0_iqm
            row["apparent_value_u0_iqr_std"] = u0_iqr_std
        logger.log(row)

    def _aggregate_returns(self, returns: torch.Tensor) -> Tuple[float, float]:
        if returns.numel() == 0:
            return 0.0, 0.0
        if self.aggregation == "mean":
            mean = returns.mean().item()
            std = returns.std(unbiased=False).item()
            return mean, std
        # IQM: mean of middle 50%; IQR-STD: std of middle 50%
        sorted_vals = torch.sort(returns.flatten()).values
        n = sorted_vals.numel()
        lower = int(0.25 * (n - 1))
        upper = int(0.75 * (n - 1))
        mid = sorted_vals[lower : upper + 1]
        mean = mid.mean().item()
        std = mid.std(unbiased=False).item()
        return mean, std

    @staticmethod
    def _make_row(
        schema: list[str], metrics: Dict[str, float] | None, episode: int
    ) -> Dict[str, float]:
        metrics = metrics or {}
        base = {
            "episode": episode,
        }
        full = {**metrics, **base}
        return {col: full.get(col, "") for col in schema}

    def run(self) -> None:
        set_seed(self.env_cfg.seed, deterministic=self.train_cfg.deterministic)
        try:
            critic_ctx = (
                CSVLogger(
                    os.path.join(self.run_dir, "critic_ablation_metrics.csv"),
                    # Strategy ablation writes its own estimation-vs-oracle schema to
                    # the same filename; the V-head schema is untouched (golden).
                    fieldnames=(
                        STRATEGY_CRITIC_ABLATION_COLUMNS
                        if self.critic_ablation.is_strategy
                        else CRITIC_ABLATION_COLUMNS
                    ),
                )
                if self.critic_ablation is not None
                else nullcontext(None)
            )
            aux_ctx = (
                CSVLogger(
                    os.path.join(self.run_dir, "aux_metrics.csv"),
                    fieldnames=AUX_METRICS_COLUMNS,
                )
                if self.aux_models is not None
                else nullcontext(None)
            )
            # Eval-per-context writer (Cell 2): opened only when --mask-indices
            # is set, so a no-mask run never creates the file.
            per_context_ctx = (
                CSVLogger(
                    os.path.join(self.run_dir, "eval_per_context.csv"),
                    fieldnames=EVAL_PER_CONTEXT_COLUMNS,
                )
                if self._mask_indices
                else nullcontext(None)
            )
            # Offline-value-trace writer (Cells 7-8): opened only for confounded
            # offline runs, so other runs never create the file.
            value_trace_ctx = (
                CSVLogger(
                    os.path.join(self.run_dir, "offline_value_trace.csv"),
                    # Runs containing a U-variant append the u=0 anchor columns to
                    # the SHARED file (run-level schema flag, so base + variant
                    # sibling runners agree on one header); pure base runs keep the
                    # frozen schema exactly.
                    fieldnames=(
                        OFFLINE_VALUE_TRACE_COLUMNS + OFFLINE_VALUE_TRACE_U0_COLUMNS
                        if self._value_trace_schema_oracle
                        else OFFLINE_VALUE_TRACE_COLUMNS
                    ),
                )
                if self._value_trace_gate_open
                else nullcontext(None)
            )
            with (
                CSVLogger(
                    os.path.join(self.run_dir, "train_metrics.csv"),
                    fieldnames=TRAIN_COLUMNS,
                ) as train_log,
                CSVLogger(
                    os.path.join(self.run_dir, "eval_metrics.csv"),
                    fieldnames=EVAL_COLUMNS,
                ) as eval_log,
                critic_ctx as critic_log,
                aux_ctx as aux_log,
                per_context_ctx as per_context_log,
                value_trace_ctx as value_trace_log,
            ):
                # evaluate() reads this to emit per-context rows when gated open.
                self._eval_per_context_logger = per_context_log
                # _train_offline reads this to emit apparent-value rows per epoch.
                self._offline_value_trace_logger = value_trace_log
                if self.algo_spec.data_regime == "offline":
                    self._train_offline(train_log, eval_log, aux_log, critic_log)
                elif self.algo_spec.kind == "on_policy":
                    self._train_on_policy(train_log, eval_log, critic_log, aux_log)
                elif self._is_online_grouped_run():
                    # Online episode-grouped off-policy (proximal latent-class variant
                    # under a fixed confounded behavior policy). Parallel to the flat
                    # _train_off_policy (untouched/golden-pinned).
                    self._train_off_policy_grouped(train_log, eval_log, aux_log)
                else:
                    self._train_off_policy(train_log, eval_log, aux_log)
        finally:
            self._eval_per_context_logger = None
            self._offline_value_trace_logger = None
            # Explicitly close vector envs to release EGL/GL contexts before interpreter shutdown.
            self.train_env.close()
            self.eval_env.close()

    def _is_recurrent_run(self) -> bool:
        """True iff the train_cfg requests any non-MLP network. Gates the
        off-policy buffer type and collection path (off-policy agents are bare
        modules without an is_recurrent attribute, so the signal comes from the
        config — the locked PR-1C2 decision)."""
        return (
            getattr(self.train_cfg, "actor_network", "mlp") != "mlp"
            or getattr(self.train_cfg, "critic_network", "mlp") != "mlp"
        )

    def _is_online_grouped_run(self) -> bool:
        """True iff this is an ONLINE off-policy run that needs episode grouping (the
        proximal latent-class variant collected live under a fixed confounded behavior
        policy — Gate B). Routes run() to _train_off_policy_grouped. Gated on the spec
        flag ONLY (not is_recurrent), so an online recurrent DQN keeps its existing
        _train_off_policy -> collect_off_policy_recurrent path untouched."""
        return (
            self.algo_spec.kind == "off_policy"
            and self.algo_spec.data_regime == "online"
            and getattr(self.algo_spec, "needs_episode_grouping", False)
        )

    def _needs_episode_grouping_run(self) -> bool:
        """True iff this run consumes episode-grouped sequences: a recurrent run OR
        an algo flagged needs_episode_grouping (the *_proximal latent-class
        variants). Gates the OFFLINE grouped path (Minari -> SequenceReplayBuffer).
        Non-grouped offline stays on the flat ReplayBuffer path, byte-identical.

        A strategy-critic ablation (observational/proximal/oracle_u) also forces the
        grouped path — the strategy critics need whole episodes (proximal's E-step)
        and the shared (B,T) window — even when the BASE algo is a plain flat learner
        (e.g. cql). The base agent's window is flattened for it in that loop."""
        if self.critic_ablation is not None and self.critic_ablation.is_strategy:
            return True
        return self._is_recurrent_run() or getattr(
            self.algo_spec, "needs_episode_grouping", False
        )

    def _make_replay_buffer(self):
        """Construct the off-policy buffer to pass to the builder: an episode-aware
        SequenceReplayBuffer for recurrent off-policy runs, else None (the builder
        keeps its own flat ReplayBuffer with its existing capacity -> goldens
        bitwise). On-policy builders ignore the buffer kwarg."""
        if self.algo_spec.kind == "off_policy" and self._is_recurrent_run():
            from src.rl.off_policy.sequence_replay_buffer import SequenceReplayBuffer

            return SequenceReplayBuffer(capacity=1_000_000, device=self.device)
        return None

    def _collect_off_policy(self, obs, total_steps, metrics_cache):
        """Route off-policy collection: recurrent runs use the sequence-buffer
        path (state threading + sequence sampling + BPTT update), everything else
        the verbatim/golden-pinned flat path."""
        if self._is_recurrent_run():
            return self.experience_source.collect_off_policy_recurrent(
                self.agent,
                self.replay_buffer,
                obs,
                collection_policy=self.collection_policy,
                n_steps=total_steps,
                n_envs=self.env_cfg.n_train_envs,
                warmup=self.offpolicy_warmup,
                batch_size=self.offpolicy_batch_size,
                seq_len=self.offpolicy_seq_len,
                metrics_cache=metrics_cache,
                aux_models=self.aux_models,
            )
        return self.experience_source.collect_off_policy(
            self.agent,
            self.replay_buffer,
            obs,
            collection_policy=self.collection_policy,
            n_steps=total_steps,
            n_envs=self.env_cfg.n_train_envs,
            warmup=self.offpolicy_warmup,
            batch_size=self.offpolicy_batch_size,
            metrics_cache=metrics_cache,
            aux_models=self.aux_models,
        )

    def _train_off_policy(
        self,
        train_logger: CSVLogger,
        eval_logger: CSVLogger,
        aux_logger: CSVLogger | None = None,
    ) -> None:
        assert self.replay_buffer is not None
        checkpoint_eps = self.train_cfg.checkpoint_episodes()
        obs, _ = self.train_env.reset()
        total_steps = self.env_cfg.rollout_len
        metrics_cache = None
        last_batch = None
        for ep in tqdm(
            range(self.train_cfg.n_episodes),
            desc=self.progress_label,
        ):
            # Collection/update loop relocated verbatim to
            # OnlineSource.collect_off_policy (Phase 1). aux_models (if any)
            # trains on each sampled batch inside the loop; last_batch is used
            # for checkpoint logging (no re-sampling, so off-policy RNG/golden
            # is unaffected). _collect_off_policy routes flat vs recurrent.
            obs, metrics_cache, last_batch = self._collect_off_policy(
                obs, total_steps, metrics_cache
            )

            if ep in checkpoint_eps:
                self._save_checkpoint(ep)
                eval_metrics = self.evaluate(ep)
                eval_metrics.update(
                    {
                        "algorithm": self.train_cfg.algorithm,
                        "environment": self.env_cfg.env_id,
                    }
                )
                eval_row = self._make_row(EVAL_COLUMNS, eval_metrics, ep)
                train_metrics = (metrics_cache or {}).copy()
                train_metrics.update(
                    {
                        "algorithm": self.train_cfg.algorithm,
                        "environment": self.env_cfg.env_id,
                    }
                )
                train_row = self._make_row(TRAIN_COLUMNS, train_metrics, ep)
                train_logger.log(train_row)
                eval_logger.log(eval_row)
                self._log_aux_rows(aux_logger, last_batch, ep)

    def _train_off_policy_grouped(
        self,
        train_logger: CSVLogger,
        eval_logger: CSVLogger,
        aux_logger: CSVLogger | None = None,
    ) -> None:
        """Online EPISODE-GROUPED off-policy training (Gate B: the proximal
        latent-class variant collected live under a FIXED confounded behavior policy).

        Parallel to _train_off_policy (flat, golden-pinned) and _train_offline_grouped
        (static Minari fill). The RUNNER owns a rolling SequenceReplayBuffer (the
        proximal builder makes a flat one that is unused here, exactly as the offline
        grouped path does). The estimator is put in ONLINE mode so its m_step cadence
        REFRESHES the rolling episode view (rebuild + re-canonicalize) rather than
        re-fitting the once-cached static batch.

        Scope: a fixed behavior policy => a stationary episode distribution (NOT the
        co-adapting cells 5-6). bias_confounded at full strength is purely U-indexed;
        train-env confounding + the collection policy were already wired at __init__.
        Five-keys: the stored transitions carry no realized U (the estimator infers).
        """
        from src.rl.off_policy.sequence_replay_buffer import SequenceReplayBuffer

        self.replay_buffer = SequenceReplayBuffer(
            capacity=1_000_000, device=self.device
        )
        # Put the proximal estimator in online (rolling-buffer) mode: m_step refreshes
        # the view on cadence. A no-op for any agent without a proximal EM attached.
        em = getattr(self.agent, "_proximal_em", None)
        if em is not None:
            em.online = True

        checkpoint_eps = self.train_cfg.checkpoint_episodes()
        obs, _ = self.train_env.reset()
        total_steps = self.env_cfg.rollout_len
        metrics_cache = None
        handed_off = False
        for ep in tqdm(range(self.train_cfg.n_episodes), desc=self.progress_label):
            obs, metrics_cache, handed_off = (
                self.experience_source.collect_off_policy_grouped(
                    self.agent,
                    self.replay_buffer,
                    obs,
                    collection_policy=self.collection_policy,
                    n_steps=total_steps,
                    n_envs=self.env_cfg.n_train_envs,
                    warmup=self.offpolicy_warmup,
                    batch_size=self.offpolicy_batch_size,
                    seq_len=self.offpolicy_seq_len,
                    metrics_cache=metrics_cache,
                    handed_off=handed_off,
                )
            )
            if ep in checkpoint_eps:
                self._save_checkpoint(ep)
                eval_metrics = self.evaluate(ep)
                eval_metrics.update(
                    {
                        "algorithm": self.train_cfg.algorithm,
                        "environment": self.env_cfg.env_id,
                    }
                )
                eval_row = self._make_row(EVAL_COLUMNS, eval_metrics, ep)
                train_metrics = (metrics_cache or {}).copy()
                train_metrics.update(
                    {
                        "algorithm": self.train_cfg.algorithm,
                        "environment": self.env_cfg.env_id,
                    }
                )
                train_row = self._make_row(TRAIN_COLUMNS, train_metrics, ep)
                train_logger.log(train_row)
                eval_logger.log(eval_row)

    def _train_offline(
        self,
        train_logger: CSVLogger,
        eval_logger: CSVLogger,
        aux_logger: CSVLogger | None = None,
        critic_logger: CSVLogger | None = None,
    ) -> None:
        """Offline (fixed-dataset) training.

        Offline has gradient steps, not env episodes, so two existing knobs are
        reinterpreted for this regime (deliberate semantic overload, kept to
        avoid config/schema churn):

          * ``n_episodes``  -> number of training EPOCHS (the outer loop and the
            checkpoint/logging axis; the frozen ``episode`` CSV column carries
            the epoch index, exactly as the online loops use it).
          * ``rollout_len`` -> number of GRADIENT STEPS per epoch.

        Behavior is the dataset, so the ``collection_policy`` seam is NOT used.
        The buffer is filled once from the Minari dataset; the training hot path
        stays batched (``sample`` -> ``agent.update``). ``train_return_*`` are
        left blank (as online off-policy already does); eval runs in the live
        env so returns stay comparable to online runs.
        """
        # Episode-grouped offline (proximal / future recurrent-offline): a parallel
        # path so the flat path below stays byte-frozen (golden 63/1). [PR-2a]
        if self._needs_episode_grouping_run():
            return self._train_offline_grouped(
                train_logger, eval_logger, aux_logger, critic_logger
            )
        assert self.replay_buffer is not None
        dataset_id = self.env_cfg.offline_dataset
        if not dataset_id:
            raise ValueError(
                "offline training requires env_cfg.offline_dataset (a Minari "
                "dataset id); pass --offline-dataset."
            )
        from src.envs.offline.minari_loader import (
            assert_dataset_matches_algo,
            fill_replay_buffer_from_minari,
            load_minari_dataset,
        )

        # Resolve the dataset (download-if-absent) and verify its action space
        # matches the consuming algo BEFORE filling — catches a continuous
        # dataset paired with a discrete offline algo (or vice versa) from the
        # dataset's metadata, not as a tensor crash mid-fill.
        dataset = load_minari_dataset(dataset_id)
        assert_dataset_matches_algo(
            dataset, self.action_type, dataset_id, self.train_cfg.algorithm
        )
        # Confounding gate (Cells 7-8): a confounded run must train on a dataset
        # whose confounding signature held at generation (docs/experimental_design
        # §7). Reject a missing signature (older dataset) or a failed gate before
        # training starts, with distinct messages.
        if self._value_trace_gate_open:
            meta = dataset.storage.metadata
            if "gate_test_passed" not in meta:
                raise ValueError(
                    f"Confounded offline run on dataset '{dataset_id}' requires "
                    "the confounding-signature metadata, but none is present "
                    "(likely generated before this metadata existed). Regenerate "
                    "the dataset with tools/generate_offline.py."
                )
            # σ=0.0 anchor: the dataset is unconfounded BY CONSTRUCTION (marginal
            # Corr(A,R) ≈ 0), so the gate test (which requires a non-zero marginal)
            # is meaningless here — it is the unconfounded baseline of the σ-sweep.
            # Skip the gate for σ=0.0 only. A missing σ field is treated as
            # σ != 0.0 (gate must pass; no silent fallback), and every σ > 0 keeps
            # the PR3 check unchanged. Exact == 0.0 is correct: σ is the operator's
            # CLI float, written verbatim by PR3 — no arithmetic drift to round off.
            if meta.get("behavior_strength_sigma") == 0.0:
                print(
                    "[runner] σ=0.0 anchor: skipping confounding gate test "
                    "(dataset is the unconfounded baseline by construction).",
                    file=sys.stderr,
                )
            elif not bool(meta["gate_test_passed"]):
                raise ValueError(
                    f"Dataset '{dataset_id}' failed the confounding gate test "
                    "(gate_test_passed=False): the confounding signature "
                    "(non-zero marginal Corr(A,R), near-zero partial Corr(A,R|U)) "
                    "did not hold at generation. Regenerate or inspect the dataset."
                )
        # For offline masked runs (Cell 4 / Cell 8) the same indices are dropped
        # from the dataset's obs/next_obs at load time — the eval env is masked
        # above, so the agent (built for the reduced dim) matches both. The
        # dataset on disk is unchanged; the projection is in-memory only.
        n_added = fill_replay_buffer_from_minari(
            dataset_id,
            self.replay_buffer,
            self.device,
            mask_indices=getattr(self.env_cfg, "mask_indices", None),
            # Oracle-U ceiling: load the per-transition latent U so the
            # U-conditioned critic can read batch["confounder_u"].
            load_u=self._requires_confounder_u,
        )
        if n_added == 0:
            raise ValueError(f"Minari dataset '{dataset_id}' yielded no transitions.")

        # Apparent-value trace queryability (Cells 7-8): warn once at run start
        # if the critic can't be queried at (s, a) so the curve is skipped rather
        # than wrong. All current offline algos are queryable (discrete q_network
        # / continuous twin-Q), so this is a defensive guard.
        value_trace_on = self._offline_value_trace_logger is not None
        if value_trace_on and self._apparent_value_unqueryable():
            print(
                f"[offline_value_trace] critic of '{self.train_cfg.algorithm}' "
                "cannot be queried at (s, a); skipping offline_value_trace.csv "
                "rows for this run.",
                file=sys.stderr,
            )
            value_trace_on = False

        checkpoint_eps = self.train_cfg.checkpoint_episodes()
        grad_steps_per_epoch = self.env_cfg.rollout_len
        batch_size = min(self.offpolicy_batch_size, len(self.replay_buffer))
        metrics_cache = None
        last_batch = None
        for ep in tqdm(range(self.train_cfg.n_episodes), desc=self.progress_label):
            for _ in range(grad_steps_per_epoch):
                batch = self.replay_buffer.sample(batch_size)
                metrics_cache = self.agent.update(batch)
                if self.aux_models is not None:
                    self.aux_models.update(batch)
                last_batch = batch

            # One apparent-value row per epoch (the Cell 7/8 training curve):
            # the critic's predicted Q on the last sampled batch's (s, a) pairs.
            # Reusing last_batch (vs a fresh sample) adds no extra RNG draw.
            if value_trace_on and last_batch is not None:
                self._log_offline_value_trace(ep, last_batch)

            if ep in checkpoint_eps:
                self._save_checkpoint(ep)
                eval_metrics = self.evaluate(ep)
                eval_metrics.update(
                    {
                        "algorithm": self.train_cfg.algorithm,
                        "environment": self.env_cfg.env_id,
                    }
                )
                eval_row = self._make_row(EVAL_COLUMNS, eval_metrics, ep)
                train_metrics = (metrics_cache or {}).copy()
                train_metrics.update(
                    {
                        "algorithm": self.train_cfg.algorithm,
                        "environment": self.env_cfg.env_id,
                    }
                )
                train_row = self._make_row(TRAIN_COLUMNS, train_metrics, ep)
                train_logger.log(train_row)
                eval_logger.log(eval_row)
                self._log_aux_rows(aux_logger, last_batch, ep)

    def _train_offline_grouped(
        self,
        train_logger: CSVLogger,
        eval_logger: CSVLogger,
        aux_logger: CSVLogger | None = None,
        critic_logger: CSVLogger | None = None,
    ) -> None:
        """Episode-grouped offline training (proximal latent-class variants; later
        offline-recurrent). Parallel to ``_train_offline`` so the flat path stays
        byte-frozen. The RUNNER owns a ``SequenceReplayBuffer`` (the offline
        builders ignore the passed buffer and make a flat one), fills it
        episode-grouped via ``fill_sequence_buffer_from_minari``, and samples
        same-episode ``(B, T, *)`` windows -> ``agent.update``; the AGENT owns
        sequence consumption (the runner only passes (B, T)). The dataset-resolve
        + confounding-gate block is duplicated from ``_train_offline`` (not shared)
        to keep that method byte-frozen.

        Deferred to PR-2b (NOT here): the offline value-trace on sequences + the
        whole-episode posterior accessor; aux-models on sequences. The Proximal
        stub degrades to the Observational floor (its wrapped learn flattens (B,T)).
        """
        dataset_id = self.env_cfg.offline_dataset
        if not dataset_id:
            raise ValueError(
                "offline training requires env_cfg.offline_dataset (a Minari "
                "dataset id); pass --offline-dataset."
            )
        from src.envs.offline.minari_loader import (
            assert_dataset_matches_algo,
            fill_sequence_buffer_from_minari,
            load_minari_dataset,
        )
        from src.rl.off_policy.sequence_replay_buffer import SequenceReplayBuffer

        dataset = load_minari_dataset(dataset_id)
        assert_dataset_matches_algo(
            dataset, self.action_type, dataset_id, self.train_cfg.algorithm
        )
        # Confounding gate (Cells 7-8) — duplicated verbatim from _train_offline.
        if self._value_trace_gate_open:
            meta = dataset.storage.metadata
            if "gate_test_passed" not in meta:
                raise ValueError(
                    f"Confounded offline run on dataset '{dataset_id}' requires "
                    "the confounding-signature metadata, but none is present "
                    "(likely generated before this metadata existed). Regenerate "
                    "the dataset with tools/generate_offline.py."
                )
            if meta.get("behavior_strength_sigma") == 0.0:
                print(
                    "[runner] sigma=0.0 anchor: skipping confounding gate test "
                    "(dataset is the unconfounded baseline by construction).",
                    file=sys.stderr,
                )
            elif not bool(meta["gate_test_passed"]):
                raise ValueError(
                    f"Dataset '{dataset_id}' failed the confounding gate test "
                    "(gate_test_passed=False): the confounding signature did not "
                    "hold at generation. Regenerate or inspect the dataset."
                )

        # The runner owns the episode-grouped buffer (offline builders ignore the
        # passed buffer). Mirror the flat fill's mask_indices / load_u args.
        self.replay_buffer = SequenceReplayBuffer(
            capacity=1_000_000, device=self.device
        )
        # Five-keys: the base algo's own load_u OR an oracle_u strategy critic in
        # the ablation (only the oracle critic's estimator ever reads the realized
        # U; observational/proximal never do). A plain-cql base has load_u=False, so
        # U is loaded here SOLELY for the oracle_u critic.
        strategy_ablation = (
            self.critic_ablation is not None and self.critic_ablation.is_strategy
        )
        load_u = self._requires_confounder_u or (
            strategy_ablation and self.critic_ablation.needs_u()
        )
        n_added = fill_sequence_buffer_from_minari(
            dataset_id,
            self.replay_buffer,
            self.device,
            mask_indices=getattr(self.env_cfg, "mask_indices", None),
            load_u=load_u,
        )
        if n_added == 0:
            raise ValueError(f"Minari dataset '{dataset_id}' yielded no transitions.")

        # Hand the episode-grouped buffer to the agent (the proximal E-step needs
        # whole episodes); a no-op for agents that don't consume it. [PR-2b]
        getattr(self.agent, "set_sequence_buffer", lambda *_a: None)(self.replay_buffer)

        # Fan the SAME shared buffer out to the strategy critics (proximal's E-step
        # warm-start + the fixed eval set for estimation-vs-oracle scoring).
        if strategy_ablation:
            self.critic_ablation.set_sequence_buffer(self.replay_buffer)
        # sigma (confounding strength) for the strategy scoring rows: prefer the
        # dataset's recorded value, else the run's behavior_strength.
        _sigma = 0.0
        if strategy_ablation:
            _meta = getattr(dataset.storage, "metadata", {}) or {}
            _sigma = float(
                _meta.get(
                    "behavior_strength_sigma",
                    getattr(self.env_cfg, "behavior_strength", 0.0) or 0.0,
                )
            )
        # A strategy ablation's base agent may be a plain flat learner (e.g. cql)
        # that cannot consume (B,T); flatten the shared window for it. Proximal /
        # recurrent bases own their (B,T) consumption, so they are left untouched.
        base_consumes_sequences = self._is_recurrent_run() or getattr(
            self.algo_spec, "needs_episode_grouping", False
        )

        checkpoint_eps = self.train_cfg.checkpoint_episodes()
        grad_steps_per_epoch = self.env_cfg.rollout_len
        seq_len = self.offpolicy_seq_len
        batch_size = self.offpolicy_batch_size
        metrics_cache = None
        critic_losses: Dict[str, float] = {}
        for ep in tqdm(range(self.train_cfg.n_episodes), desc=self.progress_label):
            for _ in range(grad_steps_per_epoch):
                if not self.replay_buffer.can_sample(seq_len):
                    break  # no episode long enough yet (skip-short semantics)
                batch = self.replay_buffer.sample_sequences(batch_size, seq_len)
                if strategy_ablation and not base_consumes_sequences:
                    # Flatten (B,T)->(B*T) for the plain base learner; the strategy
                    # critics still receive the ORIGINAL (B,T) window below, so all
                    # consumers fit the identical transitions (shared stream).
                    base_batch = {
                        k: (
                            v.flatten(0, 1)
                            if torch.is_tensor(v) and v.dim() >= 2
                            else v
                        )
                        for k, v in batch.items()
                    }
                else:
                    base_batch = batch
                metrics_cache = self.agent.update(base_batch)  # base actor
                if strategy_ablation:
                    critic_losses = self.critic_ablation.update_strategy(batch)
            if ep in checkpoint_eps:
                self._save_checkpoint(ep)
                eval_metrics = self.evaluate(ep)
                eval_metrics.update(
                    {
                        "algorithm": self.train_cfg.algorithm,
                        "environment": self.env_cfg.env_id,
                    }
                )
                eval_row = self._make_row(EVAL_COLUMNS, eval_metrics, ep)
                train_metrics = (metrics_cache or {}).copy()
                train_metrics.update(
                    {
                        "algorithm": self.train_cfg.algorithm,
                        "environment": self.env_cfg.env_id,
                    }
                )
                train_row = self._make_row(TRAIN_COLUMNS, train_metrics, ep)
                train_logger.log(train_row)
                eval_logger.log(eval_row)
                if strategy_ablation and critic_logger is not None:
                    for row in self.critic_ablation.checkpoint_rows_strategy(
                        episode=ep,
                        algorithm=self.train_cfg.algorithm,
                        environment=self.env_cfg.env_id,
                        sigma=_sigma,
                        latest_losses=critic_losses,
                    ):
                        critic_logger.log(row)
