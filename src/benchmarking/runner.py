from __future__ import annotations

import os
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
    CriticAblationConfig,
    CriticAblationManager,
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


@dataclass
class AlgorithmSpec:
    builder: Callable
    kind: str  # "on_policy" or "off_policy"
    # Data-source axis, orthogonal to ``kind``: "online" (live env interaction)
    # or "offline" (logged dataset; Stage B). Distinct from the agent's
    # vestigial ``Algorithm.paradigm`` (the on/off-policy learning regime).
    data_regime: str = "online"


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
        if getattr(env_cfg, "behavior_policy", "agent") == "bias_confounded":
            from src.envs.wrappers.confounded import ConfoundedCollectionWrapper

            _sigma = getattr(env_cfg, "behavior_strength", None)
            _sigma = 1.0 if _sigma is None else float(_sigma)
            self.train_env = ConfoundedCollectionWrapper(
                self.train_env, c_a=_sigma, c_r=_sigma
            )
        self.eval_env = build_env(
            env_id=env_cfg.env_id,
            n_envs=env_cfg.n_eval_envs,
            device=self.device,
            seed=env_cfg.seed + env_cfg.n_train_envs,
            render=True,
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
        )
        self.replay_buffer = None
        self.collection_policy = None
        self.offpolicy_batch_size = 128
        self.offpolicy_warmup = 1000
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
            if self.algo_spec.kind != "on_policy":
                raise ValueError(
                    "Critic ablation mode is supported only for on-policy algorithms."
                )
            gamma = float(getattr(self.agent, "gamma", 0.99))
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
        # Rollout loop relocated verbatim to OnlineSource.rollout (Phase 1).
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
        env.stop_video()
        mean, std = self._aggregate_returns(total_rewards)
        return {"eval_return_mean": mean, "eval_return_std": std}

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
                    fieldnames=CRITIC_ABLATION_COLUMNS,
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
            ):
                if self.algo_spec.data_regime == "offline":
                    self._train_offline(train_log, eval_log, aux_log)
                elif self.algo_spec.kind == "on_policy":
                    self._train_on_policy(train_log, eval_log, critic_log, aux_log)
                else:
                    self._train_off_policy(train_log, eval_log, aux_log)
        finally:
            # Explicitly close vector envs to release EGL/GL contexts before interpreter shutdown.
            self.train_env.close()
            self.eval_env.close()

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
            # is unaffected).
            obs, metrics_cache, last_batch = self.experience_source.collect_off_policy(
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

    def _train_offline(
        self,
        train_logger: CSVLogger,
        eval_logger: CSVLogger,
        aux_logger: CSVLogger | None = None,
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
        # For offline masked runs (Cell 4 / Cell 8) the same indices are dropped
        # from the dataset's obs/next_obs at load time — the eval env is masked
        # above, so the agent (built for the reduced dim) matches both. The
        # dataset on disk is unchanged; the projection is in-memory only.
        n_added = fill_replay_buffer_from_minari(
            dataset_id,
            self.replay_buffer,
            self.device,
            mask_indices=getattr(self.env_cfg, "mask_indices", None),
        )
        if n_added == 0:
            raise ValueError(f"Minari dataset '{dataset_id}' yielded no transitions.")

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
