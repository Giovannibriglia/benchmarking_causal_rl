from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import torch

from .device import detect_device


@dataclass
class EnvConfig:
    env_id: str = "CartPole-v1"
    n_train_envs: int = 16
    n_eval_envs: int = 16
    rollout_len: int = 1024
    # THE EVALUATION HORIZON, separated from ``rollout_len`` on 2026-09-01.
    # ``rollout_len`` carried three meanings at once -- on-policy collection
    # length, the legacy offline budget (grad steps per epoch), and the number
    # of steps every evaluation rollout runs. On an OFFLINE run the first two
    # are inert (``offline_grad_steps`` sizes the learner), so E1 set it to 2
    # for them and silently cut evaluation to two environment steps: every
    # policy scored 2.0-3.0, the metric had three reachable values, and no
    # return-based prediction could fail. Nothing errored; the CSVs were
    # well-formed. None => fall back to ``rollout_len``, so every existing run
    # is byte-identical.
    eval_rollout_len: Optional[int] = None
    seed: int = 42
    env_wrapper: str = "auto"
    env_entry_point: Optional[str] = None
    env_kwargs: dict = field(default_factory=dict)
    # Minari dataset id for data_regime="offline" runs; the live env above is
    # still built (offline eval runs in it). None for online runs.
    offline_dataset: Optional[str] = None
    # Off-policy online collection behavior policy (opt-in; default "agent" =
    # AgentBehaviorPolicy, byte-identical to the pre-A1 path). One of: agent,
    # anti_reward, curiosity, bias_skew, bias_suboptimal, bias_confounded.
    # behavior_strength maps to each policy's primary param (anti_reward=strength,
    # curiosity=strength, bias_skew=p, bias_suboptimal=beta,
    # bias_confounded=strength); for anti_reward/curiosity/bias_confounded the
    # dial is uniform: 0.0 = pure agent (baseline), 1.0 = fully active. None
    # keeps the policy default.
    behavior_policy: str = "agent"
    behavior_strength: Optional[float] = None
    # Action-dependent confounder (bias_confounded_action) reward-shift magnitude
    # c_r on the U->R edge, r += c_r * U * 1[a == a_bad]. DECOUPLED from
    # behavior_strength (sigma): sigma scales the U->A edge only; c_r is fixed
    # across the sigma sweep so the reward bonus on a_bad is invariant. None => the
    # 1.0 default. Unused by the additive bias_confounded path (cells 7/8), which
    # stays byte-frozen with c_r = c_a = sigma at its construction sites.
    confounder_c_r: Optional[float] = None
    # Deployment evaluation: keep U->R at eval while U->A is severed by the
    # learned policy acting (ruled 2026-08-31). The eval value is then
    # E_U[R|do(a),s], the causal estimand. Default False keeps every existing
    # run's eval env byte-identical (clean), so no frozen golden moves.
    eval_confounded_reward: bool = False
    # Gate firing probabilities (q0, q1) = P(bonus | U). Needed for the
    # ANALYTIC deployment return; supplied by the driver from arm_knobs (the
    # single construction site for the derivation), never re-derived here.
    gate_probs: tuple | None = None
    a_bad: int = 1
    # "analytic" (default when confounded eval is on) computes E_U[return] in
    # closed form on a CLEAN env; "sampled" draws U and is the sanity row.
    eval_confounded_mode: str = "analytic"
    # THE DATASET STORE, DECLARED. It was reachable only through the
    # MINARI_DATASETS_PATH environment variable, so every launch depended on
    # shell state -- exactly the kind of undeclared channel A1 forbids. The
    # env var still wins when set, so nothing that works today breaks; this
    # gives a config a way to say it.
    minari_datasets_path: str | None = None
    # GRACE arm: substitute interventional rewards into the offline buffer
    # before training. The variant is the base algorithm with one column
    # replaced, so this is an ARM flag rather than a critic name.
    grace_reward_transform: bool = False
    # The cell's DECLARED proxy channels (D-D: Z, W, V). Empty for cells that
    # declare none -- the diagram decides, never the config's convenience.
    grace_proxy_names: tuple = ()
    # FIXED exploration defining the SHARED base policy pi_basic (the common origin of
    # basic / biased / confounded). Read IDENTICALLY by behavior_policy="pi_basic" (the
    # basic arm) and "bias_confounded_action" (the confounded arm), so their
    # (beta=0, sigma=0) point is one identical policy. It must NOT inherit the learner's
    # decaying epsilon (that would desync the origin and, online, make the basic policy
    # non-stationary). None => the policy default (0.5: real preference, p away from 0,
    # NOT the uniform random tier). An explicit, reported parameter of the arms.
    pi_basic_epsilon: Optional[float] = None
    # Observation indices to drop from the flat obs vector (Z-hidden axis). For
    # online runs the runner wraps train+eval with MaskedObservationWrapper; for
    # offline runs the loader projects the same indices off the dataset's
    # obs/next_obs. None = no masking (default behavior unchanged).
    mask_indices: Optional[tuple] = None


@dataclass
class TrainingConfig:
    n_episodes: int = 250
    n_checkpoints: int = 25
    eval_interval: Optional[int] = None  # derived from n_episodes / n_checkpoints
    deterministic: bool = False
    device: str = field(default_factory=lambda: str(detect_device()))
    algorithm: str = "ppo"
    checkpoint_dir: Optional[str] = None
    aggregation: str = "iqm"
    # On-policy per-component network selection (separate actor/critic trunks).
    # The on-policy builders thread these into ActorCritic; off-policy builders
    # ignore them. Default mlp/mlp reproduces the plain-string algo behavior.
    # ``algorithm`` above carries the canonical id (e.g. ppo__lstm__lstm) for
    # on-policy runs; ``actor_network``/``critic_network`` drive construction.
    actor_network: str = "mlp"
    critic_network: str = "mlp"
    network_kwargs: dict = field(default_factory=dict)
    # OFFLINE gradient-step budget (feat/offline-budget-key). The offline learner reads
    # THIS as its total optimiser-step count, NOT n_episodes x rollout_len (those are
    # on-policy vectorized-rollout params that were leaking into the offline path). None
    # => the runner warns and falls back to the legacy product (keeps existing offline
    # goldens byte-identical). Production sets it in _base/budgets.yaml.
    offline_grad_steps: Optional[int] = None
    # Eval return accumulation: True counts the terminal step's reward (correct
    # under gymnasium NEXT_STEP autoreset; REQUIRED on sparse-reward envs like
    # MiniGrid/BabyAI whose only reward is terminal). False = the legacy
    # `reward * (~done)` accumulation that drops it — kept as default solely
    # because existing goldens are frozen on it (invisible off-by-one on
    # dense-reward envs). Hosted cells set it in their YAMLs.
    eval_count_terminal_reward: bool = False
    # Eval .mp4 recording. Rendering is side-effect-only (never changes numerics —
    # see runner._render_capable), but it spawns an ffmpeg writer and encodes
    # rollout_len frames at EVERY checkpoint. Default True keeps single runs
    # unchanged; the sweep driver disables it (sweep leaves never keep videos).
    record_eval_video: bool = True

    def checkpoint_episodes(self) -> list[int]:
        """Compute uniformly spaced checkpoint episodes including first and last."""
        count = max(2, min(self.n_checkpoints, self.n_episodes))
        if count == 2:
            return [0, self.n_episodes - 1]
        # linear spacing over episode indices
        indices = torch.linspace(0, self.n_episodes - 1, steps=count)
        unique = sorted({int(round(x.item())) for x in indices})
        # ensure first and last present
        unique[0] = 0
        unique[-1] = self.n_episodes - 1
        return unique

    def offline_checkpoint_steps(self) -> list[int]:
        """Uniformly spaced offline checkpoint STEP counts (1-indexed; the last is
        exactly ``offline_grad_steps``) — the step-keyed analogue of
        ``checkpoint_episodes`` for the offline_grad_steps loop (CHANGE 3). Example:
        offline_grad_steps=50_000, n_checkpoints=25 -> [2000, 4000, ..., 50000]."""
        if self.offline_grad_steps is None:
            raise ValueError(
                "offline_checkpoint_steps() requires offline_grad_steps to be set."
            )
        total = int(self.offline_grad_steps)
        count = max(1, min(self.n_checkpoints, total))
        steps = sorted({int(round(total * (i + 1) / count)) for i in range(count)})
        steps[-1] = total  # the final checkpoint is exactly offline_grad_steps
        return steps


@dataclass
class RunConfig:
    timestamp: str = field(
        default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run_dir: Optional[str] = None
    # Run-level flag (set by main.py = "any selected algo requires the confounder
    # U", i.e. any *_oracle_u variant). All sibling (env, algo) runners share one
    # offline_value_trace.csv in run_dir; when any of them is a U-variant the file
    # must use the u0-anchor SUPERSET schema so every runner writes a consistent
    # header (base runners blank-fill the u0 cells). A per-runner decision can't
    # see siblings, so this is decided once from the full algo list in main.py.
    value_trace_u0_schema: bool = False

    def resolve_run_dir(self) -> str:
        if self.run_dir is not None:
            return self.run_dir
        return f"runs/benchmark_{self.timestamp}"
