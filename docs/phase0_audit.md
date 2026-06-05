# Phase 0 Audit — `benchmarking_causal_rl` @ master (a98359b)

Date: 2026-06-05. Purpose: reconcile the names/signatures assumed in the causal-RL
execution prompt with the actual code, and record the environment facts that
constrain Phases 1–7. **No source file was modified for this audit.**

---

## 1. Actual module / class inventory

### 1.1 `src/rl/` — algorithms (prompt's inferred names → actual)

| Prompt assumed | Actual | File |
|---|---|---|
| `src/rl/base.py: Algorithm` | **does not exist** — two parallel ABCs instead (see below) | — |
| `vpg.py` | `src/rl/on_policy/vanilla.py: VanillaPolicyGradient` (registry key `"vanilla"`) | `src/rl/on_policy/vanilla.py` |
| `a2c.py` | `A2C` | `src/rl/on_policy/a2c.py` |
| `ppo.py` | `PPO` | `src/rl/on_policy/ppo.py` |
| `trpo.py` | `TRPO` (KL-penalty variant, not CG trust region) | `src/rl/on_policy/trpo.py` |
| `dqn.py` | `DQN` | `src/rl/off_policy/dqn.py` |
| `ddpg.py` | `DDPG` | `src/rl/off_policy/ddpg.py` |

**Two-ABC split (key Phase-1 fact):** there is no single `Algorithm` base.

- `src/rl/on_policy/base_actor_critic.py: BaseActorCritic(abc.ABC)` — plain object
  (NOT `nn.Module`); holds `policy`, `device`, `gamma=0.99`, `gae_lambda=0.95`;
  provides `compute_gae(rewards, dones, values, next_values)`; abstract
  `update(batch: RolloutBatch) -> Dict[str, float]`.
- `src/rl/off_policy/base_off_policy.py: BaseOffPolicy(abc.ABC)` — plain object;
  holds `device`, `gamma`; abstract `update(batch: Dict[str, Tensor]) -> Dict[str, float]`.
- On-policy agents act through a separate **policy network object**
  (`src/rl/on_policy/policy.py: ActorCriticMLP(BasePolicy)`); off-policy agents
  implement `act()` themselves (`DQN.act(obs, epsilon=None)`,
  `DDPG.act(obs, noise=True)`). Signature differences matter for the unified
  `Algorithm.act(obs, state=None, *, deterministic=False)` adapter.
- `src/rl/base_policy.py: BasePolicy(abc.ABC, torch.nn.Module)` — `act(obs)`,
  `value(obs)`. `ActorCriticMLP` adds `distribution(obs)`, `log_prob(dist, actions)`,
  `act_deterministic(obs)`. It is a **shared-encoder** actor-critic
  (`self.encoder` MLP + `self.actor` Linear head + `self.critic` Linear head) —
  the README's "strict actor–critic separation" is at the *head* level, not
  separate networks. DDPG has genuinely separate `actor`/`critic` MLPs.
- Batch type: `RolloutBatch` dataclass (obs, next_obs, actions, log_probs,
  rewards, dones, values, next_values, advantages, returns) — defined in
  `base_actor_critic.py`, imported by `runner.py` and `critic_ablation.py`.
- `src/rl/off_policy/replay_buffer.py: ReplayBuffer(capacity, device)` —
  deque of CPU tensor dicts; `sample()` uses **Python `random.sample`**
  (RNG-order relevant); returns dict batches keyed obs/actions/rewards/next_obs/dones.
- `src/rl/nets/mlp.py: MLP(input_dim, output_dim, hidden_dims=(64,64), activation=nn.Tanh, output_activation=None)`.

### 1.2 `src/benchmarking/`

- `registry.py`: `Registry` (algo name → `AlgorithmSpec`), module-level `registry`,
  `ENV_SETS` dict (`gymnasium`, `mujoco`, `gymnasium-robotics`), `expand_env_set(name)`,
  `register_default_algorithms()`. Builders return a `(policy, agent)` **tuple**;
  builder kwargs: `obs_dim, action_dim, action_type, device, action_space`.
- `runner.py`: `AlgorithmSpec(builder, kind)` with `kind ∈ {"on_policy","off_policy"}`
  (defined HERE, not in registry.py); `BenchmarkRunner(env_cfg, train_cfg, run_cfg,
  algo_spec, critic_ablation_cfg=None, progress_label=None)`.
  - `TRAIN_COLUMNS` / `EVAL_COLUMNS` schemas live at module top.
  - **The rollout loop to wrap verbatim in Phase 1** is
    `BenchmarkRunner._collect_on_policy()` (lines ~141–197): per-step order is
    `policy.value(obs)` → `policy.act(obs)` → `env.step` → `policy.value(next_obs)`;
    then `agent.compute_gae(...)`. The off-policy loop is inlined in
    `_train_off_policy()` (act → step → per-env `buffer.add` → `buffer.sample` +
    `agent.update` once warm: `len(buffer) > max(1000, 128)`).
  - `run()` calls `set_seed(seed, deterministic)` then dispatches by `kind`.
  - Eval: `evaluate(episode)` runs `rollout_len` steps on `eval_env`
    (seeded `seed + n_train_envs`), always records video, masks rewards after
    done via `total_rewards += reward * (~done)` (only counts pre-first-done
    reward per env slot, with auto-reset continuing underneath).
  - Aggregation: `_aggregate_returns` implements mean or IQM (middle-50% mean/std).
- `critic_ablation.py`: `CRITIC_ABLATION_COLUMNS` (19 cols, matches the invariant
  list + leading `episode, algorithm, environment, critic, train_loss`),
  `CriticSpec`, `CRITIC_LIBRARY` (`standard_mlp`, `residual_reward_model`),
  `default_aux_critics()`, `ResidualValueNet`, `CriticAblationConfig`,
  `AuxiliaryCritic`, `CriticAblationManager`, divergence helpers
  `_explained_variance, _pearson, _spearman, _distribution_metrics, _compute_metrics`
  (module-private — the prompt's "shared divergence utilities may be imported"
  refers to these; they are `_`-prefixed, so new code importing them should do so
  explicitly without renaming/moving).
- `checkpoints.py`: `save_checkpoint(path, state)`, `load_checkpoint(path)` (torch.save/load).
- `plotting.py`: `load_run`, `discover_metrics`, `_iqm`, `_iqr_std`,
  `compute_aggregates`, `plot_metric`, `make_latex_table`, `build_tables`,
  `run_plotting`, `main`. CLI: `--run` (required), `--split {train,eval,critic,both,all}`,
  `--x-axis {episodes,frames}` (default **frames**), `--aggregation {mean,iqm}`,
  `--outdir` (default `outputs`), `--formats` (default png pdf).
  `plot.py` is a 5-line shim calling `src.benchmarking.plotting.main`.

### 1.3 `src/config/`

- `defaults.py`: `EnvConfig(env_id, n_train_envs=16, n_eval_envs=16, rollout_len=1024,
  seed=42, env_wrapper="auto", env_entry_point=None, env_kwargs={})`;
  `TrainingConfig(n_episodes=250, n_checkpoints=25, eval_interval=None,
  deterministic=False, device=detect_device(), algorithm="ppo",
  checkpoint_dir=None, aggregation="iqm")` with `checkpoint_episodes()`;
  `RunConfig(timestamp, run_dir)` with `resolve_run_dir()`.
- `seeding.py: set_seed(seed, deterministic=False)` — seeds `random`, `np.random`,
  `torch.manual_seed`, `torch.cuda.manual_seed_all`, sets `PYTHONHASHSEED`;
  deterministic → `torch.use_deterministic_algorithms(True)` + cudnn.benchmark=False.
- `device.py: detect_device()` — cuda if available else cpu.

**Seeding call sites (Phase-1 grep baseline)** — all in `src/config/seeding.py`:
```
src/config/seeding.py:15:    random.seed(seed)
src/config/seeding.py:16:    np.random.seed(seed)
src/config/seeding.py:17:    torch.manual_seed(seed)
src/config/seeding.py:18:    torch.cuda.manual_seed_all(seed)
```
(Env seeding goes through `env.reset(seed=...)` inside the wrappers; PPO consumes
torch RNG via `torch.randperm`; DQN via `torch.rand`/`torch.randint`; DDPG via
`torch.randn_like`; ReplayBuffer via Python `random.sample`.)

### 1.4 `src/envs/`

- `base.py: BaseEnv(abc.ABC)` — `reset(seed=None)`, `step(action)`, `close()`;
  tensors in/out, 5-tuple `(obs, reward, terminated, truncated, info)`.
- `registry.py`: `EnvWrapperSpec(name, builder, match, requires_entry_point)`,
  `EnvWrapperRegistry`, module-level `registry`, `register_default_env_wrappers()`
  (registers `custom` then `gymnasium`; gymnasium is the fallback), `build_env(*,
  env_id, n_envs, device, seed, render, record_video, video_path, env_wrapper,
  env_entry_point, env_kwargs)`.
- `wrappers/gymnasium_env.py: GymnasiumEnv` — `gym.vector.SyncVectorEnv`,
  flattens obs via `gymnasium.spaces.utils.flatten`, exposes `obs_space`
  (flattened single-env space) / `act_space` / `n_envs`, manual reset-on-done
  (re-resets ALL envs when any is done if `reset_done` unavailable — note: with
  SyncVectorEnv + autoreset this code path interacts with gymnasium's own
  autoreset; relevant when adding `info["full_obs"]`), per-step video for env 0,
  `start_video(path)` / `stop_video()`. Sets `MUJOCO_GL=egl`.
  **`info` dict is passed through from `gym.vector` — the natural place to add
  `full_obs` / `confounder_u` / `behavior_logprob` in Phases 2/4.**
- `wrappers/custom_env.py: CustomEnv` + `load_entry_point` — entry-point factory envs.
- `wrappers/video.py: SingleVideoRecorder(path, fps=30)` — imageio/ffmpeg.

### 1.5 `src/logging/`, `src/marl/`

- `logger.py: CSVLogger(filepath, fieldnames)` — context manager, append mode,
  header written once, rows filtered to fieldnames, flush per row.
- `metrics.py`: `to_cpu_scalar`, `EpisodeMetrics` (appears unused by runner).
- `marl/__init__.py` — empty placeholder, as expected.

### 1.6 CLI (`main.py`) — actual flags vs prompt

- Exists as stated: `--mode {benchmark,critic_ablation}`, `--ablation`, `--envs`,
  `--algos`, `--env-set`, `--env-wrapper`, `--env-entry-point`, `--env-kwargs`,
  `--n-train-envs 16`, `--n-eval-envs 16`, `--rollout-len 1024`, `--n-episodes 250`,
  `--n-checkpoints 25`, `--seed 42`, `--reproduce`, `--deterministic`,
  `--aggregation {iqm,mean}`, `--ablation-critics/-lr/-hidden-dims/-bins`.
- **DISCREPANCY: there is NO `--device` CLI flag.** Device comes from the
  reproduce YAML (`training.device` or top-level `device`) or `detect_device()`.
  The invariant "existing flags keep their names and defaults (… `--device` …)"
  cannot apply to a flag that doesn't exist; adding `--device` would be a new,
  backward-compatible flag (decision deferred to the Phase-1 gate).
- **Precedence is `reproduce-YAML > CLI` for every overlapping key** (e.g.
  `--reproduce comoreai26 --n-episodes 2` runs **250** episodes because the YAML
  sets `training.n_episodes: 250`). The §8 verification command implicitly
  assumes CLI override; it does not exist on master. Flagged at the gate.
- Run-dir naming: `runs/benchmark_<ts>` (CLI), `runs/<repro_tag>_<ts>` (reproduce),
  `runs/<mode>_<ts>` (non-benchmark mode). `config.yaml` + `metadata.json`
  snapshot written by `main()` before any training.

---

## 2. Blocking / surprising findings (need author awareness)

0. **Master is NOT run-to-run reproducible, even with `--deterministic`.**
   Verified empirically: two runs of
   `python main.py --envs CartPole-v1 --algos ppo --n-episodes 5 --deterministic`
   differ from episode 0. Root cause (confirmed by isolation test): policy
   weights are initialized in `BenchmarkRunner.__init__` (via the registry
   builder), but `set_seed` is only called later in `BenchmarkRunner.run()` —
   so weight init consumes torch's process-random default seed. Pre-seeding the
   process (`set_seed(42, deterministic=True)` before `main()`, no source
   change) makes both the benchmark and critic-ablation pipelines **bitwise
   reproducible on GPU** (all CSVs diff-identical across two runs).
   → Golden values were generated with that pre-seed
   (`tests/_phase0_seed_probe.py`); proposed Phase-1 fix: call `set_seed`
   before network construction (e.g. top of `BenchmarkRunner.__init__`,
   keeping the existing `run()` call so the in-training RNG stream is
   untouched). Needs explicit sign-off per §3.1, since "preserve master's
   numbers exactly" is unsatisfiable — master has no stable numbers.

1. **`comoreai26.yaml` is NOT runnable on current master.** It lists
   `algos: a2c vanilla a2c_cc vanilla_cc`, but `a2c_cc` / `vanilla_cc` are
   registered nowhere (`register_default_algorithms` registers only
   vanilla/a2c/ppo/trpo/dqn/ddpg). Execution would crash at
   `registry.get("a2c_cc")` → `KeyError` after finishing the full
   `a2c` + `vanilla` sweeps. The "comoreai26 untouched and runnable" invariant
   is therefore unsatisfiable as stated on unmodified master.
2. **Ghosts of deleted causal code.** Untracked `__pycache__` artifacts exist for
   modules with no source: `src/causal_metrics/{estimators,gap}.pyc`,
   `src/envs/causal_base.pyc`, `src/envs/registry` variants,
   `src/benchmarking/offline_collector.pyc`, and a run folder
   `runs/causal_8cells_20260429_092500/` whose `config.yaml` references
   `env_set: causal_8cells`, envs `causal-sepsis-cell1..8`, algo
   `confounded_dqn`, and a `divergences` key — none exist in HEAD. A previous
   causal-cells implementation (sepsis-based) was removed or never committed.
   `tests/` also exists with empty `causal_metrics/ envs/ rl/` subdirs and no files.
3. **Staged `.pyc` files.** `git status` shows many `src/**/__pycache__/*.pyc`
   staged for commit (`A`/`AM`). Recommend unstaging and adding `__pycache__/`
   to `.gitignore` (no `.gitignore` policy change made in Phase 0 without approval).
4. **Eval-return semantics:** `evaluate()` accumulates `reward * (~done)` over a
   fixed `rollout_len` window — i.e., return until *first* episode end per env
   slot, not per-episode returns. The Phase-3+ regret protocol (`J` over ≥100
   episodes) must implement proper episode accounting in NEW code; the existing
   eval path stays as-is.
5. **Hidden-state structure:** `ActorCriticMLP` is shared-encoder. The
   "strict actor/critic separation" required by the prompt's `Algorithm` ABC
   (`self.actor` / `self.critic` distinct `nn.Module`s) holds only at head level
   for on-policy algos. Phase 1 must not change this (would change numbers);
   the ABC contract will be worded to accept head-level separation.

---

## 3. Environment facts (constrain Phases 3–6)

| Fact | Value | Impact |
|---|---|---|
| Python (venv `.venv`) | **3.13.7** (README badge says 3.10+) | d3rlpy/minari wheels must support 3.13 — verify at Phase-3 start |
| torch | 2.10.0+cu130 | deterministic CUDA runs work **without** `CUBLAS_WORKSPACE_CONFIG` (verified empirically) |
| gymnasium | 1.2.3 (+robotics 1.4.2, mujoco 3.4.0) | HalfCheetah-v5 available |
| numpy / pandas | 2.3.5 / 3.0.0 | pandas 3.0 — check d3rlpy/minari compat |
| **minari** | **NOT installed** | Phase-3 dependency to add & pin |
| **d3rlpy** | **NOT installed** | Phase-3 dependency to add & pin |
| GPU | **RTX 4070 Laptop, 8 GB** (prompt said TITAN X) | budgets OK; delphic ensemble sizing must fit 8 GB |
| pytest | 9.0.3 | scaffold ready |

## 4. Golden baseline (Phase-0 deliverable)

- Golden job: `--envs CartPole-v1 --algos ppo --n-episodes 5 --deterministic`
  (CLI defaults otherwise: seed 42, 16 train/eval envs, rollout 1024, device
  cuda), launched via `tests/_phase0_seed_probe.py` which pre-seeds the process
  before `main()` (see finding 2.0 — required for reproducibility on master;
  redundant-but-harmless after the Phase-1 fix).
- `tests/golden/benchmark/`: `train_metrics.csv`, `eval_metrics.csv`.
- `tests/golden/critic_ablation/`: `train_metrics.csv`, `eval_metrics.csv`,
  `critic_ablation_metrics.csv` (same job + `--ablation`, default
  `standard_mlp` critic).
- Grep baselines: `tests/golden/seeding_call_sites.txt` (§3 pattern),
  `tests/golden/critic_ablation_refs.txt`; both enforced by
  `tests/test_characterization.py::test_grep_snapshot_unchanged`.
- Reproducibility evidence: each golden job was run **twice**; all CSVs were
  bitwise identical across runs (GPU, deterministic, pre-seeded). Goldens are
  machine-pinned (RTX 4070 Laptop, torch 2.10.0+cu130); exact equality is only
  expected on this stack.
- Test suite: `pytest tests/ -x -q` → **11 passed** (9 characterization +
  2 golden regression) in ~46 s.

## 4.1 Phase-0 gate decisions (author, 2026-06-05)

1. Seeding: sanctioned semantic change — Phase 1 moves `set_seed()` before
   policy/network construction. Goldens = the pre-seeded probe runs.
2. comoreai26: invariant downgraded to "YAML byte-untouched + a2c/vanilla
   portion runs"; `a2c_cc`/`vanilla_cc` breakage documented in
   `docs/known_issues.md`, not fixed.
3. Precedence `reproduce-YAML > CLI` preserved as master semantics; §8
   verification command amended (see known_issues.md). `--device` removed
   from the invariant flag list; no new CLI flags in Phase 1.
4. Hygiene: `.pyc` unstaged, `__pycache__/` gitignored;
   `runs/causal_8cells_*` left in place (restored from Trash after it was
   file-manager-trashed at 15:30 on 2026-06-05, outside this work).
5. ABC contract: head-level actor/critic separation acceptable for on-policy;
   `ActorCriticMLP` not restructured. Regret evaluator (Phase 3+) implements
   its own per-episode J accounting; existing `evaluate()` untouched.
6. Phase 2+ design: causal envs are registered as Gymnasium ids under a
   `causal/` namespace + `ENV_SETS` entries (cells 1–2 runnable via plain
   benchmark mode); cell YAMLs reference these env ids; `--mode causal_cells`
   orchestrates only datasets/OPE/regret (cells 3–8).

## 5. Name mapping adopted for Phases 1+

- `Algorithm` ABC will live at `src/rl/base.py` (new file, no clash).
- Existing classes keep their names; registry keys stay
  `vanilla|a2c|ppo|trpo|dqn|ddpg` (prompt's "vpg" = `vanilla`).
- `AlgorithmSpec` stays defined in `runner.py` (imported by registry) until a
  phase explicitly moves it (none planned).
- `RolloutBatch` remains the on-policy batch type; the `ExperienceSource.rollout`
  return type aliases it (`Batch = RolloutBatch` initially).
