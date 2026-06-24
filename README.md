# Benchmarking Causal Reinforcement Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-Compatible-green.svg)](https://gymnasium.farama.org/)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-Supported-orange.svg)](https://mujoco.org/)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen.svg)](https://pre-commit.com/)

A PyTorch reinforcement-learning benchmarking framework whose purpose is to vary the data-generating mechanism along five independent axes — algorithm, environment, data regime, dataset tier, and collection behavior policy — and measure what changes in the resulting learner. The same vectorized runner trains online, loads fixed offline datasets, or generates tiered offline datasets, while behavior policies and an optional unobserved confounder let you control the provenance of the data each algorithm sees. Outputs are CSVs with a frozen schema and bitwise-reproducible golden runs, so two executions of the same configuration produce byte-identical metrics.

---

## Installation

The canonical toolchain is [uv](https://docs.astral.sh/uv/) (PEP 621 `pyproject.toml` + `uv.lock`). The development interpreter is Python 3.13; `python` is not on `PATH`, so every command below goes through `uv run` (which uses the project `./.venv`).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # install uv (skip if already installed)
uv sync                                           # base runtime + dev tooling into ./.venv
uv run pre-commit install                         # enable formatting/lint hooks
```

The PyTorch CUDA 13.0 wheel index is pinned in `pyproject.toml`, so `uv sync` resolves `torch==2.10.0+cu130` (CUDA build) rather than the CPU wheel.

Optional environment families are opt-in extras:

```bash
# Install optional env families. uv sync is EXACT by default, so running
# `uv sync --extra X` separately each time OVERWRITES the previous extra,
# leaving only the last installed. Pass them together in ONE command:
uv sync --all-extras
# ...or a subset in a single invocation, e.g. continuous control:
uv sync --extra mujoco --extra box2d --extra offline
```

The available extras are `atari` (ALE/* Atari envs), `box2d` (LunarLander,
BipedalWalker), `minigrid`, `mujoco` (HalfCheetah, Hopper, Walker2d, …),
`robotics` (gymnasium-robotics Fetch/Hand), and `offline` (Minari offline stack).
Box2D additionally needs `swig` at build time when no
`box2d-py` wheel exists for your platform; it is declared in the `box2d` extra, but
if the build still fails install it from your system package manager first
(`sudo apt-get install -y swig`, or the equivalent) before re-running `uv sync`.

If you cannot use uv, `requirements.txt` is a generated export of the base runtime for a plain virtualenv:

```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

Regenerate `requirements.txt` after changing dependencies; do not edit it by hand.

---

## Quick start

```bash
# 1. Online: one algorithm on a discrete classic-control env.
uv run python main.py --envs CartPole-v1 --algos ppo

# 2. Online: continuous control on the whole MuJoCo set with SAC (needs: uv sync --extra mujoco).
uv run python main.py --env-set mujoco --algos sac

# 3. Offline (B1): build a small discrete dataset, then load it with CQL (needs: uv sync --extra offline).
uv run python tools/make_cartpole_offline.py --dataset-id cartpole/random-v0
uv run python main.py --envs CartPole-v1 --algos cql --offline-dataset cartpole/random-v0
```

The first offline command builds a small local Minari dataset; the second trains CQL on it (you cannot `--offline-dataset` an id that does not exist locally or hosted yet).

### Reproduce the full paper matrix
The offline cells (3, 4, 7, 8) need the generated Minari datasets. Generate them
first (resumable, one-time; ~30 min cold, near-instant once present):
```bash
# 1. Generate the offline datasets:
bash tools/generate_all_datasets.sh
```
Then run Cells 1–4, 7, 8 (76 YAMLs, 8 concurrent), which plots every resulting
run dir:
```bash
# 2. Run the full paper-matrix simulation:
bash tools/run_cells_1234_parallel.sh
```
Step 1 is optional on a fresh machine: the sweep's preflight auto-builds any
missing datasets on first run via the same `tools/generate_all_datasets.sh`.
Running it explicitly first just front-loads that one-time cost.

---

## Algorithms

The `--algos` values are the keys registered in `src/benchmarking/registry.py`. There is no `argparse choices=` list; values are validated against the registry at runtime.

| `--algos` key    | File                                | Action space | Regime  | Bitwise golden |
| ---------------- | ----------------------------------- | ------------ | ------- | -------------- |
| `vanilla`        | `src/rl/on_policy/vanilla.py`       | both         | on      | —              |
| `vanilla_ac`     | `src/rl/on_policy/vanilla.py`       | both         | on      | —              |
| `a2c`            | `src/rl/on_policy/a2c.py`           | both         | on      | —              |
| `ppo`            | `src/rl/on_policy/ppo.py`           | both         | on      | ✅ `test_regression.py` |
| `trpo`           | `src/rl/on_policy/trpo.py`          | both         | on      | —              |
| `dqn`            | `src/rl/off_policy/dqn.py`          | discrete     | off     | ✅ `test_regression_offpolicy.py` |
| `sac`            | `src/rl/off_policy/sac.py`          | continuous   | off     | ✅ `test_sac.py` |
| `ddpg`           | `src/rl/off_policy/ddpg.py`         | continuous   | off     | —              |
| `offline_dqn`    | `src/rl/offline/dqn.py`             | discrete     | offline | —              |
| `bcq`            | `src/rl/offline/bcq.py`             | discrete     | offline | —              |
| `cql`            | `src/rl/offline/cql.py`             | discrete     | offline | —              |
| `iql`            | `src/rl/offline/iql.py`             | discrete     | offline | —              |
| `cql_continuous` | `src/rl/offline/cql_continuous.py`  | continuous   | offline | —              |
| `iql_continuous` | `src/rl/offline/iql_continuous.py`  | continuous   | offline | —              |
| `bcq_continuous` | `src/rl/offline/bcq_continuous.py`  | continuous   | offline | —              |

`vanilla_ac` is an alias of `vanilla` (same builder). On-policy algorithms accept both discrete and continuous action spaces; the action type is derived from the environment and dispatched at runtime. `dqn` is discrete-only and `ddpg`/`sac` are continuous-only. The offline algorithms split discrete (`offline_dqn`, `bcq`, `cql`, `iql`) from continuous (`cql_continuous`, `iql_continuous`, `bcq_continuous`) by class, and the loader rejects a dataset whose action type does not match the chosen algorithm.

---

## Environments

Named environment sets are defined in `src/benchmarking/registry.py` (`ENV_SETS`, lines 38–98). `--env-set` overrides `--envs`.

| `--env-set`          | Resolved environment IDs |
| -------------------- | ------------------------ |
| `gymnasium`          | Blackjack-v1, CartPole-v1, MountainCar-v0, MountainCarContinuous-v0, Pendulum-v1, Acrobot-v1, LunarLander-v3, LunarLanderContinuous-v3, FrozenLake-v1, FrozenLake8x8-v1, CliffWalking-v1, Taxi-v3, BipedalWalker-v3, BipedalWalkerHardcore-v3 |
| `mujoco`             | Ant-v5, Reacher-v5, Pusher-v5, InvertedPendulum-v5, InvertedDoublePendulum-v5, HalfCheetah-v5, Hopper-v5, Swimmer-v5, Walker2d-v5, Humanoid-v5, HumanoidStandup-v5 |
| `atari`              | ALE/Pong-v5, ALE/Breakout-v5 (needs `uv sync --extra atari`) |
| `minigrid`           | MiniGrid-Empty-5x5-v0, MiniGrid-DoorKey-5x5-v0 (needs `uv sync --extra minigrid`) |
| `gymnasium-robotics` | FetchReach-v4, FetchPush-v4, FetchSlide-v4, FetchPickAndPlace-v4, HandReach-v3, HandManipulateBlock-v1, HandManipulateEgg-v1, HandManipulatePen-v1 |

Environment infrastructure:

- **Custom envs** — supply a Python entry point that builds a single Gymnasium-API env via `--env-entry-point module:callable` (or a `custom:`/`user:` prefix on the env id), with `--env-kwargs` accepting a JSON dict of constructor kwargs.
- **Image observations** — a Nature-style CNN backbone is selected automatically from the observation shape, so Atari and MiniGrid run through the same code path as vector-observation envs.
- **MiniGrid** — an RGB partial-render wrapper exposes a `(3, 84, 84)` image observation.

MuJoCo is wired through the `mujoco` env-set and the standard continuous-control algorithms, but the only continuous-control end-to-end test in CI runs on **Pendulum-v1** (classic-control). MuJoCo itself is not exercised by an automated end-to-end test.

```bash
# Custom env for all envs in the run.
uv run python main.py --env-wrapper custom --env-entry-point my_pkg.envs:make_env --envs CustomEnv-v0 --algos ppo

# Mixed run: prefix only the custom env id.
uv run python main.py --envs CartPole-v1 custom:my_pkg.envs:make_env --algos ppo
```

---

## Data regimes

### Online (default)

Vectorized live interaction. No extra flags; this is the path used by the quick-start examples.

```bash
uv run python main.py --envs CartPole-v1 --algos ppo dqn      # discrete
uv run python main.py --envs Pendulum-v1 --algos ddpg sac     # continuous
```

### Offline-load (B1)

`--offline-dataset <minari-id>` routes an offline algorithm to train on a fixed Minari dataset; the live env is still built for evaluation. The loader checks the local Minari cache first and downloads only if absent. Before filling the buffer it asserts that the dataset action type matches the algorithm (`src/envs/offline/minari_loader.py:40-56`), rejecting e.g. a continuous dataset paired with a discrete offline algorithm.

```bash
uv sync --extra offline
uv run python tools/make_cartpole_offline.py --dataset-id cartpole/random-v0
uv run python main.py --envs CartPole-v1 --algos cql --offline-dataset cartpole/random-v0
```

For offline algorithms, the two training knobs are reinterpreted: `--n-episodes` becomes the number of training **epochs** and `--rollout-len` becomes the number of **gradient steps per epoch** (offline has gradient steps, not env episodes).

### Offline-generate (B2)

`tools/generate_offline.py` produces a tiered Minari dataset by training an online generator (`dqn`/`sac`/`ddpg`), snapshotting a checkpoint by episode return, and rolling it out. Tiers are `random`, `medium`, and `expert`; the medium target is range-based, `R_random + (1/3)·(R_expert − R_random)`, and `--tier-fraction` overrides the `1/3`. The generated id is consumed via B1's `--offline-dataset`. `--offline-tier` lives only in this tool, not in `main.py`.

```bash
uv sync --extra offline
uv run python tools/generate_offline.py --env CartPole-v1 --algo dqn --offline-tier expert
uv run python main.py --envs CartPole-v1 --algos offline_dqn --offline-dataset generated/cartpole/expert-v0
```

For canonical per-env fixtures, `tools/make_{atari,cartpole,pendulum}_offline.py` build small ready-made datasets consumed the same way through `--offline-dataset`.

**Bulk dataset generation.** The 56 datasets required for Cells 3/4/7/8 of the experimental matrix can be generated in a single resumable run from the repo root:

```bash
bash tools/generate_all_datasets.sh
```

It skips datasets already present in the local Minari cache, so it's safe to Ctrl-C and re-invoke (it picks up where it stopped); progress logs to `tools/.generation_progress.log`. Total wall-clock on a recent GPU laptop: ~30 minutes.

---

## Behavior policies

`--behavior-policy` selects the policy that chooses actions during off-policy online collection (defined in `src/rl/policies/behavior_policy.py`). The default `agent` is byte-identical to the standard `agent.act` collection, so it leaves the off-policy golden values unchanged. `--behavior-strength` is the single primary knob; its meaning depends on the policy.

| `--behavior-policy` | Mechanism                                                        | Strength (`--behavior-strength`) | Action spaces |
| ------------------- | ---------------------------------------------------------------- | -------------------------------- | ------------- |
| `agent`             | Delegates to the agent's own `act` (collection baseline)         | — (no knob)                      | both          |
| `anti_reward`       | Critic-pessimal: picks the lowest-`Q` action the agent values    | ε (epsilon)                      | both (off-policy) |
| `bias_skew`         | With prob. `p` emits a fixed preferred action, else the agent's  | p                                | both          |
| `bias_suboptimal`   | With prob. `β` uses the agent, else a uniform-random action      | β (beta)                         | both          |
| `curiosity`         | Steers toward novel transitions via ensemble disagreement        | intensity                        | both (vector obs only) |
| `bias_confounded`   | Per-episode latent `U` biases the action (and perturbs reward)   | σ (sigma)                        | both          |

`bias_confounded` injects unobserved confounding: a per-episode latent `U` biases the action and, paired with `ConfoundedCollectionWrapper` on the train env, also perturbs the reward, so action and reward share a common cause that is **absent from the observation**. It requires the confounded wrapper on the train env because the policy reads the current `U` from the wrapper (`env.current_u`) while `U` never enters the observation; the strength `σ` scales both the action bias and the reward perturbation. The confounding signature — non-zero marginal `Corr(A, R)` but near-zero partial `Corr(A, R | U)` — is enforced by the gate test at `tests/test_confounded_collection.py::test_confounding_signature_marginal_nonzero_partial_zero`.

---

## Auxiliary models

`--aux-models` trains a learned reward model `r̂(s, a)` and next-state model `ŝ′(s, a)` alongside RL on the same batches the agent already uses. Their losses are **logged, never folded into the RL loss**; metrics go to a separate `aux_metrics.csv` (`episode, algorithm, environment, model, train_loss, mse, mae`) and the frozen train/eval schema is unchanged. It is off by default and composes with any algorithm and any data regime. Learning rate and hidden sizes are set with `--aux-lr` and `--aux-hidden-dims`.

```bash
uv run python main.py --envs CartPole-v1 --algos ppo --aux-models --aux-lr 3e-4 --aux-hidden-dims 64,64
```

---

## CLI reference

Defaults are taken directly from `main.py` argparse.

### Environment selection

| Flag                | Default  | Description |
| ------------------- | -------- | ----------- |
| `--envs`            | `None`   | One or more environment IDs |
| `--env-set`         | `None`   | Named environment set; overrides `--envs` |
| `--env-wrapper`     | `auto`   | Wrapper to use (`auto`, `gymnasium`, `custom`, or a registered name) |
| `--env-entry-point` | `None`   | Python entry point for custom envs (`module:callable`) |
| `--env-kwargs`      | `None`   | JSON dict of kwargs for the env entry point |

### Algorithm selection

| Flag      | Default | Description |
| --------- | ------- | ----------- |
| `--algos` | `None`  | One or more algorithm keys from the registry |

### Training configuration

| Flag              | Default | Description |
| ----------------- | ------- | ----------- |
| `--n-train-envs`  | `16`    | Parallel training environments |
| `--n-eval-envs`   | `16`    | Parallel evaluation environments |
| `--rollout-len`   | `1024`  | Steps per rollout |
| `--n-episodes`    | `250`   | Training episodes |
| `--n-checkpoints` | `25`    | Checkpoint evaluations (clamped to `[2, n_episodes]`) |
| `--seed`          | `42`    | Random seed |
| `--aggregation`   | `iqm`   | Reported-stat aggregation: `iqm` or `mean` |

### Data regime (B1 load)

| Flag                | Default | Description |
| ------------------- | ------- | ----------- |
| `--offline-dataset` | `None`  | Minari dataset id for offline algorithms; live env still built for eval |

### Behavior policy

| Flag                  | Default | Description |
| --------------------- | ------- | ----------- |
| `--behavior-policy`   | `agent` | Collection policy: `agent`, `anti_reward`, `bias_skew`, `bias_suboptimal`, `curiosity`, `bias_confounded` |
| `--behavior-strength` | `None`  | Primary knob for the behavior policy; `None` keeps the policy default |

### Auxiliary models

| Flag                | Default | Description |
| ------------------- | ------- | ----------- |
| `--aux-models`      | off     | Train logged `r̂(s,a)` + `ŝ′(s,a)` models (not in the RL loss) |
| `--aux-lr`          | `3e-4`  | Learning rate for the auxiliary models |
| `--aux-hidden-dims` | `64,64` | Comma-separated hidden layer sizes for the auxiliary models |

### Mode

| Flag                     | Default     | Description |
| ------------------------ | ----------- | ----------- |
| `--mode`                 | `benchmark` | `benchmark` or `critic_ablation` |
| `--ablation`             | off         | Shortcut for `--mode critic_ablation` (on-policy only) |
| `--ablation-critics`     | `standard_mlp` | Auxiliary critics to compare in ablation mode |
| `--ablation-lr`          | `3e-4`      | Learning rate for auxiliary critics |
| `--ablation-hidden-dims` | `64,64`     | Hidden sizes for auxiliary critics (`--ablation-hidded-dims` also accepted) |
| `--ablation-bins`        | `32`        | Histogram bins for MI/KL/JS metrics |

### Device

The device is auto-detected (CUDA if available, else CPU); there is no device flag.

| Flag              | Default | Description |
| ----------------- | ------- | ----------- |
| `--deterministic` | off     | Enable deterministic PyTorch (bitwise-reproducible runs) |

### Reproducibility

| Flag          | Default | Description |
| ------------- | ------- | ----------- |
| `--reproduce` | `None`  | Name of a YAML in `reproducibility/` (with or without extension) |

---

## Reproducibility

Pinned configurations live in `reproducibility/<name>.yaml`. Values from a reproduce file take precedence over CLI flags; among environment sources the precedence is:

```
--reproduce  >  --env-set  >  --envs
```

```bash
uv run python main.py --reproduce comoreai26
```

This loads `reproducibility/comoreai26.yaml`, applies it over the CLI defaults, and saves the effective configuration in the run folder.

---

## Run artifacts

Each execution creates a timestamped run directory:

```
runs/benchmark_<datetime>/
    config.yaml
    metadata.json
    train_metrics.csv
    eval_metrics.csv
    checkpoints/<env>_<algo>_seed<seed>/
    videos/<env>_<algo>_seed<seed>_ckptXXXX.mp4
```

`train_metrics.csv` and `eval_metrics.csv` follow the frozen schemas `TRAIN_COLUMNS` and `EVAL_COLUMNS` defined in `src/benchmarking/runner.py`; both carry explicit `algorithm` and `environment` columns. The schema is pinned by `tests/test_characterization.py`, and identical configurations produce byte-identical CSVs (see [Testing](#testing)).

---

## Plotting

`plot.py` renders plots and LaTeX tables from a run directory.

```bash
uv run python plot.py --run benchmark_YYYYMMDD_HHMMSS --split eval --x-axis frames --aggregation iqm
uv run python plot.py --run benchmark_YYYYMMDD_HHMMSS --split critic --x-axis episodes
```

| Flag            | Description |
| --------------- | ----------- |
| `--run`         | Run folder name inside `runs/` |
| `--split`       | `train`, `eval`, `critic`, `per_context`, `per_context_final`, `value_trace`, `both`, or `all` |
| `--x-axis`      | `episodes` or `frames` |
| `--aggregation` | `mean` or `iqm` |
| `--formats`     | Output formats (e.g. `png pdf`) |

`--split both` plots `train` + `eval` (plus critic metrics if the run was in `critic_ablation` mode); `--split critic` plots only `critic_ablation_metrics.csv`; `--split per_context` renders per-context return bands from `eval_per_context.csv` for masked runs, skipped if the file is absent; `--split per_context_final` renders the final-checkpoint per-context return distribution from `eval_per_context.csv` (cleaner companion to `per_context`), skipped if the file is absent; `--split value_trace` renders per-config apparent-vs-true curves and a σ-sweep panel from `offline_value_trace.csv` for confounded offline runs, skipped if the file is absent (σ-sweep panels use a twin-row layout — apparent Q on top, true return on bottom — with independent y-axes per row for legibility across scale-mismatched quantities); `--split all` plots all available splits. For offline runs the checkpoint axis is labelled "Training epochs" (offline learners have gradient epochs, not env episodes). Aggregation is either `mean` (mean ± standard deviation) or `iqm` (interquartile mean ± IQR-STD). Outputs land in `outputs/<run_name>/plots/` and `outputs/<run_name>/tables/`.

---

## Testing

Three bitwise-golden tests pin exact RNG behavior: PPO (`tests/test_regression.py`), DQN (`tests/test_regression_offpolicy.py`), and SAC (`tests/test_sac.py::test_sac_deterministic_bitwise`) — two runs of a fixed configuration must produce byte-identical CSVs. A grep-snapshot drift detector in `tests/test_characterization.py` guards seeding call-sites and other pinned references against silent moves, alongside the frozen CSV-schema tests. Run the suite with:

```bash
uv run pytest tests/
```

---

## Roadmap

These are not implemented:

- **NeuralBayesianNetworks integration** — a causal critic / reward-model component (currently absent from the codebase).
- **Generic POMDP / observation-masking wrapper** — partial observability beyond the existing image and confounded paths.
- **Multi-agent (MARL) extension** — the `src/marl/` package is a placeholder.

---

## Published works

```bibtex
@inproceedings{briglia2026pervasive,
  title={Causal Models Improve Reinforcement Learning for Pervasive and Robotic Tasks},
  author={Briglia, Giovanni and Mariani, Stefano and Zambonelli, Franco},
  booktitle={2026 IEEE International Conference on Pervasive Computing and Communications Workshops},
  year={2026},
  organization={IEEE Computer Society},
  note={In press}
}
```

---

## License

GPLv3 — see [LICENSE](LICENSE).
