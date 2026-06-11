
# Benchmarking Causal Reinforcement Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-Compatible-green.svg)](https://gymnasium.farama.org/)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-Supported-orange.svg)](https://mujoco.org/)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen.svg)](https://pre-commit.com/)

A modular, PyTorch-first benchmarking framework for single- and multi-agent reinforcement learning — designed for reproducibility, extensibility, and causal augmentation.

This repository provides a clean architectural scaffold to evaluate standard (MA)RL algorithms under a unified, vectorized, and research-oriented pipeline, with built-in support for reproducible experiments and structured evaluation.

---

# Why This Framework?

Most RL benchmarking codebases grow organically and become difficult to extend, reproduce, or adapt for research-level experimentation.

This project enforces:

- Strict actor–critic separation
- Single-device propagation (CUDA-first, CPU fallback)
- Vectorized environments as a first-class abstraction
- Deterministic experiment reproducibility
- Structured checkpointing and evaluation
- Clean modular extension for causal components

The goal is not just to run (MA)RL — but to **benchmark it rigorously**.

### Planned features:
- insert *Neural Bayesian Network"
- train and load baselines
- environment wrapper for other and general usage: at the moment there is only GymnasiumEnv

---

# Core Features

## Architecture

- PyTorch-first implementation (all tensors live on one detected device).
- Vectorized Gymnasium environments.
- Automatic flattening of arbitrary observation spaces (Box, Dict, Tuple, MultiDiscrete, nested).
- Modular environment wrapper system.
- Clear separation between:
  - `rl/` (single-agent)
  - `marl/` (future multi-agent)
  - `benchmarking/` (runner, evaluation, registry)

---

## Implemented Algorithms

The names below are the exact `--algos` keys (from `register_default_algorithms`).

### On-Policy (online)

- `vanilla` — **Vanilla Policy Gradient**
- `vanilla_ac` — alias of `vanilla` (same builder/spec)
- `a2c` — **A2C**
- `ppo` — **PPO**
- `trpo` — **TRPO** (KL-penalized trust-region variant)

### Off-Policy (online)

- `dqn` — **DQN** — **discrete** action spaces only
- `ddpg` — **DDPG** — **continuous** action spaces only
- `sac` — **SAC** — **continuous** action spaces only

### Offline (fixed dataset)

- `offline_dqn` — **DQN over a fixed Minari dataset** (discrete); requires `--offline-dataset`. Eval still runs in the live env. See the offline workflow under [Running Experiments](#running-experiments).

All implementations maintain strict separation between actor and critic modules.

---

# Supported Environments

The framework currently supports **Gymnasium environments** through a unified vectorized wrapper.
It also supports **custom user environments** via a dedicated wrapper.

### Observation Spaces Supported

- `Box`
- `Discrete`
- `MultiDiscrete`
- `Tuple`
- `Dict`
- Nested combinations

Observations are automatically flattened into a tensor representation.

### Environment Sets (`--env-set`)

Named environment groups, defined in `registry.py` and easily extendable, can be used:

- `gymnasium`
- `mujoco`
- `gymnasium-robotics`

---

### Custom Environments

You can plug in a custom environment by providing a Python entry point that builds a single env.
The custom env must expose `observation_space` and `action_space` (Gymnasium spaces) and implement
`reset()` and `step()` following the Gymnasium API.

Use one of the following patterns:

```bash
# Use custom wrapper for all envs in the run
uv run python main.py --env-wrapper custom --env-entry-point my_pkg.envs:make_env --envs CustomEnv-v0 --algos ppo

# Mixed envs: prefix only the custom ones
uv run python main.py --envs CartPole-v1 custom:my_pkg.envs:make_env --algos ppo
```

You can pass JSON kwargs to the entry point with `--env-kwargs`.

# Main Training Script (`main.py`)

## Running Experiments

Command reference (all forms use `uv run`, consistent with the install section).
Registered `--algos`: on-policy `vanilla` `vanilla_ac` `a2c` `ppo` `trpo`; online off-policy `dqn` `sac` `ddpg`; offline `offline_dqn`. Named `--env-set`: `gymnasium`, `mujoco`, `gymnasium-robotics`.

```bash
# --- setup ---
uv sync                                 # base runtime + dev tooling
uv sync --extra mujoco                  # add an env family: atari | minigrid | mujoco | robotics | minari | offline

# --- online benchmark (vectorized envs) ---
uv run python main.py --envs CartPole-v1 --algos ppo dqn          # discrete env
uv run python main.py --envs Pendulum-v1 --algos ppo ddpg sac     # continuous env
uv run python main.py --env-set gymnasium --algos ppo             # whole named set
uv run python main.py --env-set mujoco --algos sac ddpg           # needs: uv sync --extra mujoco

# --- common knobs ---
uv run python main.py --envs CartPole-v1 --algos ppo \
    --n-train-envs 16 --n-eval-envs 16 --rollout-len 1024 \
    --n-episodes 250 --n-checkpoints 25 --seed 42 \
    --deterministic --aggregation iqm                            # deterministic = bitwise-reproducible

# --- critic-ablation mode (on-policy only) ---
uv run python main.py --envs CartPole-v1 --algos ppo --ablation   # == --mode critic_ablation

# --- offline training (fixed dataset; eval still runs in the live env) ---
uv sync --extra offline
uv run python tools/make_cartpole_offline.py --dataset-id cartpole/random-v0   # build a tiny dataset
uv run python main.py --envs CartPole-v1 --algos offline_dqn \
    --offline-dataset cartpole/random-v0

# --- auxiliary reward/next-state models (opt-in; any algo/regime; logged to aux_metrics.csv, not in the RL loss) ---
uv run python main.py --envs CartPole-v1 --algos ppo --aux-models --aux-lr 3e-4 --aux-hidden-dims 64,64

# --- reproduce a pinned config ---
uv run python main.py --reproduce <name>.yaml    # reads reproducibility/<name>.yaml

# --- plots + LaTeX tables from a run ---
uv run python plot.py --run <run_name>           # see the Plotting section / plot.py --help
```

Outputs: per-run CSVs and checkpoints land in `runs/<run>/`; plots and LaTeX tables in `outputs/<run>/` (see [Run Artifacts](#run-artifacts) and [Plotting and Tables](#plotting-and-tables-plotpy)).

> **Note:** `atari` and `minigrid` are installable extras but are **not** yet runnable `--env-set`s — image/grid observation backbones arrive in PR6.

The `--algos` list runs each `(algorithm, environment, seed)` combination independently; multiple algorithms and environments can be passed together (e.g. `--envs CartPole-v1 HalfCheetah-v5 --algos ppo trpo`). Custom critics for ablation: `--ablation-critics standard_mlp residual_reward_model`.

---

## CLI Parameters

### Environment Selection

| Flag        | Description                                      |
| ----------- | ------------------------------------------------ |
| `--envs`    | List of environment IDs                          |
| `--env-set` | Named group of environments (overrides `--envs`) |
| `--env-wrapper` | Env wrapper to use (`auto`, `gymnasium`, `custom`, or a registered name) |
| `--env-entry-point` | Python entry point for custom envs (`module:callable`) |
| `--env-kwargs` | JSON dict of kwargs for the env entry point |
| `--offline-dataset` | Minari dataset id for offline algorithms (`--algos offline_dqn`); the live env is still built for eval |

Precedence:
`--reproduce` > `--env-set` > `--envs`

---

### Algorithm Selection

| Flag      | Description                     |
| --------- | ------------------------------- |
| `--algos` | List of algorithms to benchmark |

Each `(algorithm, environment, seed)` combination runs independently.

---

### Training Configuration

| Flag              | Description                                              |Default|
| ----------------- | -------------------------------------------------------- |-------|
| `--n-episodes`    | Number of training episodes                              |250    |
| `--rollout-len`   | Steps per episode                                        |1024   |
| `--n-train-envs`  | Parallel training environments                           |16     |
| `--n-eval-envs`   | Parallel evaluation environments                         |16     |
| `--n-checkpoints` | Number of checkpoint evaluations (min=2, max=n_episodes) |25     |
| `--seed`          | Random seed                                              |42     |
| `--aggregation`   | Reported-stat aggregation: `iqm` or `mean`              |iqm    |

For offline algorithms (`offline_dqn`), `--n-episodes` is reinterpreted as the number of training **epochs** and `--rollout-len` as **gradient steps per epoch** (offline has gradient steps, not env episodes).

### Experiment Mode

| Flag              | Description                                                                 |
| ----------------- | --------------------------------------------------------------------------- |
| `--mode`          | `benchmark` (default) or `critic_ablation`                                 |
| `--ablation`      | Shortcut to run ablation mode (`critic_ablation`)                          |
| `--ablation-critics` | Optional override for critics to compare; default is `standard_mlp` |
| `--ablation-lr`   | Learning rate for auxiliary critics                                         |
| `--ablation-hidden-dims` | Comma-separated hidden dimensions for auxiliary critics (`--ablation-hidded-dims` also accepted) |
| `--ablation-bins` | Histogram bins used for MI/KL/JS distribution metrics                       |

In `critic_ablation` mode, the run writes `critic_ablation_metrics.csv` with checkpointed metrics for each auxiliary critic.
It tracks value-quality metrics and reward-model metrics (real environment reward vs critic-implied reward), including:
`advantage_mean`, `explained_variance`, `pearson`, `spearman`, `mutual_information`, `kl`, `js_normalized`, `mse`, `td_error_mean`, `real_reward_mean`, `pred_reward_mean`, `reward_explained_variance`, `reward_pearson`, `reward_spearman`, `reward_mutual_information`, `reward_kl`, `reward_js_normalized`, `reward_mse`, `reward_error_mean`.
If you add a new critic to the critic registry, pass it in `--ablation-critics` to compare it against `standard_mlp`.

Checkpoints are:

* Uniformly distributed
* Include episode 0 and final episode
* The only points where metrics, models, and videos are saved

### Auxiliary Models (opt-in)

Optional learned reward `r(s,a)` and next-state `s'(s,a)` models, trained alongside RL on its existing batch and **logged, not folded into the RL loss**. Off by default; composes with any algorithm/regime. When enabled, the run writes a separate `aux_metrics.csv` (`episode, algorithm, environment, model, train_loss, mse, mae`); the frozen train/eval schema is unchanged.

| Flag                 | Description                                              |Default |
| -------------------- | -------------------------------------------------------- |------- |
| `--aux-models`       | Train the auxiliary reward + next-state models           |off     |
| `--aux-lr`           | Learning rate for the auxiliary models                   |3e-4    |
| `--aux-hidden-dims`  | Comma-separated hidden layer sizes for the aux models    |64,64   |

---

### Device & Determinism

The device is **auto-detected** (CUDA if available, else CPU) via a single `detect_device()` — there is no device flag.

| Flag              | Description                           |
| ----------------- | ------------------------------------- |
| `--deterministic` | Enable deterministic PyTorch behavior (bitwise-reproducible runs) |

Default:

* Auto-detect CUDA
* Determinism disabled unless specified

---

## Reproducibility

Reproducibility configs live in: ```reproducibility/```

Run:

```bash
uv run python main.py --reproduce <config_name>
```

This will:

* Load `reproducibility/<config_name>.yaml`
* Override CLI parameters
* Save the effective configuration in the run folder

Precedence:

1. `--reproduce`
2. `--env-set`
3. `--envs`

---

# Plotting and Tables (`plot.py`)

Generate publication-ready plots and LaTeX tables:

```bash
python plot.py --run benchmark_YYYYMMDD_HHMMSS --split eval --x-axis frames --aggregation iqm
```

```bash
python plot.py --run <run_name> --split critic --x-axis episodes
```
---

## Plot Parameters

| Flag            | Description                    |
| --------------- | ------------------------------ |
| `--run`         | Run folder name inside `runs/` |
| `--split`       | `train`, `eval`, `critic`, `both`, or `all` |
| `--x-axis`      | `episodes` or `frames`         |
| `--aggregation` | `mean` or `iqm`                |
| `--formats`     | Output formats (e.g. png pdf)  |

Notes:
* `--split both` keeps the original behavior (`train` + `eval`), and if the run is in `critic_ablation` mode it also plots critic metrics automatically.
* `--split critic` plots only `critic_ablation_metrics.csv`.
* `--split all` plots `train`, `eval`, and `critic` when available.
* If `critic_ablation_metrics.csv` is missing, `--split critic` falls back to `train_metrics.csv` and plots a single critic line named `standard`.

---

## Aggregation Modes

### Mean Mode

* Center: Mean
* Spread: Standard deviation

### IQM Mode (default)

* Center: Interquartile Mean
* Spread: IQR-STD

---

## Generated Outputs

```
outputs/<run_name>/
    plots/
    tables/
```

Each metric produces:

* Per-environment plots
* Overall aggregated plot
* One LaTeX table per metric

Figures:

* High-resolution PNG (≥300 dpi)
* Vector PDF
* Consistent algorithm color mapping

---

# Run Artifacts

Each execution creates:```runs/benchmark_<datetime>/```

Contains:

* `config.yaml`
* `metadata.json`
* `train_metrics.csv`
* `eval_metrics.csv`
* `checkpoints/<env>_<algo>_seed<seed>/`
* `videos/<env>_<algo>_seed<seed>_ckptXXXX.mp4`

CSV files include explicit `algorithm` and `environment` columns.

---

# Design Philosophy

This framework prioritizes:

* Research reproducibility
* Modular structure
* Clean abstraction boundaries
* Minimal code duplication
* Extensibility toward causal RL

It is designed as long-term research infrastructure, not a one-off script.

---

# Contributing

Contributions are welcome.

You can extend:

* Algorithms (inherit from base classes in `rl/`)
* Environment wrappers
* Environment sets
* Evaluation metrics
* MARL components
* Causal critic modules

---

## Development Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management
(PEP 621 `pyproject.toml` + `uv.lock`). Install uv via the official standalone
installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create the environment and install the locked dependencies:

```bash
uv sync                       # base runtime + dev tooling, into ./.venv
uv run pre-commit install
```

The PyTorch CUDA 13.0 wheel index is pinned in `pyproject.toml`, so `uv sync`
resolves the `torch==2.10.0+cu130` build (CUDA) rather than the CPU wheel.

Optional environment families are exposed as extras:

```bash
uv sync --extra mujoco        # also: atari, minigrid, robotics, minari, offline
```

Common workflows:

```bash
uv run python main.py --envs CartPole-v1 --algos ppo   # run inside the env
uv add <package>                                       # add a dependency
uv lock                                                # re-resolve the lockfile
```

Run formatting checks:

```bash
uv run pre-commit run --all-files
```

All pull requests should pass pre-commit checks before merging.

> **pip fallback.** `requirements.txt` is a generated export of the base
> runtime (`uv export --no-dev --no-hashes --emit-index-url`) for environments
> without uv (e.g. the deployment server). Regenerate it after changing
> dependencies; do not edit it by hand.

---

# Published Works

This framework supports experiments presented in:

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

Reproduce with:```bash python main.py --reproduce comoreai26```
