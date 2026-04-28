
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

## Implemented Algorithms (v0)

### On-Policy

- **Vanilla Policy Gradient**
- **A2C**
- **PPO**
- **TRPO** (KL-penalized trust-region variant)

### Off-Policy

- **DQN** — supports **discrete action spaces only**
- **DDPG** — supports **continuous action spaces only**

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
python main.py --env-wrapper custom --env-entry-point my_pkg.envs:make_env --envs CustomEnv-v0 --algos ppo

# Mixed envs: prefix only the custom ones
python main.py --envs CartPole-v1 custom:my_pkg.envs:make_env --algos ppo
```

You can pass JSON kwargs to the entry point with `--env-kwargs`.

# Main Training Script (`main.py`)

## Basic Usage

Single algorithm:

```bash
python main.py --envs CartPole-v1 --algos ppo
````

Multiple algorithms and environments:

```bash
python main.py --envs CartPole-v1 HalfCheetah-v5 --algos ppo trpo
```

Named environment set (overrides `--envs`):

```bash
python main.py --env-set gymnasium --algos ppo a2c
```

Critic ablation mode (same actor rollout, multiple auxiliary critics in parallel):

```bash
python main.py --ablation --envs CartPole-v1 --algos ppo a2c --ablation-lr 3e-4 --ablation-hidden-dims 64,64 --ablation-bins 32
```

Compare baseline + custom critic example:

```bash
python main.py --ablation --envs CartPole-v1 --algos ppo --ablation-critics standard_mlp residual_reward_model
```

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

---

### Device & Determinism

| Flag              | Description                           |
| ----------------- | ------------------------------------- |
| `--device`        | `auto`, `cpu`, or `cuda`              |
| `--deterministic` | Enable deterministic PyTorch behavior |

Default:

* Auto-detect CUDA
* Determinism disabled unless specified

---

## Reproducibility

Reproducibility configs live in: ```reproducibility/```

Run:

```bash
python main.py --reproduce <config_name>
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

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pre-commit install
```

Run formatting checks:

```bash
pre-commit run --all-files
```

All pull requests should pass pre-commit checks before merging.

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
