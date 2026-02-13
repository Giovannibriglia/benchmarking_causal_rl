
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

---

## CLI Parameters

### Environment Selection

| Flag        | Description                                      |
| ----------- | ------------------------------------------------ |
| `--envs`    | List of environment IDs                          |
| `--env-set` | Named group of environments (overrides `--envs`) |

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

---

## Plot Parameters

| Flag            | Description                    |
| --------------- | ------------------------------ |
| `--run`         | Run folder name inside `runs/` |
| `--split`       | `train`, `eval`, or `both`     |
| `--x-axis`      | `episodes` or `frames`         |
| `--aggregation` | `mean` or `iqm`                |
| `--formats`     | Output formats (e.g. png pdf)  |

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
