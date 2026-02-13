# Benchmarking Causal Reinforcement Learning

A modular, PyTorch-first benchmarking framework for single-agent reinforcement learning — designed for reproducibility, extensibility, and future causal augmentation.

This repository provides a clean architectural scaffold to evaluate standard RL algorithms under a unified, vectorized, and research-oriented pipeline, with built-in support for reproducible experiments and structured evaluation.

---

## Why This Framework?

Most RL benchmarking codebases grow organically and become difficult to extend, reproduce, or adapt for research-level experimentation.

This project enforces:

* Strict actor–critic separation
* Single-device propagation (CUDA-first, CPU fallback)
* Vectorized environments as a first-class abstraction
* Deterministic experiment reproducibility
* Structured checkpointing and evaluation
* Clean modular extension for causal components

The goal is not just to run RL — but to *benchmark it rigorously*.

---

## Core Features

### Architecture

* PyTorch-first implementation (all tensors, buffers, and networks live on a single detected device).
* Vectorized Gymnasium environments.
* Automatic flattening of arbitrary observation spaces (Box, Dict, Tuple, MultiDiscrete, nested).
* Modular environment wrapper system.
* Clear separation between:

  * `rl/` (single-agent)
  * `marl/` (future multi-agent)
  * `benchmarking/` (runner, evaluation, registry)

### Algorithms (v0)

* PPO
* TRPO (KL-penalized trust region variant)
* A2C
* Vanilla Policy Gradient
* DQN (discrete only)
* DDPG (continuous only)

### Benchmarking Capabilities

* Multi-algorithm runs via `--algos`
* Multi-environment runs via `--envs`
* Named environment groups via `--env-set`
* Checkpoint-based logging (not per episode)
* Evaluation video recording (first eval seed only, one per checkpoint)
* CSV-only structured logging
* Unified run folder per benchmark execution

### Device & Reproducibility

* Auto CUDA detection
* Deterministic mode optional
* Fully reproducible runs via YAML configuration

---

## Usage

### Single algorithm

```bash
python main.py --envs CartPole-v1 --algos ppo
```

### Multiple algorithms and environments

```bash
python main.py --envs CartPole-v1 HalfCheetah-v5 --algos ppo trpo
```

### Named environment sets (overrides `--envs`)

```bash
python main.py --env-set gymnasium --algos ppo a2c
```

---

## Reproducibility

Reproducibility configs are stored in: ```reproducibility/```

To reproduce a published experiment:

```bash
python main.py --reproduce <filename>
```

This will:

* Load `reproducibility/<filename>.yaml`
* Override CLI parameters
* Run the benchmark exactly as specified
* Store the effective configuration in the run folder
* Respect deterministic settings if enabled

### Precedence Order

1. `--reproduce`
2. `--env-set`
3. `--envs`

---

## Plotting and Tables

Generate publication-ready plots and LaTeX tables from a completed run:

```bash
python plot.py --run benchmark_YYYYMMDD_HHMMSS --split eval --x-axis frames --aggregation iqm
```

Outputs are written to `outputs/<run_name>/` with per-env and overall plots (PNG/PDF) and per-metric LaTeX tables.

---

## Run Artifacts

Each benchmark execution creates:

```
runs/benchmark_<datetime>/
```

Contents:

* `config.yaml`
* `metadata.json`
* `train_metrics.csv` (checkpoint-only rows)
* `eval_metrics.csv` (checkpoint-only rows)
* `checkpoints/<env>_<algo>_seed<seed>/`
* `videos/<env>_<algo>_seed<seed>_ckptXXXX.mp4`

All CSV files include explicit `algorithm` and `environment` columns.

---

## Design Philosophy

This framework pursues:

* Strong modularity
* Minimal code duplication
* Clean abstraction boundaries
* Strict separation of concerns
* Extensibility toward causal RL

Future work integrates structured causal critics (e.g., VBN-based modules) seamlessly into the actor–critic pipeline.

---

# Contributing

Contributions are welcome.

We encourage improvements in:

* Algorithm implementations
* Environment wrappers
* Evaluation metrics
* MARL extensions
* Performance optimization
* Causal module integration

### Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Code Quality & Pre-commit

We enforce consistent formatting and linting.

Install pre-commit:

```bash
pip install pre-commit
pre-commit install
```

Run manually:

```bash
pre-commit run --all-files
```

The configuration ensures:

* Code formatting
* Import sorting
* Style consistency
* Prevention of accidental large commits

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

Reproduce with: ```python main.py --reproduce comoreai26 ```

---
