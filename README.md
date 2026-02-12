# Benchmarking Causal Reinforcement Learning

PyTorch-first benchmarking scaffold for single-agent RL with future causal augmentation.

## Key Features
- Single-device propagation (auto CUDA fallback to CPU).
- Vectorized Gymnasium environments with flattening for arbitrary observation spaces (Box, Dict, Tuple, MultiDiscrete, nested).
- Deterministic seeding optional via CLI; non-deterministic by default.
- CSV-only logging stored under `runs/benchmark_<datetime>/` with unified train/eval metrics, checkpoints, and videos.
- Required algorithms implemented: PPO, TRPO (simplified trust region), A2C, Vanilla PG, DQN (discrete), DDPG (continuous).
- Evaluation records video for the first eval env only, one video per checkpoint.
- Multi-algorithm benchmarking via `--algos` and `--envs` runs each (algo, env, seed) independently with tqdm bars labeled `{algo} - {env}`.

## Usage
Single algorithm:
```
python main.py --envs CartPole-v1 --algos ppo
```
Multiple algorithms and envs:
```
python main.py --envs CartPole-v1 HalfCheetah-v5 --algos ppo trpo dqn
```
Defaults: 8 train envs, 8 eval envs, rollout_len 1024, 100 episodes, 5 checkpoints, seed 42.
Enable deterministic mode with `--deterministic`.

## Reproducibility
Repro configs live in `reproducibility/`. Use `--reproduce <file_name>` (without extension) to load and override CLI arguments:
```
python main.py --reproduce comoreai26
```
- Loads `reproducibility/comoreai26.yaml` and applies all settings (envs, algos, seeds, etc.).
- Saves the effective configuration snapshot into the run folder.
- If `deterministic: true` is set, the run is deterministic given the specified seed.

## Run Artifacts
`runs/benchmark_<datetime>/`
- `config.yaml`
- `metadata.json`
- `train_metrics.csv` (checkpoint-only rows; includes algorithm/environment columns)
- `eval_metrics.csv` (checkpoint-only rows; includes algorithm/environment columns)
- `checkpoints/<env>_<algo>_seed<seed>/ckpt_epXXXX.pt`
- `videos/<env>_<algo>_seed<seed>_ckptXXXX.mp4`

## Notes
- DQN supports discrete action spaces only; DDPG supports continuous only.
- TRPO uses a KL-penalized surrogate (no CG) for simplicity; structure matches trust-region update.
- All tensors remain on the detected device; avoid scattered `.to(device)` by propagating once.
