# Dataset-generation speedup — measured bottleneck + strategy

*2026-08-13 — profiling and design memo; no code changed yet.*

## The measured bottleneck (CartPole, random-tier generator, laptop GPU)

Rollout throughput today: **71 steps/s** (policy.act 184µs + env.step 10,912µs
per step). Decomposition of the env.step cost:

| layer                          | µs/step |
|--------------------------------|--------:|
| raw `gym.make` CartPole        |      14 |
| SyncVectorEnv n=1              |      18 |
| repo `GymnasiumEnv`, CPU device|      33 |
| repo `GymnasiumEnv`, **CUDA**  | **14,845** |
| native vector env, 256 envs    | 0.62 /env-step |

**The bottleneck is one design fact: `_rollout` steps the training-oriented
`GymnasiumEnv` on the CUDA device with batch 1**, paying a ~15ms
CPU→GPU→CPU tensor round-trip (with implicit syncs) per step for an MLP whose
GPU forward at batch 256 costs the same as batch 1 (213µs vs 148µs — 177×
throughput headroom, unused). Verified end-to-end: the SAME rollout on CPU
device runs ~70× faster (100 eps: 0.4s vs 24s). At today's rate a 3000-episode
expert-tier CartPole point (~1.5M steps) costs ~6 h; on CPU ~5 min; vectorized
~seconds.

## Reproducibility framing (what "changing the bytes" costs here)

- Datasets **self-certify**: the confounding signature + gate is recomputed at
  generation and enforced at load — a regenerated dataset carries its own
  validity certificate. Byte-freeze only matters for REPRODUCING historical
  dataset ids.
- CPU-vs-CUDA forwards are not bit-equal (near-tie argmax flips): any fast path
  produces different-but-equally-valid datasets. Keep the current path behind a
  `--legacy-rollout` escape hatch for regenerating historical ids; new datasets
  use the fast path.
- `_rollout` already seeds each episode independently
  (`env.reset(seed=seed+1000+ep)`). The remaining order-dependence is the
  policy's GLOBAL torch RNG stream (epsilon draws etc.). Deriving a
  per-episode Generator from `(seed, ep)` makes generation ORDER-INDEPENDENT:
  any schedule — sequential, vectorized, multi-process, cross-seed — yields
  identical bytes. That is the right long-term contract.

## Strategies, ranked by measured value / effort

1. **S1 — Rollout on CPU (≈70×, ~an hour).** Build the rollout env and the
   collection-policy copy on CPU; generator TRAINING stays on GPU (16-env
   batches genuinely benefit). One device argument through
   `build_rollout_env` / `generate_offline_dataset` + the legacy flag.
   Side benefit: rollouts stop competing for the GPU, so the point-grain
   supervisor's workers scale with CPU cores (S3 composes for free).
2. **S2 — Vectorized generation (further 10–100×, ~a day).** N parallel env
   instances (native vector CartPole = 0.62µs/env-step; SyncVectorEnv copies
   for Acrobot), ONE batched policy forward per step, per-episode RNG streams
   (the order-independence contract above). Each vector slot works through its
   own `(seed, episode)` queue — which also means ALL SEEDS of a cell can be
   generated in a single vectorized process ("multi-env across seeds").
   Requires a vector-aware ConfoundedCollectionWrapper (per-slot U streams).
   This is the path that matters for future image-env generation
   (Atari/MiniGrid) where batched GPU forwards win.
3. **S3 — Process parallelism (exists).** The point-grain supervisor already
   fans (env, seed, point) tasks across workers; after S1 its max_workers can
   rise (no GPU contention). No new code.
4. **S4 — Cross-simulation generator cache (~2× on the training share).**
   classical and critic_ablation cells retrain the same (env, seed) generator;
   cache checkpoints keyed by (env, seed, algo, budget-hash) — the existing
   `generator_checkpoint_hash` already proves identity.

## Implementation status — S1 + S2 SHIPPED (2026-08-13)

Both landed in `generate_offline_dataset`, which now places the rollout
independently of training:

| knob | default | meaning |
|---|---|---|
| `rollout_device` | `"cpu"` | device the rollout env + collection policy run on (S1) |
| `rollout_n_envs` | 1 (library) / 16 (sweeps) | parallel rollout slots, one batched policy forward per step (S2) |
| `legacy_rollout` | `False` | restore the pre-speedup path exactly (run device, 1 slot, scalar collector) |

Surfaces: `tools/generate_offline.py` gains `--rollout-device`,
`--rollout-n-envs`, `--legacy-rollout`; the sweep driver reads the same three
keys (documented in `_base/parallel.yaml`, so every cell inherits the fast
defaults) and forwards them per point.

**Measured rollout throughput** (CartPole, random-tier generator, 600 episodes):

| path | steps/s | vs scalar-CPU |
|---|--:|--:|
| legacy (CUDA, 1 slot) | 71 | — |
| S1 scalar on CPU | 2,495 | 1.0x |
| S2 vectorized, 8 slots | 8,220 | 3.3x |
| **S2 vectorized, 16 slots** | **13,130** | **5.3x** |
| S2 vectorized, 32 slots | 11,892 | 4.8x |
| S2 vectorized, 64 slots | 10,195 | 4.1x |

16 slots is the sweet spot on CartPole: at ~10-step random-tier episodes, wider
batches spend more steps on idle/awaiting-reset slots than they win back. End to
end that is **~185x** the pre-speedup rollout.

**Production-scale check** — one real sweep-arm point (`bias_confounded`,
sigma=0.5, `pi_basic_epsilon=0.5`, `rollout_episodes=3000`, ~48k transitions),
whole-call wall clock including the Minari write:

| path | wall clock | steps/s | gate |
|---|--:|--:|:--:|
| legacy (CUDA, 1 slot) | **1,558 s** (26 min) | 30 | passed |
| S1 (CPU, 1 slot) | **30.3 s** | 1,573 | passed |
| S2 (CPU, 16 slots) | **11.6 s** | 4,170 | passed |

**51.8x (S1) and 137.2x (S2)**, and all three datasets pass the confounding gate
— the fast paths self-certify exactly as the legacy one does. Scaled to a full
`offline_mdp/classical` cell (2 envs x 3 seeds x 7 points = 42 generation
points), generation drops from **~18 hours to ~8 minutes**. The confounded
per-step readers (`action_probs`, `current_u`, `intervened`) make this point
heavier than the clean micro-benchmark, hence 4.2k vs 13k steps/s.

Correctness contract (see `tests/test_generation_speedup.py`, 7 tests): the fast
paths keep the OUTPUT contract (episode count, `T+1` observations, aligned
`infos`), stay deterministic for a fixed `(seed, n_envs)`, emit episodes in
ASSIGNMENT order (so slot races cannot reorder a dataset), pass the confounding
gate on the vectorized path, and leave the caller's training agent unmoved and
unmutated (so the stamped `generator_checkpoint_hash` is unchanged). The
existing generation/gate/sweep suites (47 tests) pass under the new defaults.

## S4 — cross-simulation reuse, RESCOPED then shipped (2026-08-14)

S4 was written up as a *generator-training* cache. Profiling killed that scope:
**every production caller builds its generator with `tier="random"`** — the
sweep (`regime_sweep` line ~1040), `tools/generate_all_datasets.sh`, and all
three probe tools — so no generator training happens anywhere in the pipeline
today and a training cache would have been dead code.

The real cross-simulation redundancy sits one level up. Dataset ids carry no
simulation component (`{prefix}/{regime}/{env}-beta_..._sigma_...-seedN-v0`), so
a regime's `classical` and `critic_ablation` cells ask for the SAME ids — and
`run_cell` used to `delete_dataset` + regenerate unconditionally. The second
simulation of every regime therefore re-generated byte-equivalent data.

**Shipped:** `generation_fingerprint()` hashes every generation-determining
input (env, algo, tier, behavior policy + strength, `c_r`, `pi_basic_epsilon`,
`a_bad`, rollout episodes, seed, the generator's parameter hash, and the rollout
MODE — device type, slot count, legacy flag) and is stamped into each dataset's
metadata. Before generating a point, `run_cell` reuses the existing dataset iff

  1. its stored fingerprint equals the one this point's inputs produce,
  2. its episode count matches the configured budget (catches an interrupted
     run, which an input-only hash cannot see), and
  3. its confounding gate did not fail.

Anything else — missing dataset, pre-S4 dataset with no fingerprint, unreadable
store — regenerates. Reuse is never the fallback. Off via `reuse_datasets:
false`.

## Measured: the generation phase of a real cell

`offline_mdp/classical`, CartPole, seed 0, all 7 L-points at the production
`rollout_episodes=3000`:

| phase | wall clock | notes |
|---|--:|---|
| legacy equivalent | **~182 min** | 7 x 1557.7 s measured per point |
| cold (S1+S2) | **87.3 s** | 4.5-24.5 s per point; confounded arms cost most |
| warm (S4 reuse) | **0.03 s** | 7/7 reused — what a regime's 2nd simulation now pays |

Generator build is 0.7 s (untrained `random` tier), confirming there is no
training to cache.

## Where the time goes now

Generation is no longer the bottleneck: a full cell's 7-point phase costs under
90 seconds cold and nothing warm. **Offline LEARNER training now dominates a
sweep** — `offline_grad_steps=50_000` per (point, algo), i.e. 4 algos x 7 points
x envs x seeds. That is the axis to attack next if sweeps need to get faster;
rollout and generation are done. S3 also needs nothing, but is worth
revisiting: with rollouts off the GPU, `max_workers` is bounded by CPU cores
rather than VRAM (the "memory-safe up to ~4" note in `_base/parallel.yaml` is
now conservative for offline cells).
