# Offline learner training — profile and speedup options

*2026-08-14 — measurement memo (no code changed). Companion to
`docs/dataset_generation_speedup.md`: with generation solved (S1/S2/S4), the
offline LEARNER is what a sweep now spends its wall clock on.*

## Setup

Real production shape: CartPole `bias_confounded` dataset (48,438 transitions,
`rollout_episodes=3000`), `offline_grad_steps=50_000`, `n_checkpoints=25`,
`rollout_len=512`, `n_eval_envs=16`, batch 128, CUDA.

## Where a run's time goes

| component | cql | offline_dqn | note |
|---|--:|--:|---|
| `agent.update` | **69.7%** | **63.7%** | 1.4–1.7 ms/step |
| `buffer.sample` | **23.9%** | **28.7%** | ~0.5 ms/step |
| `evaluate` (x25) | 3.7% | 4.6% | 0.15 s per checkpoint |
| buffer fill (once) | 2.6% | 3.1% | 2.6 s |
| **projected run** | **1.7 min** | **1.5 min** | per (point, algo) |

Two corrections to prior assumptions worth recording:

* **Eval is NOT a problem** (~4%). The 15 ms/step CUDA round trip that dominated
  *generation* does not bite here: eval steps 16 envs per call (0.29 ms/step
  total, 18 us per env-step), so the fixed per-step cost is amortized.
* **The step cost is overhead, not compute.** A 4->128->2 MLP at batch 128 is
  microseconds of arithmetic; 1.4–2.5 ms/step is kernel launches, optimizer
  bookkeeping and Python dispatch.

Per-algo step cost (sample + update, 50k-step projection):

| algo | CUDA | CPU |
|---|--:|--:|
| offline_dqn | 2.7 ms → 2.2 min | 1.9 ms → **1.6 min** |
| cql | 3.3 ms → 2.8 min | 2.1 ms → **1.8 min** |
| iql | 6.2 ms → **5.1 min** | 14.9 ms → 12.4 min |
| bcq | 4.3 ms → 3.6 min | 4.4 ms → 3.6 min |

## Options, sized

**1. Tensorized replay buffer — recommended, and BYTE-IDENTICAL.**
`ReplayBuffer` is a `deque` of per-transition dicts; `sample` does
`random.sample` over the deque (O(n) deque indexing) then `torch.stack` of 128
one-element tensors per key. Breakdown of the ~0.5–1.0 ms: index 78 us, gather
110 us, **stack 277 us**, `.to(device)` 72 us — almost all Python/dispatch.
Packing each key into one contiguous CPU tensor and sampling with an index
tensor measures **1053 us → 125 us (8.4x)**, i.e. **~20–25% off every offline
run**, algorithm-independent.

The enabling fact, verified: `random.sample(range(n), k)` draws **exactly the
same elements** as today's `random.sample(deque, k)` from the same RNG state
(CPython selects indices internally either way). Measured identical elementwise.
So the rewrite can keep goldens bitwise — the batches are the same batches.
Cost: `.storage` is currently touched directly by some call sites/tests, so the
change needs a compatibility shim or those updated.

**2. Drop the per-step `.item()` sync — NOT worth it.** `metrics_cache` is only
read at checkpoints (1 step in 2,000), so the sync in `learn()` is wasted
1,999 times. But it measures **53 us/step (~3%)**, against touching every algo's
`learn()` return contract. Skip.

**3. Per-algo device choice — real, but not byte-safe.** These tiny MLPs are
overhead-bound, so CPU beats CUDA for `offline_dqn` (1.4x) and `cql` (1.7x),
while `iql` is 2.6x *slower* on CPU and `bcq` is a wash. CPU/CUDA float ops
differ, so this changes results: it can only be an opt-in knob, and it would
have to be pinned per (algo, env-family) — image envs invert the ranking.

**4. Kernel-launch reduction (torch.compile / CUDA graphs / fused Adam).** The
only lever that attacks the remaining ~65%. `torch.compile(mode="reduce-overhead")`
on the update step is the natural first probe. Changes numerics; needs its own
golden re-freeze, so it is a project, not a patch.

## Implemented (2026-08-14) and measured

**A. Tensorized `ReplayBuffer`.** Contiguous per-key CPU tensors with index
sampling, replacing the deque of per-transition dicts. Byte-identity is pinned
by `tests/test_replay_buffer_tensorized.py`, which samples the new buffer and a
verbatim copy of the old one from the same seed and asserts equal batches —
including the ring-wraparound case, where logical index 0 must remain the oldest
live transition. Storage grows geometrically (1.5x) instead of preallocating
`capacity` (1e6 rows of image observations would be gigabytes). A `storage`
property still yields per-transition dicts for the few call sites that want
them, and a new `gather(indices)` indexes rows directly — the curiosity policy
used to call `list(buf.storage)` (materializing EVERY transition as a dict) on
every training step just to read a few rows.

**B. Intra-op thread cap** (`src/config/threads.py`, wired into `main.py` and
the supervisor's per-worker env). This is the fix for the CUDA-vs-CPU anomaly:
the earlier "iql is 2.6x slower on CPU" reading was thread OVERSUBSCRIPTION, not
a property of the algorithm. Per-update cost on a 20-core host, measured in
isolated processes:

| config | offline_dqn | cql | iql | bcq |
|---|--:|--:|--:|--:|
| CUDA | 2480 | 3149 | 5307 | 3329 |
| CPU, 14 threads (torch default) | 1061 | 1455 | **14681** | 3497 |
| CPU, 4 threads | 885 | 1162 | 2979 | 2117 |
| CPU, 1 thread | **874** | **1008** | **2481** | **1731** |

So the real story is the opposite of the original reading: **every one of these
algorithms is faster on CPU than on CUDA** once the thread pool is sane, because
a 4->128->2 MLP at batch 128 is entirely dispatch-bound (tiny-op dispatch
measures ~3.3 us on both devices — the GPU has no arithmetic advantage to
exploit at this size, and adds launch and sync overhead). IQL degrades 5.9x from
oversubscription alone because it runs the most ops per update, so it pays the
most thread-pool barriers. The cap defaults to 4 (near-optimal, leaves headroom
for genuinely parallel CPU work) and is overridable with `BCRL_NUM_THREADS`.
It cannot change production numerics: training runs on CUDA, where CPU thread
count does not touch kernel results, and the CPU-side work here is indexing and
copies, which have no parallel reductions.

### Before / after, full offline run (50k steps, CUDA, same host)

| algo | before | after | speedup |
|---|--:|--:|--:|
| offline_dqn | 2.15 min | **0.94 min** | 2.3x |
| cql | 2.54 min | **1.20 min** | 2.1x |
| iql | 4.43 min | **2.31 min** | 1.9x |
| bcq | 3.35 min | **1.65 min** | 2.0x |

Per-step cost falls 2496->1064 us (dqn), 2966->1388 (cql), 5237->2711 (iql),
3942->1917 (bcq); dataset fill also improves (4.1 s -> 2.8 s) because the same
thread cap removes the oversubscription from the fill's tensor writes. Scaled to
a full `offline_mdp/classical` cell (168 runs) that is roughly **6-8 h -> 3-4 h**
of learner training.

## Remaining headroom

The step is still dispatch-bound. Running the offline learner on CPU with a
small thread pool measures ~2.5-3x faster again than CUDA for these
classic-control models — but CPU and CUDA float ops differ, so it is an opt-in
device choice, not a default, and it inverts for image trunks. Beyond that,
`torch.compile(mode="reduce-overhead")` / CUDA graphs is the remaining lever and
needs its own golden re-freeze.

## Original recommendation

Do (1) — a contained, byte-identical ~20–25% win on every offline run and every
sweep. Skip (2). Treat (3)/(4) as opt-in experiments if sweeps still need to be
faster after (1); (4) is where the remaining factor of ~2 lives.

Scale check: a full `offline_mdp/classical` cell is 7 points x 4 algos x 2 envs
x 3 seeds = 168 runs. At today's ~1.5–5 min/run that is ~6–8 h of learner
training against ~9 min of generation — so this is now the only axis that
matters for sweep wall clock.
