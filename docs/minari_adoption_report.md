# Minari adoption report

*2026-08-11 — status of [Minari](https://minari.farama.org/) in `benchmarking_causal_rl`, and what more of it we could use.*

## TL;DR

Minari **0.5.3** is already the sole dataset backbone of this repo: every offline
dataset is written with `minari.create_dataset_from_buffers` (HDF5 backend) and read
with `minari.load_dataset`. There is no npz/custom writer anywhere. The open question
is not *whether* to adopt Minari but *which additional Minari features* to adopt.
The three worthwhile ones are:

1. **Hosted (remote) datasets** as extra classical cells — the loader already supports
   `download=True`, and the hosted *tier* axis (random/medium/expert-style) gives a
   legitimate behavior-quality dimension. But no hosted dataset exists for our current
   envs (no classic control on the remote), so this means new hosted-env cells, not
   dataset swaps; see §2.1 for the MDP/POMDP compatibility analysis (BabyAI is the
   hosted-POMDP option).
2. **Publishing our generated datasets** to the HuggingFace remote for paper
   reproducibility — one CLI command per dataset, no code changes.
3. **Namespace metadata** (`namespace_metadata.json`) to document sweep provenance
   in-store.

Not worth adopting: `DataCollector` (pulls jax; we deliberately avoid it), the Arrow
backend (would break byte-frozen goldens), and a version bump past 0.5.3 (pin is
load-bearing).

---

## 1. Where we already use Minari

| Concern | How | Where |
|---|---|---|
| **Write path** | `EpisodeBuffer` per episode → `create_dataset_from_buffers` → `storage.update_metadata(signature)` | `src/envs/offline/generate.py:202,769-774,800` |
| **Read path** | local-first `list_local_datasets()` → `load_dataset(id, download=True)` fallback | `src/envs/offline/minari_loader.py:9-30` |
| **Buffer fills** | flat 5-key fill + episode-grouped twin (deliberately duplicated to keep flat bytes frozen) | `minari_loader.py:61-137,140-208` |
| **Custom per-step data** | `infos["confounder_u"]`, `infos["intervened"]`, `infos["coverage_min"]` — written only when applicable so clean datasets stay byte-identical | `generate.py:292-306` |
| **Custom metadata** | confounding signature (4-key additive / 14-key action-dependent) + `generator_checkpoint_hash`, enforced at load by `enforce_confounding_gate` | `generate.py:358-514,796-800` |
| **Id scheme** | `generated/{env}/{tier}[-policy][-sigmaNNN]-v0` (tools) and `{prefix}/{regime}/{env}-beta_..._sigma_...-seedN-v0` (sweeps); ids can't contain dots | `generate.py:100-128`, `regime_sweep.py:556-563` |
| **Store isolation** | per-worker `MINARI_DATASETS_PATH` stores, refcounted + `rmtree` on last release; kills the `namespace_metadata.json` TOCTOU | `sweep_supervisor.py:563-627` |
| **Pinning** | `minari[hdf5]==0.5.3` (extra `minari`), `minari[hdf5,hf]==0.5.3` (extra `offline`); **not** in `requirements.txt` — strictly optional, every test `importorskip`s | `pyproject.toml:47-53` |

Datasets live in `~/.minari/datasets/` (or the per-worker store under the supervisor)
and are never committed.

## 2. What Minari offers that we don't use yet

### 2.1 Hosted datasets and their environments — **recommended, but as new cells, not drop-ins**

The Farama remote (HuggingFace-backed since 0.5.x; `[hf]` extra already in our
`offline` extra) hosts 100+ datasets. Our loader already handles them — the
network-gated test in `tests/test_offline_load_path.py` downloads
`mujoco/invertedpendulum/expert-v0` today — and a hosted id can be passed via
`--offline-dataset` / the per-env `{env_id: dataset_id}` map (`main.py:88-117`).
Discovery: `minari list remote` / `minari.list_remote_datasets()`; the eval env is
recoverable with `ds.recover_environment(eval_env=True)`.

**The zero-overlap fact first:** Minari hosts *no classic-control datasets*. Our
offline regimes run CartPole-v1 and Acrobot-v1 (`_base` env list, `discrete_only:
true`), and neither exists on the remote. So "use hosted datasets" never means
"swap the dataset under an existing sweep point" — it means adding hosted-env cells
alongside the current ones.

#### What's on the remote, against our regime requirements

| Family | Envs | Behavior tiers | Spaces | Fit |
|---|---|---|---|---|
| **MuJoCo** | 10 locomotion/control envs (Hopper, HalfCheetah, Walker2d, Ant, Humanoid, InvertedPendulum, …) | `expert` / `medium` (+ `simple` on 5 envs) | continuous Box obs+act | offline **MDP**, continuous algos only |
| **D4RL ports** | AntMaze, PointMaze, Adroit (pen/door/hammer/relocate), Kitchen | `human`/`expert`/`cloned` (Adroit), `partial`/`complete`/`mixed` (Kitchen), `play`/`diverse` (AntMaze) | continuous | offline MDP, continuous; goal-conditioned quirks |
| **Atari** | 51 games | `expert` only | image obs, discrete act | offline MDP (image); **no tier axis** |
| **MiniGrid / BabyAI** | 70+ BabyAI tasks + D4RL fourrooms | `optimal` / `optimal-fullobs` (+ `random` on fourrooms) | Dict(image 7×7 egocentric view, mission text), discrete act | the only **inherently POMDP** hosted data |

**Offline MDP.** The natural candidates are the MuJoCo sets — but they are
continuous, and both offline regime YAMLs pin `discrete_only: true` with the
discrete learner list (`offline_dqn`, `bcq`, `cql`, `iql`). The repo *does* have a
continuous offline path (`tools/make_pendulum_offline.py`, `tests/test_bcq_continuous.py`),
so a hosted-MuJoCo classical cell is feasible, just a new cell with the continuous
algo variants — not a config-only change. The discrete hosted options are Atari
(image pipeline exists; expert tier only) and D4RL fourrooms.

**Offline POMDP.** Our POMDP axis is *load-time observation masking* —
`mask_indices` does an `np.delete` on the obs vector in both fills
(`minari_loader.py:105-106,175-176`) and the runner masks the eval env with the
same indices (`runner.py:378-385`). Two consequences:

1. *Any flat-Box hosted dataset can be POMDP-ified by us*: e.g. mask the velocity
   dimensions of `mujoco/hopper/medium-v0` exactly as we mask CartPole's — the
   dataset is untouched, the axis stays ours. The catch: our only recurrent offline
   learner (`offline_dqn_recurrent`) is discrete, so a masked-MuJoCo cell today
   would have the memoryless baseline but no memory row. A masked hosted cell is
   only worth it after a continuous recurrent variant exists.
2. *BabyAI is the genuinely hosted POMDP option*: the standard datasets record the
   7×7 egocentric partial view, and the same tasks ship an `optimal-fullobs` twin —
   i.e. **the observability axis is provided as a dataset pair with the behavior
   policy held fixed**, which is exactly the comparison our offline_pomdp cell
   makes by masking. Cost: Dict observations (image + mission text) need an
   adapter in the fill (our buffers expect array obs; likely: take the image key,
   drop/embed the mission) and MiniGrid's `MissionSpace` needs Minari's custom
   space deserialization hook. The repo's MiniGrid image stack covers the env
   side; the loader side is the new work.

#### The tier axis as the behavior-policy dimension — yes, with a caveat

The user-controlled axes of our sweeps (β bias strength, σ confounding) cannot be
reproduced on hosted data: the behavior policy is whatever was recorded. But the
hosted **tier** axis (`simple`/`medium`/`expert`, `human`/`cloned`/`expert`,
`mixed`/`partial`/`complete`) is a legitimate coarse behavior-quality dimension —
and it maps directly onto the tier concept our own generator already implements
(`select_tier_episode`, `generate.py:32`: random/medium/expert via checkpoint
selection). So a hosted cell's "sweep" is: fixed env, tier ∈ {simple, medium,
expert}, algos × training-seeds. Additionally, `minari.combine_datasets` can build
the classic D4RL-style *mixtures* (e.g. medium+expert) as new local ids, extending
the axis without touching the loader.

The caveat — what the tier axis is *not*: our β arm varies bias strength around
**one shared generator checkpoint**, making cross-arm deltas paired
(`assert_shared_generator` refuses mixed hashes). Hosted tiers are different
policies from different training runs — unpaired, uncontrolled spacing. Fine for
"algo ranking vs data quality" (the standard D4RL protocol), not a substitute for
the L-sweep's paired arms. Also: one hosted dataset per tier means the seed axis
collapses to *training* seeds only (no per-seed dataset regeneration) — standard
in the offline-RL literature, but a different variance story than our sweeps, and
it should be labeled as such in any report.

**What hosted data can never do here:** anything causal. No
`infos["confounder_u"]`, no confounding signature, no `generator_checkpoint_hash`
— the σ arm, the strategy-critic ablation, and null-calibration (whose `noise_ref`
is measured per (env, algo) on *our* pipeline) all structurally require our
generator. Hosted cells are classical-simulation-only.

#### Spike results (2026-08-11): `D4RL/minigrid/fourrooms-v0` end-to-end

Run against the hosted fourrooms dataset (590 episodes / 10,010 steps, near-expert
behavior, mean return 0.847; a `fourrooms-random-v0` twin exists, so this env has a
two-point tier axis). Script: `tools/fourrooms_minari_spike.py` (run with an
absolute `MINARI_DATASETS_PATH`; downloads ~a few MB from the HF remote). Findings:

- **Gate: no carve-out needed after all.** `enforce_confounding_gate` is only
  called when `_value_trace_gate_open` is true, and that opens only for
  `behavior_policy in ("bias_confounded", "bias_confounded_action")`
  (`runner.py:277-281,1358`). A hosted classical run (default `"agent"`) never
  reaches the gate — signature-free metadata is structurally fine.
- **Space check passes untouched**: `assert_dataset_matches_algo` correctly reads
  the hosted dataset's own `action_space` (`Discrete(7)` → discrete → pass).
- **The one real break is Dict observations**: the hosted obs space is
  `Dict(direction, image 7×7×3 uint8 symbolic, mission Text)`, and
  `fill_replay_buffer_from_minari` dies at the tensor conversion
  (`minari_loader.py:107`, object-dtype array). This is the single load-path
  adapter needed for MiniGrid-family hosted data.
- **With the adapter, the chain works and learns.** Flattening the symbolic image
  to a 147-dim vector (matching `ImgObsWrapper`-wrapped eval env) and using the
  real `ReplayBuffer` + registry builders + `learn`/`update` (batch 128, 8k grad
  steps): **CQL reaches mean return 0.332 with 48% goal-success vs the random
  baseline's 0.019 / 4%** (behavior policy: 0.847). Naive `offline_dqn` scores
  0.00 — the textbook no-conservatism offline failure on narrow near-expert data,
  i.e. exactly the algorithm ordering the offline literature predicts, which is
  itself evidence the pipeline is faithful. The gap to 0.847 is expected: a
  memoryless MLP on a 7×7 egocentric view of a randomized-goal maze — closing it
  is the recurrent/POMDP story, not a bug.
- Note our MiniGrid *env* wrapper (`make_minigrid_env`, 84×84 RGB partial render)
  is a different obs encoding than the hosted symbolic 7×7 — a hosted-MiniGrid
  cell should evaluate on a symbolic-obs env build (flattened `ImgObsWrapper`),
  not the RGB pipeline, so train and eval distributions match.

#### BabyAI sweep investigation (2026-08-11): pipeline correct, two real findings

The first hosted_babyai sweep read "fullobs much worse than partial, memory
never helps" — suspicious enough to audit end-to-end. Verdict:

- **Data/encoding/adapter all correct.** Both arms' behavior policies are
  statistically identical (return 0.929 vs 0.930, 100% success, ~5-step
  episodes); the fullobs grid encoding matches today's `FullyObsWrapper`
  channel-for-channel (agent marker included); and CQL trained manually on the
  adapter-filled buffers **nearly matches the behavior policy on BOTH arms**
  (partial 0.921/98%, fullobs 0.889/96%) — fullobs is not intrinsically harder.
- **Real bug (ours): recurrent eval amnesia.** `runner.evaluate` never threaded
  the hidden state — `act(obs, state=None)` re-zeroed the LSTM every step, so
  every recurrent row in the repo was evaluated memorylessly (~20% relative
  return penalty measured at a competent BabyAI checkpoint: 0.346 vs 0.283).
  Value-estimation gates were unaffected (they never used eval return). FIXED:
  state now threads across eval steps and zeroes at episode boundaries;
  non-recurrent paths byte-identical.
- **Not a bug: the sweep's low numbers are the naive-DQN collapse.** The BabyAI
  algo rows are the DQN family (the only base with a recurrent variant); on
  narrow near-expert data naive DQN collapses (same as FourRooms), and the
  recurrent variant additionally DEGRADES with training (best @2k of 20k
  steps). A conservative memoryless reference row (cql) makes the family
  readable; recurrent-CQL remains the deferred workstream it always was.

#### Integration checklist — DONE 2026-08-11 (hosted cells are live, as SWEEPS)

All items landed. Hosted data runs as **behavior-policy sweeps** — one family
config whose `datasets:` block names the arms (the hosted analog of the
classical L-sweep's arm axis), run by the dedicated `hosted_sweep` driver
(`main.py --reproduce` dispatches on the `datasets:` key, before the
regime/sweep dispatch). Leaves mirror the classical tree
(`results/{regime}/{simulation}/{arm}/{env}/{algo}/{seed}/`, resumable), and
`render_hosted_report` produces the cross-arm comparison (grouped bars per env:
arm axis × algo, mean ± sd over seeds, plus aggregate/summary CSVs).

| Family config | Arms (the swept axis) |
|---|---|
| `offline_mdp/hosted_minigrid.yaml` | FourRooms: `random` → `near_expert` tier |
| `offline_mdp/hosted_mujoco.yaml` | Hopper-v5 + HalfCheetah-v5: `simple` → `medium` → `expert` (continuous algos) |
| `offline_pomdp/hosted_babyai.yaml` | GoToRedBallNoDists: `partial` vs `fullobs` (observability axis; full 2×2 with recurrent + memoryless on both arms, per-arm `env_kwargs` switch the eval encoding) |

Run: `uv run python main.py --reproduce rl_regimes/<cell>/<family>.yaml`
(seeds come from the file — training seeds over the fixed dataset per arm).
Report: `uv run python -m src.benchmarking.render_hosted_report <regime>
--simulation <family>`. Implementation pieces:

- **Gate**: no carve-out needed (classical runs never call it — see spike).
- **Dict-obs adapter**: `src/envs/offline/hosted_dict_obs.py` (flat + sequence
  twins, frozen fills untouched); the runner dispatches on the dataset's
  `observation_space` being a Dict, and raises clearly if `mask_indices` or an
  oracle-U variant is combined with hosted data. Handles minari 0.5.x's
  small-image JPEG write asymmetry (hosted MiniGrid data is stored raw by
  0.4.x/0.5.1; the decode branch serves locally written fixtures).
- **Eval encoding**: `env_wrapper: minigrid_symbolic`
  (`make_minigrid_symbolic_env`: ImgObsWrapper [+FullyObsWrapper via
  `env_kwargs.full_obs`] → FlattenObservation), matching the fill elementwise —
  147-dim partial / 192-dim fullobs verified against the real datasets.
- **Budget**: `offline_grad_steps` is now readable from flat reproduce YAMLs
  (previously sweep-only; legacy configs byte-unchanged).
- **Validated**: 9 new tests (`tests/test_hosted_dict_obs.py`), offline/sweep
  regression green, and real-data smokes of all three families end-to-end
  through `main.py --reproduce` (fourrooms, BabyAI partial incl. the recurrent
  grouped path, Hopper simple continuous).
- Space check via `assert_dataset_matches_algo` is already in place; wrapper
  parity needs a one-time audit per family (Atari frame-stack/resize vs
  `envs/wrappers/atari.py`; MiniGrid Dict-obs adapter as above).
- The env behind the dataset must be installed and registered (mujoco / minigrid
  extras) so the runner can build the eval env; prefer constructing it from the
  dataset's own `env_spec` / `recover_environment(eval_env=True)` to avoid
  version skew with the recorded data.
- Hosted ids should be pinned (id + `minari_version` + dataset checksum if we care)
  in the cell YAML — the remote is external state; a re-uploaded dataset would
  silently change results.

### 2.2 Publishing our datasets — **recommended for the paper**

Minari 0.5.x supports uploading to a HuggingFace remote (`minari upload <id> --remote
hf://<org>`). Publishing the 56 Cell-3/Cell-7 sweep datasets (or at least the
headline σ-sweep ones) would let readers reproduce the offline results without
running the generator (~hours of GPU). Everything needed travels with the dataset:
the confounding signature, σ, and `generator_checkpoint_hash` are ordinary Minari
metadata, and `confounder_u` rides in `infos`. The `author`/`code_permalink`/
`algorithm_name` fields (we already set `algorithm_name`) slot into Minari's standard
metadata schema. No code changes — a small `tools/upload_datasets.sh` over the id
list would do it.

One decision to make first: hosted datasets are public; the datasets embed the true
confounder U in `infos`, which is *the point* (readers can verify the oracle
ceiling), but the README for the namespace should say clearly that reported methods
must not read it (the five-keys rule).

### 2.3 Namespace metadata — nice-to-have

`namespace_metadata.json` attaches arbitrary JSON to a namespace (e.g.
`sweep/offline_mdp/`). We could stamp the sweep grid, k-pin (k=2.4), and commit hash
there so a store is self-describing. Low value while stores are ephemeral
(supervisor `rmtree`s them), real value if we publish (§2.2). Note this is exactly
the file whose concurrent-write TOCTOU the per-worker stores were built to avoid —
write it once, post-sweep, from the parent process only.

### 2.4 Utility APIs we could use opportunistically

- `minari.combine_datasets([...], new_id)` — e.g. merge per-seed datasets into one
  mixed-behavior dataset if a future cell wants that; keeps metadata lineage.
- `dataset.filter_episodes(fn)` / `minari.split_dataset(ds, sizes)` — train/val
  episode splits without touching our loader; could back an offline model-selection
  ablation.
- `dataset.set_seed()` + `sample_episodes(n)` — Minari-native episode sampling; we
  don't need it (our buffers own sampling), listed for completeness.

## 3. What we should *not* adopt

- **`DataCollector`** — deliberately avoided (`tools/make_cartpole_offline.py:11-12`):
  its per-step tree flattening pulls jax into the env step path. Our
  `EpisodeBuffer`-direct write path is simpler and byte-controlled. Keep it.
- **Arrow backend** — 0.5.x supports `data_format="arrow"`. Faster columnar reads,
  but the HDF5 bytes are what the golden/bitwise story is frozen against, and
  `[hdf5]` is the pinned extra. Switching buys nothing we need and costs golden
  re-freezes.
- **Version bump** — the `==0.5.3` pin is load-bearing: the probe orchestrators
  document a 0.5.3 `get_size()` bug (relative `MINARI_DATASETS_PATH` doubles the
  path → `FileNotFoundError`; absolute paths only), and any upgrade would have to
  re-verify byte-identity of the write path plus that workaround. Only bump with a
  golden re-run, and only for a concrete fix we need.
- **Dataset versioning via `-vN` bumps** — Minari supports version suffixes; our
  regeneration model is delete-then-recreate at `-v0` with determinism guaranteed by
  per-point re-seeding. Introducing version bumps would fragment ids across the
  results tree for no benefit. (If we publish, published ids are immutable anyway —
  publish once, from a tagged commit.)

## 4. Suggested next steps (in order of value/effort)

1. ~~**Hosted-baseline spike**~~ — **DONE 2026-08-11** (`D4RL/minigrid/fourrooms-v0`;
   spike results in §2.1).
2. ~~**Hosted cells (adapter + configs)**~~ — **DONE 2026-08-11**: Dict-obs adapter
   productionized, `minigrid_symbolic` wrapper, `offline_grad_steps` in flat
   configs, and seven hosted configs across the three viable families (see the
   integration checklist in §2.1). Deliberately deferred: **Atari** (raw
   210×160×3 frames need a grayscale+resize+stack adapter, expert tier only — no
   quality axis to buy), and Kitchen/Adroit/AntMaze (goal-conditioned quirks).
3. **Publish decision**: pick the dataset subset + HF org for the paper, write
   `tools/upload_datasets.sh`, add a namespace README documenting the five-keys rule
   and the signature schema. (~small, blocked on the org decision)
4. **Namespace metadata stamp** post-sweep, parent-process only. (~tiny, do with 3)
