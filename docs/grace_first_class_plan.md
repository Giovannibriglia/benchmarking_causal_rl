# GRACE as a first-class citizen — Phase 1 plan (2026-09-02)

Investigation and proposal only; no production code changed. The pilot campaign
(`tools/run_e1.py`, restarted 2026-09-02) keeps running and finishes as the
pilot. Everything below runs after it.

---

## 1a — the transform is algorithm-independent: VERIFIED, and the cache it licenses

### What was verified, and how far each check reaches

**Bitwise comparison of every recorded fit output, all 7 completed (cell,
dataset-seed) pairs** (`e1_provenance.json`, floats compared by `repr` — full
float64 precision, not display rounding):

| cell | seed | cql vs iql |
|---|---|---|
| danull | 0, 1, 2 | identical to the last bit, every field |
| d100s0 | 0, 2 | identical to the last bit, every field |
| d100s0 | 3 | both ABSTAIN, identical records |
| d100 | 0 | identical: `lo=0.4802500009536743`, `hi=0.5079997479915619`, `n_rewards_written=49762`, `contrast_point=0.49433374404907227` |

The strongest single witness is `contrast_point`: a float64 mean over all
49,762 substituted `r_hat` values in fixed accumulation order. Two reward
vectors differing anywhere would need an exactly cancelling difference to
produce the same 64-bit mean *and* the same interval, pessimism,
`replicate_var` and backtrack counts — those quantities are functions of the
same fit through different reductions.

**Code audit — the fit's input closure contains no algorithm.** The E1 path is
the grouped one; the buffer is runner-owned (`SequenceReplayBuffer(capacity=1e6)`
at `runner.py:1798`, same class and arguments for every algorithm), filled by
the single `fill_sequence_buffer_from_minari(dataset_id, …)` construction site,
and `_apply_grace_transform` passes only `(buffer, proxy_names, device)`. The
transform's internal seeds are fixed (`fit_seed=0`, `init_seeds=(1,2)`,
replicates at `fit_seed+1+i`); the estimator seeds itself
(`LatentClassEstimator(seed=…)`); deterministic kernels are on. The one channel
an algorithm-dependent difference could travel through is **global RNG state**
(agent construction precedes the transform and draws differently per
algorithm).

**Bitwise reward-column hash under perturbed global RNG** — the literal check
the direction asked for. Two subprocesses, same danull s0 dataset, CPU-only
(the GPU belongs to the campaign), each seeding and burning
torch/numpy/random global RNG *differently* before the fit (mimicking what a
cql vs iql agent construction leaves behind), then `sha256` over the
substituted reward column's float32 bytes:

> RESULT — see the appendix at the bottom; filled in when the runs complete.

If the hashes are equal, the transform is a pure function of
`(dataset, fit options)` and immune to the only algorithm-dependent channel;
the cache below is sound. If they differ, **stop: something reads global RNG
inside the fit and that bug outranks everything else in this plan.**

### The transform cache

**Fit once per `(dataset, fit options)`, reuse across algorithms and training
seeds.** With d-cell fits measured at ~3,600 s each and every fit currently
recomputed per algorithm per arm, the cache removes the term that makes the
campaign look expensive (see 1b costs).

**Key.** A dict, serialized canonically (sorted-keys JSON), sha256'd for the
filename — but **equality is decided on the full dict, never on the hash**: the
cache hit path loads the stored key dict and compares field-by-field; the hash
only locates the candidate. Collision is thereby impossible rather than
unlikely. Fields, with why each is in:

| field | why it must be in the key |
|---|---|
| `dataset_id` | the S6 identity — but see the next row |
| `data_sha256` — sha256 over the filled buffer's obs/actions/rewards/dones/proxy tensors, in fill order | the fingerprint bug means id ↛ content is not currently guaranteed (a regenerated dataset keeps its id). Content-addressing makes a stale or regenerated dataset a MISS, never a silent wrong hit. Costs seconds |
| `proxy_names` (ordered tuple) | changes the model's channels and the init (`proxy` vs `random`) |
| `alpha`, `b`, `fit_seed`, `init_seeds` | interval level, bootstrap width, and the determinism anchors |
| `fit_kwargs` in full (`max_iter`, `m_step_budget`, `batch_size`, and any future key — serialize the whole dict) | each changes the fit; a forgotten new kwarg must widen the key, so the WHOLE dict goes in |
| `code_version` — sha256 over the source bytes of `src/rl/offline/grace/*.py` + the vendored `nbn/` version tag + torch version | a fit is a function of the code; a synced NBN or edited estimator must invalidate. Hashing source bytes (not the git commit) catches uncommitted edits — the S12 lesson |
| `device_kind` (`cuda`/`cpu`) + deterministic flag | deterministic kernels are not bit-identical across device kinds; a CPU-fitted cache entry must not serve a CUDA run silently |

**Artifact** (`results/e1/_transform_cache/<sha>/`): `rewards.npy` (float32,
buffer order), `key.json` (the full dict above), `serving.json` (label, lo/hi,
the complete meta — the same provenance a fresh fit writes). On a hit the run's
`e1_provenance.json` records `transform_cache_hit: true` plus the cache path
and key hash (S15: the artifact says what the run DID). `apply_reward_transform`'s
existing coverage check (`n_rewards_written == n_buffer_rows`) already guards a
length mismatch.

**Abstentions are cached too** — an abstained fit at ~700–950 s is still worth
not recomputing, and a cached abstention must remain visibly an abstention in
every consuming run's provenance.

---

## 1b — two seed axes

### The split

- **dataset seed `ds`** — selects the certified dataset; one GRACE fit per
  `(cell, ds)`, shared by everything below via the cache.
- **training seed `ts`** — RL initialisation and batch order only; no new fit.

### Layout

Leaf: `.../{env}/{algo}/{arm}/ds{d}_ts{t}/` — the composite replaces the bare
seed segment; the tree depth and every other segment are unchanged, so
`iter_leaves`' glob and the beta/sigma parsing need nothing.

- `parse_results_leaf` parses the last segment with
  `^(?:(\d+)|ds(\d+)_ts(\d+))$`: a bare int (every existing leaf) sets
  `dataset_seed = training_seed = seed`; the composite sets both. The record
  gains `dataset_seed` and `training_seed` fields; `seed` stays (equal to
  `training_seed`) so existing consumers keep working.
- `results_leaf` accepts `seed: int | str`.
- `aggregate_per_seed` carries both fields; **nothing ever averages across
  `ds`** — aggregation pools over `ts` *within* a `ds` only (the D-D reporting
  constraint, extended to the new axis).
- `build_paired_report`'s match key becomes
  `(regime, β, σ, env, algo, ds, ts)` — pairing base with grace on identical
  data *and* identical training seed, which is what makes the per-pair delta a
  paired statistic.
- The pilot's existing leaves are the `ds == ts` diagonal; a one-shot migration
  renames `…/{arm}/0` → `…/{arm}/ds0_ts0` (or the parser simply accepts both
  forms forever — preferred: **no migration**, bare = diagonal, documented).

### Cost, from measured pilot numbers

Measured per-leaf wall-clock (provenance `seconds`): cql training ≈ 380 s,
iql ≈ 620 s (mean ≈ 500 s); GRACE fit ≈ 3,600 s on d-cells,
≈ 160–540 s on danull (fit time = grace leaf − matching base leaf; danull's
measured twice at both σ, agreeing within seconds). Dataset generation:
3 certified datasets in 2.6 min — generation is never the binding cost.

Per cell, target **3 ds × 5 ts** per arm, both algos, with the cache:

| component | count | time |
|---|---|---|
| fits | 3 (one per ds) | d-cells 3 × 1.0 h = **3.0 h**; danull ≈ 0.25 h |
| training runs | 2 algos × 2 arms × 15 = 60 | 30×380 s + 30×620 s = **8.3 h** |
| **per d-cell total** | | **≈ 11.3 h** |

Full campaign (danull, d100s0, d100, d025, d010asym) from scratch:
**≈ 54 h**. Reusing the pilot's diagonal (12 runs/cell already done):
**≈ 45 h**. Without the cache the same campaign would be ≈ 45 h + 15 extra
d-cell fits ≈ **+15 h**, i.e. the cache pays for itself in its first cell.

**What fits in 24 h / 48 h** (fits included, pilot diagonal reused):

- **24 h** — 3 ds × 5 ts on the two headline cells (d100s0 + d100, ≈ 22 h),
  or 3 ds × 3 ts on all five cells (≈ 27 h — slightly over; drop danull's extra
  ts to make it). Recommendation: headline cells first — P1 (no-harm) and the
  primary demonstration are what three seeds cannot support.
- **48 h** — 3 ds × 5 ts on all five cells (≈ 45 h), the full target.

**Adding dataset seeds:** each new ds costs ~1 min generation +certification,
one fit (~1 h on d-cells), and 2 algos × 2 arms × n_ts × ~500 s of training
(2.8 h at n_ts = 5) ≈ **3.8 h per (d-cell, ds)**. At 24 h of budget one can
add ≈ 6 (cell, ds) units; at 48 h ≈ 12 — e.g. 48 h buys the full 3×5 campaign
*or* 3×3 everywhere plus two extra dataset seeds on both headline cells
(5 ds × 3 ts there). The ds-vs-ts trade should follow what the pilot's variance
decomposition shows binds (dataset variance vs training variance) — decide
after the pilot completes, not now.

---

## 1c — the YAMLs drive the run

### Findings

- `load_sweep_spec` **silently drops unknown keys** — the
  `grace_reward_transform: true`, `grace_proxy_names`, `eval_confounded_reward`
  and `eval_confounded_mode` lines in `e1_*.yaml` are parsed by nothing. The
  `_grace` variants currently *document* the arm; the driver *defines* it.
- The σ=0 control cell **has no YAML at all** (`e1_d100s0*.yaml` does not
  exist); its seeds `(0, 2, 3)`, its σ override, and its rationale live only in
  `run_e1.py`'s `CELLS` tuple and a comment.
- The driver also hardcodes: the cell list, source-cell mapping, algos, envs,
  `n_eval_envs=16`, `eval_rollout_len=500`, `n_checkpoints=25`.

Two construction sites for one fact — the `c_r` / fingerprint /
`_episode_log_liks` pattern, recorded before it bites.

### Proposal

1. **SweepSpec learns the missing fields** (parsed, not tolerated):
   `grace_reward_transform: bool`, `grace_proxy_names: list`,
   `eval_confounded_reward: bool`, `eval_confounded_mode: str`,
   `eval_rollout_len: int`, `n_eval_envs: int`, `source_cell: str` (the
   generation-report cell the dataset ids resolve through), and
   `e1_cell: str` (the results-tree tag). Loader gains a **strict mode** for
   `e1_*.yaml`: an unknown top-level key raises — silence is how the current
   gap arose.
2. **One YAML pair per cell defines the two arms.** `e1_<cell>.yaml` (base) and
   `e1_<cell>_grace.yaml` (identical + the grace block) — the arm is read from
   the file, never inferred from the filename. **Add the missing
   `e1_d100s0.yaml` / `e1_d100s0_grace.yaml`** carrying σ=0, seeds `[0, 2, 3]`,
   and the s1-exclusion rationale as comments moved out of the driver.
3. **The driver shrinks to runtime**: enumerate
   `reproducibility/rl_regimes/diagrams/e1_*.yaml`, resolve certified ids from
   the generation reports named by `source_cell` (unchanged discipline: read,
   never reconstruct), keep the two pre-flight assertions, paths, device,
   resume/skip, and the staging/promote loop. `CELLS`, `SEEDS`, `ALGOS`,
   `PROXIES`, per-cell σ — all deleted in favour of the YAMLs.
4. **Test:** for every `e1_*.yaml` on disk, a dry-run resolves the full leaf
   plan — dataset ids distinct/present/stamped, `EnvConfig`/`TrainingConfig`
   constructed, arm knobs and `q1_truth` derived — **from the YAML alone**,
   with the driver's module constants monkeypatched to sentinels so any
   remaining hidden dependency fails loudly. Plus one equivalence test: the
   YAML-driven plan for the current five cells reproduces the pilot's exact
   leaf set and dataset ids.
5. The `_critics_present` renderer fix stays (already landed).

This deliberately does **not** touch `run_cell`'s two known gaps (the
`arm_generator_kwargs` splat and the cell-less `_dataset_id`) mid-campaign;
fixing those and retiring `run_e1.py` into the sweep driver proper is roadmap
item 9.

---

## 1d — plotting through the deployed tooling

### What the deployed stack covers today (after the `config.yaml` +
`_critics_present` fixes)

- `regime_report` aggregates the E1 tree; `aggregate_per_seed` keeps the seed
  axis; `build_paired_report` already pairs base/grace per seed with strict
  match verification (raises on unmatched or duplicate keys) — exactly the
  right foundation, currently reachable only from Python.
- `_REPORT_METRICS` already carries `q1_contrast_pred` / `q1_contrast_error`.
- `render_regime_report` draws the reward-sweep figure with base/grace lines.
- `read_leaf_series` (every checkpoint, not just the last) exists **with no
  consumer** — the learning-curve hook is already in the aggregator.

### Gaps against Giovanni's requirements, and where each lands

| requirement | status | extension (all in the shared stack) |
|---|---|---|
| learning curves, both arms overlaid, per seed | missing (deployed report is endpoint-only; `tools/plot_e1_seeds.py` pools seeds) | new renderer figure family consuming `read_leaf_series` per leaf: per (cell, env, algo), one panel per `ds`, base vs grace lines per `ts` — no pooling. Aggregator gains a thin `series_per_leaf()` accessor so the renderer still never walks the tree |
| final return + MC band, per seed, never averaged across ds | partial (`eval_return_mean/std` are wired; per-seed table exists) | add a final-return per-seed table + dot-with-band figure keyed by `(ds, ts)`; the ds-pooling ban enforced in the aggregator, not left to the caller |
| critic quality beside returns | partial (in the aggregate CSV) | `build_paired_report` grows a CLI (`--paired`) writing `_report/{regime}_paired.csv` with `eval_return_mean`, `q1_contrast_pred`, `q1_contrast_error` per pair, rendered as a table via `table_formatting` |
| return decomposition | missing (`eval_deployment.csv` is read by nothing deployed) | add `eval_return_base_mean`, `eval_bad_action_steps_mean` to `_METRIC_SOURCE_FILE` (additive dispatch, exactly how `arm_diagnostics` columns were added) + series panels |
| abstentions separate, never pooled | missing in deployed path (fixed only in `tools/plot_e1_seeds.py`) | `parse_results_leaf`/`aggregate_per_seed` read `e1_provenance.json` when present → `grace_abstained` flag per leaf; `aggregate_over_seeds` splits the critic label (`grace` = served only, `grace[abstained]` its own row); renderer legends carry "n=…, k abstained". The paired report pairs them but flags the column — an abstained pair is a passthrough comparison and must be readable as one |
| diagnostics per row | missing | new `_report/{regime}_grace_diagnostics.csv`: per leaf — C3 label, `lo`/`hi`, L4 kind, `pessimism_applied`, `transform_applied`, `n_rewards_written`, bootstrap failure counts + `boot_reasons`, cache hit, seconds. Pure provenance flattening, skipped for leaves without the file |

Nothing on the list argues for a parallel stack: every item is either an
additive column dispatch, a provenance read, or a renderer figure consuming the
aggregator's API. `tools/plot_e1_seeds.py` and `tools/plot_e1_partial.py`
retire once the deployed path reaches parity (kept until the parity is
demonstrated side-by-side on the pilot tree).

README gains an "E1 / GRACE" subsection: the cell YAMLs, the driver command,
and the three reporting commands (`regime_report`, `render_regime_report`,
`--paired`) with `--results-root results/e1`.

---

## 1e — the roadmap

Costs are GPU-hours (measured where possible) + build effort; ordered by
recommended sequence, blockers named. For Giovanni to re-sequence.

| # | item | cost | blocks / blocked by |
|---|---|---|---|
| 1 | **YAML-driven driver + seed split + cache** (1a–1c above) | ~2–3 days build, no new compute | blocks the seeded campaign; nothing blocks it |
| 2 | **Seeded campaign** (3 ds × 5 ts) | 22 h (headline cells) / 45 h (all five) | needs #1; blocks every claim the paper makes from E1 |
| 3 | **L4 bounds re-run under the derived stop** (`02b275f`: 4000-step safety limit, plateau tolerance) | hours–1 day GPU (12 rows; the 600-step probe was still descending at 150-step production budgets) | independent; unblocks the D-B′ coverage verdict and the CartPole instrument-value gap |
| 4 | **Fingerprint fix, 3 steps in order** (forward `n_proxies` at the store site → backfill/compat-stamp all 163 datasets → only then trust `--resume`) | ~half day | **hazard, not feature work**: until done, `--resume` deletes the store on first use. Never mid-campaign; schedule the first quiet window |
| 5 | **L2-drives-the-estimator (+ channel selection)** — the seam block's merged item; today the verdict is computed and never consulted, D-E is hand-wired | ~2–4 days build | blocks honest d_a_null serving, the general D-E mechanism, and #6 |
| 6 | **Instrument channel in the model** (the LR region is I-blind; the walk-vs-BP gap is not a clean measurement anywhere yet) | ~2–3 days build + hours of re-runs | needs #5 (verdict selects channels); CartPole-only until #8 |
| 7 | **u_card selection via the rank constraint** (catalogue: the rank statistic tests the diagram *and* selects `u_card` — one statistic, two uses; K is currently fixed at 2 everywhere) | ~2–3 days; shares permutation-null infrastructure with #9 — build once | feeds #9 |
| 8 | **Acrobot transition mechanism** (both LG and MDN ruled unusable; MDN s2 diverges outright) | design decision + new mechanism class, ~3–5 days, possibly upstream in NBN | blocks Acrobot q2 only; nothing else waits on it |
| 9 | **L5 falsification — not started, the most distinctive claim** | the big one: ~1–2 weeks build + V-D-scale compute (measured projections: 6.0 d full grid / 1.2 d pooled nulls / 0.5 d option A+B). Its design constraints are already ruled (episode granularity S1b, family-statistic nulls S3, quantile tails) | needs the stable estimator (done) and benefits from #7's shared infra; blocks the paper's headline capability |
| 10 | **Sweep driver trains diagram cells** (fix the `EnvConfig` splat + put the cell name into `_dataset_id`), then retire `run_e1.py` into `run_cell` | ~1 day + golden tests | natural successor to #1; not before the campaign ends (the bypass is the current safety) |
| 11 | **D-F / D-G (POMDP) arms** | arms themselves ~hours; but `U → S_next` means the reward-transform reduction **does not apply** — honest serving there is model-based machinery, i.e. large | recommend: declared bounds-only / out of scope for the v2 paper, per the catalogue's own D-G verdict |

**Recommended order:** 1 → 2 (the campaign runs unattended) with 3 and 4 in
its shadow; then 5 → 6 and 7 → 9 as the two build tracks; 8 and 10 slot in
around them; 11 is a scope ruling, not a work item.

---

## Appendix — 1a bitwise reward-hash control (filled when the CPU runs finish)

Two subprocesses, danull s0, CPU, deliberately different global RNG state
(`--perturb 11` vs `--perturb 999`), sha256 over the substituted reward
column's float32 bytes. On CPU the transform is slow because it is ~23 fits
(3 inits + observed + B=19 replicates), where the GPU took 160 s total.

```
run A (--perturb 11), COMPLETE:
  label: serving=Q-minus l4=interval[-0.0000,+0.0000] R=dirac[1] sep/step=0.011
  rewards sha256: 73c2a173c2232800dcb77ef8f2c4a1f75a88026a9ea316228cd7370e4ec81f99
  lo/hi: -5.960464477539061e-09 / 6.556510925292956e-08   (danull = the null
  cell; an interval indistinguishable from zero is itself a correct-behaviour
  sanity check)

run B (--perturb 999), COMPLETE (2026-09-03):
  rewards sha256: 73c2a173c2232800dcb77ef8f2c4a1f75a88026a9ea316228cd7370e4ec81f99
  lo/hi and contrast_point: bit-identical to run A.

VERDICT: EQUAL — the transform is a pure function of (dataset, fit options),
immune to global RNG state, the only algorithm-dependent channel. Together
with the 9/9 bitwise-identical production pairs (all completed cross-
algorithm (cell, dataset-seed) pairs, d100 s2 included), the cache is
licensed at the bitwise level the plan required.
```

Independent full-scale witness, available today: all **8** completed
(cell, dataset-seed) pairs (the d100 s1 pair landed after the table above was
written) are bitwise-identical across cql/iql — runs whose global RNG states
genuinely differed the way production runs differ — including
`contrast_point`, a float64 mean over the full substituted reward vector in
fixed accumulation order.
