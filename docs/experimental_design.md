# Experimental design — causal-RL eight-cell taxonomy

## 1. Purpose

This document specifies the experimental matrix for the paper *Debugging Single-Agent Reinforcement Learning through Simple Causal Lens*. The paper introduces an eight-cell taxonomy of RL regimes indexed by three binary axes — context observability (Z observed / Z hidden), data-collection mode (online / known-offline-πb / unknown-offline-πb), and presence of unobserved confounding (no U / U present). Each cell makes a distinct identification claim. The matrix realizes one empirical instance per cell where possible, with two cells (5 and 6) handled as paper-text-only theoretical distinctions.

## 2. Design principles

- **Same envs across cells.** Three discrete envs (CartPole-v1, LunarLander-v3, Acrobot-v1) and four continuous envs (Pendulum-v1, HalfCheetah-v5, Hopper-v5, Walker2d-v5) thread every cell where they're reachable. Only the data-generating regime varies cell-to-cell.
- **One seed per YAML.** `seed = [0]`. Dispersion comes from the parallel-env axis (16 train envs, 16 eval envs), not from seed averaging. IQM ± IQR-STD is reported across the 16 parallel envs at each checkpoint.
- **No cross-cell or cross-YAML auto-aggregation.** Each YAML produces its own figure. Cross-cell comparisons, if needed for the paper, are constructed manually at writing time.
- **YAML granularity is (cell-variant, env-set).** Where env-set is discrete = {CartPole-v1, LunarLander-v3, Acrobot-v1} or continuous = {Pendulum-v1, HalfCheetah-v5, Hopper-v5, Walker2d-v5}.
- **Frozen CSV schemas remain frozen.** Cell-specific data needs land in new CSVs: `eval_per_context.csv` (Cell 2, gated on `--mask-indices`) and `offline_value_trace.csv` (Cells 7–8, gated on `--behavior-policy bias_confounded`).

## 3. Common run settings

All YAMLs pin the following:

| Setting         | Value |
| --------------- | ----- |
| n-episodes      | 250   |
| rollout-len     | 1024  |
| n-train-envs    | 16    |
| n-eval-envs     | 16    |
| n-checkpoints   | 25    |
| aggregation     | iqm   |
| deterministic   | true  |
| seed            | [0]   |

For offline runs, `n-episodes` is reinterpreted as training epochs and `rollout-len` as gradient steps per epoch.

## 4. Cell-by-cell specification

### Cell 1 — Crystal Clear Clinic

**Identity.** Z observed, online (πb = agent's own policy), U absent.

**Role.** Baseline anchor — what success looks like when the causal regime is favorable.

**YAMLs at `reproducibility/rl_regimes/cell_1/`:**

| YAML                     | algos        | data regime | behavior | mask | confounded |
| ------------------------ | ------------ | ----------- | -------- | ---- | ---------- |
| `online_discrete.yaml`   | [ppo, dqn]   | online      | agent    | none | no         |
| `online_continuous.yaml` | [sac]        | online      | agent    | none | no         |

**Measurement.** Eval-return learning curve over episodes; IQM ± IQR-STD across the 16 parallel eval envs at each checkpoint. Plotted as one subplot per env, lines per algo.

**Expected result.** Stable convergence on all seven envs. Defines the success ceiling for subsequent cells.

### Cell 2 — Invisible Gene

**Identity.** Z hidden, online, U absent.

**Role.** Shows that hiding a relevant context degrades per-context performance but not the marginal — the epistemic-POMDP signature.

**YAMLs at `reproducibility/rl_regimes/cell_2/`:**

| YAML                            | algos        | data regime | behavior | mask              | confounded |
| ------------------------------- | ------------ | ----------- | -------- | ----------------- | ---------- |
| `online_masked_discrete.yaml`   | [ppo, dqn]   | online      | agent    | per-env spec (§5) | no         |
| `online_masked_continuous.yaml` | [sac]        | online      | agent    | per-env spec (§5) | no         |

**Measurement.** Eval-return curve (IQM ± IQR-STD across 16 parallel envs) plus per-context return binned by the masked Z component, written to `eval_per_context.csv`. The bins are derived from the unwrapped env's state at eval time so they're recoverable even though the agent never sees them.

**Expected result.** Marginal eval return holds reasonably well; per-context dispersion grows substantially relative to the unwrapped env's behavior in Cell 1.

### Cell 3 — Perfect Archive

**Identity.** Z observed, known-offline-πb, U absent.

**Role.** Offline RL with backdoor adjustment when coverage holds. Tier sweep separates the coverage axis from the regime axis.

**YAMLs at `reproducibility/rl_regimes/cell_3/`:**

| YAML                              | algos                                        | data regime | behavior | tier   | mask | confounded |
| --------------------------------- | -------------------------------------------- | ----------- | -------- | ------ | ---- | ---------- |
| `offline_random_discrete.yaml`    | [offline_dqn, bcq, cql, iql]                 | B2 dataset  | agent    | random | none | no         |
| `offline_medium_discrete.yaml`    | same                                         | B2 dataset  | agent    | medium | none | no         |
| `offline_expert_discrete.yaml`    | same                                         | B2 dataset  | agent    | expert | none | no         |
| `offline_random_continuous.yaml`  | [cql_continuous, iql_continuous, bcq_continuous] | B2 dataset | agent | random | none | no         |
| `offline_medium_continuous.yaml`  | same                                         | B2 dataset  | agent    | medium | none | no         |
| `offline_expert_continuous.yaml`  | same                                         | B2 dataset  | agent    | expert | none | no         |

**Measurement.** Eval-return curve over offline training epochs; IQM ± IQR-STD across the 16 parallel eval envs.

**Expected result.** Expert tier approaches Cell 1; medium degrades modestly; random collapses, with conservative learners (e.g. offline_dqn, cql_continuous) surviving longer.

### Cell 4 — Burned Files

**Identity.** Z hidden, known-offline-πb, U absent.

**Role.** Backdoor adjustment fails because the conditioning variables are missing. Shows structural rather than statistical degradation.

**YAMLs at `reproducibility/rl_regimes/cell_4/`:**

| YAML                                     | algos                                        | tier   | mask              |
| ---------------------------------------- | -------------------------------------------- | ------ | ----------------- |
| `offline_random_masked_discrete.yaml`    | [offline_dqn, bcq, cql, iql]                 | random | per-env spec (§5) |
| `offline_medium_masked_discrete.yaml`    | same                                         | medium | per-env spec (§5) |
| `offline_expert_masked_discrete.yaml`    | same                                         | expert | per-env spec (§5) |
| `offline_random_masked_continuous.yaml`  | [cql_continuous, iql_continuous, bcq_continuous] | random | per-env spec (§5) |
| `offline_medium_masked_continuous.yaml`  | same                                         | medium | per-env spec (§5) |
| `offline_expert_masked_continuous.yaml`  | same                                         | expert | per-env spec (§5) |

**Measurement.** Eval-return curve over offline training epochs; IQM ± IQR-STD across 16 parallel eval envs. The same B2 dataset is loaded as in Cell 3 with the mask projection applied at load time — datasets are not regenerated.

**Expected result.** Each tier underperforms the matched Cell 3 tier, with a roughly constant gap across tiers (the gap is structural, not coverage-driven).

### Cell 5 — Doctor's Intuition

**Identity.** Z observed, unknown-offline-πb, U absent.

**Role.** Paper-text only. Empirically inseparable from Cell 3 because the entire offline algorithm stack in this repo is direct-method (regression on (Z, A)) and never consumes πb. The cell's distinctness is theoretical.

**YAMLs.** None. The paper section points to Cell 3 figures and notes the πb-unknown layer.

### Cell 6 — Fog of History

**Identity.** Z hidden, unknown-offline-πb, U absent.

**Role.** Paper-text only. Empirically inseparable from Cell 4 for the same reason as Cell 5.

**YAMLs.** None. The paper section points to Cell 4 figures.

### Cell 7 — Shadowed Vitals

**Identity.** Z observed, known-offline-πb, U present.

**Role.** The paper's centerpiece. Confounded offline data drives a wedge between apparent training value and true eval return; the wedge grows with confounding strength σ.

**YAMLs at `reproducibility/rl_regimes/cell_7/`:**

| YAML                                | algos                                        | behavior        | σ    | mask | confounded |
| ----------------------------------- | -------------------------------------------- | --------------- | ---- | ---- | ---------- |
| `confounded_sigma_000_discrete.yaml`   | [offline_dqn, bcq, cql, iql]              | bias_confounded | 0.00 | none | yes        |
| `confounded_sigma_025_discrete.yaml`   | same                                      | bias_confounded | 0.25 | none | yes        |
| `confounded_sigma_050_discrete.yaml`   | same                                      | bias_confounded | 0.50 | none | yes        |
| `confounded_sigma_075_discrete.yaml`   | same                                      | bias_confounded | 0.75 | none | yes        |
| `confounded_sigma_100_discrete.yaml`   | same                                      | bias_confounded | 1.00 | none | yes        |
| `confounded_sigma_000_continuous.yaml` | [cql_continuous, iql_continuous, bcq_continuous] | bias_confounded | 0.00 | none | yes |
| `confounded_sigma_025_continuous.yaml` | same                                      | bias_confounded | 0.25 | none | yes        |
| `confounded_sigma_050_continuous.yaml` | same                                      | bias_confounded | 0.50 | none | yes        |
| `confounded_sigma_075_continuous.yaml` | same                                      | bias_confounded | 0.75 | none | yes        |
| `confounded_sigma_100_continuous.yaml` | same                                      | bias_confounded | 1.00 | none | yes        |

The `ConfoundedCollectionWrapper` is active on the train env only; evaluation always runs on the clean (unwrapped) env.

**Measurement.** Within each run, two curves over offline training epochs (both IQM ± IQR-STD across 16 parallel envs where the parallel-env axis applies):

- **True eval return** — measured on the clean eval env. Written to the standard `eval_metrics.csv`.
- **Apparent training value** — the critic's predicted Q on sampled dataset transitions, averaged across the batch. Written to `offline_value_trace.csv`.

Additionally, the per-dataset confounding signature — Corr(A, R) and Corr(A, R \| U) — is recorded once at dataset generation time and stored in the dataset's metadata. The gate test (`tests/test_confounded_collection.py::test_confounding_signature_marginal_nonzero_partial_zero`) is run on each generated dataset as a sanity check before training.

**Expected result.** Apparent training value inflates with σ while true eval return decays with σ. The gap is the cell's signature.

### Cell 8 — Dark Ages

**Identity.** Z hidden, unknown-offline-πb, U present.

**Role.** Caps the taxonomy. Shows the worst case when all three identification axes fail simultaneously.

**YAMLs at `reproducibility/rl_regimes/cell_8/`:**

| YAML                                        | algos                                        | behavior        | σ    | mask              |
| ------------------------------------------- | -------------------------------------------- | --------------- | ---- | ----------------- |
| `confounded_sigma_000_masked_discrete.yaml`   | [offline_dqn, bcq, cql, iql]               | bias_confounded | 0.00 | per-env spec (§5) |
| `confounded_sigma_025_masked_discrete.yaml`   | same                                       | bias_confounded | 0.25 | per-env spec (§5) |
| `confounded_sigma_050_masked_discrete.yaml`   | same                                       | bias_confounded | 0.50 | per-env spec (§5) |
| `confounded_sigma_075_masked_discrete.yaml`   | same                                       | bias_confounded | 0.75 | per-env spec (§5) |
| `confounded_sigma_100_masked_discrete.yaml`   | same                                       | bias_confounded | 1.00 | per-env spec (§5) |
| `confounded_sigma_000_masked_continuous.yaml` | [cql_continuous, iql_continuous, bcq_continuous] | bias_confounded | 0.00 | per-env spec (§5) |
| `confounded_sigma_025_masked_continuous.yaml` | same                                       | bias_confounded | 0.25 | per-env spec (§5) |
| `confounded_sigma_050_masked_continuous.yaml` | same                                       | bias_confounded | 0.50 | per-env spec (§5) |
| `confounded_sigma_075_masked_continuous.yaml` | same                                       | bias_confounded | 0.75 | per-env spec (§5) |
| `confounded_sigma_100_masked_continuous.yaml` | same                                       | bias_confounded | 1.00 | per-env spec (§5) |

The wrapper stack is base env → `ConfoundedCollectionWrapper` → `MaskedObservationWrapper` — masking on the outside, confounding underneath.

**Measurement.** Same as Cell 7 — true eval return in `eval_metrics.csv`, apparent training value in `offline_value_trace.csv`, per-dataset signature in dataset metadata.

**Expected result.** Worse degradation than Cell 7 at matched σ; the apparent-vs-true gap is widest here.

## 5. Per-env mask specification

| Env            | Obs dim | Mask indices | What's hidden |
| -------------- | ------- | ------------ | ------------- |
| CartPole-v1    | 4       | 1,3          | cart velocity, pole angular velocity |
| LunarLander-v3 | 8       | 2,3,5        | x velocity, y velocity, angular velocity |
| Acrobot-v1     | 6       | 4,5          | both joint angular velocities |
| Pendulum-v1    | 3       | 2            | angular velocity |
| HalfCheetah-v5 | 17      | 8,9          | one representative hip+knee angular velocity pair |
| Hopper-v5      | 11      | 5,6          | one representative angular velocity pair |
| Walker2d-v5    | 17      | 9,10         | one representative joint velocity pair |

The masked components are velocities throughout — both genuinely informative for control and not recoverable by single-step inference. This makes Cell 2's per-context-variance prediction visible.

## 6. New CSV outputs

Both CSVs are written in addition to the frozen `train_metrics.csv` and `eval_metrics.csv`. The frozen schemas are not modified.

### 6.1 `eval_per_context.csv` — gated on `--mask-indices`

Written during evaluation when `--mask-indices` is non-empty. Columns:

| Column             | Type  | Description |
| ------------------ | ----- | ----------- |
| episode            | int   | Training episode at the checkpoint |
| algorithm          | str   | Algorithm key |
| environment        | str   | Env ID |
| context_bin        | int   | Bin index over the masked Z component |
| context_value_low  | float | Lower edge of the bin |
| context_value_high | float | Upper edge of the bin |
| n_episodes_in_bin  | int   | Number of eval episodes that fell in this bin |
| return_iqm         | float | IQM return for episodes in this bin |
| return_iqr_std     | float | IQR-STD spread |

Used only for Cell 2 plotting.

### 6.2 `offline_value_trace.csv` — gated on `--behavior-policy bias_confounded` and offline algorithm

Written during offline training. Columns:

| Column                 | Type  | Description |
| ---------------------- | ----- | ----------- |
| epoch                  | int   | Training epoch |
| algorithm              | str   | Algorithm key |
| environment            | str   | Env ID |
| apparent_value_iqm     | float | IQM of critic-predicted Q over sampled dataset transitions |
| apparent_value_iqr_std | float | IQR-STD spread |

Used only for Cell 7 and Cell 8 plotting.

## 7. Dataset metadata for confounded runs

Each B2-generated dataset for Cells 7 and 8 carries the per-dataset confounding signature in its metadata:

| Field                    | Type  | Description |
| ------------------------ | ----- | ----------- |
| corr_a_r_marginal        | float | Marginal Corr(A, R) on the dataset |
| corr_a_r_partial_given_u | float | Partial Corr(A, R \| U) on the dataset |
| gate_test_passed         | bool  | Whether test_confounding_signature_marginal_nonzero_partial_zero passed |
| behavior_strength_sigma  | float | The σ used during rollout |

Confounded runs read this metadata at training start and reject any dataset where `gate_test_passed == False`.

## 8. Implementation phasing

This document is PR0. Subsequent PRs against this design:

- **PR1.** `MaskedObservationWrapper` + `--mask-indices` CLI flag + loader hook in `src/envs/offline/minari_loader.py`. The masking machinery, complete and testable. No CSV writers.
- **PR2.** `eval_per_context.csv` writer, gated on `--mask-indices` and eval-time.
- **PR3.** `offline_value_trace.csv` writer, gated on `--behavior-policy bias_confounded` and offline algorithm. Dataset-metadata confounding signature.
- **PR4+.** YAML files per cell, one cell at a time.

Each PR is independently mergeable, has its own tests, and references this document by section anchor.

## 9. What this document does not cover

- The eight-cell taxonomy itself (defined in the paper).
- The `bias_confounded` mechanism (already implemented; see `src/rl/policies/behavior_policy.py` and the gate test).
- Plotting code (handled by `plot.py` once new CSVs exist; one figure per YAML).
- Multi-agent extension (out of scope for the single-agent paper).
