# RL Regimes Restructure — Phase 1 Investigation

**Status:** Investigation only (no code changes, no commits). This document is the
sole deliverable. It audits the current `reproducibility/rl_regimes/` layout against
the proposed taxonomy and reports every coupling that must change.

**New taxonomy (target).**
```
cells    = {offline, online} × {mdp, pomdp}          # 4 cells
subcells = {basic, biased, confounded}               # 3 slices of two 1-D sweeps
  basic      = (beta=0, sigma=0)                      # shared origin
  biased     = (beta>0, sigma=0)                      # beta = behavior-policy bias
  confounded = (beta=0, sigma>0)                      # sigma = confounding strength
```
`(beta>0, sigma>0)` is out of scope; the two axes are orthogonal.

**Definitions held fixed (from the brief).**
- Confounding is **action-dependent**: `r += c_r * U * 1[a == a_bad]`.
- `sigma` scales the `U→A` edge only. `c_r` is **fixed** across the sigma sweep. At
  `sigma=0`, `U` still perturbs reward but `A ⊥ U`, so no backdoor path exists.
- Behavior policy is **marginally matched**: `E_U[pi_b(a|s,U)] == pi_basic(a|s)`.
- POMDP hidden state `S_hid` and confounder `U` are **separate, independent** env
  knobs — never the same variable.

> **Headline for reviewers.** Three of the four "definitions held fixed" are
> **violated by the current implementation**, and all three sit on the critical path:
> 1. The runtime confounding gate is calibrated for the *additive* confounder and
>    **rejects** the action-dependent one (Task 3).
> 2. The confounded behavior policy is **not** marginally matched — its marginal
>    action distribution shifts with sigma (Task 4).
> 3. `c_r` is **coupled** to sigma in code (`c_a = c_r = sigma`), not fixed
>    (Tasks 3–4). At `sigma=0` the current code sets `c_r=0`, so `U` does *not*
>    perturb reward — the opposite of the required semantics.

> **PR-1 status — CONTINUOUS CONFOUNDED CELLS ARE NOT RUNNABLE.** PR 1 lands the
> marginally-matched action-dependent confounder as a **discrete-only** binary
> partition swap (cells `{a_good}`/`{a_bad}`). The continuous arm is **hard-gated**:
> `MarginallyMatchedConfoundedBehaviorPolicy.__init__` raises `NotImplementedError` for
> `action_type == "continuous"`, and `ConfoundedCollectionWrapper` raises when
> `action_gated` is requested on a continuous (`Box`) action space. So any
> `{offline,online}×{mdp,pomdp}` **confounded** cell on a continuous env (Pendulum,
> MuJoCo) cannot be generated or run today. Building the continuous confounder is a
> dedicated follow-up PR whose scope is the three defects in the earlier
> half-space-reflection prototype:
> 1. **Post-squash noise.** The prototype added Gaussian noise *after* the policy's
>    squash, so `pi_basic` was an unbounded Gaussian the env then **clips**: marginal
>    matching survives as a pushforward, but the *realized* `pi_basic` is a clipped
>    Gaussian with **atoms at the bounds**, and the reported noise scale does not
>    describe the policy that ran. The construction must move **pre-squash** — `tanh`
>    is monotone, so the half-space, the reflection, and the cell indicator all commute
>    with it and the action is bounded with **no clipping**.
> 2. **Untested `deterministic=True` API assumption.** The prototype called
>    `agent.act(obs, deterministic=True)`; the test mocks swallow it via `**kwargs`, so
>    tests passed while a real SAC/TD3 would `TypeError`. The continuous path must be
>    validated against a **real registered algorithm**, not a mock.
> 3. **Silent `current_h` reward-gate side-channel.** The continuous reward gate
>    `r += c_r·U·1[h(a)==1]` was routed through `env.current_h` with a `try/except
>    pass` and an `action == a_bad` fallback that is **false everywhere on a float
>    action** — so the gate would never fire, `c_r` would have no effect, and the
>    dataset would be **silently unconfounded**. The follow-up needs a checked,
>    non-silent gate wired to the pre-squash cell indicator.

The **discrete** action-dependent confounder (cell 9-style, CartPole/Acrobot) is
unaffected and fully runnable.

---

## 1. Audit of the current tree

### 1.1 Directory inventory

`reproducibility/rl_regimes/` is a flat `cell_1 … cell_9` layout. What each encodes
(observability, data regime, behavior/confounder), drawn from the YAMLs and
`docs/experimental_design.md §4`:

| Cell | Observability | Data regime | U? | YAML families |
|---|---|---|---|---|
| `cell_1` | MDP (full) | online | no | `online_{discrete,continuous}.yaml`; `…_anti_reward_{000..100}`, `…_curiosity_{000..100}` behavior sweeps (22 files) |
| `cell_2` | POMDP (masked) | online | no | `online_masked_{discrete,continuous}.yaml` + same anti_reward/curiosity sweeps (22 files) |
| `cell_3` | MDP | offline | no | `offline_{random,medium,expert}_{discrete,continuous}.yaml` — **coverage-tier sweep** (6 files) |
| `cell_4` | POMDP (masked) | offline | no | `offline_{tier}_masked_{discrete,continuous}.yaml` (6 files) |
| `cell_5` | MDP, **πb unknown** | offline | no | **none** — paper-text only (`experimental_design.md §4 Cell 5`) |
| `cell_6` | POMDP, **πb unknown** | offline | no | **none** — paper-text only |
| `cell_7` | MDP | offline | **yes (additive)** | `confounded_sigma_{000..100}_{discrete,continuous}[_gated][_deconfounded]`; `online_confounded_sigma_*_discrete_gated`; `sensitivity_sweep_gamma_{100,200,400,800}` |
| `cell_8` | POMDP (masked) | offline | **yes (additive)** | `confounded_sigma_*_masked_*[_gated]`; `…_recurrent_proximal`; `ablation_strategy_recurrent_sigma_{000,050}`; `sensitivity_sweep_gamma_*_masked_discrete_recurrent`; `online_confounded_sigma_*_masked_discrete_gated` |
| `cell_9` | MDP | offline | **yes (ACTION-GATED)** | `action_gated_sigma_100_discrete.yaml` (1 file, CartPole, σ=1.0 only) |

**Config-schema fields that carry the semantics** (`src/config/defaults.py`):
- `behavior_policy` (defaults.py:33) + `behavior_strength` (defaults.py:34) — the
  single strength dial; its meaning depends on the policy (`anti_reward`/`curiosity`
  strength, `bias_suboptimal` β, `bias_confounded[_action]` σ). See `_STRENGTH_PARAM`
  in `src/rl/policies/behavior_policy.py:450-457`.
- `mask_indices` (defaults.py:39) — the POMDP marker (Cells 2/4/8).
- `offline_dataset` (defaults.py:24) — presence ⇒ offline regime.
- `critic_network` (defaults.py:58) — `mlp` vs `lstm/gru/rnn` recurrent arm.
- `networks: {gamma_sensitivity: …}` — the Kallus-Zhou Γ (a *method* parameter,
  read from config, `src/rl/offline/sensitivity.py:135`).

**Results directories.** `runs/rl_regimes/cell_N/<yaml_basename>_<YYYYMMDD>_<HHMMSS>/`,
each holding `train_metrics.csv`, `eval_metrics.csv`, `config.yaml`, `metadata.json`,
and (gated) `eval_per_context.csv` / `offline_value_trace.csv` /
`critic_ablation_metrics.csv`. Confirmed under `runs/rl_regimes/cell_{1,2,3,4,7,8,9}/`.

### 1.2 Everything that assumes flat `cell_N` numbering

| Location | What it assumes | Evidence |
|---|---|---|
| `tools/run_cells_1234_parallel.sh` | globs `cell_{1,2,3,4}/*.yaml` and **`cell_{7,8}/*_gated.yaml`** for the paper matrix; preflight `grep -rhoE 'generated/…' cell_{7,8}/*_gated.yaml` | the "`*_gated.yaml`" glob the brief flags lives **here** (the dataset-id grep + the coverage comment "Cells 7, 8: ONLY *_gated.yaml") |
| `src/benchmarking/plotting.py:1546` | LaTeX caption parses cell number: `re.search(r"cell_(\d+)", run_name)` → `f"Cell {…}"` | degrades silently under new paths (no `cell_N` ⇒ bare-metric caption) |
| `src/benchmarking/plotting.py:1879-1882` | **1:1 cell_path across three trees**: `cell_path = Path(run_name).parent`; `reproducibility/<cell_path>`, `runs/<cell_path>`, `<outdir>/<cell_path>` | `render_sweep_tables` — highest-risk coupling |
| `src/benchmarking/plotting.py:561, 793` | σ-sibling globs `run_dir.parent.glob("confounded_sigma_*_{arm}_*")` and `"online_confounded_sigma_*_{arm}_*")`, each with `("_gated_" in d.name)` partition (564, 800) | assumes all σ-values co-located in one flat dir; keyed on basename prefix |
| `src/benchmarking/table_formatting.py:106,109,123` | `_STRENGTH_SUFFIX_RE = r"^(.+?)_(\d{3})(?:_(.+))?$"`; `detect_sweep_families` globs `<cell_dir>/*.yaml` | the ×100 sweep-family convention (σ and Γ) |
| `src/benchmarking/table_formatting.py:140` | `_LABEL_DROP_TOKENS = {"discrete","continuous","gated"}` | regime/observability descriptor tokens |
| `src/benchmarking/plotting.py:255,264,277` | `_sigma_from_run`: `r"sigma_?(\d{3})"` on run dir, fallback `r"-sigma(\d{3})-"` on dataset id | σ decoded from path/name |
| `tools/__pycache__/{aggregate_matrix,grid_heatmaps,make_plot_data}.pyc` | **source deleted**; `.pyc` hard-code cell env-ids (`causal/cartpole/cell7-b1-d0p5-v0`), CLI positions `<cell8_run_dir> <cell3_run_dir>`, a `cell` group column, title "…(cell 7, CartPole)…" | dead but cell-numbered; not part of the live path |
| `tests/test_yamls_cell_{1,2,3,4,7,8}.py`, `…_7_8_gated.py`, `…_7_8_online.py`, `test_table_formatting.py`, `test_sigma_zero_anchor.py`, `test_eval_per_context.py` | construct/glob `cell_N/…` paths and `*_gated.yaml` filenames | test-suite coupling |
| `src/benchmarking/registry.py:487,536`; `runner.py` comments (57,243,310,1178,1221); `critic_ablation.py:318`; `identification.py:123`; `main.py:98,244` | comments/messages naming "Cell N" | mostly cosmetic, but message strings leak the old taxonomy |

`README.md` contains no `cell_N` literals. The `--split` plot dispatch
(`plotting.py:2060-2089`) keys on split names (`per_context`, `value_trace`,
`online_sigma_sweep`, `both`), **not** cell numbers — those survive the restructure.

---

## 2. Old → new mapping table

Target coordinates: `(regime ∈ {offline,online}×{mdp,pomdp}, subcell, beta, sigma)`.

| Old | New regime | Subcell | beta | sigma | Notes |
|---|---|---|---|---|---|
| `cell_1` base | online_mdp | basic | 0 | 0 | clean anchor |
| `cell_1` anti_reward/curiosity `{025..100}` | online_mdp | **biased** | >0 | 0 | β = `behavior_strength`; **but** anti_reward/curiosity are exploration/pessimism shapers, not classic πb-bias — *ambiguous whether they are the intended `biased` dial* |
| `cell_2` base | online_pomdp | basic | 0 | 0 | |
| `cell_2` anti_reward/curiosity `{025..100}` | online_pomdp | biased | >0 | 0 | same ambiguity |
| `cell_3` (tiers) | offline_mdp | basic | 0 | 0 | **coverage tier (random/medium/expert) is a THIRD axis with no (β,σ) home — ambiguous** |
| `cell_4` (tiers) | offline_pomdp | basic | 0 | 0 | same coverage-tier ambiguity |
| `cell_5` | offline_mdp | — | — | — | **no clean home**: πb-unknown is a *data-mode* value the new schema drops; paper-text only |
| `cell_6` | offline_pomdp | — | — | — | same as cell_5 |
| `cell_7` `confounded_sigma_*` | offline_mdp | confounded | 0 | >0 | **confounder-kind mismatch**: additive `r+=c_r·U`, not action-dependent |
| `cell_7` `online_confounded_*` | online_mdp | confounded | 0 | >0 | additive |
| `cell_7` `sensitivity_sweep_gamma_*` | offline_mdp | confounded | 0 | 0.5 | adds an orthogonal **Γ method-axis** (not β/σ) |
| `cell_7` `*_deconfounded` triad | offline_mdp | confounded | 0 | ≥0 | method comparison (floor/proximal/oracle), not a new grid point |
| `cell_8` `confounded_sigma_*_masked_*` | offline_pomdp | confounded | 0 | >0 | additive |
| `cell_8` `online_confounded_*_masked_*` | online_pomdp | confounded | 0 | >0 | additive |
| `cell_8` `sensitivity_*_recurrent` | offline_pomdp | confounded | 0 | 0.5 | Γ-axis, recurrent |
| `cell_9` `action_gated_sigma_100` | offline_mdp | confounded | 0 | 1.0 | **the ONLY action-dependent config; matches the new confounder definition** |

### Old cells with no clean new home
- **`cell_5`, `cell_6`** — πb-unknown data mode is not representable in `(regime, β, σ)`.
  (They already carry no YAMLs; paper-text only.)
- **`cell_3`/`cell_4` coverage tiers** — `random/medium/expert` is a genuine data axis
  orthogonal to β and σ; the new schema has no slot for it. Either fold into `basic`
  (losing the coverage sweep) or add a third parameter. **Ambiguous — needs a decision.**
- **`cell_7`/`cell_8` additive confounded** — these exist and are byte-frozen, but the
  new "confounded" is *action-dependent*. They map to `confounded` only if the schema
  admits a `confounder.kind` field distinguishing `additive` vs `action_gated`
  (see Task 3). Otherwise they have no home under the action-dependent definition.

### New (regime, β, σ) points with no old equivalent
- `offline_pomdp / confounded (action-dependent)` — cell_9 is MDP-only.
- `online_mdp / confounded (action-dependent)` and `online_pomdp / confounded
  (action-dependent)` — only *additive* online variants exist.
- action-dependent `confounded` at `σ ∈ {0.25, 0.5, 0.75}` — cell_9 only ships σ=1.0.
- `offline_mdp / biased` and `offline_pomdp / biased` (β>0, σ=0 via
  `bias_suboptimal`/`bias_skew`) — **no offline biased dataset exists**:
  `generate_all_datasets.sh` only produces `agent` (tiers) and `bias_confounded`
  rollouts; `grep behavior_policy` over all YAMLs returns only `anti_reward`,
  `curiosity`, `bias_confounded`, `bias_confounded_action` (no `bias_suboptimal`/
  `bias_skew` in any config). So offline-biased is unrealized.

---

## 3. Confounder audit (highest priority)

### 3.1 Where the gate lives and what it asserts

**Computation** — `src/envs/offline/generate.py:264-284`,
`compute_confounding_signature`:
```python
r_ar, r_au, r_ru = _pearson(a, r), _pearson(a, u), _pearson(r, u)
denom = np.sqrt((1 - r_au**2) * (1 - r_ru**2))
partial = float((r_ar - r_au * r_ru) / denom) if denom > 0 else 0.0
gate = bool(abs(r_ar) > 0.2 and abs(partial) < 0.05)
```
i.e. **marginal `|Corr(A,R)| > 0.2` AND partial `|Corr(A,R | U)| < 0.05`**. Same
thresholds as the unit gate `tests/test_confounded_collection.py:89-92`.

**Enforcement** — the runner rejects a dataset before training, in **two** places
(duplicated verbatim): `src/benchmarking/runner.py:1149-1177` (`_train_offline`, flat)
and `runner.py:1288+` (`_train_offline_grouped`, the proximal/episode-grouped path).
The gate opens when `_value_trace_gate_open` is true — `behavior_policy ∈
{bias_confounded, bias_confounded_action}` AND `data_regime == "offline"`
(`runner.py:256-260`). σ=0.0 is the only exemption (`runner.py:1165`).

### 3.2 Why the action-dependent confounder fails it

Two independent failures, both on the critical path:

1. **The signature is never computed for the action-dependent confounder.**
   `generate.py:405` guards on `behavior_policy == "bias_confounded"` **only**:
   ```python
   if behavior_policy == "bias_confounded" and sig_samples is not None:
       signature = compute_confounding_signature(sig_samples, behavior_strength)
   else:
       signature = {... "gate_test_passed": None, "behavior_strength_sigma": None ...}
   ```
   So a `bias_confounded_action` dataset carries `gate_test_passed=None`,
   `behavior_strength_sigma=None`. At load the gate does: key present ⇒ skip the
   "missing metadata" branch; `meta.get("behavior_strength_sigma") == 0.0` is
   `None == 0.0 → False` ⇒ skip the σ=0 exemption; `not bool(meta["gate_test_passed"])`
   is `not bool(None) → True` ⇒ **`raise ValueError("failed the confounding gate test")`**.

2. **Even if the additive signature *were* computed on action-gated data, it would
   fail by design.** The additive gate proves *pure* confounding (backdoor only, no
   direct effect): partial `Corr(A,R|U) ≈ 0`. The action-dependent confounder
   `r += c_r·U·1[a=a_bad]` deliberately introduces a **direct A→R effect** (the bonus
   is a real consequence of taking `a_bad`), so conditioning on `U` does **not** null
   the A–R association — `partial Corr(A,R|U)` is materially non-zero, breaking the
   `< 0.05` clause. The gate mislabels a legitimate action-dependent confounder as
   "gate failed."

**Empirical corroboration.** The one committed action-gated run,
`runs/rl_regimes/cell_9/action_gated_sigma_100_discrete_20260710_100624/`, has
`eval_metrics.csv`, `train_metrics.csv`, and `offline_value_trace.csv` that are **header
only (1 line each)** and a stub `metadata.json` (`{"timestamp": …}`) — i.e. **no
training rows were produced**, consistent with the gate raising before training. (The
unit gates in `tests/test_action_dep_confounder_gate_b.py` — which the memory records
as "4/4 gates pass" — exercise the *wrapper/estimator* directly and never touch the
dataset-generation signature gate, so they don't cover this path.)

### 3.3 What a correct action-dependent gate would assert

The action-dependent regime's operational signature is **not** "association vanishes
given U." It is: *the U→A edge is live, the gated U→R bonus is live, and a naive
observational value of `a_bad` is inflated relative to the U=0 stratum.* Concretely,
over the confounded rollout's `(a, r, u)` samples:

1. **U→A edge active:** `Corr(1[a==a_bad], U) > τ_ua` (behavior takes `a_bad` more when
   `U=1`). This is what `sigma>0` is supposed to switch on.
2. **Gated U→R active:** within `a==a_bad` transitions, `Corr(R, U) > τ_ur`
   (the bonus is present); within `a!=a_bad` transitions, `Corr(R, U) ≈ 0`
   (no bonus off the gated action).
3. **Backdoor bias present (optional, strongest):** the observational advantage of
   `a_bad`, `E[R | a_bad] − E[R | a_good]`, exceeds its U=0-stratum counterpart
   `E[R | a_bad, U=0] − E[R | a_good, U=0]` by a margin — the quantity the
   deconfounding methods are meant to remove.

At `sigma=0`, condition (1) must be **absent** (`Corr(1[a==a_bad], U) ≈ 0`) while the
reward bonus (2) may still be present — matching the brief's "at σ=0, U still perturbs
reward but A ⊥ U, so no backdoor path exists." So the σ=0 exemption should assert
"no U→A edge," not "no reward shift."

### 3.4 Proposal: gate as a declarative YAML field, not a hard-coded assert

Replace the branch-on-`behavior_policy` string comparison and the hard-coded
thresholds with a declarative block the sweep YAML owns. The generator reads it,
computes the named signature, and stamps `gate_test_passed`; the runner enforces
whatever the dataset was stamped with.

```yaml
confounder:
  kind: action_gated          # additive | action_gated  (selects the wrapper + gate)
  a_bad: 1
  c_r: 1.0                    # FIXED reward-shift magnitude (decoupled from sigma; see Task 4)
  gate:
    type: action_dependent    # additive | action_dependent  (selects the assertion)
    # action_dependent thresholds:
    u_to_a_corr_min: 0.10     # Corr(1[a==a_bad], U) > this   (U->A edge live at sigma>0)
    gated_reward_corr_min: 0.10  # within a_bad: Corr(R,U) > this (U->R bonus live)
    ungated_reward_corr_max: 0.05  # within a_good: Corr(R,U) < this
    sigma_zero_exemption: u_to_a   # at sigma=0 assert NO U->A edge (not "no reward shift")
    # additive thresholds (back-compat, byte-frozen cells 7/8):
    # marginal_min: 0.20 ; partial_max: 0.05
```
Benefits: (a) cells 7/8's additive path keeps its exact thresholds (byte-frozen);
(b) the action-dependent gate is *declared*, not string-matched on
`bias_confounded_action`; (c) `c_r` becomes an explicit, sigma-independent field.

---

## 4. Marginal-matching check

**Does the current confounded behavior policy satisfy `E_U[pi_b(a|s,U)] = pi_basic(a|s)`?
No.**

`ConfoundedBehaviorPolicy.act` (`src/rl/policies/behavior_policy.py:435-443`), discrete:
```python
pref = u.round().long().clamp(0, n - 1)
take = torch.rand(...) < min(self.c_a, 1.0)   # c_a = base_c_a * strength = sigma
return ActionOutput(action=torch.where(take, pref, agent_a))
```
With `U ~ Bernoulli(0.5)`, `pref = round(U) ∈ {0,1}`. As `sigma` grows, the policy
mixes in `pref` with probability `min(c_a,1)`, so the **marginal** action distribution
`E_U[pi_b(a|s,U)]` shifts toward `{0,1}` and away from `pi_basic = pi_agent`. Only at
`sigma=0` (⇒ `c_a=0`, `take` always false) does it reduce to the agent's action and
match trivially. **For every σ>0 the confounder smuggles in behavior bias** — exactly
the failure the brief warns against. (Continuous case, line 443:
`agent_a + c_a·U` — an additive mean shift; its marginal also moves with σ unless
`E_U[U]=0` *and* symmetric, which Bernoulli(0.5) is not.)

**Compounding issue — `c_r` coupled to σ.** Both the generation path
(`generate.py:151-153`, `c_a=sig, c_r=sig`) and the runner
(`runner.py:313-314`, `c_a=_sigma, c_r=_sigma`) set **`c_r = c_a = sigma`**. The brief
requires `c_r` *fixed* across the σ-sweep and `sigma` to scale *only* `U→A`. Under the
current coupling, `sigma=0` also zeroes the reward shift (`c_r=0`), so `U` does not
perturb reward at all — contradicting "at σ=0, U still perturbs reward."

### What would have to change
1. Decouple: `c_a = f(sigma)` (scales `U→A` only), `c_r = const` (a fixed YAML field).
2. Make the behavior policy marginally matched — e.g. define `pi_b(a|s,U)` as a
   **mean-preserving** perturbation: split probability mass symmetrically around
   `U=0/U=1` so that `0.5·pi_b(a|s,U=0) + 0.5·pi_b(a|s,U=1) = pi_basic(a|s)` exactly.
   (Concretely: when `U=1` shift mass toward `a_bad` by δ, when `U=0` shift mass toward
   `a_good` by δ — the U-average cancels.)

### Proposed generation-time assertion (every sigma)
At dataset generation, compare the empirical marginal action distribution at `σ` to the
`σ=0` reference and assert TV distance within tolerance:
```
p_sigma[a]  = empirical P(A=a) over the rollout at strength sigma
p_basic[a]  = same at sigma=0 (the shared-origin reference)
assert 0.5 * sum_a |p_sigma[a] - p_basic[a]| < tol_marginal   # e.g. 0.02
```
This is cheap (the rollout already collects `sig_a`), runs per σ, and would have caught
the current marginal drift. It belongs next to `compute_confounding_signature` and can
be surfaced as another declarative gate field (`gate.marginal_tv_max: 0.02`).

---

## 5. Proposed schema

### 5.1 Target layout
```
reproducibility/rl_regimes/
  _base/
    envs.yaml          # env lists (discrete/continuous, per-env mask_indices)
    algos.yaml         # algo rosters (base + recurrent + strategy variants)
    seeds.yaml         # seed set
    budgets.yaml       # n_episodes, rollout_len, n_train/eval_envs, n_checkpoints, aggregation
  offline_mdp/sweep.yaml
  offline_pomdp/sweep.yaml
  online_mdp/sweep.yaml
  online_pomdp/sweep.yaml
  _legacy/
    cell_1/ … cell_9/  # FROZEN, read-only — existing YAMLs moved verbatim
```
Each `sweep.yaml` declares the two orthogonal 1-D sweeps (β-axis and σ-axis) sharing
the origin; the subcell labels (`basic/biased/confounded`) are *derived at reporting
time* from `(beta, sigma)`, never stored in a path.

### 5.2 Example `offline_mdp/sweep.yaml`
```yaml
# offline_mdp — {basic, biased, confounded} as slices of two 1-D sweeps sharing (0,0).
extends: [_base/envs.yaml, _base/algos.yaml, _base/seeds.yaml, _base/budgets.yaml]

regime:
  observability: mdp          # no mask_indices
  data: offline

# Behavior-policy bias axis (sigma pinned 0): the `biased` slice.
beta_sweep:
  behavior_policy: bias_suboptimal
  values: [0.0, 0.25, 0.50, 0.75, 1.0]     # 0.0 == basic origin

# Confounding axis (beta pinned 0): the `confounded` slice.
sigma_sweep:
  behavior_policy: bias_confounded_action  # ACTION-DEPENDENT (new primary)
  confounder:
    kind: action_gated
    a_bad: 1
    c_r: 1.0                  # FIXED across the sweep (decoupled from sigma)
    gate:
      type: action_dependent
      u_to_a_corr_min: 0.10
      gated_reward_corr_min: 0.10
      ungated_reward_corr_max: 0.05
      marginal_tv_max: 0.02   # marginal-matching assertion (Task 4)
      sigma_zero_exemption: u_to_a
  values: [0.0, 0.25, 0.50, 0.75, 1.0]     # 0.0 == basic origin, shared with beta_sweep

# (beta>0, sigma>0) is OUT OF SCOPE — no cross-product is emitted.

critics: [observational, proximal, oracle_u]   # every cell supports the ablation (Task 8)
```

### 5.3 Parameterized results paths (labels derived, never baked)
```
results/{regime}/beta_{beta*100:03d}_sigma_{sigma*100:03d}/{env}/{algo}/{critic}/{seed}/
```
- ×100 zero-padded encoding, matching the existing `gamma_100`/`gamma_400` and
  `sigma_050` conventions (`table_formatting.py:106`, `strength_to_float_label`).
- Example: `results/offline_mdp/beta_000_sigma_050/CartPole-v1/cql/proximal/0/`.
- `basic = beta_000_sigma_000`, `biased = beta_XXX_sigma_000`,
  `confounded = beta_000_sigma_XXX` are computed by the reporting layer, not stored.
- The `{critic}` segment is required by Task 8 (below). Non-ablation runs can use a
  sentinel `{critic}=none`.

**Reporting-layer changes this forces** (from Task 1): `plotting.py:1546` caption regex
(replace `cell_(\d+)` with a `(regime, beta, sigma)` parser), `plotting.py:1879-1882`
cell_path 1:1 mapping, and the σ-sibling globs `plotting.py:561,793` (repoint from
`run_dir.parent` flat dirs to the `beta_*_sigma_*` grid). `table_formatting.py`'s
`detect_sweep_families` glob can stay if `sweep.yaml` expands into per-point stems.

---

## 6. Metric superset

**Currently logged per run** (all CSV; no JSON — `grep json.dump runner.py` empty):

| File | Columns (constant / location) |
|---|---|
| `train_metrics.csv` | `TRAIN_COLUMNS` = episode, algorithm, environment, train_return_mean/std, loss, policy_loss, value_loss, entropy, kl, critic_loss, actor_loss, q_loss (`runner.py:32-46`) |
| `eval_metrics.csv` | `EVAL_COLUMNS` = episode, algorithm, environment, eval_return_mean/std (`runner.py:48-54`) |
| `eval_per_context.csv` | episode, algorithm, environment, context_bin, context_value_low/high, n_episodes_in_bin, return_iqm, return_iqr_std (`runner.py:59-69`; gated on `mask_indices`) |
| `offline_value_trace.csv` | epoch, algorithm, environment, apparent_value_iqm/iqr_std (+`apparent_value_u0_iqm/iqr_std` for `*_oracle_u`) (`runner.py:74-88`) |
| `critic_ablation_metrics.csv` | `CRITIC_ABLATION_COLUMNS` (V-head, 24 cols) **or** `STRATEGY_CRITIC_ABLATION_COLUMNS` (`critic_ablation.py:15-40 / 45-62`) |
| `aux_metrics.csv` | episode, algorithm, environment, model, train_loss, mse, mae (`aux_models.py:12-20`) |

**Status of the six requested report metrics** (single sweep, no re-runs):

| Metric | Status | Evidence / gap |
|---|---|---|
| **return / IQM** | ✅ **logged** | `eval_return_mean/std` are IQM by default (`aggregation="iqm"`, defaults.py:51; `_aggregate_returns` middle-50% at `runner.py:776-791`); also `return_iqm` per context bin, `apparent_value_iqm` in value trace |
| **value MSE vs oracle Q** | ⚠️ **logged only in strategy ablation** | `value_mse_to_oracle` (`critic_ablation.py:56`, computed `:668` vs `UMarginalizedQ.forward` oracle). **Not** emitted on normal offline runs — `offline_value_trace.csv` logs the critic's own `apparent_value_iqm` but not its MSE to oracle. To report per-cell, the ablation must run in every cell (Task 8). |
| **OPE error** | ❌ **not computable** | no OPE/IS/WIS/DR/FQE estimator anywhere; logged behavior propensities are **discarded** at load (`experience_source.py:118`, `ep.pop("behavior_logprob", None)`). Must be added *and* propensities retained. |
| **coverage stats** | ⚠️ **computed but dead (never logged)** | `Proximal.statistical_diagnostic` returns `separability`, `action_overlap` (`identification.py:125-157`) but has **zero callers** and no schema column. Needs a caller + columns. |
| **pi_b recovery error** | ❌ **not computable** | no BC/behavior estimator (`sensitivity.py:19-22` "no behavioral-cloning estimator"); propensities discarded (`experience_source.py:118`). Requires a πb estimator + retained propensities. |
| **Γ-bound width** | ❌ **not computable** | sensitivity produces only a one-sided pessimistic `Q_lower` (`sensitivity.py:14,102-115`); **no upper bound, no interval width**. `SensitivityBounds.statistical_diagnostic` exposes only the input Γ (`identification.py:218-219`), and that is dead (no caller). Requires an upper-bound backup + a width column. |

**Net:** to report all three subcells from one sweep, add columns/writers for
`value_mse_to_oracle` on the main offline path (or always run the ablation), a coverage
writer wiring `statistical_diagnostic`, an OPE estimator + column, a πb-recovery
estimator + column (and stop discarding `behavior_logprob`), and a two-sided Γ bound +
width. Only **return/IQM** is fully in place today; **value-MSE-vs-oracle** is logged
but path-restricted; the other four need new computation.

---

## 7. Online cells — πb dependence and drift

**Reframing (important).** The brief's premise is that IPW, the proximal E-step, and
the sensitivity bounds *estimate or depend on πb*. In this codebase **none of the four
identification strategies estimates or depends on πb, and there is no IPW anywhere in
the causal path.** So the "non-stationary ε-greedy πb drift" failure mode has **no
πb-estimator to break**. What *does* drift and get re-canonicalized is the **latent-U
label**, not a propensity. Details:

| Component | Depends on πb? | Re-estimated per buffer refresh? | Evidence |
|---|---|---|---|
| **IPW / propensity weighting** | **No estimator exists** in the causal path | N/A | only *logged* `behavior_logprob` exists, discarded at load (`experience_source.py:44-52,118`); none of the 4 strategies consume it |
| **Observational** | no | n/a | `critic_value` returns `net(x)` (`identification.py:52-66`) |
| **OracleU** | no (reads *realized U*, not πb) | reads U each batch | `net.q_su(x, batch["confounder_u"])` (`identification.py:81`) |
| **Proximal E-step** | **no** (infers latent U from a reward-mixture LLR) | **yes, in online mode** | `run_em` LLR over rewards (`proximal.py:324-336`); `m_step` cadence: `if online: self.refresh() else: self.run_em()` (`proximal.py:370-377`); `refresh()` re-applies warm-start + label canonicalization every refresh (`proximal.py:193-229`), with the guarded label-swap flip (`:338-359`) and the `softplus(delta)≥0` unreachable-basin convention (`:67-78`) |
| **Sensitivity / MSM** | **no** (residual-based on the agent's own Q) | recomputed **per batch** | `reweight`: `delta = r − q_sa(batch)`, `w = where(delta>0, 1/Γ, Γ)` (`sensitivity.py:102-115`); wrapped around every `learn` (`sensitivity.py:117-132`); docstring "RESIDUAL-BASED (no behavioral-cloning estimator)" (`:19-22`) |

**Stationarity scope of the online-confounded path.** Gate B
(`_train_off_policy_grouped`, `runner.py:1026-1050`, sets `em.online=True`) is
**explicitly scoped to a fixed/stationary behavior policy** — the comment at
`runner.py:1036-1039` says "a fixed behavior policy ⇒ a stationary episode distribution
(NOT the co-adapting cells 5-6)." The drifting ε-greedy-over-own-Q case the brief
describes is therefore **declared out of scope** in the current design, and the only
refresh hook is `ProximalEM.refresh()` (the flat `ReplayBuffer` has none). Fresh
online transitions are seeded with `r_tau = prior` (`experience_source.py:529`) until a
refresh overwrites them.

**Verdict per requested component:**
- **IPW** — no code; nothing to re-estimate. Would need to be built if the new schema
  wants propensity-based identification under drift.
- **Proximal E-step** — does **not** depend on πb; **re-estimates the latent U (with
  label re-canonicalization) per refresh** in online mode. This is the real, handled
  drift (the "label-swap canonicalization" the memory cites) — but it is U-label drift,
  not πb drift.
- **Sensitivity bounds** — does **not** depend on πb; reweights per batch from the live
  Q. No πb, no refresh dependence.

If the restructure genuinely wants online cells with **drifting πb** *and* πb-dependent
identification (IPW / proximal-with-propensities / MSM-with-BC-weights), that machinery
**does not exist yet** and would be new work; today's online-confounded path assumes a
stationary behavior policy.

---

## 8. Critic ablation (addendum)

### 8.1 Registered critics and base learners

**Ablation-critic library** (`critic_ablation.py:102-128`, `CRITIC_LIBRARY`):
- V-head: `standard_mlp`, `residual_reward_model` (scored vs returns).
- Strategy: **`observational`, `proximal`, `oracle_u`** (scored estimation-vs-oracle).
- **No `sensitivity` critic** is registered in the ablation library.

**Flat vs recurrent** (`_build_strategy_critic`, `critic_ablation.py:231-363`): the
`encoder` axis is orthogonal to the strategy. `encoder="mlp"` → flat (Cell-7 byte-frozen
arm); `encoder ∈ {rnn,lstm,gru}` → recurrent (Cell-8 arm), **DQN-base only**:
- `observational` → `build_offline_dqn_recurrent` (:263)
- `proximal` → `build_recurrent_proximal_dqn` (:327)
- `oracle_u` → `build_recurrent_oracle_u_dqn` (:346)

So recurrent variants of observational/proximal/oracle_u **do** exist.

### 8.2 Is `pomdp × confounded × sensitivity` runnable?

**As a standalone algorithm: yes. As an ablation critic: no.**
- A recurrent sensitivity **algo** exists: `build_sensitivity_dqn_recurrent`
  (`sensitivity.py:171-208`), registered as `offline_dqn_recurrent_sensitivity`
  (`registry.py:546`), with live Cell-8 configs
  `cell_8/sensitivity_sweep_gamma_{100,200,400,800}_masked_discrete_recurrent.yaml`.
  So a POMDP×confounded sensitivity *run* is possible.
- **But there is no `sensitivity` entry in `CRITIC_LIBRARY`**, so it cannot be placed in
  the estimation-vs-oracle ablation alongside observational/proximal/oracle_u. It was
  **left as an unlanded follow-up for the ablation harness specifically.** Explicitly:
  **`pomdp × confounded × sensitivity` is NOT runnable as a head-to-head ablation
  critic** — only as a full standalone training run. To make "every cell supports the
  critic ablation" literally true, a `sensitivity` `CriticSpec` + a `StrategyCritic`
  branch (flat and recurrent) must be added.

### 8.3 Do all critics share the same base learner? (the bare-DQN bug)

**Fixed.** The prior bug — the observational baseline constructed as bare `DQN`
regardless of `--algos`, inflating the deconfounding gap — is explicitly addressed in
`critic_ablation.py:272-308`. The observational floor now builds the **base algo's own
class** with an `Observational()` strategy (a literal pass-through Q):
`build_cql/iql/bcq(..., strategy=Observational())` (:296-308). The `dqn` suffix keeps a
bare `DQN` (:286-292) — but that is DQN-vs-DQN, so there is no base confound. The
in-code comment (:272-279) documents exactly the failure mode the addendum names ("a
DQN floor inflates the observational→proximal gap with a CQL-vs-DQN base confound"). So
proximal/oracle_u/observational share one base learner per run.

### 8.4 Results path with critic

Adopt the addendum's path (folded into §5.3):
```
results/{regime}/beta_{beta*100:03d}_sigma_{sigma*100:03d}/{env}/{algo}/{critic}/{seed}/
```

### 8.5 Proposed NULL-CALIBRATION gate

**Assertion:** at the shared origin `(beta=0, sigma=0)`, all critics must agree within
seed noise. There is **no such assertion today** — `gap_closed_fraction` is merely left
*blank* at σ=0 (`critic_ablation.py:684-691`); `value_mse_to_oracle` is logged but never
asserted-upon. Concretely, assert that every critic's `value_mse_to_oracle` at
`(0,0)` is below the estimator-noise floor (reuse `_GAP_NOISE_FLOOR_MSE = 1e-2`,
`critic_ablation.py:69`) and that pairwise critic disagreement is within seed spread:
```
for critic in {observational, proximal, oracle_u}:
    assert value_mse_to_oracle(critic) @ (beta=0,sigma=0) < _GAP_NOISE_FLOOR_MSE
assert max_pairwise |apparent_q_mean_i - apparent_q_mean_j| @ (0,0) < k * seed_std
```
**Where it would live:** two options —
(1) a **declarative check** emitted from `CriticAblationManager.checkpoint_rows_strategy`
(it already computes `value_mse_to_oracle` and knows `sigma`; add a `null_calibrated`
boolean column asserted when `beta==sigma==0`), and
(2) a **gate test** in `tests/` (sibling to `test_action_dep_confounder_gate_b.py` and
`test_sigma_zero_anchor.py`, which already assert σ=0 *collapse* for the estimator but
not *cross-critic agreement*). Option (1) makes it a CI-checkable artifact of every
`basic`-slice run; option (2) pins it as a unit invariant. Recommend both, with the
threshold sourced from the `basic` run's own across-seed std (so "seed noise" is
measured, not assumed).

---

## Uncertainties / evidence notes
- **cell_9 gate failure** (§3.2) is proven from the code path (signature never computed
  for `bias_confounded_action`; both offline gate blocks treat `None` as failure) and
  is *corroborated* by the empty/stub run artifacts. I did not re-run the pipeline to
  observe the raised exception directly; the header-only CSVs are consistent-with but
  not a live capture of it.
- The `anti_reward`/`curiosity` → `biased` mapping (§2) is flagged **ambiguous**: these
  are exploration/pessimism shapers, and whether they are the intended `beta` behavior-
  bias dial (vs `bias_suboptimal`/`bias_skew`) is a modeling decision, not a fact in the
  code.
- The deleted `tools/*.py` (`aggregate_matrix`, `grid_heatmaps`, `make_plot_data`) were
  read from `.pyc` string constants only (source gone); their exact current behavior is
  inferred from those constants.
- `separability`/`action_overlap`/`gamma_sensitivity` diagnostics (§6) are confirmed
  **caller-less** via repo-wide grep; "computable" here means the function exists, not
  that anything invokes it.
```
