# Phase 1.5 — Review memo: recovered prior causal implementation

Date: 2026-06-05. Scope: verdict {**port** | **adapt** | **discard**} per module
against the new architecture (§4 of the execution plan) and the
CartPole/HalfCheetah + Minari anchors. **No code was integrated**; porting
decisions are to be approved at the Phase-2 gate.

The old implementation was a *synthetic-env* eight-cell benchmark
(sepsis / Block-MDP simulators with exact analytic do-oracles, gap-regularized
DQN) — a different experimental design from the new plan (real Gymnasium
anchors, Minari datasets, OPE + regret vs a Cell-1 reference). That difference
drives most verdicts.

---

## Headline answers

### (a) Does `causal_dispatch`/`_causal_cell` already implement the `causal/` env-id registration pattern (gate decision 6)?

**Partially — same naming idea, wrong mechanism.** `causal_dispatch.build_causal_env`
routes `causal-*` string prefixes through the repo's own `EnvWrapperRegistry`
to custom `BaseEnv` subclasses. It never touches **Gymnasium registration**
(`gym.register`), so these envs are invisible to `gym.make`, vector wrappers,
and — critically — `minari.DataCollector`, which Phase 3+ requires.
Gate decision 6 needs true Gymnasium ids under a `causal/` namespace.

- **Adapt:** the id grammar (`causal-<family>-cell<N>`) and
  `_causal_cell.parse_cell_from_env_id` (clean, tested) — reuse the parsing
  convention for `causal/...` ids and YAML specs.
- **Discard:** the dispatcher itself; Phase 2 registers wrapped envs via
  `gymnasium.register(id="causal/...", entry_point=...)` + `ENV_SETS` entries.
- **Do NOT port `CELL_CONFIGS`:** the old table conflates *confounding* with
  *U-observability* and *contradicts the paper taxonomy* — old cells 4/6 have
  `z_exposed=True` whereas the paper's Cells 4 ("Burned Files") and 6
  ("Fog of History") have **Z hidden**; the old table also has no U→(A,R)
  confounding axis (cells 7–8 differ only by `alpha` plumbed elsewhere).
  `src/causal/cells.py` must be rebuilt from the paper table (§1), not ported.

### (b) Do `estimators.py`/`gap.py` contain reusable IPW/DM/gap logic for Phase 3 §6.5?

**Gap: yes. IPW/DM: no.**

- `estimators.py` — **port** (env-agnostic, clean, typed):
  - `plugin_tv`: tabular TV divergence — fine.
  - `mmd_gauss`: unbiased Gaussian-kernel MMD² with median-heuristic bandwidth
    — directly useful for the §6.4 oracle's *continuous* reward-distribution
    comparison (a better-behaved alternative alongside the spec'd KL/JS; keep
    both, report KL/JS as spec'd).
  - `dice_chi2`: clipped importance-ratio χ² over `behavior_logprob` — not an
    OPE value estimator, but a 10-line precedent for propensity-ratio
    plumbing and a useful *overlap diagnostic* for the §6.2 confounding gate
    (condition (i) support) and tier-coverage reporting.
- `gap.py` — **adapt**: `compute_gap` + `_divergence` (tv/kl/chi2/sup)
  implement exactly the secondary metric pipeline (`gap_kl`,
  `gap_js_normalized` columns, §6.6) but are coupled to the `CausalEnv` ABC
  (`env.do_reward`, `env.reward_support`, `supports_oracle`). Decouple to
  consume the new `src/eval/oracle.py` interface (set_state / analytic-state
  oracles over real envs). The silent broad `try/except` fallback to MMD must
  go (quality flag — masks oracle errors).
- **No IPW / DM / DR / FQE value estimators exist anywhere in the archive.**
  §6.5 (`NaiveEstimator`, `DirectMethod`/FQE, `IPWEstimator`, `DoublyRobust`
  returning `(value, ci)`) must be written fresh.

### (c) Does `biased_explorer.py` implement a usable U→action coupling for Phase 4 §6.2?

**Yes — adapt, with two changes.** `ConfoundedExplorer` implements
`logits' = β·logits + α·u·tanh(logits)` with **exact `behavior_logprob` under
the biased policy** (the §6.2 hard requirement) and a `latent=` injection
point that maps cleanly onto per-episode U.

- Change 1 — coupling form: §6.2 specifies `logits' = logits + β·U·e_{a*}`
  (bias a *single preferred action's* logit), not a tanh gate over all
  logits. Keep the class interface (`select_action(obs, latent) ->
  (action, logp)`), swap the formula.
- Change 2 — continuous axis: add a Gaussian behavior variant implementing
  `a' = clip(a + γ·U·v)` with exact Normal log-prob (needed for HalfCheetah,
  Cells 7–8 continuous).
- `UniformExplorer` / `EpsilonGreedyExplorer` — **port** nearly as-is: exact
  log-prob bookkeeping makes them the right logging policies for the
  CartPole `{simple, medium, expert}` tiers (§6.3, known π_b).

---

## Per-module verdicts

| Module | Purpose | Quality | Verdict |
|---|---|---|---|
| `causal_metrics/estimators.py` | TV / MMD² / χ²-ratio divergences | High; env-agnostic, typed | **port** (Phase 3, into `src/eval/`) |
| `causal_metrics/gap.py` | Δ_φ obs-vs-do reward gap (tv/kl/chi2/sup) | Good; bad broad `except` fallback; coupled to `CausalEnv` | **adapt** (Phase 3/6, retarget to new oracle; remove silent fallback) |
| `benchmarking/offline_collector.py` | biased collection + per-cell expose switches | OK; loop is fine but pre-Minari | **discard impl, keep convention**: collect ONCE with full info; apply cell switches (drop propensities / mask latent) at *load* time, exactly as §6.3 does via Minari infos |
| `rl/off_policy/biased_explorer.py` | behavior policies w/ exact log-probs; confounded variant | High | **adapt** (Phase 4, see (c)) |
| `rl/off_policy/confounded_dqn.py` | DQN + λ·Δ_φ oracle-regularized TD loss | OK code, but reads the env oracle *inside the learner* — violates the §8 no-oracle-leak rule; superseded by delphic as the Cells-7/8 variant; pre-ABC `update()` signature | **discard** |
| `envs/causal_base.py` | `CausalEnv` ABC: `latent_state`, `do_reward`, `do_transition` | Clean | **discard for anchors** (oracle lives in standalone `src/eval/oracle.py` over real envs); pattern may inform the oracle's interface |
| `envs/wrappers/sepsis.py` | vectorized sepsis-like tabular env, exact do-oracles | High (seeded `torch.Generator`, exact oracles) | **adapt as test fixture** (Phase 3): exact oracle ⇒ ground truth for unit-testing IPW/DR unbiasedness and gap estimators; NOT a paper experiment |
| `envs/wrappers/block_mdp.py` | synthetic Block-MDP, configurable leak ρ / confounding α | High | **adapt as test fixture** (same rationale) |
| `envs/wrappers/_causal_cell.py` | cell→exposure table + id parsing | Parsing good; **cell table contradicts paper taxonomy** (see (a)) | **adapt parsing only; discard `CELL_CONFIGS`** |
| `envs/wrappers/causal_dispatch.py` | string-prefix env routing | OK | **discard** (replaced by `gym.register` under `causal/`, gate decision 6) |
| `reproducibility/causal_8cells.yaml`, `causal_blockmdp_8cells.yaml` | old runner-integrated sweeps (`divergences:` key, `confounded_dqn`) | n/a | **discard** (superseded by §4.3 cell YAMLs); historical reference only |
| `docs/causal_extensions.md` | old cell table + conventions | n/a | **discard as spec** (taxonomy mismatch), keep as context |
| `tests/causal_metrics/test_mmd.py`, `test_plugin_tv.py` | estimator sanity tests | Good | **port with estimators** |
| `tests/causal_metrics/test_runner_integration.py` | runner `divergences` integration | references removed `TrainingConfig(divergences=...)` | **discard** |
| `tests/envs/test_{causal_base,sepsis,block_mdp}.py` | env contract tests | Good | **adapt with the fixtures** |
| `tests/rl/off_policy/test_biased_explorer.py` | explorer log-prob correctness | Good | **adapt with explorer** |
| `tests/rl/off_policy/test_{confounded_dqn,offline_collector}.py` | tests of discarded modules | n/a | **discard** |

## Summary of porting requests for the Phase-2 gate

1. Phase 2 (now): reuse only the `cell<N>` id-parsing convention.
2. Phase 3: port `estimators.py` (+2 tests) into `src/eval/`; adapt `gap.py`
   onto the new oracle; adapt sepsis/block-MDP as OPE-correctness test
   fixtures; port Uniform/EpsilonGreedy explorers for tier collection.
3. Phase 4: adapt `ConfoundedExplorer` to the §6.2 coupling (+ continuous
   variant).
4. Everything else: discard (kept in archive).
