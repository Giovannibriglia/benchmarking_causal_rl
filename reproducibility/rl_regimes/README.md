# rl_regimes — composable cells

Four cells (regime × observability), each shipping **two simulation components**
with a production YAML + a tiny-budget smoke:

| cell | classical (algo × env benchmark) | critic ablation (strategy comparison) |
|---|---|---|
| `offline_mdp`   | `classical.yaml` / `classical_smoke.yaml` | `critic_ablation.yaml` / `critic_ablation_smoke.yaml` |
| `offline_pomdp` | `classical.yaml` / `classical_smoke.yaml` | `critic_ablation.yaml` / `critic_ablation_smoke.yaml` |
| `online_mdp`    | `classical.yaml` / `classical_smoke.yaml` | `critic_ablation.yaml` / `critic_ablation_smoke.yaml` |
| `online_pomdp`  | `classical.yaml` / `classical_smoke.yaml` | `critic_ablation.yaml` / `critic_ablation_smoke.yaml` |

(`sweep.yaml` / `sweep_smoke.yaml` are legacy aliases of the critic-ablation
simulation and stay runnable.)

## Run

```bash
# any cell, either simulation, one command (smoke first!):
uv run python main.py --reproduce rl_regimes/online_mdp/critic_ablation_smoke.yaml
uv run python main.py --reproduce rl_regimes/online_mdp/critic_ablation.yaml

# equivalent module form with extra flags (--max-workers, --envs, --seeds, --device):
uv run python -m src.benchmarking.regime_sweep \
  reproducibility/rl_regimes/offline_mdp/classical.yaml --max-workers 4
```

Smoke files route to `results_smoke/` (keyed on "smoke" in the filename);
production lands in `results/`.

## What each simulation runs

* **classical** — every algo on every env at every point of the L
  (basic origin + biased β-arm + confounded σ-arm), NO critic axis. Leaves:
  `results/{regime}/classical/beta_*_sigma_*/{env}/{algo}/{seed}/`. Every
  classical config compares **at least two** designed-for-regime algo rows
  (offline_mdp: the four offline learners; offline/online_pomdp: recurrent
  learner vs an explicit memoryless `name__mlp__mlp` baseline on the same
  masked obs; online_mdp: `dqn` vs `online_dqn_proximal`).
* **critic_ablation** — the identification-strategy comparison at every point.
  Offline: the arm's critic set (`observational/proximal/oracle_u/sensitivity`)
  trains on ONE shared episode-grouped stream and explodes into per-critic
  leaves. Online: strategies are ALGO VARIANTS (observational = `dqn`,
  proximal = `online_dqn_proximal`); `online_pomdp` is observational-only (no
  recurrent online proximal exists). Leaves:
  `results/{regime}/beta_*_sigma_*/{env}/{algo}/{critic}/{seed}/`.

The `sweep:` and `critics:` blocks are **real, validated config** (off-L
declarations and unknown/unavailable strategies are refused). `envs`, `seeds`,
`budgets`, `max_workers` inherit from `_base/*.yaml`; a cell key wins. **`algos`
is declared explicitly in every cell file** (never inherited) and validated
against the registry: an entry whose `data_regime` doesn't match the cell is
refused. Entries are `"name"` (auto trunks: mlp on mdp, lstm critic on pomdp)
or the explicit `"name__actor__critic"` form (e.g. `dqn__mlp__mlp` pins a
memoryless baseline in a pomdp cell; the leaf path uses the entry verbatim).

## Parallelism

`max_workers` = how many (env, seed) GROUPS run concurrently (one subprocess
owns a group's whole L, so the shared-generator invariant is preserved; offline
workers get isolated Minari stores). 1 = serial, byte-identical ordering; the
production YAMLs default to 4 (memory-safe on a 16 GB GPU). Override per run
with `--max-workers` on the module form. `main.py --reproduce` honors the
cell's `max_workers`.

## Reports & figures

```bash
# classical:
uv run python -m src.benchmarking.regime_report <regime> --simulation classical
uv run python -m src.benchmarking.render_classical_report <regime>
# critic ablation (+ null-calibration gate):
uv run python -m src.benchmarking.regime_report <regime>
uv run python -m src.benchmarking.render_regime_report <regime>
```

Tables land in `<results-root>/_report/`, figures in
`<results-root>/_report/figures/` (stems are regime-prefixed).
