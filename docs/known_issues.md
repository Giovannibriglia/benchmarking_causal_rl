# Known issues (pre-existing on master — documented, NOT fixed by the causal-RL refactor)

## 1. `reproducibility/comoreai26.yaml` references unregistered algorithms

`comoreai26.yaml` lists `algos: a2c vanilla a2c_cc vanilla_cc`, but
`register_default_algorithms()` registers only
`vanilla | a2c | ppo | trpo | dqn | ddpg`. A full
`python main.py --reproduce comoreai26` therefore completes the `a2c` and
`vanilla` sweeps over the `mujoco` env set, then crashes with
`KeyError: "Algorithm a2c_cc not registered"`.

The `a2c_cc` / `vanilla_cc` ("causal critic") implementations existed in an
earlier, uncommitted version of the codebase (orphaned `__pycache__` artifacts
for `src/causal_metrics/{estimators,gap}.py`, `src/envs/causal_base.py` etc.
are the remaining traces) and were removed before commit `a98359b`.

**Status (Phase-0 gate decision, 2026-06-05):** the reproducibility invariant
for this file is downgraded to *"YAML byte-untouched + the a2c/vanilla portion
runs"*. The missing algorithms are pre-existing breakage and are explicitly
out of scope for this project.

## 2. Reproduce-YAML keys override CLI flags

Precedence is `reproduce-YAML > CLI` for every key present in the YAML, e.g.
`python main.py --reproduce comoreai26 --n-episodes 2` runs **250** episodes
(`training.n_episodes` from the YAML wins). This is master semantics and is
preserved by decision at the Phase-0 gate. The pre-merge verification command
is amended accordingly:

```bash
# verify comoreai26 config-loads and starts training (kill after startup);
# CLI episode overrides are intentionally ineffective under --reproduce
timeout 150 python main.py --reproduce comoreai26
```

## 3. Unmodified master is not run-to-run reproducible

Policy weights are initialized in `BenchmarkRunner.__init__` before
`set_seed()` runs in `BenchmarkRunner.run()`. Sanctioned semantic change at
the Phase-0 gate: Phase 1 moves seeding before network construction; golden
values (`tests/golden/`) were generated with the equivalent pre-seeded probe
(`tests/_phase0_seed_probe.py`), whose runs were verified bitwise identical.
The golden command is:

```bash
PYTHONPATH=. python tests/_phase0_seed_probe.py \
    --envs CartPole-v1 --algos ppo --n-episodes 5 --deterministic [--ablation]
```
