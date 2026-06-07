#!/usr/bin/env bash
# Hardened discrete-matrix sweep harness (post-interrupt direction, 2026-06-06):
# - pipefail + fail-fast on the first failing spec
# - full output tee'd to runs/sweep_<ts>/sweep.log (no live grep filtering)
# - post-hoc validation + aggregation via tools.aggregate_matrix
set -euo pipefail

PY=.venv/bin/python
TS=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR="runs/sweep_${TS}"
mkdir -p "$SWEEP_DIR"
LOG="$SWEEP_DIR/sweep.log"
SPECS=(cell3_cartpole cell3_cartpole_h50 cell4_cartpole cell5_cartpole
       cell6_cartpole cell7_cartpole cell8_cartpole)

run_dirs=()
for spec in "${SPECS[@]}"; do
    echo "=== $spec ===" | tee -a "$LOG"
    $PY main.py --mode causal_cells --reproduce "causal_cells/$spec" 2>&1 | tee -a "$LOG"
    run_dirs+=("$(ls -td runs/causal_cells_${spec}_* | head -1)")
done

echo "=== cell2 online sweep ===" | tee -a "$LOG"
for s in 0 1 2 3 4; do
    $PY main.py --envs causal/cartpole-cell2 causal/cartpole-cell2fs \
        --algos ppo --n-episodes 150 --rollout-len 512 \
        --n-train-envs 8 --n-eval-envs 8 --seed "$s" --deterministic \
        2>&1 | tee -a "$LOG"
done

echo "=== validate + aggregate ===" | tee -a "$LOG"
$PY -m tools.aggregate_matrix "${run_dirs[@]}" 2>&1 | tee -a "$LOG"
echo "SWEEP COMPLETE: $SWEEP_DIR" | tee -a "$LOG"
