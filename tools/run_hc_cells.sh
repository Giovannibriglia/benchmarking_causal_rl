#!/usr/bin/env bash
# Hardened HC continuous-cell sweep (fail-fast; a variant crash STOPS the
# chain instead of silently skipping - the Step-2 lesson). Cells 7/8 require
# the gated confounded dataset; all cells use the constant normalizer (no
# per-seed reference training).
set -euo pipefail

PY=.venv/bin/python
TS=$(date +%Y%m%d_%H%M%S)
LOG="runs/hc_cells_sweep_${TS}.log"
CELLS=("$@")
[ ${#CELLS[@]} -eq 0 ] && CELLS=(3 4 5 6 7 8)

for c in "${CELLS[@]}"; do
    echo "=== HC cell $c ===" | tee -a "$LOG"
    $PY main.py --mode causal_cells \
        --reproduce "causal_cells/cell${c}_halfcheetah" 2>&1 | tee -a "$LOG"
done
echo "HC-CELLS-SWEEP-DONE: ${CELLS[*]}" | tee -a "$LOG"
