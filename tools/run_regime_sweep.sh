#!/usr/bin/env bash
# tools/run_regime_sweep.sh — the (regime × L-sweep) driver.
#
# Replaces tools/run_cells_1234_parallel.sh (which globbed cell_{1,2,3,4}/*.yaml and
# the cell_{7,8}/*_gated.yaml special-case). NEW model: ONE cell = ONE job. The 7
# sweep points (an L: a shared basic origin + a biased arm + a confounded arm) are the
# inner loop and share ONE generator checkpoint per (env, seed) — so the arm deltas
# are PAIRED and never confounded by generator variance (PR 5, CHANGE 1). We do NOT
# pair across cells (different obs spaces make that impossible).
#
# Results land in a PARALLEL tree whose PATH SEGMENTS carry the parameters:
#   results/{regime}/beta_{beta*100:03d}_sigma_{sigma*100:03d}/{env}/{algo}/{critic}/{seed}/
# A leaf is an ordinary run dir. No current renderer reads this tree — PR 6 wires
# reporting; the legacy runs/rl_regimes/cell_N/ renderers keep serving _legacy/ runs.
# NO plotting phase here (that was the old script's Phase 2; it belongs to PR 6).
#
# Usage:
#   tools/run_regime_sweep.sh                       # offline_mdp + offline_pomdp
#   tools/run_regime_sweep.sh offline_mdp           # one cell
#   tools/run_regime_sweep.sh offline_mdp offline_pomdp
#
# Online cells (online_mdp / online_pomdp) have no offline generator to share — their
# behavior policy IS the learner — so the offline driver refuses them by design; run
# those through the on-policy benchmark path.

set -uo pipefail
cd "$(dirname "$0")/.."

CELLS=("$@")
if [ ${#CELLS[@]} -eq 0 ]; then
  CELLS=(offline_mdp offline_pomdp)
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p runs/_sweep_logs
MASTER_LOG="runs/_sweep_logs/regime_sweep_${TIMESTAMP}.log"
log() { echo "[$(date -Is)] $*" | tee -a "$MASTER_LOG"; }

log "=== regime sweep started (cells: ${CELLS[*]}) ==="
for cell in "${CELLS[@]}"; do
  yaml="reproducibility/rl_regimes/${cell}/sweep.yaml"
  if [ ! -f "$yaml" ]; then
    log "!!! SKIP ${cell}: ${yaml} not found"
    continue
  fi
  t0=$(date +%s)
  log ">>> START ${cell} (${yaml})"
  if uv run python -m src.benchmarking.regime_sweep "$yaml" >>"$MASTER_LOG" 2>&1; then
    log "<<< DONE  ${cell} ($(( $(date +%s) - t0 ))s)"
  else
    log "!!! FAIL  ${cell} (rc nonzero after $(( $(date +%s) - t0 ))s; see ${MASTER_LOG})"
  fi
done
log "=== regime sweep finished ==="
