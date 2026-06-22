#!/usr/bin/env bash
# tools/run_cells_1234_parallel.sh — comprehensive paper-matrix sweep.
#
# Sweeps the current paper-matrix YAMLs across Cells 1, 2, 3, 4, 7, 8 at 8
# concurrent, then plots every newly-created run dir. Single-tier (the previous
# tiered 4+2+1 structure was conservative pacing; the 2026-06-19 diagnostic
# showed each leg is single-core because SyncVectorEnv steps its 16 envs
# serially in-process, so 8 concurrent legs ~= 8 cores used, ~12 cores headroom
# for the OS / ffmpeg eval-video encoding / editor / agent / desktop).
#
# Coverage (76 YAMLs):
#   - Cells 1, 2: all YAMLs (MLP + LSTM baselines + behavior strength sweeps).
#   - Cells 3, 4: all offline-tier YAMLs (discrete + continuous).
#   - Cells 7, 8: ONLY *_gated.yaml — the offline σ-wedge on CartPole+Acrobot
#     (PR #34's dense-reward finding) plus the PR #45 online σ-sweeps. The
#     non-gated and continuous σ-sweep YAMLs remain in the repo for reference
#     but are NOT paper-matrix (LunarLander's dense reward and the continuous
#     envs defeat the σ·U mechanism: Corr(A,R)~=0 at σ<=1), so they are excluded
#     from automated sweeps.
#   => 22 + 22 + 6 + 6 + 10 + 10 = 76.
#
# Continue-on-failure: a failed leg/plot is logged and the sweep proceeds — one
# bad config never blocks the matrix.
#
# Requires bash 4.3+ for `wait -n` (verified: laptop has 5.2.x).

set -uo pipefail
cd "$(dirname "$0")/.."

# --- Preflight: ensure the offline datasets THIS sweep consumes exist --------
# Derive the exact ids the sweep will run (same globs as Phase 1: cells 1-4 all
# YAMLs, cells 7-8 ONLY *_gated), build any missing via the resumable generator,
# and abort only if a REQUIRED id is still absent. We gate on the required set,
# not the generator's exit code: the generator builds a 56-id superset whose
# heaviest, most failure-prone slice (continuous- + LunarLander-confounded, 25
# ids) the sweep never touches, so those failures must not block a runnable run.
req=$(mktemp); present=$(mktemp); miss=$(mktemp)
grep -rhoE 'generated/[a-z0-9_]+/[a-z0-9_.-]+' \
  reproducibility/rl_regimes/cell_{1,2,3,4}/*.yaml \
  reproducibility/rl_regimes/cell_{7,8}/*_gated.yaml 2>/dev/null | sort -u >"$req"
n_req=$(wc -l <"$req")
recheck() {
  uv run python -c "import minari;[print(d) for d in minari.list_local_datasets()]" \
    2>/dev/null | sort -u >"$present"
  comm -23 "$req" "$present" >"$miss"
}
recheck
if [ -s "$miss" ]; then
  echo "[preflight] missing offline datasets:"; sed 's/^/    /' "$miss"
  echo "[preflight] building via tools/generate_all_datasets.sh (resumable)..."
  bash tools/generate_all_datasets.sh || true   # superset build; required set verified below
  recheck
  if [ -s "$miss" ]; then
    echo "[preflight] STILL missing REQUIRED datasets after generation — aborting:" >&2
    sed 's/^/    /' "$miss" >&2; rm -f "$req" "$present" "$miss"; exit 1
  fi
fi
rm -f "$req" "$present" "$miss"
echo "[preflight] all $n_req required offline datasets present; proceeding."
# ----------------------------------------------------------------------------

SWEEP_CONCURRENCY=8
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SWEEP_START_EPOCH=$(date +%s)
MASTER_LOG="runs/_sweep_logs/cells_all_parallel_master_${TIMESTAMP}.log"
mkdir -p runs/_sweep_logs

log() {
  echo "[$(date -Is)] $*" | tee -a "$MASTER_LOG"
}

# Run one YAML to its own log; record DONE/FAIL + wall-clock in the master log.
# Never returns non-zero (continue-on-failure relies on this swallowing the
# leg's exit code).
run_leg() {
  local yaml="$1"
  local base leg_log t0 dt
  base=$(echo "$yaml" | sed 's|/|_|g')
  leg_log="runs/_sweep_logs/parallel_${base}_${TIMESTAMP}.log"
  log ">>> START $yaml (log: $leg_log)"
  t0=$(date +%s)
  if uv run python main.py --reproduce "$yaml" >"$leg_log" 2>&1; then
    dt=$(( $(date +%s) - t0 ))
    log "<<< DONE  $yaml (${dt}s)"
  else
    dt=$(( $(date +%s) - t0 ))
    log "!!! FAIL  $yaml (rc nonzero after ${dt}s; see $leg_log)"
  fi
}

# Launch a list of YAMLs with bounded concurrency. `wait -n` releases a slot as
# soon as ANY in-flight leg finishes (a slow leg never stalls the others); a
# trailing `wait` drains before return.
run_tier() {
  local concurrency="$1"; shift
  local yamls=("$@")
  local running=0 yaml
  for yaml in "${yamls[@]}"; do
    run_leg "$yaml" &
    running=$((running + 1))
    if (( running >= concurrency )); then
      wait -n
      running=$((running - 1))
    fi
  done
  wait
}

# ============================================================================
# Phase 1 — Sweep
# ============================================================================

log "=== Comprehensive sweep started (timestamp ${TIMESTAMP}) ==="

ALL_YAMLS=()
# Cells 1, 2, 3, 4 — all YAMLs.
for cell in cell_1 cell_2 cell_3 cell_4; do
  for y in reproducibility/rl_regimes/$cell/*.yaml; do
    rel="${y#reproducibility/}"
    rel="${rel%.yaml}"
    ALL_YAMLS+=("$rel")
  done
done
# Cells 7, 8 — only the *_gated families (offline σ-wedge + online σ-sweeps).
# PR #34 found LunarLander's dense reward and the continuous envs defeat the
# σ·U mechanism (Corr(A,R)~=0), so the σ-wedge experiments gate on the
# unit-reward-scale envs (CartPole, Acrobot). The non-gated and continuous
# σ-sweep YAMLs stay in the repo for reference but are not part of the matrix.
for cell in cell_7 cell_8; do
  for y in reproducibility/rl_regimes/$cell/*_gated.yaml; do
    rel="${y#reproducibility/}"
    rel="${rel%.yaml}"
    ALL_YAMLS+=("$rel")
  done
done

log "--- Sweep: ${#ALL_YAMLS[@]} YAMLs at ${SWEEP_CONCURRENCY} concurrent ---"
run_tier "$SWEEP_CONCURRENCY" "${ALL_YAMLS[@]}"

log "=== Sweep finished ==="
log "--- per-leg summary (DONE/FAIL) ---"
grep -E "<<< DONE|!!! FAIL" "$MASTER_LOG" | tail -100

# ============================================================================
# Phase 2 — Plotting
# ============================================================================
# Render figures for every run dir created during THIS sweep (mtime filter on
# SWEEP_START_EPOCH avoids re-plotting pre-existing runs). The plotter's --run
# takes a path relative to runs/ with no trailing slash. Continue-on-failure.

log "=== Plotting phase: rendering figures for sweep run dirs ==="
PLOT_START=$(date +%s)
PLOT_SUCCESS=0
PLOT_FAIL=0
PLOT_SKIPPED=0

for run_dir in runs/rl_regimes/cell_*/*/; do
  if [ "$(stat -c %Y "$run_dir")" -lt "$SWEEP_START_EPOCH" ]; then
    PLOT_SKIPPED=$((PLOT_SKIPPED + 1))
    continue
  fi
  name=${run_dir#runs/}
  name=${name%/}
  if uv run python -m src.benchmarking.plotting --run "$name" --split all >>"$MASTER_LOG" 2>&1; then
    PLOT_SUCCESS=$((PLOT_SUCCESS + 1))
  else
    PLOT_FAIL=$((PLOT_FAIL + 1))
    log "!!! PLOT FAIL  $name (see master log)"
  fi
done

PLOT_DT=$(( $(date +%s) - PLOT_START ))
log "=== Plotting phase done: $PLOT_SUCCESS success, $PLOT_FAIL fail, $PLOT_SKIPPED skipped (${PLOT_DT}s) ==="

log "=== Comprehensive sweep + plotting finished ==="
log "--- plotting summary ---"
log "Plots: $PLOT_SUCCESS success, $PLOT_FAIL fail, $PLOT_SKIPPED skipped (${PLOT_DT}s)"
