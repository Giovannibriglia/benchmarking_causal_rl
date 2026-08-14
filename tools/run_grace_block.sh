#!/usr/bin/env bash
# Run ONE GRACE experiment block (the unit that replaced the night schedule).
#
#   bash tools/run_grace_block.sh <reproduce-path> [label]
#
# e.g. bash tools/run_grace_block.sh rl_regimes/offline_mdp/critic_ablation_grace_e1.yaml blockA
#
# Blocks NEVER chain: each ends in a push and a relay pause. One block per
# invocation, deliberately.
#
# Suspension: the whole run is wrapped in `systemd-inhibit --what=sleep:idle`
# so an idle laptop cannot suspend mid-block. The Aug-12 run lost most of a
# night to 46 suspend/resume cycles (each one pauses every worker). A manual
# lid-close can still override an inhibitor — disable auto-suspend too for
# long blocks.
#
# Launch DETACHED so a harness cull cannot kill it:
#   setsid nohup bash tools/run_grace_block.sh <cfg> <label> >/dev/null 2>&1 &
set -u
cd "$(dirname "$0")/.."
CFG="${1:?usage: run_grace_block.sh <reproduce-path> [label]}"
LABEL="${2:-block}"
LOG_DIR=runs/_sweep_logs
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/grace_${LABEL}_$STAMP.log"

INHIBIT=(systemd-inhibit --what=sleep:idle --who=grace-block \
         --why="GRACE experiment block $LABEL" --mode=block)
command -v systemd-inhibit >/dev/null 2>&1 || INHIBIT=()

{
  echo "[$LABEL] START $(date)"
  echo "[$LABEL] config=$CFG commit=$(git rev-parse --short HEAD) branch=$(git branch --show-current)"
  if [ "${#INHIBIT[@]}" -eq 0 ]; then
    echo "[$LABEL] WARNING: systemd-inhibit unavailable — suspends will stall this run"
  fi
  # The router must be calibrated before any defect cell is scored (E0).
  if [ ! -f reproducibility/rl_regimes/_base/grace_router_reference.yaml ]; then
    echo "[$LABEL] ABORT: grace_router_reference.yaml missing (run E0 first)"
    exit 2
  fi
  "${INHIBIT[@]}" uv run python main.py --reproduce "$CFG"
  RC=$?
  echo "[$LABEL] exit=$RC $(date)"
  echo "[$LABEL] DONE $(date)"
} >>"$LOG" 2>&1
