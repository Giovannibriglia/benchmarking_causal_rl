#!/usr/bin/env bash
# GRACE Night 1 (feat/grace-critic, approved schedule D6(a)):
#   E1 = offline_mdp critic_ablation (full L, 6 critic arms, production budget)
#   E2 = offline_mdp classical_grace_returns (return-level G2 leg)
# Sequential; each block's exit code is logged. Launch DETACHED (setsid nohup)
# — harness-tracked background jobs can be culled mid-run (documented gotcha).
# PREREQUISITE: tools/calibrate_grace_router.py has written
# _base/grace_router_reference.yaml (the router reads it at fit time).
set -u
cd "$(dirname "$0")/.."
LOG_DIR=runs/_sweep_logs
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/grace_night1_$STAMP.log"
{
  if [ ! -f reproducibility/rl_regimes/_base/grace_router_reference.yaml ]; then
    echo "[night1] ABORT: grace_router_reference.yaml missing (run E0 first)"
    exit 2
  fi
  echo "[night1] E1 offline_mdp critic_ablation START $(date)"
  uv run python main.py --reproduce rl_regimes/offline_mdp/critic_ablation.yaml
  E1=$?
  echo "[night1] E1 exit=$E1 $(date)"
  echo "[night1] E2 classical_grace_returns START $(date)"
  uv run python main.py --reproduce rl_regimes/offline_mdp/classical_grace_returns.yaml
  E2=$?
  echo "[night1] E2 exit=$E2 $(date)"
  echo "[night1] DONE e1=$E1 e2=$E2 $(date)"
} >>"$LOG" 2>&1
