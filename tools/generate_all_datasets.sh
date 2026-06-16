#!/usr/bin/env bash
# generate_all_datasets.sh — resumable generation of the 56 Cell 3 + Cell 7
# offline datasets for the eight-cell causal-RL matrix.
#
#   Cell 3 (unconfounded, 21): 3 discrete + 4 continuous envs x 3 tiers.
#   Cell 7 (confounded, 35):   3 discrete + 4 continuous envs x 5 sigma (random tier).
#   Cells 4 and 8 REUSE these (mask applied at load time) — no extra generation.
#
# Resumable: each dataset is skipped if already present in the local Minari
# cache, so re-running after a Ctrl-C / crash picks up exactly where it stopped.
# Progress (timestamp | id | skip/done | wall-clock) is appended to the log.
#
# Place at repo root as tools/generate_all_datasets.sh and run:  bash tools/generate_all_datasets.sh
set -uo pipefail
cd "$(dirname "$0")/.."                 # repo root (script lives in tools/)

LOG="tools/.generation_progress.log"
GEN=(uv run python tools/generate_offline.py)

DISC_ENVS=(CartPole-v1 LunarLander-v3 Acrobot-v1)
CONT_ENVS=(Pendulum-v1 HalfCheetah-v5 Hopper-v5 Walker2d-v5)
TIERS=(random medium expert)
SIGMAS=(0.0 0.25 0.5 0.75 1.0)
declare -A NNN=([0.0]=000 [0.25]=025 [0.5]=050 [0.75]=075 [1.0]=100)
DISC_GEN=dqn ; CONT_GEN=sac

slug() { echo "${1%-v*}" | tr '[:upper:]' '[:lower:]'; }   # CartPole-v1 -> cartpole

# Build the work list IN ORDER: discrete-unconf, cont-unconf, disc-conf, cont-conf.
IDS=() ; ARGS=()
add() { IDS+=("$1") ; ARGS+=("$2") ; }
for e in "${DISC_ENVS[@]}"; do s=$(slug "$e"); for t in "${TIERS[@]}"; do
  add "generated/$s/$t-v0" "--env $e --algo $DISC_GEN --offline-tier $t"; done; done
for e in "${CONT_ENVS[@]}"; do s=$(slug "$e"); for t in "${TIERS[@]}"; do
  add "generated/$s/$t-v0" "--env $e --algo $CONT_GEN --offline-tier $t"; done; done
for e in "${DISC_ENVS[@]}"; do s=$(slug "$e"); for sig in "${SIGMAS[@]}"; do
  add "generated/$s/random-bias_confounded-sigma${NNN[$sig]}-v0" \
      "--env $e --algo $DISC_GEN --offline-tier random --behavior-policy bias_confounded --behavior-strength $sig"; done; done
for e in "${CONT_ENVS[@]}"; do s=$(slug "$e"); for sig in "${SIGMAS[@]}"; do
  add "generated/$s/random-bias_confounded-sigma${NNN[$sig]}-v0" \
      "--env $e --algo $CONT_GEN --offline-tier random --behavior-policy bias_confounded --behavior-strength $sig"; done; done

TOTAL=${#IDS[@]}

# Snapshot the existing local datasets once (fast membership test).
EXISTING=$(mktemp)
uv run python -c "import minari;[print(d) for d in minari.list_local_datasets()]" >"$EXISTING" 2>/dev/null
have() { grep -qxF "$1" "$EXISTING"; }

present=0; for id in "${IDS[@]}"; do have "$id" && present=$((present+1)); done
remain=$((TOTAL - present))
echo "=== Starting generation: $remain of $TOTAL datasets remain ($present already present) ==="
echo "$(date -Is) | START | $remain of $TOTAL remain" >>"$LOG"

gen=0; skip=0; fail=0; t_start=$(date +%s)
for i in "${!IDS[@]}"; do
  id="${IDS[$i]}"; args="${ARGS[$i]}"
  if have "$id"; then
    printf 'SKIP  [%2d/%2d] %s\n' "$((i+1))" "$TOTAL" "$id"
    echo "$(date -Is) | SKIP | $id" >>"$LOG"; skip=$((skip+1)); continue
  fi
  printf 'GEN   [%2d/%2d] %s\n' "$((i+1))" "$TOTAL" "$id"
  t0=$(date +%s)
  if "${GEN[@]}" $args >>"$LOG" 2>&1; then
    dt=$(( $(date +%s) - t0 ))
    printf 'DONE  [%2d/%2d] %s  (%ds)\n' "$((i+1))" "$TOTAL" "$id" "$dt"
    echo "$(date -Is) | DONE | $id | ${dt}s" >>"$LOG"
    gen=$((gen+1)); echo "$id" >>"$EXISTING"   # mark present for the in-run summary
  else
    printf 'FAIL  [%2d/%2d] %s  (see %s) — continuing\n' "$((i+1))" "$TOTAL" "$id" "$LOG"
    echo "$(date -Is) | FAIL | $id" >>"$LOG"; fail=$((fail+1))
  fi
done

total_t=$(( $(date +%s) - t_start ))
echo "=== Generated $gen, skipped $skip, failed $fail of $TOTAL; total wall-clock ${total_t}s ($((total_t/60))m) ==="
echo "$(date -Is) | END | gen=$gen skip=$skip fail=$fail total=${total_t}s" >>"$LOG"
rm -f "$EXISTING"
[ "$fail" -eq 0 ]
