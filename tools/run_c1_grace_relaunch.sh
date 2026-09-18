#!/usr/bin/env bash
# Relaunch of the c1 grace chain after the s1-k=1 memory-guard stall (2026-09-04).
# The mitigation (peer-ruled, no code change): return cached allocator blocks to
# the driver above 60% reserved so nbn's query_batch mem_get_info() guard sees
# truthful free memory and chunks sanely. Memory management only — served numbers
# unaffected (VE exact, chunk-independent, deterministic kernels). Frozen grace
# package untouched. Finished ds0 leaves + the stored k=0 entries skip/hit.
set -uo pipefail
export MINARI_DATASETS_PATH="${MINARI_DATASETS_PATH:-$HOME/.minari-grace-v2}"
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6
cd /home/giovanni-briglia/PycharmProjects/benchmarking_causal_rl
for W in c1_tmdp_grace_dmdp.yaml c1_tmdp_grace_dpomdp.yaml c1_tpomdp_grace_dmdp.yaml c1_tpomdp_grace_dpomdp.yaml; do
  echo "=== GRACE WAVE (mitigated): $W ($(date -Is))"
  uv run python tools/run_e1.py --campaign=c1 "--yamls=$W" || { echo "GRACE WAVE $W FAILED rc=$?"; exit 1; }
done
echo "=== GRACE CELLS COMPLETE ($(date -Is))"
