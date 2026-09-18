#!/usr/bin/env bash
# C1 base cells, iv' shape (ruled 2026-09-04), launched AFTER the grace chain
# finishes — GPU alone. Waves per the ruling: (5) tmdp critics+base,
# (6) tpomdp critics+base, (7) the two sigma-0 anchors. Finished ds0_ts0
# critic leaves skip (verified). GPU pre-flight + S7 completion invariant
# between waves. Run from repo root, on the grace chain's completion.
set -uo pipefail   # NOT -e: a fallback returning non-zero must not abort
export MINARI_DATASETS_PATH="${MINARI_DATASETS_PATH:-$HOME/.minari-grace-v2}"
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6   # memory-guard mitigation (2026-09-04)

WAVES=(
  "c1_tmdp_critics.yaml,c1_tmdp_base.yaml"
  "c1_tpomdp_critics.yaml,c1_tpomdp_base.yaml"
  "c1_tmdp_base_s0.yaml,c1_tpomdp_base_s0.yaml"
)

gpu_free() {
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  if [ "$used" -gt 500 ]; then
    echo "GPU pre-flight FAILED: ${used} MiB in use — the grid runs alone." >&2
    exit 1
  fi
}

for i in "${!WAVES[@]}"; do
  w="${WAVES[$i]}"
  echo "=== BASE WAVE $((i+5)): $w  ($(date -Is))"
  gpu_free
  uv run python tools/run_e1.py --campaign=c1 "--yamls=$w"
  uv run python - "$w" <<'PY'
import sys
from pathlib import Path
from src.benchmarking.regime_sweep import results_leaf, arm_label
from tools.run_e1 import CAMPAIGN_ROOTS, REGIME, enumerate_plan
want = set(sys.argv[1].split(","))
missing = []
for e in enumerate_plan("c1"):
    if e["yaml"] not in want:
        continue
    for c in e["spec"].critics_for(arm_label(0.0, e["sigma"])):
        leaf = results_leaf(CAMPAIGN_ROOTS["c1"], f"{REGIME}_{e['tag']}", 0.0,
                            e["sigma"], e["env"], e["algo"], c, e["seed_segment"])
        if not (leaf / "eval_metrics.csv").exists():
            missing.append(str(leaf))
if missing:
    print(f"BASE WAVE INCOMPLETE: {len(missing)} leaves missing, e.g. {missing[:3]}")
    sys.exit(1)
print("base wave complete: every expected leaf present")
PY
done
echo "=== ALL BASE WAVES COMPLETE ($(date -Is))"
