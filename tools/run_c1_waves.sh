#!/usr/bin/env bash
# The c1 contract grid, launched in cache-priming wave order (ruled
# 2026-09-04) with an S7 completion invariant and a GPU pre-flight between
# waves. Each wave is a set of cell YAMLs; assertions always run on the FULL
# plan. Usage:  bash tools/run_c1_waves.sh  (from the repo root, on GO only)
set -euo pipefail
export MINARI_DATASETS_PATH="${MINARI_DATASETS_PATH:-$HOME/.minari-grace-v2}"

WAVES=(
  "c1_tmdp_base.yaml,c1_tmdp_grace_dmdp.yaml"
  "c1_tmdp_grace_dpomdp.yaml"
  "c1_tpomdp_base.yaml,c1_tpomdp_grace_dmdp.yaml,c1_tpomdp_grace_dpomdp.yaml"
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

# Cache pre-flight: every stored entry's code_version must match the launch
# tree, else "hits" silently become re-fits and the cost projection lies.
uv run python - <<'PY'
import json, sys
from pathlib import Path
from src.rl.offline.grace.transform_cache import code_version
cv = code_version()
stale = []
for k in Path("results/grace_cache").glob("*/key.json"):
    if json.loads(k.read_text()).get("code_version") != cv:
        stale.append(str(k.parent.name))
if stale:
    print(f"[cache pre-flight] STALE entries (code_version mismatch): {stale}")
    print("  These will MISS and re-fit. Re-run the reference into the cache "
          "on the launch tree, or accept the re-fit cost knowingly.")
else:
    print(f"[cache pre-flight] all entries current (code_version {cv[:12]})")
PY

for i in "${!WAVES[@]}"; do
  w="${WAVES[$i]}"
  echo "=== WAVE $((i+1)): $w  ($(date -Is))"
  gpu_free
  uv run python tools/run_e1.py --campaign=c1 "--yamls=$w"
  # S7: surviving leaves == the filtered plan, checked BEFORE the next wave.
  uv run python - "$w" <<'PY'
import sys
from pathlib import Path
from src.benchmarking.regime_sweep import results_leaf
from tools.run_e1 import CAMPAIGN_ROOTS, REGIME, enumerate_plan
from src.benchmarking.regime_sweep import arm_label

want = set(sys.argv[1].split(","))
missing = []
for e in enumerate_plan("c1"):
    if e["yaml"] not in want:
        continue
    spec = e["spec"]
    critics = (
        [e["arm"]] if False else
        (list(spec.critics_for(arm_label(0.0, e["sigma"]))))
    )
    for c in critics:
        leaf = results_leaf(
            CAMPAIGN_ROOTS["c1"], f"{REGIME}_{e['tag']}", 0.0, e["sigma"],
            e["env"], e["algo"], c, e["seed_segment"],
        )
        if not (leaf / "eval_metrics.csv").exists():
            missing.append(str(leaf))
if missing:
    print(f"WAVE INCOMPLETE: {len(missing)} expected leaves missing, e.g. {missing[:3]}")
    sys.exit(1)
print("wave complete: every expected leaf present")
PY
done
echo "=== ALL WAVES COMPLETE ($(date -Is))"
