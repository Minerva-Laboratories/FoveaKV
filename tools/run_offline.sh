#!/usr/bin/env bash
# Run the offline study (NumPy, CPU) on one captured model, writing results/<tag>/.
# Usage: tools/run_offline.sh <tag> [experiments...]
#   needs FOVEAKV_MODEL and FOVEAKV_KV_DIR set for that model; FOVEAKV_LAYERS optionally restricts the layers.
# Each experiment runs through tools/guarded.sh: 4 GB cap (OFFLINE_CAP), and it backs off when other jobs need the memory.
set -u
cd "$(dirname "$0")/.."
tag=$1; shift
exps=${*:-"e01_gaussianization e02_04_selection e11_improvements e13_selection_criteria e14_weighted_mse e16_selection_gap e17_specache e18_static_mixed"}
export FOVEAKV_RESULTS="$PWD/results/$tag"
mkdir -p "$FOVEAKV_RESULTS"
for e in $exps; do
  echo "== $e $(date +%T)"
  CAP=${OFFLINE_CAP:-4G} START_MB=${OFFLINE_START_MB:-5000} tools/guarded.sh "$FOVEAKV_RESULTS/$e.log" python3 "exp/$e.py"
  echo "   exit $? $(grep -o 'runtime [0-9]*s\|time [0-9]*s' "$FOVEAKV_RESULTS/$e.log" | tail -1)"
done
