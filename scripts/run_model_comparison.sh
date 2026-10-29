#!/usr/bin/env bash
# Equal-budget fine-tuning of the five paper models, one process per model, then the comparison report.
# Safe to re-run: finished folds and activation caches are reused. Usage:
#   bash scripts/run_model_comparison.sh data/ct-kidney [threads]
set -euo pipefail
cd "$(dirname "$0")/.."
DATA="${1:-data/ct-kidney}"
THREADS="${2:-6}"
for MODEL in MobileNet ResNet50 InceptionV3 Xception VGG16; do
  echo "=== $(date '+%F %T') $MODEL"
  python3 scripts/run_stagewise.py --data-dir "$DATA" --models "$MODEL" --threads "$THREADS" --nice 15 --cache-batch 16
done
echo "=== $(date '+%F %T') comparison report"
python3 scripts/compare_models.py --data-dir "$DATA"
echo "=== $(date '+%F %T') ALL DONE"
