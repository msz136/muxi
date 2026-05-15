#!/bin/bash
# 一键评估脚本：运行所有 pointing 相关 benchmark

set -e

CHECKPOINT=${1:-"/data/checkpoints/opd_pointing_expert/best"}
OUTPUT_DIR=${2:-"./evaluation/results"}

echo "=== Full Evaluation Suite ==="
echo "Checkpoint: ${CHECKPOINT}"
echo "Output: ${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

# 1. RoboPoint-Test (Full)
echo ""
echo "[1/4] RoboPoint-Test (Full)..."
python evaluation/eval_pointing.py \
    --predictions "${OUTPUT_DIR}/robopoint_predictions.jsonl" \
    --output "${OUTPUT_DIR}/robopoint_metrics.json" \
    --thresholds 50 100 150

# 2. ViewSpatial-Bench
echo ""
echo "[2/4] ViewSpatial-Bench..."
python -m benchmarks.viewspatial_eval \
    --model-path "${CHECKPOINT}" \
    --data-path "/data/benchmarks/ViewSpatial-Bench/test.jsonl" \
    --output "${OUTPUT_DIR}/viewspatial_metrics.json"

# 3. Format Check
echo ""
echo "[3/4] Format Correctness..."
python evaluation/eval_format.py \
    --predictions "${OUTPUT_DIR}/robopoint_predictions.jsonl" \
    --output "${OUTPUT_DIR}/format_metrics.json"

# 4. General VLM Regression (MMBench)
echo ""
echo "[4/4] MMBench Regression..."
python -m benchmarks.mmbench_eval \
    --model-path "${CHECKPOINT}" \
    --output "${OUTPUT_DIR}/mmbench_metrics.json"

# 汇总
echo ""
echo "=== Summary ==="
python -c "
import json, glob
for f in sorted(glob.glob('${OUTPUT_DIR}/*_metrics.json')):
    name = f.split('/')[-1].replace('_metrics.json', '')
    with open(f) as fp:
        m = json.load(fp)
    print(f'  {name}:')
    for k, v in m.items():
        print(f'    {k}: {v:.4f}' if isinstance(v, float) else f'    {k}: {v}')
"

echo ""
echo "=== Evaluation Complete ==="
