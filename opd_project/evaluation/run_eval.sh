#!/usr/bin/env bash
# OPD Pointing Expert - 评估脚本 (集群版)
# 评估模型在 pointing 任务上的表现
#
# 用法: bash /data/msz/opd_project/evaluation/run_eval.sh [checkpoint_path]
set -euo pipefail

ROOT="/data/msz/opd_project"
CHECKPOINT="${1:-/data/msz/models/Qwen3-VL-8B-Instruct}"
OUTPUT_DIR="${ROOT}/evaluation/results/$(basename ${CHECKPOINT})_$(date +%Y%m%d_%H%M%S)"

mkdir -p "${OUTPUT_DIR}"

echo "============================================"
echo " OPD Pointing Expert - Evaluation"
echo " Checkpoint: ${CHECKPOINT}"
echo " Output: ${OUTPUT_DIR}"
echo "============================================"

# Step 1: 生成 RoboPoint 预测
echo ""
echo "[1/3] Generating predictions on eval subset..."
python3 << EOF
import json
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from PIL import Image
import os

CHECKPOINT = "${CHECKPOINT}"
EVAL_DATA = "${ROOT}/data/eval_robopoint_500.jsonl"
OUTPUT = "${OUTPUT_DIR}/robopoint_predictions.jsonl"

print(f"Loading model: {CHECKPOINT}")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    CHECKPOINT,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(CHECKPOINT)
model.eval()

results = []
with open(EVAL_DATA) as f:
    items = [json.loads(line) for line in f if line.strip()]

print(f"Evaluating {len(items)} samples...")
for i, item in enumerate(items):
    if i % 50 == 0:
        print(f"  {i}/{len(items)}")

    try:
        # Load image
        img_path = item["images"][0]
        if not os.path.exists(img_path):
            results.append({"response": "", "gt_points": item["gt_points"]})
            continue

        image = Image.open(img_path).convert("RGB")

        # Build messages
        messages = item["messages"]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs = processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )

        # Decode only new tokens
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        response = processor.decode(generated, skip_special_tokens=True)

        results.append({
            "response": response,
            "gt_points": item["gt_points"],
        })

    except Exception as e:
        results.append({"response": "", "gt_points": item.get("gt_points", [])})

# Save predictions
with open(OUTPUT, "w") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Predictions saved: {OUTPUT}")
print(f"Total: {len(results)}")
EOF

# Step 2: 计算 pointing 指标
echo ""
echo "[2/3] Computing pointing metrics..."
python3 "${ROOT}/evaluation/eval_pointing.py" \
    --predictions "${OUTPUT_DIR}/robopoint_predictions.jsonl" \
    --output "${OUTPUT_DIR}/pointing_metrics.json" \
    --thresholds 50 100 150

# Step 3: 格式正确性
echo ""
echo "[3/3] Checking format accuracy..."
python3 << EOF
import json, re

with open("${OUTPUT_DIR}/robopoint_predictions.jsonl") as f:
    preds = [json.loads(line) for line in f if line.strip()]

# Check for point format
point_pattern = r'<point>\s*\[.*?\]\s*</point>'
tuple_pattern = r'\(\s*[\d.]+\s*,\s*[\d.]+\s*\)'

format_stats = {"point_tag": 0, "tuple": 0, "other": 0, "empty": 0}
for p in preds:
    resp = p["response"]
    if not resp:
        format_stats["empty"] += 1
    elif re.search(point_pattern, resp, re.DOTALL):
        format_stats["point_tag"] += 1
    elif re.search(tuple_pattern, resp):
        format_stats["tuple"] += 1
    else:
        format_stats["other"] += 1

total = len(preds)
print(f"\nFormat Distribution ({total} samples):")
for fmt, count in format_stats.items():
    print(f"  {fmt}: {count} ({count/total*100:.1f}%)")

# Save
with open("${OUTPUT_DIR}/format_metrics.json", "w") as f:
    json.dump(format_stats, f, indent=2)
EOF

echo ""
echo "============================================"
echo " Evaluation Complete"
echo " Results: ${OUTPUT_DIR}"
echo "============================================"
