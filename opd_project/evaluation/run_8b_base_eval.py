#!/usr/bin/env python3
"""Evaluation: ACE-Brain-0-8B (8b_base) on pointing tasks."""
import json, os, re, math, torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

CHECKPOINT = "/data/msz/models/8b_base"
EVAL_DATA = "/data/msz/opd_project/data/eval_robopoint_500.jsonl"
OUTPUT_DIR = "/data/msz/opd_project/evaluation/results/8b_base"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading model...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    CHECKPOINT, dtype=torch.bfloat16, device_map="auto",
)
processor = AutoProcessor.from_pretrained(CHECKPOINT)
model.eval()

print("Loading eval data...")
with open(EVAL_DATA) as f:
    items = [json.loads(line) for line in f if line.strip()]
print(f"Samples: {len(items)}")

results = []
for i, item in enumerate(items):
    if i % 25 == 0:
        print(f"  [{i}/{len(items)}]")
    try:
        img_path = item["images"][0]
        if not os.path.exists(img_path):
            results.append({"response": "", "gt_points": item["gt_points"]})
            continue

        # Build messages for Qwen3-VL
        sys_msg = next((m["content"] for m in item["messages"] if m["role"] == "system"), None)
        user_msg = next((m["content"] for m in item["messages"] if m["role"] == "user"), "")
        # Remove <image> placeholder
        user_msg = user_msg.replace("<image>\n", "").replace("<image>", "").strip()

        messages = []
        if sys_msg:
            messages.append({"role": "system", "content": sys_msg})
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{img_path}"},
                {"type": "text", "text": user_msg},
            ]
        })

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            return_tensors="pt", padding=True,
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        generated = out[0][inputs["input_ids"].shape[1]:]
        response = processor.decode(generated, skip_special_tokens=True)
        results.append({"response": response, "gt_points": item["gt_points"]})

    except Exception as e:
        if i < 5:
            print(f"  Error@{i}: {e}")
        results.append({"response": "", "gt_points": item.get("gt_points", [])})

# Save predictions
pred_file = f"{OUTPUT_DIR}/predictions.jsonl"
with open(pred_file, "w") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"\nPredictions: {pred_file}")

# Compute metrics
def parse_points(text):
    # <point>[[x,y]]</point>
    m = re.search(r"<point>\s*(\[.*?\])\s*</point>", text, re.DOTALL)
    if m:
        try:
            pts = json.loads(m.group(1))
            return pts if isinstance(pts[0], list) else [pts]
        except: pass
    # (x, y) tuple format (0-1 float)
    matches = re.findall(r"\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)", text)
    if matches:
        return [[int(float(x)*1000), int(float(y)*1000)] for x, y in matches]
    # bare [[x,y]] (0-1000 int)
    m2 = re.search(r"\[\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]", text)
    if m2:
        return [[int(m2.group(1)), int(m2.group(2))]]
    return []

distances = []
format_ok = 0
for r in results:
    pred = parse_points(r["response"])
    gt = r["gt_points"]
    if pred: format_ok += 1
    if pred and gt:
        for g in gt:
            d = min(math.sqrt((p[0]-g[0])**2 + (p[1]-g[1])**2) for p in pred)
            distances.append(d)

dists = np.array(distances) if distances else np.array([])
metrics = {
    "model": "ACE-Brain-0-8B (8b_base)",
    "total_samples": len(results),
    "non_empty_responses": sum(1 for r in results if r["response"]),
    "format_accuracy": format_ok / len(results),
    "num_point_pairs": len(distances),
}
if len(dists) > 0:
    metrics.update({
        "mean_distance": float(np.mean(dists)),
        "median_distance": float(np.median(dists)),
        "acc@50": float(np.mean(dists <= 50)),
        "acc@100": float(np.mean(dists <= 100)),
        "acc@150": float(np.mean(dists <= 150)),
    })

print(f"\n{'='*50}")
print(f" ACE-Brain-0-8B (8b_base)")
print(f"{'='*50}")
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

with open(f"{OUTPUT_DIR}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\nSaved: {OUTPUT_DIR}/metrics.json")

# Show sample predictions
print(f"\nSample predictions:")
for r in results[:5]:
    resp = r["response"][:150]
    gt = r["gt_points"][:3]
    print(f"  GT: {gt}")
    print(f"  Pred: {resp}")
    print()
