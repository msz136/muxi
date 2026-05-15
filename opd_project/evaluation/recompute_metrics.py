#!/usr/bin/env python3
"""Recompute baseline metrics with corrected point parsing."""
import json, re, math
import numpy as np

PRED_FILE = "/data/msz/opd_project/evaluation/results/baseline_instruct/predictions.jsonl"
OUTPUT_DIR = "/data/msz/opd_project/evaluation/results/baseline_instruct"

with open(PRED_FILE) as f:
    results = [json.loads(line) for line in f if line.strip()]

def parse_points(text):
    """Parse model output coordinates, normalize to 0-1000."""
    points = []

    # 1. <point>[[x,y]]</point>
    m = re.search(r'<point>\s*(\[.*?\])\s*</point>', text, re.DOTALL)
    if m:
        try:
            pts = json.loads(m.group(1))
            if isinstance(pts[0], list):
                return [[int(p[0]), int(p[1])] for p in pts]
            return [[int(pts[0]), int(pts[1])]]
        except:
            pass

    # 2. (x, y) tuple format
    matches = re.findall(r'\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)', text)
    if matches:
        for x_str, y_str in matches:
            x, y = float(x_str), float(y_str)
            if x <= 1.0 and y <= 1.0:
                points.append([int(x * 1000), int(y * 1000)])
            else:
                points.append([int(x), int(y)])
        return points

    # 3. [[x, y]] bare
    m2 = re.findall(r'\[\s*(\d+)\s*,\s*(\d+)\s*\]', text)
    if m2:
        return [[int(x), int(y)] for x, y in m2]

    return []

distances = []
format_ok = 0
format_types = {"tuple_01": 0, "tuple_1000": 0, "point_tag": 0, "bracket": 0, "none": 0}

for r in results:
    resp = r["response"]
    pred = parse_points(resp)
    gt = r["gt_points"]

    if pred:
        format_ok += 1
        if "<point>" in resp:
            format_types["point_tag"] += 1
        elif re.search(r'\(\s*0\.\d+', resp):
            format_types["tuple_01"] += 1
        elif re.search(r'\(\s*\d{2,3}\s*,', resp):
            format_types["tuple_1000"] += 1
        else:
            format_types["bracket"] += 1
    else:
        format_types["none"] += 1

    if pred and gt:
        for g in gt:
            d = min(math.sqrt((p[0]-g[0])**2 + (p[1]-g[1])**2) for p in pred)
            distances.append(d)

dists = np.array(distances)
metrics = {
    "model": "Qwen3-VL-8B-Instruct (baseline)",
    "total_samples": len(results),
    "format_accuracy": format_ok / len(results),
    "format_types": format_types,
    "num_point_pairs": len(distances),
    "mean_distance": float(np.mean(dists)),
    "median_distance": float(np.median(dists)),
    "acc@50": float(np.mean(dists <= 50)),
    "acc@100": float(np.mean(dists <= 100)),
    "acc@150": float(np.mean(dists <= 150)),
    "acc@200": float(np.mean(dists <= 200)),
}

print("=" * 55)
print(" Baseline Evaluation: Qwen3-VL-8B-Instruct (corrected)")
print("=" * 55)
print(f"  Samples: {metrics['total_samples']}")
print(f"  Format accuracy: {metrics['format_accuracy']:.4f}")
print(f"  Format types: {format_types}")
print(f"  Point pairs evaluated: {metrics['num_point_pairs']}")
print(f"  Mean distance: {metrics['mean_distance']:.2f}")
print(f"  Median distance: {metrics['median_distance']:.2f}")
print(f"  Acc@50:  {metrics['acc@50']:.4f}")
print(f"  Acc@100: {metrics['acc@100']:.4f}")
print(f"  Acc@150: {metrics['acc@150']:.4f}")
print(f"  Acc@200: {metrics['acc@200']:.4f}")

with open(f"{OUTPUT_DIR}/metrics_corrected.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\nSaved: {OUTPUT_DIR}/metrics_corrected.json")

# Examples
print("\nExamples (parsed):")
for r in results[:8]:
    pred = parse_points(r["response"])
    gt = r["gt_points"][:2]
    resp_short = r["response"][:100]
    print(f"  GT={gt}  Pred={pred[:2]}  Raw={resp_short}")
