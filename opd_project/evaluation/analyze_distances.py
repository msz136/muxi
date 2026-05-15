#!/usr/bin/env python3
"""Analyze pointing distance distribution to inform metric design."""
import json, re, math
import numpy as np

PRED_FILE = "/data/msz/opd_project/evaluation/results/baseline_instruct/predictions.jsonl"

with open(PRED_FILE) as f:
    results = [json.loads(line) for line in f if line.strip()]

def parse_points(text):
    m = re.search(r'<point>\s*(\[.*?\])\s*</point>', text, re.DOTALL)
    if m:
        try:
            pts = json.loads(m.group(1))
            if isinstance(pts[0], list):
                return [[int(p[0]), int(p[1])] for p in pts]
            return [[int(pts[0]), int(pts[1])]]
        except:
            pass
    matches = re.findall(r'\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)', text)
    if matches:
        pts = []
        for x_str, y_str in matches:
            x, y = float(x_str), float(y_str)
            if x <= 1.0 and y <= 1.0:
                pts.append([int(x * 1000), int(y * 1000)])
            else:
                pts.append([int(x), int(y)])
        return pts
    m2 = re.findall(r'\[\s*(\d+)\s*,\s*(\d+)\s*\]', text)
    if m2:
        return [[int(x), int(y)] for x, y in m2]
    return []

# Per-sample: average min-distance across GT points
sample_distances = []
per_point_distances = []
for r in results:
    pred = parse_points(r["response"])
    gt = r["gt_points"]
    if pred and gt:
        dists = []
        for g in gt:
            d = min(math.sqrt((p[0] - g[0]) ** 2 + (p[1] - g[1]) ** 2) for p in pred)
            dists.append(d)
            per_point_distances.append(d)
        sample_distances.append(np.mean(dists))

dists_sample = np.array(sample_distances)
dists_point = np.array(per_point_distances)

print("=" * 60)
print(" Pointing Distance Distribution Analysis")
print("=" * 60)
print(f"\n  Coordinate range: 0-1000 (normalized)")
print(f"  Image diagonal: ~1414 units")
print(f"  Samples evaluated: {len(dists_sample)}")
print(f"  Total GT points: {len(dists_point)}")

print("\n--- Per-Point Distance Distribution ---")
print(f"  Mean:   {np.mean(dists_point):.2f}")
print(f"  Std:    {np.std(dists_point):.2f}")
print(f"  Min:    {np.min(dists_point):.2f}")
print(f"  25th:   {np.percentile(dists_point, 25):.2f}")
print(f"  Median: {np.median(dists_point):.2f}")
print(f"  75th:   {np.percentile(dists_point, 75):.2f}")
print(f"  90th:   {np.percentile(dists_point, 90):.2f}")
print(f"  95th:   {np.percentile(dists_point, 95):.2f}")
print(f"  Max:    {np.max(dists_point):.2f}")

print("\n--- Per-Sample Avg Distance Distribution ---")
print(f"  Mean:   {np.mean(dists_sample):.2f}")
print(f"  Median: {np.median(dists_sample):.2f}")
print(f"  75th:   {np.percentile(dists_sample, 75):.2f}")
print(f"  90th:   {np.percentile(dists_sample, 90):.2f}")

print("\n--- Acc@T Curves ---")
print("  Threshold | Per-Point | Per-Sample")
print("  ----------|-----------|----------")
for t in [10, 25, 50, 75, 100, 125, 150, 200, 250, 300, 500]:
    acc_pt = float(np.mean(dists_point <= t))
    acc_sp = float(np.mean(dists_sample <= t))
    print(f"  @{t:>4}     | {acc_pt:.4f}    | {acc_sp:.4f}")

# What does threshold mean physically?
print("\n--- Physical Interpretation ---")
print("  In a 1000x1000 coordinate space:")
print("  - @50  = 5% of image width/height")
print("  - @100 = 10% of image width/height")
print("  - @150 = 15% of image width/height")
print("  For a 640x480 image at 1000-scale:")
print("  - @50  ~ 32 pixels")
print("  - @100 ~ 64 pixels")
print("  - @150 ~ 96 pixels")

# Number of predicted points vs GT points
pred_counts = []
gt_counts = []
for r in results:
    pred = parse_points(r["response"])
    gt = r["gt_points"]
    pred_counts.append(len(pred))
    gt_counts.append(len(gt))

print(f"\n--- Point Count Mismatch ---")
print(f"  Avg GT points/sample:   {np.mean(gt_counts):.1f}")
print(f"  Avg Pred points/sample: {np.mean(pred_counts):.1f}")
exact_match = sum(1 for p, g in zip(pred_counts, gt_counts) if p == g)
print(f"  Exact count match: {exact_match}/{len(results)} ({exact_match/len(results)*100:.1f}%)")
