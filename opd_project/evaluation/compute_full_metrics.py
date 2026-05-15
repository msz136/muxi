#!/usr/bin/env python3
"""Compute comprehensive pointing metrics and save."""
import json, re, math
import numpy as np

PRED_FILE = "/data/msz/opd_project/evaluation/results/baseline_instruct/predictions.jsonl"
OUTPUT = "/data/msz/opd_project/evaluation/results/baseline_instruct/metrics_full.json"

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

# Compute all distances
per_point_distances = []
per_sample_avg_distances = []
pred_counts = []
gt_counts = []
point_recall_at_100 = []  # per GT point, was it matched within 100?

for r in results:
    pred = parse_points(r["response"])
    gt = r["gt_points"]
    pred_counts.append(len(pred))
    gt_counts.append(len(gt))

    if pred and gt:
        sample_dists = []
        for g in gt:
            d = min(math.sqrt((p[0] - g[0]) ** 2 + (p[1] - g[1]) ** 2) for p in pred)
            per_point_distances.append(d)
            sample_dists.append(d)
            point_recall_at_100.append(1 if d <= 100 else 0)
        per_sample_avg_distances.append(np.mean(sample_dists))
    elif gt:
        # No prediction but has GT
        per_sample_avg_distances.append(1414.0)
        for _ in gt:
            per_point_distances.append(1414.0)
            point_recall_at_100.append(0)

dists_pt = np.array(per_point_distances)
dists_sp = np.array(per_sample_avg_distances)

# AUC: area under Acc@T curve from T=0 to T=500, normalized
thresholds = list(range(0, 501, 10))
acc_curve = [float(np.mean(dists_pt <= t)) for t in thresholds]
auc = np.trapz(acc_curve, thresholds) / 500.0  # normalized to [0, 1]

# F1@100: treating each GT point as a "target"
# Precision: of predicted points, how many are within 100 of some GT?
# Recall: of GT points, how many have a pred within 100?
precision_at_100_list = []
recall_at_100_list = []
for r in results:
    pred = parse_points(r["response"])
    gt = r["gt_points"]
    if not pred or not gt:
        continue
    # Recall: for each GT, is there a pred within 100?
    matched_gt = sum(1 for g in gt if min(math.sqrt((p[0]-g[0])**2 + (p[1]-g[1])**2) for p in pred) <= 100)
    recall = matched_gt / len(gt)
    # Precision: for each pred, is there a GT within 100?
    matched_pred = sum(1 for p in pred if min(math.sqrt((p[0]-g[0])**2 + (p[1]-g[1])**2) for g in gt) <= 100)
    precision = matched_pred / len(pred)
    precision_at_100_list.append(precision)
    recall_at_100_list.append(recall)

avg_precision = np.mean(precision_at_100_list)
avg_recall = np.mean(recall_at_100_list)
f1_at_100 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0

metrics = {
    "model": "Qwen3-VL-8B-Instruct (baseline)",
    "total_samples": len(results),

    # Distance metrics (primary)
    "per_point_mean_distance": float(np.mean(dists_pt)),
    "per_point_median_distance": float(np.median(dists_pt)),
    "per_point_std_distance": float(np.std(dists_pt)),
    "per_sample_mean_distance": float(np.mean(dists_sp)),
    "per_sample_median_distance": float(np.median(dists_sp)),

    # Acc@T (per-point)
    "acc@50_per_point": float(np.mean(dists_pt <= 50)),
    "acc@100_per_point": float(np.mean(dists_pt <= 100)),
    "acc@150_per_point": float(np.mean(dists_pt <= 150)),
    "acc@200_per_point": float(np.mean(dists_pt <= 200)),

    # Acc@T (per-sample, avg distance)
    "acc@50_per_sample": float(np.mean(dists_sp <= 50)),
    "acc@100_per_sample": float(np.mean(dists_sp <= 100)),
    "acc@150_per_sample": float(np.mean(dists_sp <= 150)),

    # AUC
    "auc_0_500": float(auc),

    # F1@100 (precision + recall of point matching)
    "precision@100": float(avg_precision),
    "recall@100": float(avg_recall),
    "f1@100": float(f1_at_100),

    # Point count stats
    "avg_gt_points_per_sample": float(np.mean(gt_counts)),
    "avg_pred_points_per_sample": float(np.mean(pred_counts)),
    "point_count_exact_match": sum(1 for p, g in zip(pred_counts, gt_counts) if p == g) / len(results),

    # Format
    "format_accuracy": sum(1 for r in results if parse_points(r["response"])) / len(results),
}

print("=" * 60)
print(" Complete Baseline Metrics: Qwen3-VL-8B-Instruct")
print("=" * 60)
for k, v in metrics.items():
    if isinstance(v, float):
        print(f"  {k}: {v:.4f}")
    else:
        print(f"  {k}: {v}")

with open(OUTPUT, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\nSaved: {OUTPUT}")
