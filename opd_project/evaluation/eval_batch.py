#!/usr/bin/env python3
"""Batch evaluation of multiple models on pointing task.
Uses batched inference for speed. Supports multi-GPU via device_map=auto.
"""
import json, re, math, os, sys, time
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

EVAL_DATA = "/data/msz/opd_project/data/eval_robopoint_500.jsonl"
OUTPUT_ROOT = "/data/msz/opd_project/evaluation/results"
BATCH_SIZE = 8  # batched generation

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


def compute_metrics(results):
    per_point_distances = []
    per_sample_avg_distances = []
    pred_counts = []
    gt_counts = []
    precision_list = []
    recall_list = []

    for r in results:
        pred = parse_points(r["response"])
        gt = r["gt_points"]
        pred_counts.append(len(pred))
        gt_counts.append(len(gt))

        if pred and gt:
            sample_dists = []
            for g in gt:
                d = min(math.sqrt((p[0]-g[0])**2 + (p[1]-g[1])**2) for p in pred)
                per_point_distances.append(d)
                sample_dists.append(d)
            per_sample_avg_distances.append(np.mean(sample_dists))

            # Precision/Recall@100
            matched_gt = sum(1 for g in gt if min(math.sqrt((p[0]-g[0])**2+(p[1]-g[1])**2) for p in pred) <= 100)
            matched_pred = sum(1 for p in pred if min(math.sqrt((p[0]-g[0])**2+(p[1]-g[1])**2) for g in gt) <= 100)
            recall_list.append(matched_gt / len(gt))
            precision_list.append(matched_pred / len(pred))
        elif gt:
            per_sample_avg_distances.append(1414.0)
            for _ in gt:
                per_point_distances.append(1414.0)

    dists_pt = np.array(per_point_distances) if per_point_distances else np.array([1414.0])
    dists_sp = np.array(per_sample_avg_distances) if per_sample_avg_distances else np.array([1414.0])

    # AUC
    thresholds = list(range(0, 501, 10))
    acc_curve = [float(np.mean(dists_pt <= t)) for t in thresholds]
    auc = np.trapz(acc_curve, thresholds) / 500.0

    avg_prec = np.mean(precision_list) if precision_list else 0
    avg_rec = np.mean(recall_list) if recall_list else 0
    f1 = 2 * avg_prec * avg_rec / (avg_prec + avg_rec) if (avg_prec + avg_rec) > 0 else 0

    return {
        "total_samples": len(results),
        "per_point_mean_distance": float(np.mean(dists_pt)),
        "per_point_median_distance": float(np.median(dists_pt)),
        "per_sample_mean_distance": float(np.mean(dists_sp)),
        "per_sample_median_distance": float(np.median(dists_sp)),
        "acc@50_per_point": float(np.mean(dists_pt <= 50)),
        "acc@100_per_point": float(np.mean(dists_pt <= 100)),
        "acc@150_per_point": float(np.mean(dists_pt <= 150)),
        "acc@200_per_point": float(np.mean(dists_pt <= 200)),
        "acc@50_per_sample": float(np.mean(dists_sp <= 50)),
        "acc@100_per_sample": float(np.mean(dists_sp <= 100)),
        "acc@150_per_sample": float(np.mean(dists_sp <= 150)),
        "auc_0_500": float(auc),
        "precision@100": float(avg_prec),
        "recall@100": float(avg_rec),
        "f1@100": float(f1),
        "avg_gt_points": float(np.mean(gt_counts)),
        "avg_pred_points": float(np.mean(pred_counts)),
        "format_accuracy": sum(1 for r in results if parse_points(r["response"])) / len(results),
    }


def evaluate_model(checkpoint, model_name, eval_items):
    """Evaluate a single model."""
    out_dir = f"{OUTPUT_ROOT}/{model_name}"
    os.makedirs(out_dir, exist_ok=True)

    pred_file = f"{out_dir}/predictions.jsonl"

    # Skip if already done
    if os.path.exists(pred_file):
        with open(pred_file) as f:
            existing = sum(1 for _ in f)
        if existing >= len(eval_items):
            print(f"  [{model_name}] Already evaluated ({existing} predictions), computing metrics...")
            with open(pred_file) as f:
                results = [json.loads(line) for line in f if line.strip()]
            metrics = compute_metrics(results)
            metrics["model"] = model_name
            with open(f"{out_dir}/metrics_full.json", "w") as f:
                json.dump(metrics, f, indent=2)
            return metrics

    print(f"  [{model_name}] Loading model from {checkpoint}...")
    t0 = time.time()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        checkpoint, dtype=torch.bfloat16, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(checkpoint)
    model.eval()
    print(f"  [{model_name}] Model loaded in {time.time()-t0:.1f}s")

    results = []
    t0 = time.time()

    for i in range(0, len(eval_items), BATCH_SIZE):
        batch = eval_items[i:i+BATCH_SIZE]
        if i % 50 == 0:
            elapsed = time.time() - t0
            speed = i / elapsed if elapsed > 0 else 0
            print(f"  [{model_name}] {i}/{len(eval_items)} ({speed:.1f} samples/s)")

        texts = []
        images_list = []
        valid_indices = []

        for j, item in enumerate(batch):
            img_path = item["images"][0]
            if not os.path.exists(img_path):
                results.append({"response": "", "gt_points": item["gt_points"]})
                continue

            sys_msg = next((m["content"] for m in item["messages"] if m["role"] == "system"), None)
            user_msg = next((m["content"] for m in item["messages"] if m["role"] == "user"), "")
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

            try:
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, _ = process_vision_info(messages)
                texts.append(text)
                images_list.append(image_inputs)
                valid_indices.append(j)
            except Exception as e:
                results.append({"response": "", "gt_points": item["gt_points"]})

        if not texts:
            continue

        # Process one by one (batched vision processing is tricky with variable sizes)
        for idx, (text, imgs) in enumerate(zip(texts, images_list)):
            try:
                inputs = processor(
                    text=[text], images=imgs, return_tensors="pt", padding=True,
                ).to(model.device)

                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
                generated = out[0][inputs["input_ids"].shape[1]:]
                response = processor.decode(generated, skip_special_tokens=True)
                results.append({"response": response, "gt_points": batch[valid_indices[idx]]["gt_points"]})
            except Exception as e:
                results.append({"response": "", "gt_points": batch[valid_indices[idx]]["gt_points"]})

    elapsed = time.time() - t0
    print(f"  [{model_name}] Done: {len(results)} predictions in {elapsed:.1f}s ({len(results)/elapsed:.1f} samples/s)")

    # Save predictions
    with open(pred_file, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Compute metrics
    metrics = compute_metrics(results)
    metrics["model"] = model_name
    metrics["inference_time_s"] = elapsed

    with open(f"{out_dir}/metrics_full.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    return metrics


def main():
    models = {
        "expert3": "/data/msz/models/expert3",
        "expert4": "/data/msz/models/expert4",
    }

    # Load eval data
    print("Loading eval data...")
    with open(EVAL_DATA) as f:
        eval_items = [json.loads(line) for line in f if line.strip()]
    print(f"Eval samples: {len(eval_items)}")

    all_metrics = {}
    for name, path in models.items():
        print(f"\n{'='*60}")
        print(f" Evaluating: {name}")
        print(f"{'='*60}")
        metrics = evaluate_model(path, name, eval_items)
        all_metrics[name] = metrics

    # Print comparison table
    print(f"\n{'='*70}")
    print(" COMPARISON TABLE")
    print(f"{'='*70}")

    # Load baseline too
    baseline_file = f"{OUTPUT_ROOT}/baseline_instruct/metrics_full.json"
    if os.path.exists(baseline_file):
        with open(baseline_file) as f:
            all_metrics["baseline_instruct"] = json.load(f)

    header = f"{'Metric':<30} | " + " | ".join(f"{n:>15}" for n in all_metrics.keys())
    print(header)
    print("-" * len(header))

    key_metrics = [
        "per_point_mean_distance", "per_sample_mean_distance",
        "acc@50_per_point", "acc@100_per_point", "acc@150_per_point",
        "auc_0_500", "precision@100", "recall@100", "f1@100",
        "avg_pred_points", "format_accuracy",
    ]
    for k in key_metrics:
        row = f"{k:<30} | "
        for name in all_metrics:
            v = all_metrics[name].get(k, "N/A")
            if isinstance(v, float):
                row += f"{v:>15.4f} | "
            else:
                row += f"{str(v):>15} | "
        print(row)

    # Save comparison
    comparison_file = f"{OUTPUT_ROOT}/comparison.json"
    with open(comparison_file, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nComparison saved: {comparison_file}")


if __name__ == "__main__":
    main()
