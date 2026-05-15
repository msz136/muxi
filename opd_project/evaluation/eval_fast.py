#!/usr/bin/env python3
"""Fast eval - uses max_memory to pin model to specific GPUs, avoiding CUDA_VISIBLE_DEVICES bug."""
import json, re, math, os, sys, time, argparse
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

EVAL_DATA = "/data/msz/opd_project/data/eval_robopoint_500.jsonl"
OUTPUT_ROOT = "/data/msz/opd_project/evaluation/results"


def parse_points(text):
    json_matches = re.findall(r'"point_2d"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]', text)
    if json_matches:
        return [[int(x), int(y)] for x, y in json_matches]
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
            pts.append([int(x*1000), int(y*1000)] if x <= 1.0 and y <= 1.0 else [int(x), int(y)])
        return pts
    m2 = re.findall(r'\[\s*(\d+)\s*,\s*(\d+)\s*\]', text)
    if m2:
        return [[int(x), int(y)] for x, y in m2]
    return []


def compute_metrics(results):
    per_point_distances, per_sample_avg = [], []
    pred_counts, gt_counts = [], []
    precision_list, recall_list = [], []

    for r in results:
        pred = parse_points(r["response"])
        gt = r["gt_points"]
        pred_counts.append(len(pred))
        gt_counts.append(len(gt))
        if pred and gt:
            sd = []
            for g in gt:
                d = min(math.sqrt((p[0]-g[0])**2+(p[1]-g[1])**2) for p in pred)
                per_point_distances.append(d)
                sd.append(d)
            per_sample_avg.append(np.mean(sd))
            matched_gt = sum(1 for g in gt if min(math.sqrt((p[0]-g[0])**2+(p[1]-g[1])**2) for p in pred) <= 100)
            matched_pred = sum(1 for p in pred if min(math.sqrt((p[0]-g[0])**2+(p[1]-g[1])**2) for g in gt) <= 100)
            recall_list.append(matched_gt / len(gt))
            precision_list.append(matched_pred / len(pred))
        elif gt:
            per_sample_avg.append(1414.0)
            for _ in gt:
                per_point_distances.append(1414.0)

    dists_pt = np.array(per_point_distances) if per_point_distances else np.array([1414.0])
    dists_sp = np.array(per_sample_avg) if per_sample_avg else np.array([1414.0])
    thresholds = list(range(0, 501, 10))
    auc = np.trapz([float(np.mean(dists_pt <= t)) for t in thresholds], thresholds) / 500.0
    avg_prec = np.mean(precision_list) if precision_list else 0
    avg_rec = np.mean(recall_list) if recall_list else 0
    f1 = 2*avg_prec*avg_rec/(avg_prec+avg_rec) if (avg_prec+avg_rec) > 0 else 0

    return {
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
        "format_accuracy": sum(1 for r in results if parse_points(r["response"])) / max(len(results), 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--start-gpu", type=int, default=0)
    parser.add_argument("--num-gpus", type=int, default=4)
    args = parser.parse_args()

    out_dir = f"{OUTPUT_ROOT}/{args.name}"
    os.makedirs(out_dir, exist_ok=True)

    with open(EVAL_DATA) as f:
        eval_items = [json.loads(line) for line in f if line.strip()]

    gpu_ids = list(range(args.start_gpu, args.start_gpu + args.num_gpus))
    print(f"[{args.name}] {len(eval_items)} samples, GPUs={gpu_ids}", flush=True)

    # Pin model to specific GPUs using max_memory, block all others
    max_memory = {}
    for g in range(8):
        if g in gpu_ids:
            max_memory[g] = "60GiB"
        else:
            max_memory[g] = "0GiB"

    print(f"[{args.name}] Loading model...", flush=True)
    t0 = time.time()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
    )
    processor = AutoProcessor.from_pretrained(args.model)
    model.eval()
    print(f"[{args.name}] Loaded in {time.time()-t0:.0f}s, device={model.device}", flush=True)

    results = [None] * len(eval_items)
    t0 = time.time()

    for i, item in enumerate(eval_items):
        if i % 16 == 0:
            elapsed = time.time() - t0
            eta = elapsed / i * (len(eval_items) - i) if i > 0 else 0
            print(f"[{args.name}] {i}/{len(eval_items)} | {elapsed:.0f}s elapsed | ETA {eta:.0f}s", flush=True)

        try:
            img_path = item["images"][0]
            if not os.path.exists(img_path):
                results[i] = {"response": "", "gt_points": item["gt_points"]}
                continue

            image = Image.open(img_path).convert("RGB")
            sys_msg = next((m["content"] for m in item["messages"] if m["role"] == "system"), None)
            user_msg = next((m["content"] for m in item["messages"] if m["role"] == "user"), "")
            user_msg = user_msg.replace("<image>\n", "").replace("<image>", "").strip()

            messages = []
            if sys_msg:
                messages.append({"role": "system", "content": sys_msg})
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_msg},
                ]
            })

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True).to(model.device)

            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            gen = out[0][inputs["input_ids"].shape[1]:]
            results[i] = {"response": processor.decode(gen, skip_special_tokens=True), "gt_points": item["gt_points"]}

        except Exception as e:
            results[i] = {"response": "", "gt_points": item.get("gt_points", [])}

    for i in range(len(results)):
        if results[i] is None:
            results[i] = {"response": "", "gt_points": eval_items[i]["gt_points"]}

    elapsed = time.time() - t0
    print(f"[{args.name}] Done: {elapsed:.0f}s ({elapsed/len(eval_items):.1f}s/sample)", flush=True)

    with open(f"{out_dir}/predictions.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    metrics = compute_metrics(results)
    metrics.update({
        "model": args.name,
        "checkpoint": args.model,
        "total_samples": len(results),
        "inference_time_s": elapsed,
    })
    with open(f"{out_dir}/metrics_full.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[{args.name}] === RESULTS ===", flush=True)
    for k in ["per_point_mean_distance", "acc@50_per_point", "acc@100_per_point",
              "acc@150_per_point", "auc_0_500", "f1@100", "avg_pred_points", "format_accuracy"]:
        print(f"  {k}: {metrics[k]:.4f}", flush=True)


if __name__ == "__main__":
    main()
