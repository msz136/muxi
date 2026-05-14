#!/usr/bin/env python3
"""Extract and diagnose samples at step 418 of expert SFT training.

Reproduces HF Trainer DistributedSampler ordering:
  8 GPUs, per_device_batch_size=6, grad_accum=4, seed=42.
"""
import json, sys, traceback
import numpy as np
from pathlib import Path
from PIL import Image
from io import BytesIO
import torch

DATA_PATH = "/data/msz/point/data_expert/expert_mix_v1_shuffled.jsonl"
NUM_GPUS = 8
PER_DEVICE_BS = 6
GRAD_ACCUM = 4
SEED = 42
TARGET_STEP = 418
N_TOTAL = 1546201

# ── Reproduce DistributedSampler ───────────────────────────────────────
g = torch.Generator()
g.manual_seed(SEED)  # epoch 0
shuffled = torch.randperm(N_TOTAL, generator=g).tolist()  # sampler_pos → file_line

# For each rank, collect file lines needed at target step
start_mb = TARGET_STEP * GRAD_ACCUM  # 1672
end_mb = start_mb + GRAD_ACCUM        # 1676

needed_lines = set()
sample_info = {}  # file_line → (rank, local_pos)

for rank in range(NUM_GPUS):
    for mb in range(start_mb, end_mb):
        for s in range(PER_DEVICE_BS):
            local_pos = mb * PER_DEVICE_BS + s
            sampler_pos = rank + NUM_GPUS * local_pos
            if sampler_pos >= N_TOTAL:
                continue
            file_line = shuffled[sampler_pos]
            needed_lines.add(file_line)
            sample_info[file_line] = (rank, local_pos)

print(f"Step {TARGET_STEP}: need {len(needed_lines)} file lines out of {N_TOTAL}")
print(f"Line range: {min(needed_lines)} – {max(needed_lines)}")

# ── Read samples from file ─────────────────────────────────────────────
samples = {}

# Sort needed lines for efficient streaming read
needed_sorted = sorted(needed_lines)
needed_set = set(needed_lines)

with open(DATA_PATH, encoding="utf-8") as f:
    needed_idx = 0
    for i, line in enumerate(f):
        if i == needed_sorted[needed_idx]:
            line = line.strip()
            if line:
                try:
                    samples[i] = json.loads(line)
                except Exception as e:
                    samples[i] = {"_error": f"json parse: {e}"}
            else:
                samples[i] = {"_error": "empty line"}
            needed_idx += 1
            if needed_idx >= len(needed_sorted):
                break

# ── Load all samples (any we missed) ───────────────────────────────────
for ln in needed_sorted:
    if ln not in samples:
        samples[ln] = {"_error": "NOT FOUND"}
print(f"Loaded {len(samples)} samples")

# ── Check functions ────────────────────────────────────────────────────
import requests as _req

def check_image(img_path: str) -> list[str]:
    problems = []
    if img_path.startswith("http://") or img_path.startswith("https://"):
        try:
            resp = _req.get(img_path, timeout=5, stream=True)
            if resp.status_code != 200:
                problems.append(f"HTTP {resp.status_code}")
                return problems
            data = resp.content
        except Exception as e:
            problems.append(f"fetch failed: {e}")
            return problems
    else:
        p = Path(img_path)
        if not p.exists():
            alt = f"/data/msz/dataset/RoboPoint/images/{img_path}"
            if Path(alt).exists():
                p = Path(alt)
        if not p.exists():
            problems.append(f"file not found: {img_path}")
            return problems
        try:
            data = p.read_bytes()
        except Exception as e:
            problems.append(f"read failed: {e}")
            return problems

    try:
        img = Image.open(BytesIO(data)).convert("RGB")
        arr = np.array(img, dtype=np.float32)
        if arr.size == 0:
            problems.append("zero-size image")
        else:
            if not np.isfinite(arr).all():
                problems.append("NaN/Inf pixels")
            if arr.max() == arr.min():
                problems.append("constant image")
    except Exception as e:
        problems.append(f"decode failed: {e}")
    return problems

def check_text(text, label: str) -> list[str]:
    problems = []
    if text is None:
        problems.append(f"{label}: None")
        return problems
    if not isinstance(text, str):
        problems.append(f"{label}: not str ({type(text).__name__})")
        return problems
    if len(text) == 0:
        problems.append(f"{label}: empty")
    elif text.isspace():
        problems.append(f"{label}: whitespace only")
    if len(text) > 200000:
        problems.append(f"{label}: too long ({len(text)} chars)")
    return problems

# ── Diagnose ───────────────────────────────────────────────────────────
total_issues = 0
clean = 0
missing_img = 0
corrupt_img = 0
text_bad = 0

for file_line in sorted(samples.keys()):
    rank, local_pos = sample_info[file_line]
    s = samples[file_line]
    issues = []

    if "_error" in s:
        issues.append(f"LOAD ERROR: {s['_error']}")

    # Images
    for j, img in enumerate(s.get("image", [])):
        ps = check_image(img)
        for p in ps:
            issues.append(f"image[{j}]: {p}")
        if any("not found" in p for p in ps):
            missing_img += 1
        if any("NaN/Inf" in p or "constant" in p or "decode" in p or "fetch" in p for p in ps):
            corrupt_img += 1

    # Text
    convs = s.get("conversations", [])
    for j, turn in enumerate(convs):
        role = turn.get("from", "?")
        val = turn.get("value", "")
        ps = check_text(val, f"conv[{j}].{role}")
        issues.extend(ps)
        if ps:
            text_bad += 1

    # Check for video presence (video encoding can be heavy)
    if s.get("video"):
        issues.append("has video (heavy encoding)")

    if issues:
        total_issues += 1
        print(f"--- [rank={rank} local={local_pos} file_line={file_line}] ---")
        print(f"  dataset: {s.get('dataset', '?')}  data_path: {s.get('data_path', '?')}")
        print(f"  images ({len(s.get('image', []))}): {s.get('image', [])}")
        print(f"  video ({len(s.get('video', []))}): {s.get('video', [])}")
        for iss in issues:
            print(f"  ** {iss}")
        for j, turn in enumerate(convs):
            val = turn.get("value", "")
            role = turn.get("from", "?")
            print(f"  [{j}] {role}: {val[:200]}{'...' if len(val)>200 else ''}")
        print()
    else:
        clean += 1

print("=" * 70)
print(f"SUMMARY: step {TARGET_STEP} — {len(samples)} samples")
print(f"  Clean:          {clean}/{len(samples)}")
print(f"  With issues:    {total_issues}/{len(samples)}")
print(f"  Missing images: {missing_img}")
print(f"  Corrupt/bad img:{corrupt_img}")
print(f"  Text anomalies: {text_bad}")
