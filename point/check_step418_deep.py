#!/usr/bin/env python3
"""DEEP diagnosis of step 418 samples: check image dimensions, pixel stats,
text lengths, coordinate values, and any edge cases that might cause NaN.
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

g = torch.Generator()
g.manual_seed(SEED)
shuffled = torch.randperm(N_TOTAL, generator=g).tolist()

start_mb = TARGET_STEP * GRAD_ACCUM
end_mb = start_mb + GRAD_ACCUM

needed_lines = {}
for rank in range(NUM_GPUS):
    for mb in range(start_mb, end_mb):
        for s in range(PER_DEVICE_BS):
            local_pos = mb * PER_DEVICE_BS + s
            sampler_pos = rank + NUM_GPUS * local_pos
            if sampler_pos >= N_TOTAL:
                continue
            file_line = shuffled[sampler_pos]
            needed_lines[file_line] = (rank, local_pos)

needed_sorted = sorted(needed_lines.keys())
needed_set = set(needed_lines.keys())

samples = {}
with open(DATA_PATH, encoding="utf-8") as f:
    ni = 0
    for i, line in enumerate(f):
        if i == needed_sorted[ni]:
            line = line.strip()
            if line:
                try:
                    samples[i] = json.loads(line)
                except Exception as e:
                    samples[i] = {"_error": f"json: {e}"}
            ni += 1
            if ni >= len(needed_sorted):
                break

# ── DEEP CHECK ─────────────────────────────────────────────────────────

print(f"=== DEEP DIAGNOSIS: step {TARGET_STEP}, {len(samples)} samples ===\n")

issues_detail = []
img_sizes = []
img_mins = []
img_maxs = []
text_lens = []
coord_values = []

import requests as _req

for file_line in sorted(samples.keys()):
    rank, local_pos = needed_lines[file_line]
    s = samples[file_line]
    ds = s.get("dataset", "?")
    issues = []

    # ── Image deep check ────────────────────────────────────────────────
    for j, img_path in enumerate(s.get("image", [])):
        # Resolve path
        resolved = img_path
        if img_path.startswith("http://") or img_path.startswith("https://"):
            try:
                resp = _req.get(img_path, timeout=5)
                if resp.status_code == 200:
                    img = Image.open(BytesIO(resp.content)).convert("RGB")
                    arr = np.array(img, dtype=np.float32)
                    img_sizes.append(arr.shape)
                    img_mins.append(arr.min())
                    img_maxs.append(arr.max())
                    if not np.isfinite(arr).all():
                        issues.append(f"img[{j}] URL NaN/Inf pixels")
                    if arr.min() < 0 or arr.max() > 255:
                        issues.append(f"img[{j}] URL pixel range [{arr.min()}, {arr.max()}]")
                else:
                    issues.append(f"img[{j}] URL HTTP {resp.status_code}")
            except Exception as e:
                issues.append(f"img[{j}] URL error: {e}")
        else:
            p = Path(img_path)
            if not p.exists():
                alt = f"/data/msz/dataset/RoboPoint/images/{img_path}"
                if Path(alt).exists():
                    resolved = alt
                    p = Path(alt)
            if p.exists():
                try:
                    img = Image.open(p).convert("RGB")
                    arr = np.array(img, dtype=np.float32)
                    img_sizes.append(arr.shape)
                    img_mins.append(arr.min())
                    img_maxs.append(arr.max())
                    if not np.isfinite(arr).all():
                        issues.append(f"img[{j}] NaN/Inf pixels")
                    # Check for corrupted PNG (single-color blocks, bad encoding)
                    if arr.std() < 1.0:
                        issues.append(f"img[{j}] near-constant (std={arr.std():.2f})")
                    if arr.min() < 0 or arr.max() > 255:
                        issues.append(f"img[{j}] bad pixel range [{arr.min()}, {arr.max()}]")
                    # Dimensions
                    h, w = arr.shape[:2]
                    if h < 10 or w < 10:
                        issues.append(f"img[{j}] too small: {w}x{h}")
                    if h > 8000 or w > 8000:
                        issues.append(f"img[{j}] too large: {w}x{h}")
                except Exception as e:
                    issues.append(f"img[{j}] decode: {e}")
            else:
                issues.append(f"img[{j}] MISSING: {img_path}")

    # ── Text checks ─────────────────────────────────────────────────────
    convs = s.get("conversations", [])
    for j, turn in enumerate(convs):
        val = turn.get("value", "")
        role = turn.get("from", "?")
        text_lens.append(len(val))
        if len(val) > 10000:
            issues.append(f"conv[{j}].{role} very long: {len(val)} chars")
        if val is None:
            issues.append(f"conv[{j}].{role} is None")
        if role == "gpt":
            # Extract coordinate values
            import re
            coords = re.findall(r'\[\[?\s*(\d+)\s*,\s*(\d+)\s*\]\]?', val)
            for x_str, y_str in coords:
                x, y = int(x_str), int(y_str)
                coord_values.append((x, y))
                if x < 0 or x > 1000 or y < 0 or y > 1000:
                    issues.append(f"conv[{j}] out-of-range coord: [{x},{y}]")
                if x == 0 and y == 0:
                    issues.append(f"conv[{j}] zero coord [0,0]")

    # Track dataset distribution
    if issues:
        issues_detail.append((file_line, rank, local_pos, ds, issues, s))

# ── Print summary ──────────────────────────────────────────────────────
print(f"Image stats ({len(img_sizes)} loaded):")
if img_sizes:
    sizes = np.array([s[:2] for s in img_sizes])
    print(f"  width:  min={sizes[:,1].min()}, max={sizes[:,1].max()}, mean={sizes[:,1].mean():.0f}")
    print(f"  height: min={sizes[:,0].min()}, max={sizes[:,0].max()}, mean={sizes[:,0].mean():.0f}")
if img_mins:
    print(f"  pixel min range: {min(img_mins)} – {max(img_mins)}")
if img_maxs:
    print(f"  pixel max range: {min(img_maxs)} – {max(img_maxs)}")

print(f"\nText stats ({len(text_lens)} utterances):")
if text_lens:
    print(f"  len: min={min(text_lens)}, max={max(text_lens)}, mean={np.mean(text_lens):.0f}")

if coord_values:
    xs = [c[0] for c in coord_values]
    ys = [c[1] for c in coord_values]
    print(f"\nCoordinate stats ({len(coord_values)} points):")
    print(f"  x: min={min(xs)}, max={max(xs)}")
    print(f"  y: min={min(ys)}, max={max(ys)}")

# Dataset distribution in this batch
from collections import Counter
ds_counts = Counter()
for file_line in sorted(samples.keys()):
    s = samples[file_line]
    ds_counts[s.get("dataset", "?")] += 1
print(f"\nDataset distribution:")
for ds, cnt in ds_counts.most_common():
    print(f"  {ds}: {cnt}")

print(f"\n{'='*60}")
print(f"SAMPLES WITH ISSUES: {len(issues_detail)}/{len(samples)}")
for file_line, rank, local_pos, ds, issues, s in issues_detail:
    print(f"\n--- [rank={rank} local={local_pos} line={file_line}] {ds} ---")
    for iss in issues:
        print(f"  !! {iss}")
    imgs = s.get("image", [])
    if imgs:
        print(f"  images: {imgs}")
    convs = s.get("conversations", [])
    for j, turn in enumerate(convs):
        val = turn.get("value", "")
        print(f"  [{j}] {turn.get('from','?')}: {val[:150]}{'...' if len(val)>150 else ''}")
