#!/usr/bin/env python3
"""Regenerate data sans pixmo, then build three mix versions with different RoboPoint multipliers.

Usage: python /data/msz/point/build_mixes.py

Output:
  data_expert/expert_mix_v1.jsonl   – natural mix (1x everything)
  data_expert/expert_mix_v2.jsonl   – RoboPoint 3x, others 1x
  data_expert/expert_mix_v3.jsonl   – RoboPoint 6x, others 1x
"""

import json, os, random, sys
from pathlib import Path

DATA_ROOT = Path("/data/msz/dataset")
OUT_DIR = Path("/data/msz/point/data_expert")
OLD_ROBOPOINT = Path("/data/msz/point/data/grounding_sft.jsonl")
SEED = 42

def log(msg: str) -> None:
    print(msg, flush=True)

# ── step 1: run point_data_only.py ────────────────────────────────────
log("=== Step 1: regenerate base data (skip pixmo + robopoint reprocess) ===")
# First clean up
import shutil
if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)
extracted = Path("/data/msz/dataset/Robo2VLM-1/_extracted_images")
if extracted.exists():
    shutil.rmtree(extracted)

# We'll import the converter directly instead of subprocess
sys.path.insert(0, str(Path("/data/msz/point")))
import point_data_only as pdo

class Args:
    data_root = DATA_ROOT
    out_dir = OUT_DIR
    seed = SEED
    limit_per_dataset = 0
    allow_url_images = True
    skip_datasets = "robopoint,pixmopoints"
    import_existing = OLD_ROBOPOINT

pdo.log = log
writer = pdo.Writer(OUT_DIR)
try:
    # Import old RoboPoint data
    if Args.import_existing and Args.import_existing.exists():
        log(f"[import] converting existing data from {Args.import_existing}")
        n = pdo.convert_old_robopoint_format(Args.import_existing, writer)
        log(f"[import] imported {n} samples")

    # Process other datasets
    pdo.convert_pixmo(Args.data_root, writer, Args.limit_per_dataset, Args.allow_url_images)  # no-op, skipped
    pdo.convert_sharerobot_json(Args.data_root, writer, Args.limit_per_dataset)
    pdo.convert_embspatial(Args.data_root, writer, Args.limit_per_dataset, Args.allow_url_images)
    pdo.convert_robo2vlm1(Args.data_root, writer, Args.limit_per_dataset)
finally:
    writer.close()

counts = pdo.merge_outputs(OUT_DIR, SEED)
log(f"[step1] base mix: {dict(counts)} total={sum(counts.values())}")

# ── step 2: separate RoboPoint lines from other lines ─────────────────
log("=== Step 2: separate RoboPoint from others ===")
mix_path = OUT_DIR / "expert_grounding_mix.jsonl"
rng = random.Random(SEED)

robo_lines = []
other_lines = []
with mix_path.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if obj.get("dataset") == "robopoint":
            robo_lines.append(line)
        else:
            other_lines.append(line)

log(f"[step2] robopoint={len(robo_lines)} other={len(other_lines)}")

# ── step 3: build three versions ─────────────────────────────────────
log("=== Step 3: build v1, v2, v3 ===")

# v1: natural mix (already done, just rename)
(mix_path).rename(OUT_DIR / "expert_mix_v1.jsonl")
log(f"[v1] expert_mix_v1.jsonl = {len(robo_lines) + len(other_lines)} lines (1x)")

# v2: RoboPoint 3x
rng.shuffle(other_lines)
rng.shuffle(robo_lines)
with (OUT_DIR / "expert_mix_v2.jsonl").open("w", encoding="utf-8") as f:
    for line in other_lines:
        f.write(line)
    for _ in range(3):
        for line in robo_lines:
            f.write(line)
        rng.shuffle(robo_lines)
v2_total = len(other_lines) + 3 * len(robo_lines)
log(f"[v2] expert_mix_v2.jsonl = {v2_total} lines (robopoint 3x)")

# v3: RoboPoint 6x
rng.shuffle(other_lines)
rng.shuffle(robo_lines)
with (OUT_DIR / "expert_mix_v3.jsonl").open("w", encoding="utf-8") as f:
    for line in other_lines:
        f.write(line)
    for _ in range(6):
        for line in robo_lines:
            f.write(line)
        rng.shuffle(robo_lines)
v3_total = len(other_lines) + 6 * len(robo_lines)
log(f"[v3] expert_mix_v3.jsonl = {v3_total} lines (robopoint 6x)")

# ── report ────────────────────────────────────────────────────────────
log("=== Done ===")
for p in sorted(OUT_DIR.glob("expert_mix_v*.jsonl")):
    n = sum(1 for _ in open(p, encoding="utf-8"))
    log(f"  {p.name}: {n} lines")

# Update manifest
manifest = {
    "data_root": str(DATA_ROOT),
    "out_dir": str(OUT_DIR),
    "seed": SEED,
    "versions": {
        "v1": "1x all (natural)",
        "v2": "RoboPoint 3x, others 1x",
        "v3": "RoboPoint 6x, others 1x",
    },
    "counts": {k: v for k, v in counts.items()},
    "robopoint_lines": len(robo_lines),
    "other_lines": len(other_lines),
}
(OUT_DIR / "manifest_mixes.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
)
log("[done]")
