#!/usr/bin/env python3
"""Build a raw-source holdout evaluation set for 8-model OPD comparisons.

The sampler excludes rows seen by the five 100k experts and the final OPD
student train set using source-line keys and content fingerprints. It also
tracks image overlap with train data so the final report can distinguish strict
image-disjoint rows from row-disjoint rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


EXPERTS = [
    "general_obj_expert",
    "general_reasoning_expert",
    "region_expert",
    "robopoint_expert",
    "spatial_rel_expert",
]
TRAIN_NAME = "train_shuffled_seed20260520.jsonl"
TRAIN_SEEN_LIMIT = 100_000

SOURCE_TO_POOL = {
    "refcoco": "refcoco",
    "flickr30k": "flickr30k_entities",
    "vg_object": "visual_genome_object",
    "vg_region": "visual_genome_region",
    "vg_relationship": "visual_genome_relationship",
    "vg_relationship_balanced": "visual_genome_relationship",
    "keepalive": "keepalive_vqa",
    "robopoint": "grounding_point",
}

POOL_FILES = {
    "refcoco": "refcoco_clean_v1.jsonl",
    "flickr30k_entities": "flickr30k_entities_clean_v1.jsonl",
    "visual_genome_object": "visual_genome_object_clean_v1.jsonl",
    "visual_genome_region": "visual_genome_region_clean_v1.jsonl",
    "visual_genome_relationship": "visual_genome_relationship_clean_v1.jsonl",
    "grounding_point": "grounding_point_clean_v1_mediaok.jsonl",
    "keepalive_vqa": "keepalive_vqa_clean_v1_mediaok.jsonl",
    "semantic_nav_box": "semantic_nav_box_clean_v1.jsonl",
}

QUOTAS = {
    "refcoco": 1100,
    "flickr30k_entities": 900,
    "visual_genome_object": 1100,
    "visual_genome_region": 1100,
    "visual_genome_relationship": 1100,
    "grounding_point": 1400,
    "keepalive_vqa": 2500,
    "semantic_nav_box": 800,
}

COORD_RE = re.compile(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]")


def now() -> str:
    return time.strftime("%F %T")


def log(obj: dict[str, Any]) -> None:
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def answer_of(row: dict[str, Any]) -> str:
    for turn in row.get("conversations") or []:
        if str(turn.get("from", "")).lower() in {"gpt", "assistant"}:
            return str(turn.get("value", ""))
    return ""


def first_image(row: dict[str, Any]) -> str:
    images = row.get("image") or []
    return str(images[0]) if images else ""


def row_fingerprint(row: dict[str, Any]) -> str:
    payload = {
        "image": row.get("image") or [],
        "dataset": row.get("dataset"),
        "target": row.get("target") or {},
        "answer": answer_of(row),
    }
    return stable_hash(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def expected_format(answer: str) -> str:
    has_point = "<point>" in answer
    has_box = "<box>" in answer
    if has_point and not has_box:
        return "point"
    if has_box and not has_point:
        return "box"
    if has_point and has_box:
        return "mixed_grounding"
    return "text"


def source_line_key_from_mix(row: dict[str, Any]) -> tuple[str, int] | None:
    mix = ((row.get("metadata") or {}).get("expert_mix") or {})
    source = SOURCE_TO_POOL.get(str(mix.get("source")))
    line_no = mix.get("source_line")
    if not source or line_no is None:
        return None
    try:
        return source, int(line_no)
    except Exception:
        return None


def source_line_key(pool: str, line_no: int) -> tuple[str, int]:
    return pool, line_no


def image_group(row: dict[str, Any], pool: str) -> str:
    if pool == "grounding_point":
        img = first_image(row)
        return img.split("/images/", 1)[-1].split("/", 1)[0] if "/images/" in img else "unknown"
    if pool == "refcoco":
        meta = row.get("metadata") or {}
        return f"{meta.get('source')}:{meta.get('split')}"
    if pool == "semantic_nav_box":
        meta = row.get("metadata") or {}
        return f"{meta.get('prompt_mode')}:{meta.get('relation')}"
    if pool == "visual_genome_relationship":
        target = row.get("target") or {}
        return str(target.get("relation") or "unknown")
    return pool


def valid_row(row: dict[str, Any]) -> bool:
    if not row.get("conversations") or not row.get("image") or not answer_of(row):
        return False
    answer = answer_of(row)
    fmt = expected_format(answer)
    if fmt == "mixed_grounding":
        return False
    total = 0
    gpt = 0
    for turn in row.get("conversations") or []:
        value = str(turn.get("value", ""))
        total += len(value)
        if str(turn.get("from", "")).lower() in {"gpt", "assistant"}:
            gpt += len(value)
    if total > 900 or (fmt == "point" and gpt > 500):
        return False
    coords = [(float(x), float(y)) for x, y in COORD_RE.findall(answer)]
    if fmt == "point":
        return 0 < len(coords) <= 50 and all(0 <= x <= 1000 and 0 <= y <= 1000 for x, y in coords)
    if fmt == "box":
        if len(coords) < 2:
            return False
        (x1, y1), (x2, y2) = coords[:2]
        return 0 <= x1 <= 1000 and 0 <= y1 <= 1000 and 0 <= x2 <= 1000 and 0 <= y2 <= 1000 and x2 > x1 and y2 > y1
    return True


def heap_add(heap: list[tuple[str, int, dict[str, Any]]], item: dict[str, Any], quota: int, seed: int) -> None:
    key = stable_hash(f"{seed}\0{item['pool']}\0{item['line_no']}\0{item['fingerprint']}")
    record = (key, item["line_no"], item)
    if len(heap) < quota:
        heap.append(record)
        if len(heap) == quota:
            heap.sort(reverse=True)
    elif key < heap[0][0]:
        heap[0] = record
        heap.sort(reverse=True)


def build_blocklists(train_root: Path, opd_train: Path) -> dict[str, Any]:
    blocked_keys: set[tuple[str, int]] = set()
    blocked_fp: set[str] = set()
    train_images: set[str] = set()
    counts = Counter()

    for expert in EXPERTS:
        path = train_root / expert / TRAIN_NAME
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if line_no > TRAIN_SEEN_LIMIT:
                    break
                row = json.loads(line)
                key = source_line_key_from_mix(row)
                if key:
                    blocked_keys.add(key)
                blocked_fp.add(row_fingerprint(row))
                img = first_image(row)
                if img:
                    train_images.add(img)
                counts[f"expert_seen:{expert}"] += 1

    with opd_train.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            key = source_line_key_from_mix(row)
            if key:
                blocked_keys.add(key)
            opd = ((row.get("metadata") or {}).get("opd") or {})
            fp = opd.get("fingerprint") or row_fingerprint(row)
            blocked_fp.add(str(fp))
            img = first_image(row)
            if img:
                train_images.add(img)
            counts["opd_train"] += 1

    return {
        "blocked_keys": blocked_keys,
        "blocked_fingerprints": blocked_fp,
        "train_images": train_images,
        "counts": dict(counts),
    }


def sample_pool(
    pool: str,
    path: Path,
    quota: int,
    block: dict[str, Any],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    over_quota = quota + max(100, quota // 4)
    strict_quota = over_quota
    relaxed_quota = over_quota * 2
    strict_heap: list[tuple[str, int, dict[str, Any]]] = []
    relaxed_heap: list[tuple[str, int, dict[str, Any]]] = []
    stats = Counter()
    group_counts = Counter()
    blocked_keys = block["blocked_keys"]
    blocked_fp = block["blocked_fingerprints"]
    train_images = block["train_images"]

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stats["rows_seen"] += 1
            try:
                row = json.loads(line)
            except Exception:
                stats["bad_json"] += 1
                continue
            if not valid_row(row):
                stats["invalid"] += 1
                continue
            key = source_line_key(pool, line_no)
            fp = row_fingerprint(row)
            img = first_image(row)
            if key in blocked_keys:
                stats["drop_source_line_seen"] += 1
                continue
            if fp in blocked_fp:
                stats["drop_fingerprint_seen"] += 1
                continue
            item = {
                "pool": pool,
                "source_file": str(path),
                "line_no": line_no,
                "row": row,
                "fingerprint": fp,
                "image_seen_in_train": img in train_images,
                "group": image_group(row, pool),
                "expected_format": expected_format(answer_of(row)),
            }
            if item["image_seen_in_train"]:
                stats["candidate_row_disjoint_image_seen"] += 1
                heap_add(relaxed_heap, item, relaxed_quota, seed + 19)
            else:
                stats["candidate_image_disjoint"] += 1
                heap_add(strict_heap, item, strict_quota, seed + 7)
            if line_no % 500_000 == 0:
                log({"stage": "pool_progress", "pool": pool, "line_no": line_no, "strict_kept": len(strict_heap), "relaxed_kept": len(relaxed_heap), "time": now()})

    selected_records = sorted(strict_heap)
    selected = [rec[2] for rec in selected_records[:over_quota]]
    if len(selected) < over_quota:
        need = over_quota - len(selected)
        strict_fps = {item["fingerprint"] for item in selected}
        relaxed = [rec[2] for rec in sorted(relaxed_heap) if rec[2]["fingerprint"] not in strict_fps]
        selected.extend(relaxed[:need])
    for item in selected:
        group_counts[item["group"]] += 1
    stats["selected_candidates"] = len(selected)
    stats["selected_image_seen"] = sum(1 for item in selected if item["image_seen_in_train"])
    return selected, {"stats": dict(stats), "groups": dict(group_counts.most_common())}


def annotate(item: dict[str, Any], idx: int) -> dict[str, Any]:
    row = dict(item["row"])
    meta = dict(row.get("metadata") or {})
    meta["raw_holdout_eval"] = {
        "version": "raw_holdout_eval_v1",
        "eval_index": idx,
        "source_pool": item["pool"],
        "source_file": item["source_file"],
        "source_line": item["line_no"],
        "group": item["group"],
        "fingerprint": item["fingerprint"],
        "expected_format": item["expected_format"],
        "image_seen_in_train": bool(item["image_seen_in_train"]),
        "dedupe": "source_line_and_fingerprint_excluded_from_expert_seen100k_and_opd_train",
    }
    row["metadata"] = meta
    row["gold"] = answer_of(row)
    return row


def verify(path: Path) -> dict[str, Any]:
    rows = 0
    bad_json = 0
    fps: set[str] = set()
    dup_fp = 0
    images = Counter()
    pools = Counter()
    formats = Counter()
    image_seen = Counter()
    for line in path.open("r", encoding="utf-8"):
        if not line.strip():
            continue
        rows += 1
        try:
            row = json.loads(line)
        except Exception:
            bad_json += 1
            continue
        meta = ((row.get("metadata") or {}).get("raw_holdout_eval") or {})
        fp = str(meta.get("fingerprint") or row_fingerprint(row))
        if fp in fps:
            dup_fp += 1
        fps.add(fp)
        img = first_image(row)
        if img:
            images[img] += 1
        pools[str(meta.get("source_pool"))] += 1
        formats[str(meta.get("expected_format"))] += 1
        image_seen[str(bool(meta.get("image_seen_in_train")))] += 1
    return {
        "rows": rows,
        "bad_json": bad_json,
        "unique_fingerprints": len(fps),
        "duplicate_fingerprints": dup_fp,
        "unique_images": len(images),
        "reused_images_within_eval": sum(1 for v in images.values() if v > 1),
        "pool_counts": dict(pools.most_common()),
        "format_counts": dict(formats.most_common()),
        "image_seen_in_train_counts": dict(image_seen.most_common()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-root", default="/data/msz/point/data_grounding_clean_v1")
    parser.add_argument("--train-root", default="/data/msz/point/data_expert_seed0_v1_shuffled")
    parser.add_argument("--opd-train", default="/data/msz/point/opd_student_v1/train_prompts.jsonl")
    parser.add_argument("--out-root", default="/data/msz/point/eval_raw_holdout_v1")
    parser.add_argument("--seed", type=int, default=20260525)
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    block = build_blocklists(Path(args.train_root), Path(args.opd_train))
    log({
        "stage": "blocklists",
        "blocked_source_keys": len(block["blocked_keys"]),
        "blocked_fingerprints": len(block["blocked_fingerprints"]),
        "train_images": len(block["train_images"]),
        "time": now(),
    })

    candidates_by_pool: dict[str, list[dict[str, Any]]] = {}
    pool_summaries = {}
    for i, (pool, quota) in enumerate(QUOTAS.items()):
        path = Path(args.clean_root) / POOL_FILES[pool]
        selected, summary = sample_pool(pool, path, quota, block, args.seed + i * 1000)
        candidates_by_pool[pool] = selected
        pool_summaries[pool] = summary
        log({"stage": "selected_pool", "pool": pool, "quota": quota, "candidates": len(selected), "image_seen": summary["stats"].get("selected_image_seen", 0), "time": now()})

    all_items: list[dict[str, Any]] = []
    global_fps: set[str] = set()
    final_pool_counts = Counter()
    final_duplicate_skips = Counter()
    for pool, quota in QUOTAS.items():
        pool_candidates = sorted(
            candidates_by_pool[pool],
            key=lambda item: stable_hash(f"{args.seed}\0final\0{pool}\0{item['line_no']}\0{item['fingerprint']}"),
        )
        for item in pool_candidates:
            if final_pool_counts[pool] >= quota:
                break
            if item["fingerprint"] in global_fps:
                final_duplicate_skips[pool] += 1
                continue
            global_fps.add(item["fingerprint"])
            all_items.append(item)
            final_pool_counts[pool] += 1
        if final_pool_counts[pool] < quota:
            raise RuntimeError(f"pool {pool} underfilled after global dedupe: {final_pool_counts[pool]} < {quota}")

    random.Random(args.seed).shuffle(all_items)
    eval_path = out_root / "raw_holdout_eval_v1_10k.jsonl"
    tmp = eval_path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as wf:
        for idx, item in enumerate(all_items):
            wf.write(json.dumps(annotate(item, idx), ensure_ascii=False) + "\n")
    os.replace(tmp, eval_path)
    summary = {
        "version": "raw_holdout_eval_v1",
        "created_at": now(),
        "seed": args.seed,
        "eval_path": str(eval_path),
        "quotas": QUOTAS,
        "pool_files": {k: str(Path(args.clean_root) / v) for k, v in POOL_FILES.items()},
        "blocklist_counts": {
            "blocked_source_keys": len(block["blocked_keys"]),
            "blocked_fingerprints": len(block["blocked_fingerprints"]),
            "train_images": len(block["train_images"]),
            "inputs": block["counts"],
        },
        "pool_summaries": pool_summaries,
        "final_pool_counts": dict(final_pool_counts),
        "final_duplicate_skips": dict(final_duplicate_skips),
        "verify": verify(eval_path),
        "skipped_remote_datasets": {
            "PhraseCut": "present but not converted in the active clean pool; needs mask/phrase conversion for a separate OOD eval",
            "Talk2Car-Slim": "JSONL present but full image set is not present locally",
            "RoboRefIt": "parquet present but previously excluded for abnormal boxes; needs re-cleaning before OOD eval",
            "EmbSpatial": "URL-image rows excluded from media-ok keepalive; current media-ok general eval uses Robo2VLM-1",
            "Phys100k/Struct2D-Set/embodied_jsons": "empty directories",
        },
    }
    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log({"stage": "done", "eval_path": str(eval_path), "verify": summary["verify"], "time": now()})


if __name__ == "__main__":
    main()
