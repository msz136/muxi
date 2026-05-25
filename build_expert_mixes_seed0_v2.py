#!/usr/bin/env python3
"""Build seed=0 train/eval mixes for five expert models.

Inputs are the cleaned, media-ok public/remote pools only. Old synthetic region
datasets, PhraseCut, Talk2Car image data, and RoboRefIt are intentionally absent.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import random
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


TRAIN_TOTAL = 800_000
EVAL_TOTAL = 20_000
EVAL_HASH_MOD = 10_000
EVAL_HASH_THRESHOLD = 500  # 5% image-level candidate split.


SOURCE_FILES = {
    "refcoco": "refcoco_clean_v1.jsonl",
    "flickr30k": "flickr30k_entities_clean_v1.jsonl",
    "vg_object": "visual_genome_object_clean_v1.jsonl",
    "vg_region": "visual_genome_region_clean_v1.jsonl",
    "vg_relationship": "visual_genome_relationship_clean_v1.jsonl",
    "keepalive": "keepalive_vqa_clean_v1_mediaok.jsonl",
    "robopoint": "grounding_point_clean_v1_mediaok.jsonl",
}


EXPERT_TRAIN_PLAN: dict[str, dict[str, int]] = {
    "general_reasoning_expert": {
        "keepalive": 640_000,
        "refcoco": 40_000,
        "vg_region": 40_000,
        "vg_object": 30_000,
        "vg_relationship": 20_000,
        "robopoint": 30_000,
    },
    "robopoint_expert": {
        "robopoint": 640_000,
        "keepalive": 80_000,
        "vg_object": 30_000,
        "refcoco": 20_000,
        "vg_region": 20_000,
        "vg_relationship": 10_000,
    },
    "general_obj_expert": {
        "refcoco": 200_000,
        "flickr30k": 180_000,
        "vg_object": 220_000,
        "vg_region": 80_000,
        "vg_relationship": 60_000,
        "keepalive": 40_000,
        "robopoint": 20_000,
    },
    "region_expert": {
        "vg_region": 640_000,
        "refcoco": 40_000,
        "flickr30k": 40_000,
        "vg_object": 30_000,
        "vg_relationship": 30_000,
        "keepalive": 10_000,
        "robopoint": 10_000,
    },
    "spatial_rel_expert": {
        "vg_relationship_balanced": 640_000,
        "vg_region": 50_000,
        "vg_object": 40_000,
        "refcoco": 30_000,
        "flickr30k": 20_000,
        "keepalive": 10_000,
        "robopoint": 10_000,
    },
}

REL_BUCKET_WEIGHTS = {
    "on": 0.18,
    "in_inside": 0.18,
    "near_next_beside": 0.14,
    "under_below": 0.10,
    "above_over": 0.10,
    "front": 0.08,
    "behind": 0.08,
    "left_right": 0.06,
    "between": 0.04,
    "other_spatial": 0.04,
}


@dataclass(frozen=True)
class Bucket:
    expert: str
    split: str
    source: str


def stable_int(text: str, seed: int) -> int:
    digest = hashlib.sha1(f"{seed}\t{text}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def split_for_image(row: dict[str, Any], source: str, seed: int) -> str:
    images = row.get("image") or []
    videos = row.get("video") or []
    key = images[0] if images else (videos[0] if videos else json.dumps(row.get("target") or row.get("metadata") or {}, sort_keys=True))
    value = stable_int(str(key), seed) % EVAL_HASH_MOD
    return "eval" if value < EVAL_HASH_THRESHOLD else "train"


def scaled_eval_counts(train_counts: dict[str, int]) -> dict[str, int]:
    raw = {k: v * EVAL_TOTAL / TRAIN_TOTAL for k, v in train_counts.items()}
    counts = {k: int(math.floor(v)) for k, v in raw.items()}
    remainder = EVAL_TOTAL - sum(counts.values())
    order = sorted(raw, key=lambda k: (raw[k] - counts[k], raw[k]), reverse=True)
    for k in order[:remainder]:
        counts[k] += 1
    return counts


def real_source(source: str) -> str:
    return "vg_relationship" if source == "vg_relationship_balanced" else source


def rel_bucket(row: dict[str, Any]) -> str:
    rel = str((row.get("target") or {}).get("relation") or "").lower()
    if "front" in rel:
        return "front"
    if "behind" in rel:
        return "behind"
    if "left" in rel or "right" in rel:
        return "left_right"
    if "between" in rel:
        return "between"
    if any(x in rel for x in ("next to", "beside", "near")):
        return "near_next_beside"
    if any(x in rel for x in ("under", "below", "beneath")):
        return "under_below"
    if any(x in rel for x in ("above", "over")):
        return "above_over"
    if any(x in rel for x in ("inside", " in ", "in a", "in the")) or rel == "in":
        return "in_inside"
    if "on" in rel:
        return "on"
    return "other_spatial"


def weighted_bucket_counts(total: int) -> dict[str, int]:
    raw = {k: total * v for k, v in REL_BUCKET_WEIGHTS.items()}
    counts = {k: int(math.floor(v)) for k, v in raw.items()}
    remainder = total - sum(counts.values())
    order = sorted(raw, key=lambda k: raw[k] - counts[k], reverse=True)
    for k in order[:remainder]:
        counts[k] += 1
    return counts


def add_sample(reservoir: list[int], line_no: int, seen: int, target: int, rng: random.Random) -> None:
    if target <= 0:
        return
    if len(reservoir) < target:
        reservoir.append(line_no)
        return
    j = rng.randrange(seen)
    if j < target:
        reservoir[j] = line_no


def source_counts_by_split(
    clean_dir: Path,
    source: str,
    seed: int,
    normal_targets: dict[Bucket, int],
    balanced_targets: dict[tuple[str, str], dict[str, int]],
) -> tuple[dict[Bucket, list[int]], dict[tuple[str, str, str], list[int]], dict[str, Any]]:
    path = clean_dir / SOURCE_FILES[source]
    rngs: dict[Any, random.Random] = {}
    normal_seen: dict[Bucket, int] = collections.Counter()
    normal_samples: dict[Bucket, list[int]] = {bucket: [] for bucket in normal_targets}
    balanced_seen: dict[tuple[str, str, str], int] = collections.Counter()
    balanced_samples: dict[tuple[str, str, str], list[int]] = {}
    for (expert, split), target_by_bucket in balanced_targets.items():
        for bucket_name in target_by_bucket:
            balanced_samples[(expert, split, bucket_name)] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            row = json.loads(line)
            split = split_for_image(row, source, seed)
            for bucket, target in normal_targets.items():
                if bucket.split != split:
                    continue
                normal_seen[bucket] += 1
                rng = rngs.setdefault(("normal", bucket), random.Random(stable_int(str(bucket), seed)))
                add_sample(normal_samples[bucket], line_no, normal_seen[bucket], target, rng)
            if source == "vg_relationship":
                rb = rel_bucket(row)
                for (expert, bsplit), target_by_bucket in balanced_targets.items():
                    if bsplit != split or rb not in target_by_bucket:
                        continue
                    key = (expert, split, rb)
                    balanced_seen[key] += 1
                    rng = rngs.setdefault(("balanced", key), random.Random(stable_int(str(key), seed)))
                    add_sample(balanced_samples[key], line_no, balanced_seen[key], target_by_bucket[rb], rng)

    # Oversample rare relation buckets when needed.
    oversample_stats: dict[str, Any] = {}
    for key, samples in balanced_samples.items():
        expert, split, bucket_name = key
        target = balanced_targets[(expert, split)][bucket_name]
        if len(samples) < target:
            if not samples:
                raise RuntimeError(f"no samples for balanced bucket {key}")
            rng = random.Random(stable_int(f"oversample\t{key}", seed))
            original = len(samples)
            while len(samples) < target:
                samples.append(rng.choice(samples[:original]))
            oversample_stats[str(key)] = {"unique": original, "target": target, "oversampled": target - original}

    stats = {
        "source": source,
        "path": str(path),
        "normal_seen": {str(k): v for k, v in normal_seen.items()},
        "normal_selected": {str(k): len(v) for k, v in normal_samples.items()},
        "balanced_seen": {str(k): v for k, v in balanced_seen.items()},
        "balanced_selected": {str(k): len(v) for k, v in balanced_samples.items()},
        "balanced_oversample": oversample_stats,
    }
    return normal_samples, balanced_samples, stats


def build_targets() -> tuple[dict[str, dict[Bucket, int]], dict[str, dict[tuple[str, str], dict[str, int]]], dict[str, Any]]:
    normal: dict[str, dict[Bucket, int]] = collections.defaultdict(dict)
    balanced: dict[str, dict[tuple[str, str], dict[str, int]]] = collections.defaultdict(dict)
    plans: dict[str, Any] = {}
    for expert, train_plan in EXPERT_TRAIN_PLAN.items():
        eval_plan = scaled_eval_counts(train_plan)
        plans[expert] = {"train": train_plan, "eval": eval_plan}
        for split, source_plan in (("train", train_plan), ("eval", eval_plan)):
            for src, count in source_plan.items():
                if src == "vg_relationship_balanced":
                    balanced["vg_relationship"][(expert, split)] = weighted_bucket_counts(count)
                else:
                    normal[real_source(src)][Bucket(expert, split, src)] = count
    return normal, balanced, plans


def make_index(samples: dict[Bucket, list[int]], balanced_samples: dict[tuple[str, str, str], list[int]]) -> dict[int, list[tuple[str, str, str, str | None]]]:
    index: dict[int, list[tuple[str, str, str, str | None]]] = collections.defaultdict(list)
    for bucket, line_numbers in samples.items():
        for line_no in line_numbers:
            index[line_no].append((bucket.expert, bucket.split, bucket.source, None))
    for (expert, split, rel_bucket_name), line_numbers in balanced_samples.items():
        for line_no in line_numbers:
            index[line_no].append((expert, split, "vg_relationship_balanced", rel_bucket_name))
    return index


def prepare_row(row: dict[str, Any], expert: str, split: str, source: str, source_line: int, seed: int, rel_bucket_name: str | None) -> dict[str, Any]:
    out = dict(row)
    metadata = dict(out.get("metadata") or {})
    metadata["expert_mix"] = {
        "expert": expert,
        "split": split,
        "source": source,
        "source_line": source_line,
        "seed": seed,
    }
    if rel_bucket_name is not None:
        metadata["expert_mix"]["relation_bucket"] = rel_bucket_name
    out["metadata"] = metadata
    return out


def write_outputs(
    clean_dir: Path,
    out_dir: Path,
    seed: int,
    by_source_normal: dict[str, dict[Bucket, list[int]]],
    by_source_balanced: dict[str, dict[tuple[str, str, str], list[int]]],
) -> dict[str, Any]:
    handles: dict[tuple[str, str], Any] = {}
    counts: dict[str, dict[str, int]] = {expert: {"train": 0, "eval": 0} for expert in EXPERT_TRAIN_PLAN}
    source_counts: dict[str, dict[str, collections.Counter[str]]] = {
        expert: {"train": collections.Counter(), "eval": collections.Counter()} for expert in EXPERT_TRAIN_PLAN
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        for expert in EXPERT_TRAIN_PLAN:
            expert_dir = out_dir / expert
            expert_dir.mkdir(parents=True, exist_ok=True)
            for split in ("train", "eval"):
                handles[(expert, split)] = (expert_dir / f"{split}.jsonl").open("w", encoding="utf-8")

        for source, file_name in SOURCE_FILES.items():
            index = make_index(by_source_normal.get(source, {}), by_source_balanced.get(source, {}))
            if not index:
                continue
            path = clean_dir / file_name
            with path.open("r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    assignments = index.get(line_no)
                    if not assignments:
                        continue
                    row = json.loads(line)
                    for expert, split, mix_source, rel_bucket_name in assignments:
                        out = prepare_row(row, expert, split, mix_source, line_no, seed, rel_bucket_name)
                        handles[(expert, split)].write(json.dumps(out, ensure_ascii=False) + "\n")
                        counts[expert][split] += 1
                        source_counts[expert][split][mix_source] += 1
    finally:
        for handle in handles.values():
            handle.close()

    return {
        "counts": counts,
        "source_counts": {
            expert: {split: dict(counter) for split, counter in split_map.items()}
            for expert, split_map in source_counts.items()
        },
    }


def extract_box_or_point(row: dict[str, Any]) -> tuple[str, list[list[int]]] | None:
    target = row.get("target") or {}
    box = target.get("box")
    if isinstance(box, list) and len(box) == 2:
        return "box", box
    answer = " ".join(
        str(t.get("value", ""))
        for t in row.get("conversations") or []
        if str(t.get("from", "")).lower() in {"gpt", "assistant"}
    )
    box_match = re.search(r"<box>\s*\[\[(\d+),(\d+)\],\[(\d+),(\d+)\]\]\s*</box>", answer.replace(" ", ""))
    if box_match:
        x1, y1, x2, y2 = map(int, box_match.groups())
        return "box", [[x1, y1], [x2, y2]]
    point_match = re.search(r"<point>\s*(\[\[.*?\]\])\s*</point>", answer.replace(" ", ""))
    if point_match:
        try:
            pts = json.loads(point_match.group(1))
            if pts:
                return "point", pts[:8]
        except Exception:
            return None
    return None


def label_for(row: dict[str, Any]) -> str:
    target = row.get("target") or {}
    if target.get("description"):
        return str(target["description"])
    if target.get("object_name") and target.get("relation"):
        return f"{target['object_name']} {target['relation']} {target.get('anchor_object', '')}"
    if target.get("object_name"):
        return str(target["object_name"])
    for turn in row.get("conversations") or []:
        if str(turn.get("from", "")).lower() in {"human", "user"}:
            return str(turn.get("value", "")).replace("<image>", "").strip()
    return str(row.get("dataset", ""))


def preview_expert(expert_dir: Path, seed: int, n: int = 24) -> dict[str, Any]:
    rng = random.Random(stable_int(str(expert_dir), seed))
    train_path = expert_dir / "train.jsonl"
    sample: list[str] = []
    with train_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            if len(sample) < n:
                sample.append(line)
            else:
                j = rng.randrange(idx)
                if j < n:
                    sample[j] = line
    tiles: list[Image.Image] = []
    skipped = 0
    font = ImageFont.load_default()
    for line in sample:
        row = json.loads(line)
        media = extract_box_or_point(row)
        images = row.get("image") or []
        if not media or not images or not Path(str(images[0])).exists():
            skipped += 1
            continue
        kind, coords = media
        try:
            img = Image.open(str(images[0])).convert("RGB")
        except Exception:
            skipped += 1
            continue
        img.thumbnail((260, 200))
        canvas = Image.new("RGB", (280, 250), "white")
        xoff = (280 - img.width) // 2
        canvas.paste(img, (xoff, 8))
        draw = ImageDraw.Draw(canvas)
        sx, sy = img.width / 1000.0, img.height / 1000.0
        if kind == "box":
            (x1, y1), (x2, y2) = coords[:2]
            draw.rectangle([xoff + x1 * sx, 8 + y1 * sy, xoff + x2 * sx, 8 + y2 * sy], outline=(0, 114, 255), width=3)
        else:
            for x, y in coords:
                cx, cy = xoff + int(x) * sx, 8 + int(y) * sy
                draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=(0, 114, 255))
        mix = (row.get("metadata") or {}).get("expert_mix") or {}
        title = f"{mix.get('source', row.get('dataset', ''))}"[:38]
        wrapped = textwrap.wrap(label_for(row), width=42)[:2]
        draw.text((8, 212), title, fill=(0, 0, 0), font=font)
        draw.text((8, 226), "\n".join(wrapped), fill=(40, 40, 40), font=font)
        tiles.append(canvas)
    if not tiles:
        return {"preview_rows": 0, "preview_skipped": skipped}
    cols = 4
    rows = math.ceil(len(tiles) / cols)
    sheet = Image.new("RGB", (cols * 280, rows * 250), "white")
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % cols) * 280, (i // cols) * 250))
    path = expert_dir / "preview_train.png"
    sheet.save(path)
    return {"preview_path": str(path), "preview_rows": len(tiles), "preview_skipped": skipped}


def _valid_media_for_verify(media: tuple[str, list[list[int]]] | None) -> bool:
    if media is None:
        return False
    kind, coords = media
    if kind == "box":
        if not isinstance(coords, list) or len(coords) != 2:
            return False
        try:
            (x1, y1), (x2, y2) = coords
            values = [int(x1), int(y1), int(x2), int(y2)]
        except Exception:
            return False
        if any(v < 0 or v > 1000 for v in values):
            return False
        return values[0] <= values[2] and values[1] <= values[3]
    if kind == "point":
        if not isinstance(coords, list) or not coords:
            return False
        for point in coords:
            if not isinstance(point, list) or len(point) < 2:
                return False
            try:
                x, y = int(point[0]), int(point[1])
            except Exception:
                return False
            if x < 0 or x > 1000 or y < 0 or y > 1000:
                return False
        return True
    return False


def verify_outputs(out_dir: Path) -> dict[str, Any]:
    report: dict[str, Any] = {}
    sample_n = 1000
    for expert in EXPERT_TRAIN_PLAN:
        train_images: set[str] = set()
        eval_images: set[str] = set()
        sampled_media_missing = {"train": 0, "eval": 0}
        sampled_media_checked = {"train": 0, "eval": 0}
        bad_json = {"train": 0, "eval": 0}
        bad_conversations = {"train": 0, "eval": 0}
        bad_format = {"train": 0, "eval": 0}
        empty_image = {"train": 0, "eval": 0}
        row_count = {"train": 0, "eval": 0}
        samples = {"train": [], "eval": []}
        rngs = {
            "train": random.Random(stable_int(f"{expert}\ttrain\tverify", 0)),
            "eval": random.Random(stable_int(f"{expert}\teval\tverify", 0)),
        }
        for split, bucket in (("train", train_images), ("eval", eval_images)):
            with (out_dir / expert / f"{split}.jsonl").open("r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    row_count[split] = line_no
                    try:
                        row = json.loads(line)
                    except Exception:
                        bad_json[split] += 1
                        continue
                    images = row.get("image") or []
                    if images:
                        image = str(images[0])
                        bucket.add(image)
                        if len(samples[split]) < sample_n:
                            samples[split].append(image)
                        else:
                            j = rngs[split].randrange(line_no)
                            if j < sample_n:
                                samples[split][j] = image
                    else:
                        empty_image[split] += 1
                    conversations = row.get("conversations") or []
                    if not isinstance(conversations, list) or len(conversations) < 2:
                        bad_conversations[split] += 1
                    row_text = json.dumps(row, ensure_ascii=False)
                    if ("<box>" in row_text or "<point>" in row_text) and not _valid_media_for_verify(extract_box_or_point(row)):
                        bad_format[split] += 1
        for split in ("train", "eval"):
            for image in samples[split]:
                sampled_media_checked[split] += 1
                if not Path(image).exists():
                    sampled_media_missing[split] += 1
        report[expert] = {
            "row_count": row_count,
            "train_unique_images": len(train_images),
            "eval_unique_images": len(eval_images),
            "image_overlap_train_eval": len(train_images & eval_images),
            "sampled_media_checked": sampled_media_checked,
            "sampled_media_missing": sampled_media_missing,
            "empty_image": empty_image,
            "bad_json": bad_json,
            "bad_conversations": bad_conversations,
            "bad_format": bad_format,
            "media_check_mode": "full media existence is guaranteed by upstream clean media-ok pools; this verifies a deterministic sample per expert/split",
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-dir", default="/data/msz/point/data_grounding_clean_v1")
    parser.add_argument("--out-dir", default="/data/msz/point/data_expert_seed0_v1")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    clean_dir = Path(args.clean_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    normal_targets, balanced_targets, plans = build_targets()

    by_source_normal: dict[str, dict[Bucket, list[int]]] = {}
    by_source_balanced: dict[str, dict[tuple[str, str, str], list[int]]] = {}
    source_stats: dict[str, Any] = {}
    for source in SOURCE_FILES:
        normal_samples, balanced_samples, stats = source_counts_by_split(
            clean_dir,
            source,
            args.seed,
            normal_targets.get(source, {}),
            balanced_targets.get(source, {}),
        )
        by_source_normal[source] = normal_samples
        by_source_balanced[source] = balanced_samples
        source_stats[source] = stats
        print(json.dumps({"stage": "sampled", "source": source, "stats": stats}, ensure_ascii=False), flush=True)

    output_stats = write_outputs(clean_dir, out_dir, args.seed, by_source_normal, by_source_balanced)
    preview_stats = {expert: preview_expert(out_dir / expert, args.seed) for expert in EXPERT_TRAIN_PLAN}
    verify_stats = verify_outputs(out_dir)
    summary = {
        "seed": args.seed,
        "train_total_per_expert": TRAIN_TOTAL,
        "eval_total_per_expert": EVAL_TOTAL,
        "clean_dir": str(clean_dir),
        "out_dir": str(out_dir),
        "plans": plans,
        "source_sampling": source_stats,
        "outputs": output_stats,
        "preview": preview_stats,
        "verify": verify_stats,
        "excluded_sources": ["old synthetic semantic-nav/region data", "PhraseCut", "Talk2Car image version", "RoboRefIt"],
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({"summary": str(out_dir / "summary.json"), "outputs": output_stats["counts"], "verify": verify_stats}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
