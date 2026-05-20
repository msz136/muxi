#!/usr/bin/env python3
"""Build held-out semantic-navigation box eval sets from RoboPoint point rows.

This intentionally excludes prompt_ids already used by the full object_ref SFT
set, then rewrites point targets into box-grounding prompts.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = (
    "You are a semantic navigation grounding assistant. Given an image and "
    "target region information, return the target region's bounding box in "
    "coordinates from 0 to 1000. Return only <box>[[x1,y1],[x2,y2]]</box>."
)

RELATION_PHRASES = {
    "on": "on the highlighted area or surface",
    "left": "to the left of the highlighted object",
    "right": "to the right of the highlighted object",
    "inside": "inside the highlighted container or area",
    "beside": "beside the highlighted object",
    "front": "in front of the highlighted object",
    "behind": "behind the highlighted object",
    "between": "between the highlighted objects",
    "above": "above the highlighted object",
    "below": "below the highlighted object",
    "on-front": "on the front part of the highlighted area or surface",
    "on-back": "on the back part of the highlighted area or surface",
    "on-left": "on the left part of the highlighted area or surface",
    "on-right": "on the right part of the highlighted area or surface",
}

ANCHOR_BY_RELATION = {
    "on": "highlighted area or surface",
    "inside": "highlighted container or area",
    "between": "highlighted objects",
    "on-front": "highlighted area or surface",
    "on-back": "highlighted area or surface",
    "on-left": "highlighted area or surface",
    "on-right": "highlighted area or surface",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def user_text(row: dict[str, Any]) -> str:
    for msg in row.get("messages", []):
        if isinstance(msg, dict) and msg.get("role") == "user":
            return str(msg.get("content", ""))
    return ""


def relation_from_prompt_id(prompt_id: str) -> str:
    for rel in sorted(RELATION_PHRASES, key=len, reverse=True):
        if prompt_id.endswith("_" + rel):
            return rel
    tail = prompt_id.rsplit("_", 1)[-1] if "_" in prompt_id else ""
    return tail if tail in RELATION_PHRASES else ""


def make_box(points: list[list[Any]], margin_ratio: float, min_margin: int) -> list[list[int]]:
    xs = [int(round(float(p[0]))) for p in points if isinstance(p, (list, tuple)) and len(p) >= 2]
    ys = [int(round(float(p[1]))) for p in points if isinstance(p, (list, tuple)) and len(p) >= 2]
    if not xs or not ys:
        raise ValueError("empty point set")
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    mx = max(min_margin, round(width * margin_ratio))
    my = max(min_margin, round(height * margin_ratio))
    return [[max(0, x1 - mx), max(0, y1 - my)], [min(1000, x2 + mx), min(1000, y2 + my)]]


def box_stats(box: list[list[int]]) -> tuple[int, int, int]:
    width = box[1][0] - box[0][0]
    height = box[1][1] - box[0][1]
    return width, height, width * height


def infer_region_name(text: str, prefix: str) -> str:
    t = text.lower()
    if any(s in t for s in ("unoccupied", "empty", "free space", "vacant")):
        return "target free space"
    if any(s in t for s in ("container", "inside", "drawer", "cabinet")) and "inside" in t:
        return "target container interior"
    if any(s in t for s in ("surface", "plane", "area", "region")):
        return "target surface region"
    if prefix == "region_ref":
        return "target region"
    return "target object"


def relation_phrase(relation: str) -> str:
    return RELATION_PHRASES[relation]


def anchor_object(relation: str) -> str:
    return ANCHOR_BY_RELATION.get(relation, "highlighted object")


def box_answer(box: list[list[int]]) -> str:
    return "<box>" + json.dumps(box, separators=(",", ":")) + "</box>"


def make_eval_rows(base: dict[str, Any], variants: set[str]) -> list[dict[str, Any]]:
    relation = base["relation"]
    rel_phrase = relation_phrase(relation)
    anchor = anchor_object(relation)
    region_name = base["region_name"]
    answer = box_answer(base["box"])
    prompts: list[tuple[str, str, dict[str, Any]]] = []

    if "relation_plain" in variants:
        prompts.append(
            (
                "relation_plain",
                f"<image>\nFind the target region {rel_phrase}. Return only <box>[[x1,y1],[x2,y2]]</box> with integer coordinates from 0 to 1000.",
                {"object_name": None, "relation": rel_phrase, "anchor_object": anchor},
            )
        )
    if "semantic_json" in variants:
        info = {"target_region": region_name, "relation": rel_phrase, "anchor_object": anchor}
        prompts.append(
            (
                "semantic_json",
                "<image>\nTarget region information:\n"
                + json.dumps(info, ensure_ascii=False, separators=(",", ":"))
                + "\nReturn only the bounding box as <box>[[x1,y1],[x2,y2]]</box>.",
                {"object_name": region_name, "relation": rel_phrase, "anchor_object": anchor},
            )
        )
    if "object_relation_compat" in variants:
        info = {"object_name": region_name, "relation": rel_phrase, "anchor_object": anchor}
        prompts.append(
            (
                "object_relation_compat",
                "<image>\nTarget object information:\n"
                + json.dumps(info, ensure_ascii=False, separators=(",", ":"))
                + "\nReturn only the bounding box as <box>[[x1,y1],[x2,y2]]</box>.",
                {"object_name": region_name, "relation": rel_phrase, "anchor_object": anchor},
            )
        )

    rows = []
    for mode, prompt, target in prompts:
        rows.append(
            {
                "dataset": "semantic_nav_box_grounding_heldout_eval_v1",
                "image": [base["image"]],
                "video": [],
                "target": {
                    **target,
                    "region_name": region_name,
                    "box": base["box"],
                    "point_count": base["point_count"],
                },
                "conversations": [
                    {"from": "system", "value": SYSTEM_PROMPT},
                    {"from": "human", "value": prompt},
                    {"from": "gpt", "value": answer},
                ],
                "metadata": {
                    "source": "robopoint_heldout_point_to_box",
                    "task_type": "box_grounding_eval",
                    "prompt_mode": mode,
                    "split": "eval",
                    "heldout_reason": base["heldout_reason"],
                    "original_prompt_id": base["prompt_id"],
                    "relation": relation,
                    "relation_phrase": rel_phrase,
                    "prefix": base["prefix"],
                    "region_name_rule": base["region_name_rule"],
                    "conversion": "points_bbox_margin_v1",
                    "point_count": base["point_count"],
                    "box_area": base["box_area"],
                    "old_user": base["old_user"],
                },
            }
        )
    return rows


def load_train_ids(paths: list[Path]) -> set[str]:
    ids: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for row in read_jsonl(path):
            prompt_id = row.get("prompt_id") or row.get("metadata", {}).get("original_prompt_id")
            if prompt_id:
                ids.add(str(prompt_id))
    return ids


def build_candidates(args: argparse.Namespace) -> tuple[list[dict[str, Any]], Counter]:
    train_ids = load_train_ids(args.exclude_manifests)
    prefixes = set(args.prefixes.split(","))
    relations = set(args.relations.split(","))
    dropped: Counter = Counter()
    candidates = []
    for row in read_jsonl(args.input):
        if row.get("metadata", {}).get("source") != "robopoint":
            dropped["not_robopoint"] += 1
            continue
        prompt_id = str(row.get("prompt_id", ""))
        prefix = prompt_id.split("/", 1)[0] if "/" in prompt_id else ""
        if prefix not in prefixes:
            dropped["prefix"] += 1
            continue
        heldout_reason = "prefix_not_trained"
        if prompt_id in train_ids:
            dropped["in_train_manifest"] += 1
            continue
        if prefix == "object_ref":
            heldout_reason = "object_ref_prompt_id_not_in_train_manifest"
        relation = relation_from_prompt_id(prompt_id)
        if relation not in relations:
            dropped["relation"] += 1
            continue
        images = row.get("images") or row.get("image") or []
        if isinstance(images, str):
            images = [images]
        image = str(images[0]) if images else ""
        if not image or not Path(image).exists():
            dropped["missing_image"] += 1
            continue
        points = row.get("gt_points") or []
        if len(points) < args.min_points:
            dropped["point_count"] += 1
            continue
        try:
            box = make_box(points, args.margin_ratio, args.min_margin)
        except Exception:
            dropped["bad_points"] += 1
            continue
        width, height, area = box_stats(box)
        if width < args.min_box_side or height < args.min_box_side:
            dropped["small_box"] += 1
            continue
        if area < args.min_box_area:
            dropped["small_area"] += 1
            continue
        if area > args.max_box_area:
            dropped["large_area"] += 1
            continue
        old = user_text(row)
        region_name = infer_region_name(old, prefix)
        candidates.append(
            {
                "prompt_id": prompt_id,
                "image": image,
                "old_user": old,
                "box": box,
                "relation": relation,
                "prefix": prefix,
                "point_count": len(points),
                "box_area": area,
                "region_name": region_name,
                "region_name_rule": "keyword_inference_v1",
                "heldout_reason": heldout_reason,
            }
        )
    return candidates, dropped


def stratified_sample(candidates: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    rng = random.Random(args.seed)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        groups[(row["prefix"], row["relation"])].append(row)
    selected: list[dict[str, Any]] = []
    for key in sorted(groups):
        values = groups[key]
        rng.shuffle(values)
        selected.extend(values[: args.max_per_group])
    rng.shuffle(selected)
    if len(selected) > args.max_base_samples:
        selected = selected[: args.max_base_samples]
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("/data/msz/opd_project/data/prompt_pool_clean.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("/data/msz/opd_project/data/semantic_nav_box_v1/eval"))
    parser.add_argument("--name", default="semantic_nav_box_heldout_eval_v1")
    parser.add_argument(
        "--exclude-manifests",
        type=Path,
        nargs="*",
        default=[Path("/data/msz/opd_project/data/semantic_nav_box_v1/semantic_nav_box_grounding_full_object_ref_v1_manifest.jsonl")],
    )
    parser.add_argument("--prefixes", default="region_ref,object_ref")
    parser.add_argument("--relations", default="on,left,right,inside,beside,front,behind,between,on-front,on-back,on-left,on-right,above,below")
    parser.add_argument("--variants", default="relation_plain,semantic_json,object_relation_compat")
    parser.add_argument("--max-base-samples", type=int, default=512)
    parser.add_argument("--max-per-group", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--margin-ratio", type=float, default=0.15)
    parser.add_argument("--min-margin", type=int, default=10)
    parser.add_argument("--min-points", type=int, default=4)
    parser.add_argument("--min-box-side", type=int, default=24)
    parser.add_argument("--min-box-area", type=int, default=1200)
    parser.add_argument("--max-box-area", type=int, default=400000)
    args = parser.parse_args()

    candidates, dropped = build_candidates(args)
    selected = stratified_sample(candidates, args)
    variants = {v.strip() for v in args.variants.split(",") if v.strip()}
    rows: list[dict[str, Any]] = []
    for base in selected:
        rows.extend(make_eval_rows(base, variants))

    out_jsonl = args.output_dir / f"{args.name}.jsonl"
    manifest_jsonl = args.output_dir / f"{args.name}_base_manifest.jsonl"
    summary_json = args.output_dir / f"{args.name}_summary.json"
    write_jsonl(out_jsonl, rows)
    write_jsonl(manifest_jsonl, selected)

    summary = {
        "dataset": args.name,
        "input": str(args.input),
        "output_jsonl": str(out_jsonl),
        "base_manifest_jsonl": str(manifest_jsonl),
        "base_candidates": len(candidates),
        "base_selected": len(selected),
        "eval_rows": len(rows),
        "variants": sorted(variants),
        "dropped": dict(dropped),
        "by_prefix": dict(Counter(r["prefix"] for r in selected)),
        "by_relation": dict(Counter(r["relation"] for r in selected)),
        "by_prompt_mode": dict(Counter(r["metadata"]["prompt_mode"] for r in rows)),
        "heldout_reason": dict(Counter(r["heldout_reason"] for r in selected)),
        "box_area": {
            "min": min((r["box_area"] for r in selected), default=None),
            "max": max((r["box_area"] for r in selected), default=None),
            "mean": round(sum(r["box_area"] for r in selected) / len(selected), 2) if selected else None,
        },
        "notes": [
            "All rows exclude prompt_ids present in the SFT manifest.",
            "region_name is rule-inferred; use crop VLM labeling for object-name-specific eval.",
            "This set evaluates semantic relation-to-box grounding, not memorized train prompts.",
        ],
    }
    write_json(summary_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
