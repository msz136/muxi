#!/usr/bin/env python3
"""Build a bidirectionally-validated RegionRef Solution-A eval set.

The eval target is still the current point-derived region bbox, but the prompt
uses a unique full-image description generated from a temporary box overlay.

Bidirectional acceptance:
1. box -> description: a VLM sees the image with the target box and writes a
   unique reference expression without mentioning the temporary overlay.
2. description -> box: the same VLM sees the original image without overlay and
   predicts a box from the description. We keep only samples whose prediction
   substantially covers the target box and source points without being a whole
   image shortcut.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from build_region_ref_solution_pilots import (
    SYSTEM_PROMPT,
    anchor_object,
    box_answer,
    call_qwen,
    draw_overlay,
    fetch_image,
    image_data_url,
    read_jsonl,
    relation_phrase,
    remove_overlay_artifacts,
    write_json,
    write_jsonl,
)


PROMPT_MODES = ["description_plain", "semantic_json", "description_relation"]


def area(box: list[list[int]]) -> int:
    return max(0, int(box[1][0]) - int(box[0][0])) * max(0, int(box[1][1]) - int(box[0][1]))


def intersection(box_a: list[list[int]], box_b: list[list[int]]) -> int:
    x1 = max(int(box_a[0][0]), int(box_b[0][0]))
    y1 = max(int(box_a[0][1]), int(box_b[0][1]))
    x2 = min(int(box_a[1][0]), int(box_b[1][0]))
    y2 = min(int(box_a[1][1]), int(box_b[1][1]))
    return max(0, x2 - x1) * max(0, y2 - y1)


def iou(box_a: list[list[int]], box_b: list[list[int]]) -> float:
    inter = intersection(box_a, box_b)
    denom = area(box_a) + area(box_b) - inter
    return inter / denom if denom > 0 else 0.0


def center_distance(box_a: list[list[int]], box_b: list[list[int]]) -> float:
    ax = (box_a[0][0] + box_a[1][0]) / 2
    ay = (box_a[0][1] + box_a[1][1]) / 2
    bx = (box_b[0][0] + box_b[1][0]) / 2
    by = (box_b[0][1] + box_b[1][1]) / 2
    return math.hypot(ax - bx, ay - by)


def point_recall(points: list[list[int]], pred_box: list[list[int]]) -> float:
    valid = [p for p in points if isinstance(p, list) and len(p) >= 2]
    if not valid:
        return 0.0
    inside = 0
    for x, y in valid:
        if pred_box[0][0] <= x <= pred_box[1][0] and pred_box[0][1] <= y <= pred_box[1][1]:
            inside += 1
    return inside / len(valid)


def clean_description(text: str) -> str:
    text = remove_overlay_artifacts(text)
    text = re.sub(r"^(the\s+)?target region is\s+", "", text, flags=re.I)
    text = re.sub(r"^(the\s+)?target area is\s+", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text


def annotation_prompt(row: dict[str, Any]) -> str:
    original = row.get("old_user", "").replace("<image>", "").strip()
    return (
        "You are writing a unique reference expression for semantic navigation region grounding.\n"
        "The source image may already contain red rectangles. Those red rectangles are real highlighted anchors "
        "from the original task and may be mentioned.\n"
        "A temporary blue rectangle marks the target region only for annotation. Do not mention blue, box, rectangle, "
        "overlay, annotation mark, coordinates, pixels, or target rectangle in the final description. Mention the "
        "visible scene, red-highlighted anchor, spatial relation, nearby objects, surface/container, and relative "
        "position so that the region is uniquely recoverable in the original image without the temporary blue rectangle.\n\n"
        f"Original RoboPoint instruction:\n{original}\n\n"
        f"Known relation: {relation_phrase(row['relation'])}\n"
        f"Target box for your reference only: {row['box']}\n\n"
        "Return strict JSON only:\n"
        "{\n"
        '  "region_category": "free_space|surface|container|object|unclear",\n'
        '  "anchor_phrase": "short phrase for the red-highlighted anchor or nearest stable anchor",\n'
        '  "relation_to_anchor": "spatial relation to the anchor",\n'
        '  "unique_description": "one sentence, no coordinates, uniquely locating this exact region",\n'
        '  "confidence": "high|medium|low"\n'
        "}"
    )


def validate_prompt(description: str, annotation: dict[str, Any]) -> str:
    info = {
        "region_category": annotation["region_category"],
        "description": description,
        "relation": annotation["relation_to_anchor"],
        "anchor_object": annotation["anchor_phrase"],
    }
    return (
        "You are validating semantic navigation grounding. Given the original image and the navigation target "
        "information, return the target region's bounding box in normalized coordinates from 0 to 1000.\n"
        "Return strict JSON only: {\"box\": [[x1,y1],[x2,y2]]}.\n\n"
        "Navigation target information:\n"
        + json.dumps(info, ensure_ascii=False, separators=(",", ":"))
    )


def normalize_annotation(obj: dict[str, Any], relation: str) -> dict[str, Any]:
    def s(value: Any, fallback: str) -> str:
        text = str(value or "").strip().strip("\"'")
        text = re.sub(r"\s+", " ", text)
        return text or fallback

    category = s(obj.get("region_category") or obj.get("region_type"), "unclear").lower()
    if category not in {"free_space", "surface", "container", "object", "unclear"}:
        category = "unclear"
    anchor = s(obj.get("anchor_phrase") or obj.get("anchor_object"), anchor_object(relation))
    rel = s(obj.get("relation_to_anchor") or obj.get("relation"), relation_phrase(relation))
    desc = clean_description(s(obj.get("unique_description") or obj.get("description"), ""))
    confidence = s(obj.get("confidence"), "medium").lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "medium"
    return {
        "region_category": category,
        "anchor_phrase": remove_overlay_artifacts(anchor),
        "relation_to_anchor": remove_overlay_artifacts(rel),
        "unique_description": desc,
        "confidence": confidence,
        "raw_annotation": obj,
    }


def parse_box_from_obj(obj: Any) -> list[list[int]] | None:
    if isinstance(obj, list):
        for item in obj:
            parsed = parse_box_from_obj(item)
            if parsed is not None:
                return parsed
        return None
    if not isinstance(obj, dict):
        return None
    box = obj.get("box") or obj.get("bbox_2d") or obj.get("bbox")
    if not isinstance(box, list) or len(box) != 2:
        if isinstance(box, list) and len(box) == 4:
            try:
                x1, y1, x2, y2 = [int(round(float(v))) for v in box]
            except Exception:
                return None
            x1, x2 = sorted([max(0, min(1000, x1)), max(0, min(1000, x2))])
            y1, y2 = sorted([max(0, min(1000, y1)), max(0, min(1000, y2))])
            if x2 <= x1 or y2 <= y1:
                return None
            return [[x1, y1], [x2, y2]]
        return None
    try:
        x1, y1 = int(round(float(box[0][0]))), int(round(float(box[0][1])))
        x2, y2 = int(round(float(box[1][0]))), int(round(float(box[1][1])))
    except Exception:
        return None
    x1, x2 = sorted([max(0, min(1000, x1)), max(0, min(1000, x2))])
    y1, y2 = sorted([max(0, min(1000, y1)), max(0, min(1000, y2))])
    if x2 <= x1 or y2 <= y1:
        return None
    return [[x1, y1], [x2, y2]]


def parse_box(obj: dict[str, Any], raw: str = "") -> list[list[int]] | None:
    parsed = parse_box_from_obj(obj)
    if parsed is not None:
        return parsed
    # Qwen sometimes returns a JSON array or a bbox_2d object even when asked
    # for strict JSON. Recover those common forms from the raw text.
    patterns = [
        r'"bbox_2d"\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]',
        r'"box"\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]',
        r'\[\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]\s*,\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]\s*\]',
    ]
    for pattern in patterns:
        match = re.search(pattern, raw, flags=re.S)
        if not match:
            continue
        vals = [int(round(float(v))) for v in match.groups()]
        if len(vals) == 4:
            x1, y1, x2, y2 = vals
            x1, x2 = sorted([max(0, min(1000, x1)), max(0, min(1000, x2))])
            y1, y2 = sorted([max(0, min(1000, y1)), max(0, min(1000, y2))])
            if x2 > x1 and y2 > y1:
                return [[x1, y1], [x2, y2]]
    return None


def description_is_valid(desc: str) -> tuple[bool, str]:
    lower = desc.lower()
    if len(desc.split()) < 8:
        return False, "description_too_short"
    banned_patterns = [
        r"\b(?:blue|green)\s+(?:rectangle|rectangular|box|outline|outlined|border|highlight|highlighted|overlay|target)",
        r"\b(?:coordinate|pixel|annotation|overlay|bounding box)\b",
    ]
    if any(re.search(pattern, lower) for pattern in banned_patterns):
        return False, "overlay_artifact"
    # Red rectangles are real anchors in RoboPoint images; other spatial words
    # make it harder for generic labels like "empty floor space" to slip in.
    spatial = [
        "left",
        "right",
        "front",
        "behind",
        "above",
        "below",
        "inside",
        "within",
        "between",
        "directly",
        "near",
        "adjacent",
        "beneath",
        "under",
        "on",
        "red",
    ]
    if not any(word in lower for word in spatial):
        return False, "no_spatial_anchor"
    return True, "ok"


def validate_geometry(target: list[list[int]], pred: list[list[int]], points: list[list[int]], args: argparse.Namespace) -> tuple[bool, dict[str, Any]]:
    target_area = area(target)
    pred_area = area(pred)
    inter = intersection(target, pred)
    metrics = {
        "pred_box": pred,
        "iou": round(iou(target, pred), 4),
        "target_coverage": round(inter / target_area, 4) if target_area else 0.0,
        "pred_coverage": round(inter / pred_area, 4) if pred_area else 0.0,
        "area_ratio_pred_over_target": round(pred_area / target_area, 4) if target_area else 999.0,
        "center_distance": round(center_distance(target, pred), 2),
        "point_recall": round(point_recall(points, pred), 4),
    }
    ok = (
        metrics["target_coverage"] >= args.min_target_coverage
        and metrics["point_recall"] >= args.min_point_recall
        and metrics["center_distance"] <= args.max_center_distance
        and metrics["area_ratio_pred_over_target"] <= args.max_area_ratio
    )
    return ok, metrics


def make_eval_rows(record: dict[str, Any], dataset_name: str) -> list[dict[str, Any]]:
    box = record["box"]
    answer = box_answer(box)
    target = {
        "region_category": record["region_category"],
        "description": record["unique_description"],
        "relation": record["relation_to_anchor"],
        "anchor_object": record["anchor_phrase"],
        "box": box,
    }
    prompts = [
        (
            "description_plain",
            f"<image>\nFind the navigation region described as: {record['unique_description']}\n"
            "Return only <box>[[x1,y1],[x2,y2]]</box> with integer coordinates from 0 to 1000.",
        ),
        (
            "semantic_json",
            "<image>\nNavigation target information:\n"
            + json.dumps(
                {
                    "region_category": record["region_category"],
                    "description": record["unique_description"],
                    "relation": record["relation_to_anchor"],
                    "anchor_object": record["anchor_phrase"],
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\nReturn only the bounding box as <box>[[x1,y1],[x2,y2]]</box>.",
        ),
        (
            "description_relation",
            f"<image>\nFind the target navigation region {record['relation_to_anchor']}. "
            f"It is specifically described as: {record['unique_description']}\n"
            "Return only <box>[[x1,y1],[x2,y2]]</box> with integer coordinates from 0 to 1000.",
        ),
    ]
    rows = []
    for mode, prompt in prompts:
        rows.append(
            {
                "dataset": dataset_name,
                "image": [record["image"]],
                "video": [],
                "target": target,
                "conversations": [
                    {"from": "system", "value": SYSTEM_PROMPT},
                    {"from": "human", "value": prompt},
                    {"from": "gpt", "value": answer},
                ],
                "metadata": {
                    "source": "robopoint_region_ref_solution_a_bidir_eval",
                    "task_type": "region_box_grounding_eval",
                    "prompt_mode": mode,
                    "split": "eval",
                    "original_prompt_id": record["prompt_id"],
                    "relation": record["relation"],
                    "point_count": record["point_count"],
                    "box_area": record["box_area"],
                    "description_to_box_validation": record["description_to_box_validation"],
                    "box_to_description_source": "qwen_full_image_target_overlay",
                },
            }
        )
    return rows


def choose_candidates(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    rng = random.Random(args.seed)
    used_images: set[str] = set()
    by_rel: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("quality") != "high_quality":
            continue
        if row.get("annotation", {}).get("confidence") == "low":
            continue
        if len(row.get("points") or []) < args.min_points:
            continue
        box_area = area(row["box"])
        if box_area < args.min_box_area or box_area > args.max_box_area:
            continue
        if args.exclude_prompt_ids and row["prompt_id"] in args.exclude_prompt_ids:
            continue
        by_rel[row["relation"]].append(row)

    ordered_relations = [
        "on",
        "inside",
        "beside",
        "left",
        "right",
        "front",
        "behind",
        "between",
        "above",
        "below",
        "on-front",
        "on-left",
        "on-right",
        "on-back",
    ]
    selected: list[dict[str, Any]] = []
    rounds = 0
    while len(selected) < args.max_candidates and rounds < args.max_candidates * 4:
        progressed = False
        for rel in ordered_relations:
            bucket = by_rel.get(rel) or []
            if not bucket:
                continue
            if rounds == 0:
                rng.shuffle(bucket)
            while bucket:
                row = bucket.pop()
                if args.unique_images and row["image"] in used_images:
                    continue
                selected.append(row)
                used_images.add(row["image"])
                progressed = True
                break
            if len(selected) >= args.max_candidates:
                break
        if not progressed:
            break
        rounds += 1
    return selected


def render_preview(path: Path, records: list[dict[str, Any]], args: argparse.Namespace) -> None:
    if not records:
        return
    try:
        title_font = ImageFont.truetype("arial.ttf", 14)
        small_font = ImageFont.truetype("arial.ttf", 11)
    except Exception:
        title_font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    cards: list[Image.Image] = []
    for idx, rec in enumerate(records[: args.preview_size], start=1):
        image = fetch_image(args, rec["image"])
        width, height = image.size
        image.thumbnail((420, 260), Image.Resampling.LANCZOS)
        dw, dh = image.size
        sx, sy = dw / width, dh / height
        draw = ImageDraw.Draw(image, "RGBA")

        def draw_box(box: list[list[int]], color: tuple[int, int, int], line_width: int) -> None:
            x1 = round(box[0][0] / 1000 * width * sx)
            y1 = round(box[0][1] / 1000 * height * sy)
            x2 = round(box[1][0] / 1000 * width * sx)
            y2 = round(box[1][1] / 1000 * height * sy)
            draw.rectangle([x1, y1, x2, y2], fill=(*color, 35), outline=(*color, 255), width=line_width)

        draw_box(rec["box"], (0, 110, 255), 4)
        pred_box = rec["description_to_box_validation"]["pred_box"]
        draw_box(pred_box, (255, 132, 0), 3)
        for px, py in rec.get("points", [])[:40]:
            x = round(px / 1000 * width * sx)
            y = round(py / 1000 * height * sy)
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(255, 40, 40, 220))

        card = Image.new("RGB", (460, 430), (248, 250, 252))
        card.paste(image, ((460 - dw) // 2, 8))
        d = ImageDraw.Draw(card)
        y = dh + 18
        v = rec["description_to_box_validation"]
        d.text((10, y), f"{idx}. rel={rec['relation']} blue=target orange=desc->box", fill=(15, 23, 42), font=title_font)
        y += 20
        d.text((10, y), f"cov={v['target_coverage']} pts={v['point_recall']} cd={v['center_distance']} area={v['area_ratio_pred_over_target']}", fill=(71, 85, 105), font=small_font)
        y += 16
        for line in wrap(rec["unique_description"], 64)[:5]:
            d.text((10, y), line, fill=(15, 23, 42), font=small_font)
            y += 15
        cards.append(card)

    cols = 2
    rows_n = (len(cards) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 460, rows_n * 430), (226, 232, 240))
    for i, card in enumerate(cards):
        sheet.paste(card, ((i % cols) * 460, (i // cols) * 430))
    sheet.save(path)


def wrap(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        if current and sum(len(w) + 1 for w in current) + len(word) > width:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", default="semantic_nav_region_solution_a_bidir_eval_v1")
    parser.add_argument("--image-base-url", required=True)
    parser.add_argument("--remote-image-prefix", default="/data/msz/dataset/RoboPoint/images")
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", default="qwen-122b")
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--target-base-samples", type=int, default=60)
    parser.add_argument("--max-candidates", type=int, default=180)
    parser.add_argument("--min-points", type=int, default=4)
    parser.add_argument("--min-box-area", type=int, default=1800)
    parser.add_argument("--max-box-area", type=int, default=180000)
    parser.add_argument("--min-target-coverage", type=float, default=0.45)
    parser.add_argument("--min-point-recall", type=float, default=0.75)
    parser.add_argument("--max-center-distance", type=float, default=150.0)
    parser.add_argument("--max-area-ratio", type=float, default=8.0)
    parser.add_argument("--unique-images", action="store_true", default=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=260)
    parser.add_argument("--request-timeout", type=int, default=120)
    parser.add_argument("--image-timeout", type=int, default=30)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=1.5)
    parser.add_argument("--preview-size", type=int, default=40)
    args = parser.parse_args()
    args.exclude_prompt_ids = set()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.annotations)
    candidates = choose_candidates(rows, args)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for idx, row in enumerate(candidates, start=1):
        image = fetch_image(args, row["image"])
        overlay = draw_overlay(image, row["box"], box_color=(0, 110, 255))
        ann_obj, ann_raw = call_qwen(args, image_data_url(overlay), annotation_prompt(row))
        annotation = normalize_annotation(ann_obj, row["relation"])
        valid_desc, desc_reason = description_is_valid(annotation["unique_description"])
        if not valid_desc or annotation["confidence"] == "low":
            rejected.append({"prompt_id": row["prompt_id"], "reason": desc_reason, "annotation": annotation})
            continue

        pred_obj, pred_raw = call_qwen(args, image_data_url(image), validate_prompt(annotation["unique_description"], annotation))
        pred_box = parse_box(pred_obj, pred_raw)
        if pred_box is None:
            rejected.append({"prompt_id": row["prompt_id"], "reason": "no_pred_box", "annotation": annotation, "raw": pred_raw})
            continue
        ok, metrics = validate_geometry(row["box"], pred_box, row.get("points") or [], args)
        if not ok:
            rejected.append(
                {
                    "prompt_id": row["prompt_id"],
                    "reason": "desc_to_box_geometry",
                    "annotation": annotation,
                    "validation": metrics,
                    "raw": pred_raw,
                }
            )
            continue

        record = {
            "prompt_id": row["prompt_id"],
            "image": row["image"],
            "old_user": row.get("old_user", ""),
            "relation": row["relation"],
            "relation_phrase": relation_phrase(row["relation"]),
            "box": row["box"],
            "points": row.get("points") or [],
            "point_count": len(row.get("points") or []),
            "box_area": area(row["box"]),
            "region_category": annotation["region_category"],
            "anchor_phrase": annotation["anchor_phrase"],
            "relation_to_anchor": annotation["relation_to_anchor"],
            "unique_description": annotation["unique_description"],
            "confidence": annotation["confidence"],
            "box_to_description": {
                "method": "qwen_full_image_target_overlay",
                "raw_response": ann_raw,
                "annotation": annotation,
            },
            "description_to_box_validation": {
                **metrics,
                "method": "qwen_original_image_no_overlay",
                "raw_response": pred_raw,
            },
            "heldout_reason": "solution_a_eval_holdout_for_future_training",
        }
        accepted.append(record)
        print(
            f"[accept {len(accepted)}/{args.target_base_samples}] cand={idx}/{len(candidates)} "
            f"rel={row['relation']} cov={metrics['target_coverage']} pts={metrics['point_recall']} "
            f"desc={annotation['unique_description'][:80]}",
            flush=True,
        )
        if len(accepted) >= args.target_base_samples:
            break
        time.sleep(0.02)

    eval_rows: list[dict[str, Any]] = []
    for record in accepted:
        eval_rows.extend(make_eval_rows(record, args.dataset_name))

    base_manifest = args.output_dir / f"{args.dataset_name}_base_manifest.jsonl"
    eval_jsonl = args.output_dir / f"{args.dataset_name}.jsonl"
    rejected_jsonl = args.output_dir / f"{args.dataset_name}_rejected.jsonl"
    summary_json = args.output_dir / f"{args.dataset_name}_summary.json"
    preview_png = args.output_dir / f"{args.dataset_name}_preview.png"
    holdout_txt = args.output_dir / f"{args.dataset_name}_holdout_prompt_ids.txt"

    write_jsonl(base_manifest, accepted)
    write_jsonl(eval_jsonl, eval_rows)
    write_jsonl(rejected_jsonl, rejected)
    holdout_txt.write_text("\n".join(r["prompt_id"] for r in accepted) + "\n", encoding="utf-8")
    render_preview(preview_png, accepted, args)

    counters = {
        "by_relation": Counter(r["relation"] for r in accepted),
        "by_region_category": Counter(r["region_category"] for r in accepted),
        "by_prompt_mode": Counter(row["metadata"]["prompt_mode"] for row in eval_rows),
        "rejected_reasons": Counter(r["reason"] for r in rejected),
    }
    metrics = [r["description_to_box_validation"] for r in accepted]
    summary = {
        "dataset": args.dataset_name,
        "annotations": str(args.annotations),
        "output_jsonl": str(eval_jsonl),
        "base_manifest_jsonl": str(base_manifest),
        "rejected_jsonl": str(rejected_jsonl),
        "preview_png": str(preview_png),
        "holdout_prompt_ids": str(holdout_txt),
        "base_candidates_scanned": len(candidates),
        "base_selected": len(accepted),
        "eval_rows": len(eval_rows),
        "variants": PROMPT_MODES,
        "acceptance_thresholds": {
            "min_target_coverage": args.min_target_coverage,
            "min_point_recall": args.min_point_recall,
            "max_center_distance": args.max_center_distance,
            "max_area_ratio": args.max_area_ratio,
        },
        "counters": {k: dict(v) for k, v in counters.items()},
        "validation_metrics_mean": {
            key: round(sum(m[key] for m in metrics) / len(metrics), 4) if metrics else 0
            for key in ["iou", "target_coverage", "pred_coverage", "area_ratio_pred_over_target", "center_distance", "point_recall"]
        },
        "notes": [
            "Solution A eval: target boxes remain point-derived, but prompts use unique descriptions.",
            "Every base sample passed box->description generation and description->box geometric validation.",
            "Future Solution A training should exclude prompt_ids listed in holdout_prompt_ids.",
        ],
    }
    write_json(summary_json, summary)
    print("[done] " + json.dumps(summary, ensure_ascii=False, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
