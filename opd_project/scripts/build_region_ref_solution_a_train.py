#!/usr/bin/env python3
"""Regenerate RegionRef Solution-A training data with bidirectional cleaning.

Solution A keeps the current RoboPoint point-derived region bbox, but replaces
the ambiguous crop label with a unique full-image reference expression. A sample
is kept only when a VLM can recover the target region from the generated
description on the original image without the temporary target overlay.

The script is resumable: accepted/rejected base records are appended as they
finish, and reruns skip completed prompt_ids before rebuilding final train JSONL
and summary files.
"""

from __future__ import annotations

import argparse
import json
import random
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from build_region_ref_solution_a_eval import (
    PROMPT_MODES,
    annotation_prompt,
    area,
    box_answer,
    clean_description,
    description_is_valid,
    iou,
    parse_box,
    point_recall,
    relation_phrase,
    validate_geometry,
    validate_prompt,
)
from build_region_ref_solution_pilots import (
    SYSTEM_PROMPT,
    call_qwen,
    draw_overlay,
    fetch_image,
    image_data_url,
    read_jsonl,
    remove_overlay_artifacts,
    write_json,
    write_jsonl,
)


def append_jsonl(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    line = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
    with lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(line)


def load_prompt_ids(path: Path) -> set[str]:
    if not path or not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def normalize_annotation(obj: dict[str, Any], relation: str) -> dict[str, Any]:
    def s(value: Any, fallback: str) -> str:
        text = str(value or "").strip().strip("\"'")
        return " ".join(text.split()) or fallback

    category = s(obj.get("region_category") or obj.get("region_type"), "unclear").lower()
    if category not in {"free_space", "surface", "container", "object", "unclear"}:
        category = "unclear"
    anchor = remove_overlay_artifacts(s(obj.get("anchor_phrase") or obj.get("anchor_object"), "highlighted anchor"))
    rel = remove_overlay_artifacts(s(obj.get("relation_to_anchor") or obj.get("relation"), relation_phrase(relation)))
    desc = clean_description(s(obj.get("unique_description") or obj.get("description"), ""))
    confidence = s(obj.get("confidence"), "medium").lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "medium"
    return {
        "region_category": category,
        "anchor_phrase": anchor,
        "relation_to_anchor": rel,
        "unique_description": desc,
        "confidence": confidence,
        "raw_annotation": obj,
    }


def process_one(args: argparse.Namespace, row: dict[str, Any]) -> dict[str, Any]:
    try:
        image = fetch_image(args, row["image"])
        overlay = draw_overlay(image, row["box"], box_color=(0, 110, 255))
        ann_obj, ann_raw = call_qwen(args, image_data_url(overlay, max_side=args.annotation_image_max_side), annotation_prompt(row))
        annotation = normalize_annotation(ann_obj, row["relation"])

        valid_desc, desc_reason = description_is_valid(annotation["unique_description"])
        if not valid_desc or annotation["confidence"] == "low":
            return {
                "status": "rejected",
                "prompt_id": row["prompt_id"],
                "reason": "low_confidence" if annotation["confidence"] == "low" else desc_reason,
                "annotation": annotation,
                "raw_response": ann_raw,
                "relation": row["relation"],
            }

        pred_obj, pred_raw = call_qwen(
            args,
            image_data_url(image, max_side=args.validation_image_max_side),
            validate_prompt(annotation["unique_description"], annotation),
        )
        pred_box = parse_box(pred_obj, pred_raw)
        if pred_box is None:
            return {
                "status": "rejected",
                "prompt_id": row["prompt_id"],
                "reason": "no_pred_box",
                "annotation": annotation,
                "raw_response": pred_raw,
                "relation": row["relation"],
            }

        ok, metrics = validate_geometry(row["box"], pred_box, row.get("points") or [], args)
        if not ok:
            return {
                "status": "rejected",
                "prompt_id": row["prompt_id"],
                "reason": "desc_to_box_geometry",
                "annotation": annotation,
                "validation": metrics,
                "raw_response": pred_raw,
                "relation": row["relation"],
            }

        return {
            "status": "accepted",
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
            "heldout_excluded": row.get("heldout_excluded", False),
        }
    except Exception as exc:  # noqa: BLE001 - long jobs should keep moving.
        return {
            "status": "rejected",
            "prompt_id": row.get("prompt_id"),
            "reason": "exception",
            "error": repr(exc),
            "relation": row.get("relation"),
        }


def make_train_rows(record: dict[str, Any], dataset_name: str) -> list[dict[str, Any]]:
    box = record["box"]
    answer = box_answer(box)
    target = {
        "region_category": record["region_category"],
        "description": record["unique_description"],
        "relation": record["relation_to_anchor"],
        "anchor_object": record["anchor_phrase"],
        "box": box,
    }
    prompt_specs = [
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
    rows: list[dict[str, Any]] = []
    for mode, prompt in prompt_specs:
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
                    "source": "robopoint_region_ref_solution_a_bidir_cleaned",
                    "task_type": "region_box_grounding",
                    "prompt_mode": mode,
                    "original_prompt_id": record["prompt_id"],
                    "relation": record["relation"],
                    "point_count": record["point_count"],
                    "box_area": record["box_area"],
                    "description_to_box_validation": record["description_to_box_validation"],
                    "box_to_description_source": "qwen_full_image_target_overlay",
                    "quality": "bidir_cleaned",
                },
            }
        )
    return rows


def candidate_rows(raw_rows: list[dict[str, Any]], holdout_ids: set[str], args: argparse.Namespace) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in raw_rows:
        if row.get("prompt_id") in holdout_ids:
            continue
        if row.get("quality") != "high_quality":
            continue
        if row.get("annotation", {}).get("confidence") == "low":
            continue
        if len(row.get("points") or []) < args.min_points:
            continue
        box_area = area(row["box"])
        if box_area < args.min_box_area or box_area > args.max_box_area:
            continue
        candidates.append(row)

    rng = random.Random(args.seed)
    by_rel: dict[str, list[dict[str, Any]]] = {}
    for row in candidates:
        by_rel.setdefault(row["relation"], []).append(row)
    for bucket in by_rel.values():
        rng.shuffle(bucket)

    relation_order = [
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
    balanced: list[dict[str, Any]] = []
    progressed = True
    while progressed:
        progressed = False
        for rel in relation_order:
            bucket = by_rel.get(rel) or []
            if bucket:
                balanced.append(bucket.pop())
                progressed = True
    return balanced[: args.max_candidates] if args.max_candidates > 0 else balanced


def load_done(accepted_path: Path, rejected_path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    accepted: dict[str, dict[str, Any]] = {}
    rejected: dict[str, dict[str, Any]] = {}
    for path, out in [(accepted_path, accepted), (rejected_path, rejected)]:
        if not path.exists():
            continue
        for row in read_jsonl(path):
            prompt_id = row.get("prompt_id")
            if prompt_id:
                out[prompt_id] = row
    return accepted, rejected


def render_preview(path: Path, records: list[dict[str, Any]], args: argparse.Namespace) -> None:
    if not records:
        return
    sample = records[:]
    random.Random(args.seed).shuffle(sample)
    sample = sample[: args.preview_size]
    try:
        title_font = ImageFont.truetype("arial.ttf", 14)
        small_font = ImageFont.truetype("arial.ttf", 11)
    except Exception:
        title_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    cards: list[Image.Image] = []
    for idx, rec in enumerate(sample, start=1):
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
        draw_box(rec["description_to_box_validation"]["pred_box"], (255, 132, 0), 3)
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


def finalize_outputs(
    *,
    args: argparse.Namespace,
    accepted_records: list[dict[str, Any]],
    rejected_records: list[dict[str, Any]],
    total_candidates: int,
) -> None:
    train_rows: list[dict[str, Any]] = []
    for record in accepted_records:
        train_rows.extend(make_train_rows(record, args.dataset_name))

    train_jsonl = args.output_dir / f"{args.dataset_name}.jsonl"
    high_quality_jsonl = args.output_dir / f"{args.dataset_name}_high_quality.jsonl"
    summary_json = args.output_dir / f"{args.dataset_name}_summary.json"
    preview_png = args.output_dir / f"{args.dataset_name}_preview.png"

    write_jsonl(train_jsonl, train_rows)
    write_jsonl(high_quality_jsonl, train_rows)
    render_preview(preview_png, accepted_records, args)

    metrics = [r["description_to_box_validation"] for r in accepted_records]
    summary = {
        "dataset": args.dataset_name,
        "annotations": str(args.annotations),
        "holdout_prompt_ids": str(args.holdout_prompt_ids),
        "accepted_base_jsonl": str(args.accepted_base_path),
        "rejected_base_jsonl": str(args.rejected_base_path),
        "train_jsonl": str(train_jsonl),
        "high_quality_jsonl": str(high_quality_jsonl),
        "preview_png": str(preview_png),
        "total_candidates_after_holdout_and_filters": total_candidates,
        "base_accepted": len(accepted_records),
        "base_rejected": len(rejected_records),
        "train_rows": len(train_rows),
        "variants": PROMPT_MODES,
        "acceptance_thresholds": {
            "min_target_coverage": args.min_target_coverage,
            "min_point_recall": args.min_point_recall,
            "max_center_distance": args.max_center_distance,
            "max_area_ratio": args.max_area_ratio,
        },
        "counters": {
            "accepted_by_relation": dict(Counter(r["relation"] for r in accepted_records)),
            "accepted_by_region_category": dict(Counter(r["region_category"] for r in accepted_records)),
            "rejected_reasons": dict(Counter(r.get("reason") for r in rejected_records)),
            "prompt_modes": dict(Counter(row["metadata"]["prompt_mode"] for row in train_rows)),
        },
        "validation_metrics_mean": {
            key: round(sum(m[key] for m in metrics) / len(metrics), 4) if metrics else 0
            for key in ["iou", "target_coverage", "pred_coverage", "area_ratio_pred_over_target", "center_distance", "point_recall"]
        },
        "notes": [
            "RegionRef Solution A full train set: target boxes are point-derived boxes, descriptions are unique full-image references.",
            "Every accepted base sample passed box->description and description->box validation.",
            "Eval holdout prompt_ids were excluded before generation.",
        ],
    }
    write_json(summary_json, summary)
    print("[finalize] " + json.dumps(summary, ensure_ascii=False, separators=(",", ":")), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--holdout-prompt-ids", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", default="semantic_nav_region_solution_a_bidir_train_v1")
    parser.add_argument("--image-base-url", required=True)
    parser.add_argument("--remote-image-prefix", default="/data/msz/dataset/RoboPoint/images")
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", default="qwen-122b")
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--stop-after", type=int, default=0, help="Process at most this many pending rows in this invocation.")
    parser.add_argument("--min-points", type=int, default=4)
    parser.add_argument("--min-box-area", type=int, default=1800)
    parser.add_argument("--max-box-area", type=int, default=180000)
    parser.add_argument("--min-target-coverage", type=float, default=0.45)
    parser.add_argument("--min-point-recall", type=float, default=0.75)
    parser.add_argument("--max-center-distance", type=float, default=150.0)
    parser.add_argument("--max-area-ratio", type=float, default=8.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=260)
    parser.add_argument("--request-timeout", type=int, default=120)
    parser.add_argument("--image-timeout", type=int, default=30)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=1.5)
    parser.add_argument("--annotation-image-max-side", type=int, default=1100)
    parser.add_argument("--validation-image-max-side", type=int, default=1100)
    parser.add_argument("--preview-size", type=int, default=48)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.accepted_base_path = args.output_dir / f"{args.dataset_name}_base_accepted.jsonl"
    args.rejected_base_path = args.output_dir / f"{args.dataset_name}_base_rejected.jsonl"

    holdout_ids = load_prompt_ids(args.holdout_prompt_ids)
    raw_rows = read_jsonl(args.annotations)
    candidates = candidate_rows(raw_rows, holdout_ids, args)
    accepted_done, rejected_done = load_done(args.accepted_base_path, args.rejected_base_path)
    done_ids = set(accepted_done) | set(rejected_done)
    pending = [row for row in candidates if row["prompt_id"] not in done_ids]
    if args.stop_after > 0:
        pending = pending[: args.stop_after]

    print(
        f"[start] candidates={len(candidates)} accepted_done={len(accepted_done)} "
        f"rejected_done={len(rejected_done)} pending_this_run={len(pending)} workers={args.workers}",
        flush=True,
    )

    lock = threading.Lock()
    completed = 0
    accepted_new = 0
    rejected_new = 0
    if pending:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_one, args, row): row for row in pending}
            for future in as_completed(futures):
                result = future.result()
                if result.get("status") == "accepted":
                    append_jsonl(args.accepted_base_path, result, lock)
                    accepted_done[result["prompt_id"]] = result
                    accepted_new += 1
                else:
                    append_jsonl(args.rejected_base_path, result, lock)
                    if result.get("prompt_id"):
                        rejected_done[result["prompt_id"]] = result
                    rejected_new += 1
                completed += 1
                if completed % args.log_every == 0 or completed == len(pending):
                    total_done = len(accepted_done) + len(rejected_done)
                    print(
                        f"[progress] run={completed}/{len(pending)} total_done={total_done}/{len(candidates)} "
                        f"accepted={len(accepted_done)} rejected={len(rejected_done)} "
                        f"new_accept={accepted_new} new_reject={rejected_new}",
                        flush=True,
                    )

    # Re-read from disk to include all append-completed records exactly once.
    accepted_done, rejected_done = load_done(args.accepted_base_path, args.rejected_base_path)
    accepted_records = [accepted_done[row["prompt_id"]] for row in candidates if row["prompt_id"] in accepted_done]
    rejected_records = [rejected_done[row["prompt_id"]] for row in candidates if row["prompt_id"] in rejected_done]
    finalize_outputs(args=args, accepted_records=accepted_records, rejected_records=rejected_records, total_candidates=len(candidates))


if __name__ == "__main__":
    main()
