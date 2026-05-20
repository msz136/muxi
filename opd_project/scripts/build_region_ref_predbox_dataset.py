#!/usr/bin/env python3
"""Build RegionRef data using description->box predictions as final labels.

This is the follow-up to Solution A. Solution A generated a unique full-image
description from a temporary target overlay, then asked a VLM to recover a box
from that description on the original image. Earlier data kept the original
point-derived seed box as the label and rejected samples when the recovered box
was larger or shifted. This script instead uses the recovered box as the label.

The original point-derived box is kept as metadata.seed_box for provenance.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import random
import re
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


SYSTEM_PROMPT = (
    "You are a semantic navigation region-grounding assistant. Given an image and a uniquely "
    "described target navigation region, return the target region's bounding box in coordinates "
    "from 0 to 1000. Return only <box>[[x1,y1],[x2,y2]]</box>."
)

PROMPT_MODES = ["description_plain", "semantic_json", "description_relation"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def area(box: list[list[int]]) -> int:
    return max(0, box[1][0] - box[0][0]) * max(0, box[1][1] - box[0][1])


def intersection(a: list[list[int]], b: list[list[int]]) -> int:
    x1 = max(a[0][0], b[0][0])
    y1 = max(a[0][1], b[0][1])
    x2 = min(a[1][0], b[1][0])
    y2 = min(a[1][1], b[1][1])
    return max(0, x2 - x1) * max(0, y2 - y1)


def iou(a: list[list[int]], b: list[list[int]]) -> float:
    inter = intersection(a, b)
    union = area(a) + area(b) - inter
    return inter / union if union else 0.0


def center_distance(a: list[list[int]], b: list[list[int]]) -> float:
    ax = (a[0][0] + a[1][0]) / 2
    ay = (a[0][1] + a[1][1]) / 2
    bx = (b[0][0] + b[1][0]) / 2
    by = (b[0][1] + b[1][1]) / 2
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def point_recall(points: list[list[int]], box: list[list[int]]) -> float:
    if not points:
        return 1.0
    hits = 0
    for x, y in points:
        if box[0][0] <= x <= box[1][0] and box[0][1] <= y <= box[1][1]:
            hits += 1
    return hits / len(points)


def relation_phrase(rel: str) -> str:
    mapping = {
        "on": "on the highlighted area or surface",
        "inside": "inside the highlighted container or area",
        "front": "in front of the highlighted object",
        "behind": "behind the highlighted object",
        "left": "to the left of the highlighted object",
        "right": "to the right of the highlighted object",
        "beside": "beside the highlighted object",
        "between": "between the highlighted objects",
        "above": "above the highlighted object",
        "below": "below the highlighted object",
        "on-left": "on the left side of the highlighted area",
        "on-right": "on the right side of the highlighted area",
        "on-front": "on the front side of the highlighted area",
        "on-back": "on the back side of the highlighted area",
    }
    return mapping.get(rel, f"{rel} the highlighted object")


def box_answer(box: list[list[int]]) -> str:
    return f"<box>[[{box[0][0]},{box[0][1]}],[{box[1][0]},{box[1][1]}]]</box>"


def parse_jsonish(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"(\{.*\}|\[.*\])", text, re.S)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            return {}
    return {}


def parse_box_from_any(obj: Any, raw: str = "") -> list[list[int]] | None:
    def from_nums(vals: list[Any]) -> list[list[int]] | None:
        try:
            nums = [int(round(float(x))) for x in vals]
        except Exception:
            return None
        if len(nums) != 4:
            return None
        x1, y1, x2, y2 = nums
        x1, x2 = sorted([max(0, min(1000, x1)), max(0, min(1000, x2))])
        y1, y2 = sorted([max(0, min(1000, y1)), max(0, min(1000, y2))])
        if x2 <= x1 or y2 <= y1:
            return None
        return [[x1, y1], [x2, y2]]

    queue: list[Any] = [obj]
    while queue:
        cur = queue.pop(0)
        if isinstance(cur, dict):
            for key in ("pred_box", "bbox_2d", "bbox", "box"):
                val = cur.get(key)
                if isinstance(val, list):
                    if len(val) == 4:
                        parsed = from_nums(val)
                        if parsed:
                            return parsed
                    if len(val) == 2 and all(isinstance(x, list) for x in val):
                        parsed = from_nums([val[0][0], val[0][1], val[1][0], val[1][1]])
                        if parsed:
                            return parsed
                    if len(val) == 1 and isinstance(val[0], list) and len(val[0]) == 4:
                        parsed = from_nums(val[0])
                        if parsed:
                            return parsed
            queue.extend(cur.values())
        elif isinstance(cur, list):
            if len(cur) == 4 and all(isinstance(x, (int, float, str)) for x in cur):
                parsed = from_nums(cur)
                if parsed:
                    return parsed
            queue.extend(cur)

    nums = [int(x) for x in re.findall(r"-?\d+", raw)]
    for i in range(0, max(0, len(nums) - 3)):
        parsed = from_nums(nums[i : i + 4])
        if parsed:
            return parsed
    return None


def clean_text(value: Any, fallback: str = "") -> str:
    text = str(value or "").strip().strip("\"'")
    text = re.sub(r"\s+", " ", text)
    return text or fallback


def remove_overlay_artifacts(text: str) -> str:
    # Blue/green overlays are temporary target marks. Red rectangles are part of
    # the source RoboPoint images and may be legitimate anchor references.
    text = re.sub(
        r"\b(?:blue|green)\s+(?:rectangle|rectangular|box|outline|outlined|border|highlight|highlighted|overlay|target)\b",
        "target area",
        text,
        flags=re.I,
    )
    text = re.sub(r"\b(?:coordinate|pixel|annotation|overlay|bounding box)\b", "", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip(" .")


def description_is_valid(desc: str) -> tuple[bool, str]:
    lower = desc.lower()
    if len(desc.split()) < 8:
        return False, "description_too_short"
    banned = [
        r"\b(?:blue|green)\s+(?:rectangle|rectangular|box|outline|outlined|border|highlight|highlighted|overlay|target)\b",
        r"\b(?:coordinate|pixel|annotation|overlay|bounding box)\b",
    ]
    if any(re.search(pattern, lower) for pattern in banned):
        return False, "temporary_overlay_artifact"
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


def normalize_annotation(obj: dict[str, Any], relation: str) -> dict[str, Any]:
    category = clean_text(obj.get("region_category") or obj.get("region_type"), "unclear").lower()
    if category not in {"free_space", "surface", "container", "object", "unclear"}:
        category = "unclear"
    anchor = clean_text(obj.get("anchor_phrase") or obj.get("anchor_object"), "highlighted anchor")
    rel = clean_text(obj.get("relation_to_anchor") or obj.get("relation"), relation_phrase(relation))
    desc = clean_text(obj.get("unique_description") or obj.get("description"), "")
    confidence = clean_text(obj.get("confidence"), "medium").lower()
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


def draw_overlay(image: Image.Image, box: list[list[int]], box_color: tuple[int, int, int] = (0, 110, 255)) -> Image.Image:
    out = image.convert("RGBA")
    draw = ImageDraw.Draw(out, "RGBA")
    w, h = out.size
    x1 = int(box[0][0] / 1000 * w)
    y1 = int(box[0][1] / 1000 * h)
    x2 = int(box[1][0] / 1000 * w)
    y2 = int(box[1][1] / 1000 * h)
    line_width = max(3, int(min(w, h) * 0.008))
    draw.rectangle([x1, y1, x2, y2], fill=(*box_color, 35), outline=(*box_color, 255), width=line_width)
    return out.convert("RGB")


def image_data_url(image: Image.Image, max_side: int) -> str:
    image = image.convert("RGB")
    w, h = image.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        image = image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def call_qwen(args: argparse.Namespace, image: Image.Image, prompt: str, max_tokens: int = 384) -> tuple[Any, str]:
    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url(image, args.image_max_side)}},
                ],
            }
        ],
        "temperature": args.temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = "Bearer " + args.api_key
    body = json.dumps(payload).encode("utf-8")
    last_error = ""
    for attempt in range(1, args.retries + 1):
        try:
            req = urllib.request.Request(args.api_url, data=body, headers=headers)
            with urllib.request.urlopen(req, timeout=args.request_timeout) as response:
                resp = json.loads(response.read().decode("utf-8"))
            raw = resp["choices"][0]["message"]["content"]
            return parse_jsonish(raw), raw
        except Exception as exc:  # noqa: BLE001
            last_error = repr(exc)
            if attempt < args.retries:
                time.sleep(args.retry_sleep * attempt)
    return {"error": last_error}, last_error


def annotation_prompt(row: dict[str, Any]) -> str:
    return (
        "You are creating a semantic navigation region reference. The source image may already contain "
        "red rectangles; those are real highlighted anchors. A temporary blue rectangle marks the target "
        "region only for annotation. Do not mention blue, box, rectangle, overlay, annotation mark, "
        "coordinates, pixels, or target rectangle in the final description. Mention the visible scene, "
        "red-highlighted anchor when useful, spatial relation, nearby objects, surface/container, and "
        "relative position so that the region is uniquely recoverable in the original image without the "
        "temporary blue rectangle.\n\n"
        f"Original relation: {relation_phrase(row.get('relation', ''))}\n"
        "Return JSON only:\n"
        "{\n"
        '  "region_category": "free_space|surface|container|object|unclear",\n'
        '  "anchor_phrase": "short phrase for the visible anchor",\n'
        '  "relation_to_anchor": "short spatial relation phrase",\n'
        '  "unique_description": "one sentence unique region description without mentioning the temporary mark",\n'
        '  "confidence": "high|medium|low"\n'
        "}"
    )


def validation_prompt(desc: str, annotation: dict[str, Any]) -> str:
    info = {
        "region_category": annotation["region_category"],
        "description": desc,
        "relation": annotation["relation_to_anchor"],
        "anchor_object": annotation["anchor_phrase"],
    }
    return (
        "Locate the navigation target region in the original image. Return JSON only as "
        '[{"bbox_2d":[x1,y1,x2,y2],"label":"region"}] with integer coordinates from 0 to 1000.\n'
        "Target region information:\n"
        + json.dumps(info, ensure_ascii=False, separators=(",", ":"))
    )


def metrics(seed_box: list[list[int]], final_box: list[list[int]], points: list[list[int]]) -> dict[str, Any]:
    seed_area = area(seed_box)
    final_area = area(final_box)
    inter = intersection(seed_box, final_box)
    return {
        "seed_box": seed_box,
        "final_box": final_box,
        "iou_seed_final": round(iou(seed_box, final_box), 4),
        "seed_coverage_by_final": round(inter / seed_area, 4) if seed_area else 0.0,
        "final_coverage_by_seed": round(inter / final_area, 4) if final_area else 0.0,
        "area_ratio_final_over_seed": round(final_area / seed_area, 4) if seed_area else 999.0,
        "center_distance": round(center_distance(seed_box, final_box), 2),
        "point_recall": round(point_recall(points, final_box), 4),
        "seed_area": seed_area,
        "final_area": final_area,
    }


def final_box_is_valid(seed_box: list[list[int]], final_box: list[list[int]], points: list[list[int]], args: argparse.Namespace) -> tuple[bool, str, dict[str, Any]]:
    m = metrics(seed_box, final_box, points)
    if m["final_area"] < args.min_final_box_area:
        return False, "final_box_too_small", m
    if m["final_area"] > args.max_final_box_area:
        return False, "final_box_too_large", m
    if m["point_recall"] < args.min_point_recall:
        return False, "low_point_recall", m
    return True, "ok", m


def extract_existing_record(solution_row: dict[str, Any], base_row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    prompt_id = base_row["prompt_id"]
    relation = base_row.get("relation", "")
    reason = solution_row.get("reason") or "accepted"
    annotation = solution_row.get("annotation")
    if not annotation and solution_row.get("box_to_description", {}).get("annotation"):
        annotation = solution_row["box_to_description"]["annotation"]
    annotation = normalize_annotation(annotation or {}, relation)
    valid_desc, desc_reason = description_is_valid(annotation["unique_description"])
    if not valid_desc or annotation["confidence"] == "low":
        return {"status": "rejected", "prompt_id": prompt_id, "reason": desc_reason if not valid_desc else "low_confidence"}

    pred_box = None
    raw = solution_row.get("raw_response") or solution_row.get("raw") or ""
    validation = solution_row.get("description_to_box_validation") or solution_row.get("validation") or {}
    pred_box = parse_box_from_any(validation, raw)
    if pred_box is None:
        pred_box = parse_box_from_any(parse_jsonish(raw), raw)
    if pred_box is None:
        return {"status": "rejected", "prompt_id": prompt_id, "reason": "no_pred_box"}

    ok, box_reason, m = final_box_is_valid(base_row["box"], pred_box, base_row.get("points") or [], args)
    if not ok:
        return {"status": "rejected", "prompt_id": prompt_id, "reason": box_reason, "metrics": m}

    return {
        "status": "accepted",
        "prompt_id": prompt_id,
        "image": base_row["image"],
        "old_user": base_row.get("old_user", ""),
        "relation": relation,
        "relation_phrase": relation_phrase(relation),
        "seed_box": base_row["box"],
        "final_box": pred_box,
        "points": base_row.get("points") or [],
        "point_count": len(base_row.get("points") or []),
        "region_category": annotation["region_category"],
        "anchor_phrase": annotation["anchor_phrase"],
        "relation_to_anchor": annotation["relation_to_anchor"],
        "unique_description": annotation["unique_description"],
        "confidence": annotation["confidence"],
        "seed_to_final_metrics": m,
        "description_to_box": {
            "method": "qwen_original_image_no_overlay",
            "raw_response": raw,
            "parsed_box": pred_box,
        },
        "box_to_description": {
            "method": "qwen_full_image_target_overlay",
            "annotation": annotation,
            "raw_response": solution_row.get("box_to_description", {}).get("raw_response") or solution_row.get("raw_response") or "",
        },
        "source_stage": "existing_solution_a",
        "source_solution_status": solution_row.get("status", "accepted"),
        "source_solution_reason": reason,
    }


def process_missing_row(args: argparse.Namespace, base_row: dict[str, Any]) -> dict[str, Any]:
    prompt_id = base_row["prompt_id"]
    relation = base_row.get("relation", "")
    try:
        if base_row.get("quality") == "weak" and not args.include_weak:
            return {"status": "rejected", "prompt_id": prompt_id, "reason": "weak_source_quality"}
        if len(base_row.get("points") or []) < args.min_points:
            return {"status": "rejected", "prompt_id": prompt_id, "reason": "too_few_points"}
        seed_area = area(base_row["box"])
        if seed_area < args.min_seed_box_area:
            return {"status": "rejected", "prompt_id": prompt_id, "reason": "seed_box_too_small"}

        image = Image.open(base_row["image"]).convert("RGB")
        overlay = draw_overlay(image, base_row["box"])
        ann_obj, ann_raw = call_qwen(args, overlay, annotation_prompt(base_row))
        if not isinstance(ann_obj, dict):
            ann_obj = {}
        annotation = normalize_annotation(ann_obj, relation)
        valid_desc, desc_reason = description_is_valid(annotation["unique_description"])
        if not valid_desc or annotation["confidence"] == "low":
            return {
                "status": "rejected",
                "prompt_id": prompt_id,
                "reason": desc_reason if not valid_desc else "low_confidence",
                "annotation": annotation,
                "raw_response": ann_raw,
            }
        pred_obj, pred_raw = call_qwen(args, image, validation_prompt(annotation["unique_description"], annotation))
        pred_box = parse_box_from_any(pred_obj, pred_raw)
        if pred_box is None:
            return {
                "status": "rejected",
                "prompt_id": prompt_id,
                "reason": "no_pred_box",
                "annotation": annotation,
                "raw_response": pred_raw,
            }
        ok, box_reason, m = final_box_is_valid(base_row["box"], pred_box, base_row.get("points") or [], args)
        if not ok:
            return {
                "status": "rejected",
                "prompt_id": prompt_id,
                "reason": box_reason,
                "annotation": annotation,
                "metrics": m,
                "raw_response": pred_raw,
            }
        return {
            "status": "accepted",
            "prompt_id": prompt_id,
            "image": base_row["image"],
            "old_user": base_row.get("old_user", ""),
            "relation": relation,
            "relation_phrase": relation_phrase(relation),
            "seed_box": base_row["box"],
            "final_box": pred_box,
            "points": base_row.get("points") or [],
            "point_count": len(base_row.get("points") or []),
            "region_category": annotation["region_category"],
            "anchor_phrase": annotation["anchor_phrase"],
            "relation_to_anchor": annotation["relation_to_anchor"],
            "unique_description": annotation["unique_description"],
            "confidence": annotation["confidence"],
            "seed_to_final_metrics": m,
            "description_to_box": {
                "method": "qwen_original_image_no_overlay",
                "raw_response": pred_raw,
                "parsed_box": pred_box,
            },
            "box_to_description": {
                "method": "qwen_full_image_target_overlay",
                "annotation": annotation,
                "raw_response": ann_raw,
            },
            "source_stage": "newly_processed_missing",
            "source_solution_status": "not_previously_processed",
            "source_solution_reason": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": "rejected", "prompt_id": prompt_id, "reason": "exception", "error": repr(exc)}


def make_train_rows(record: dict[str, Any], dataset_name: str) -> list[dict[str, Any]]:
    box = record["final_box"]
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
    rows = []
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
                    {"from": "gpt", "value": box_answer(box)},
                ],
                "metadata": {
                    "source": "robopoint_region_ref_predbox_label_v1",
                    "task_type": "region_box_grounding",
                    "prompt_mode": mode,
                    "original_prompt_id": record["prompt_id"],
                    "relation": record["relation"],
                    "point_count": record["point_count"],
                    "seed_box": record["seed_box"],
                    "final_box_source": "description_to_box_pred",
                    "seed_to_final_metrics": record["seed_to_final_metrics"],
                    "source_stage": record["source_stage"],
                    "source_solution_status": record["source_solution_status"],
                    "source_solution_reason": record["source_solution_reason"],
                    "quality": "predbox_label_cleaned",
                },
            }
        )
    return rows


def build_existing_map(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    existing: dict[str, dict[str, Any]] = {}
    for path in args.existing_solution_files:
        for row in read_jsonl(path):
            pid = row.get("prompt_id")
            if not pid:
                continue
            # Prefer rows that already made it into accepted manifests, but keep
            # rejected rows when no accepted record exists for the prompt_id.
            if pid not in existing or row.get("status", "accepted") == "accepted":
                existing[pid] = row
    return existing


def draw_preview(records: list[dict[str, Any]], path: Path, limit: int = 30) -> None:
    if not records:
        return
    rng = random.Random(7)
    chosen = list(records)
    rng.shuffle(chosen)
    chosen = chosen[:limit]
    thumbs = []
    for rec in chosen:
        try:
            image = Image.open(rec["image"]).convert("RGB")
            w, h = image.size
            scale = min(260 / w, 180 / h)
            thumb = image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
            d = ImageDraw.Draw(thumb, "RGBA")

            def draw_norm_box(box: list[list[int]], color: tuple[int, int, int], width: int) -> None:
                x1 = int(box[0][0] / 1000 * thumb.size[0])
                y1 = int(box[0][1] / 1000 * thumb.size[1])
                x2 = int(box[1][0] / 1000 * thumb.size[0])
                y2 = int(box[1][1] / 1000 * thumb.size[1])
                d.rectangle([x1, y1, x2, y2], outline=(*color, 255), width=width)

            draw_norm_box(rec["seed_box"], (148, 163, 184), 2)
            draw_norm_box(rec["final_box"], (0, 112, 255), 4)
            d.rectangle([0, 0, thumb.size[0], 34], fill=(255, 255, 255, 235))
            d.text((5, 4), rec["unique_description"][:58], fill=(15, 23, 42))
            d.text((5, 18), f"pts={rec['seed_to_final_metrics']['point_recall']} area={rec['seed_to_final_metrics']['area_ratio_final_over_seed']}", fill=(71, 85, 105))
            thumbs.append(thumb.convert("RGB"))
        except Exception:
            continue
    if not thumbs:
        return
    cols = 3
    cell_w, cell_h = 270, 218
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h + 42), (248, 250, 252))
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        font = None
    d.text((12, 10), "RegionRef predbox label v1: gray=seed point box, blue=final description->box label", fill=(15, 23, 42), font=font)
    for i, thumb in enumerate(thumbs):
        x = (i % cols) * cell_w + 5
        y = (i // cols) * cell_h + 42
        sheet.paste(thumb, (x, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def summarize(records: list[dict[str, Any]], rejected: list[dict[str, Any]], rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    metrics_by_key: dict[str, list[float]] = defaultdict(list)
    for rec in records:
        for key, value in rec["seed_to_final_metrics"].items():
            if isinstance(value, (int, float)):
                metrics_by_key[key].append(float(value))
    mode_counter = Counter(row["metadata"]["prompt_mode"] for row in rows)
    return {
        "dataset": "semantic_nav_region_predbox_label_v1",
        "base_input": str(args.base_annotations),
        "output_dir": str(args.output_dir),
        "base_accepted": len(records),
        "base_rejected": len(rejected),
        "train_rows": len(rows),
        "prompt_modes": dict(mode_counter),
        "by_relation": dict(Counter(rec["relation"] for rec in records)),
        "by_region_category": dict(Counter(rec["region_category"] for rec in records)),
        "by_source_stage": dict(Counter(rec["source_stage"] for rec in records)),
        "by_source_solution_reason": dict(Counter(str(rec["source_solution_reason"]) for rec in records)),
        "rejected_reasons": dict(Counter(rec.get("reason") for rec in rejected)),
        "metrics_mean": {key: round(sum(vals) / len(vals), 4) for key, vals in metrics_by_key.items() if vals},
        "thresholds": {
            "min_point_recall": args.min_point_recall,
            "min_final_box_area": args.min_final_box_area,
            "max_final_box_area": args.max_final_box_area,
            "min_points": args.min_points,
        },
        "notes": [
            "Final target.box is the VLM description->box prediction on the original image.",
            "The old point-derived box is preserved as metadata.seed_box and base seed_box.",
            "Samples previously rejected only because pred_box differed from seed_box are recovered when pred_box is valid and covers the original points.",
            "Temporary blue/green overlay references are banned from target descriptions; red rectangles may be real RoboPoint anchors.",
        ],
    }


def validate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    box_re = re.compile(r"<box>\[\[(\d+),(\d+)\],\[(\d+),(\d+)\]\]</box>")
    bad = 0
    overlay = 0
    overlay_re = re.compile(
        r"\b(?:blue|green)\s+(?:rectangle|rectangular|box|outline|outlined|border|highlight|highlighted|overlay|target)\b|"
        r"\b(?:coordinate|pixel|annotation|overlay|bounding box)\b",
        re.I,
    )
    for row in rows:
        answer = row["conversations"][-1]["value"].strip()
        m = box_re.fullmatch(answer)
        if not m:
            bad += 1
            continue
        got = [[int(m.group(1)), int(m.group(2))], [int(m.group(3)), int(m.group(4))]]
        if got != row["target"]["box"]:
            bad += 1
        if overlay_re.search(row["target"].get("description", "")):
            overlay += 1
    return {"bad_box_answers": bad, "temporary_overlay_artifact_mentions": overlay}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--existing-solution-files", type=Path, nargs="*", default=[])
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", default="qwen-122b")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--image-max-side", type=int, default=1000)
    parser.add_argument("--request-timeout", type=float, default=90.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=1.5)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--min-points", type=int, default=4)
    parser.add_argument("--min-seed-box-area", type=int, default=100)
    parser.add_argument("--min-final-box-area", type=int, default=300)
    parser.add_argument("--max-final-box-area", type=int, default=360000)
    parser.add_argument("--min-point-recall", type=float, default=0.75)
    parser.add_argument("--include-weak", action="store_true")
    parser.add_argument("--process-missing", action="store_true")
    parser.add_argument("--seed", type=int, default=20260519)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_rows = read_jsonl(args.base_annotations)
    base_by_id = {row["prompt_id"]: row for row in base_rows}
    existing = build_existing_map(args)

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for row in base_rows:
        pid = row["prompt_id"]
        if pid in existing:
            result = extract_existing_record(existing[pid], row, args)
            if result.get("status") == "accepted":
                accepted.append(result)
            else:
                result.setdefault("source_stage", "existing_solution_a")
                rejected.append(result)
        else:
            missing.append(row)

    if args.process_missing and missing:
        rng = random.Random(args.seed)
        rng.shuffle(missing)
        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process_missing_row, args, row) for row in missing]
            for fut in as_completed(futures):
                result = fut.result()
                done += 1
                if result.get("status") == "accepted":
                    accepted.append(result)
                else:
                    result.setdefault("source_stage", "newly_processed_missing")
                    rejected.append(result)
                if done % 100 == 0:
                    print(json.dumps({"processed_missing": done, "accepted_total": len(accepted), "rejected_total": len(rejected)}, ensure_ascii=False), flush=True)
    else:
        for row in missing:
            rejected.append({"status": "rejected", "prompt_id": row["prompt_id"], "reason": "not_previously_processed", "source_stage": "missing_not_processed"})

    accepted.sort(key=lambda x: x["prompt_id"])
    rejected.sort(key=lambda x: str(x.get("prompt_id")))

    dataset_name = "semantic_nav_region_predbox_label_v1"
    train_rows: list[dict[str, Any]] = []
    for rec in accepted:
        train_rows.extend(make_train_rows(rec, dataset_name))

    high_quality_rows = train_rows
    write_jsonl(args.output_dir / f"{dataset_name}_base_accepted.jsonl", accepted)
    write_jsonl(args.output_dir / f"{dataset_name}_base_rejected.jsonl", rejected)
    write_jsonl(args.output_dir / f"{dataset_name}.jsonl", train_rows)
    write_jsonl(args.output_dir / f"{dataset_name}_high_quality.jsonl", high_quality_rows)
    draw_preview(accepted, args.output_dir / f"{dataset_name}_preview.png")

    summary = summarize(accepted, rejected, train_rows, args)
    summary["row_validation"] = validate_rows(train_rows)
    summary["base_input_count"] = len(base_rows)
    summary["base_input_ids"] = len(base_by_id)
    write_json(args.output_dir / f"{dataset_name}_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
