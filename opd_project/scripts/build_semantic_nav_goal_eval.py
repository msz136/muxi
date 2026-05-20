#!/usr/bin/env python3
"""Build precise semantic-navigation eval sets for object and region grounding.

The output is aligned with the intended inference task:
image + semantic target information -> <box>[[x1,y1],[x2,y2]]</box>.

Region eval reuses Solution-A bidirectionally validated holdout records.
Object eval samples supported-relation object_ref records and keeps only rows
whose object label can recover the target box on the original image.
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


SUPPORTED_OBJECT_RELATIONS = ["on", "beside", "inside", "left", "right", "behind", "front", "between"]
REGION_MODES = ["description_plain", "semantic_json", "description_relation"]
OBJECT_MODES = ["object_text", "object_json", "object_relation_compact"]

REGION_SYSTEM = (
    "You are a semantic navigation region-grounding assistant. Given an image and a uniquely "
    "described target navigation region, return the target region's bounding box in coordinates "
    "from 0 to 1000. Return only <box>[[x1,y1],[x2,y2]]</box>."
)
OBJECT_SYSTEM = (
    "You are a semantic navigation grounding assistant. Given an image and target object "
    "information, return the target object's bounding box in coordinates from 0 to 1000. "
    "Return only <box>[[x1,y1],[x2,y2]]</box>."
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def box_answer(box: list[list[int]]) -> str:
    return f"<box>[[{box[0][0]},{box[0][1]}],[{box[1][0]},{box[1][1]}]]</box>"


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
    hit = 0
    for x, y in points:
        if box[0][0] <= x <= box[1][0] and box[0][1] <= y <= box[1][1]:
            hit += 1
    return hit / len(points)


def relation_phrase(rel: str) -> str:
    mapping = {
        "on": "on the highlighted object or surface",
        "inside": "inside the highlighted container",
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


def image_to_data_url(image: Image.Image, max_side: int = 1000) -> str:
    image = image.convert("RGB")
    w, h = image.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        image = image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


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


def parse_box(obj: Any, raw: str) -> list[list[int]] | None:
    candidates: list[Any] = []
    if isinstance(obj, list):
        candidates.extend(obj)
    elif isinstance(obj, dict):
        candidates.append(obj)
    for item in candidates:
        if not isinstance(item, dict):
            continue
        val = item.get("bbox_2d") or item.get("box") or item.get("bbox")
        if isinstance(val, list) and len(val) == 4:
            nums = val
        elif isinstance(val, list) and len(val) == 2 and all(isinstance(x, list) for x in val):
            nums = [val[0][0], val[0][1], val[1][0], val[1][1]]
        else:
            nums = None
        if nums is not None:
            try:
                vals = [int(round(float(x))) for x in nums]
                x1, y1, x2, y2 = vals
                x1, x2 = sorted([max(0, min(1000, x1)), max(0, min(1000, x2))])
                y1, y2 = sorted([max(0, min(1000, y1)), max(0, min(1000, y2))])
                if x2 > x1 and y2 > y1:
                    return [[x1, y1], [x2, y2]]
            except Exception:
                pass
    nums = [int(x) for x in re.findall(r"-?\d+", raw)]
    if len(nums) >= 4:
        x1, y1, x2, y2 = nums[:4]
        x1, x2 = sorted([max(0, min(1000, x1)), max(0, min(1000, x2))])
        y1, y2 = sorted([max(0, min(1000, y1)), max(0, min(1000, y2))])
        if x2 > x1 and y2 > y1:
            return [[x1, y1], [x2, y2]]
    return None


def call_qwen(args: argparse.Namespace, image: Image.Image, prompt: str, max_tokens: int = 256) -> tuple[Any, str]:
    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_to_data_url(image, args.validation_image_max_side)}},
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = "Bearer " + args.api_key
    body = json.dumps(payload).encode("utf-8")
    last = ""
    for attempt in range(1, args.retries + 1):
        try:
            req = urllib.request.Request(args.api_url, data=body, headers=headers)
            with urllib.request.urlopen(req, timeout=args.request_timeout) as response:
                resp = json.loads(response.read().decode("utf-8"))
            raw = resp["choices"][0]["message"]["content"]
            return parse_jsonish(raw), raw
        except Exception as exc:  # noqa: BLE001
            last = repr(exc)
            time.sleep(args.retry_sleep * attempt)
    return {}, last


def object_name_is_precise(name: str) -> bool:
    lower = re.sub(r"\s+", " ", name.strip().lower())
    if len(lower.split()) < 1:
        return False
    banned = {
        "object",
        "target object",
        "item",
        "target item",
        "thing",
        "region",
        "area",
        "surface",
        "container interior",
        "empty space",
    }
    return lower not in banned


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
        and metrics["area_ratio_pred_over_target"] <= args.max_area_ratio
        and metrics["center_distance"] <= args.max_center_distance
    )
    return ok, metrics


def object_validation_prompt(row: dict[str, Any], object_name: str, attrs: list[str], rel_phrase: str) -> str:
    info = {
        "object_name": object_name,
        "attributes": attrs[:6],
        "relation": rel_phrase,
        "anchor_object": "highlighted object",
    }
    return (
        "Locate exactly one target object in the original image. The image may contain a red-highlighted "
        "anchor object; use it only as the spatial anchor. Return JSON only, as "
        '[{"bbox_2d":[x1,y1,x2,y2],"label":"short label"}], with integer coordinates from 0 to 1000.\n'
        "Target object information:\n"
        + json.dumps(info, ensure_ascii=False, separators=(",", ":"))
    )


def process_object_candidate(args: argparse.Namespace, row: dict[str, Any]) -> dict[str, Any]:
    ann = row.get("annotation") or {}
    object_name = str(ann.get("object_name") or "").strip()
    attrs = [str(x).strip() for x in ann.get("attributes") or [] if str(x).strip()]
    region_type = str(ann.get("region_type") or "").lower()
    confidence = str(ann.get("confidence") or "").lower()
    if row.get("quality") != "high_quality":
        return {"status": "rejected", "prompt_id": row.get("prompt_id"), "reason": "not_high_quality"}
    if row.get("relation") not in SUPPORTED_OBJECT_RELATIONS:
        return {"status": "rejected", "prompt_id": row.get("prompt_id"), "reason": "unsupported_relation"}
    if region_type != "object":
        return {"status": "rejected", "prompt_id": row.get("prompt_id"), "reason": "not_object_region"}
    if confidence != "high":
        return {"status": "rejected", "prompt_id": row.get("prompt_id"), "reason": "not_high_confidence"}
    if not object_name_is_precise(object_name):
        return {"status": "rejected", "prompt_id": row.get("prompt_id"), "reason": "generic_object_name", "object_name": object_name}
    if len(row.get("points") or []) < args.min_points:
        return {"status": "rejected", "prompt_id": row.get("prompt_id"), "reason": "few_points"}
    box_area = area(row["box"])
    if box_area < args.min_box_area or box_area > args.max_box_area:
        return {"status": "rejected", "prompt_id": row.get("prompt_id"), "reason": "box_area"}

    try:
        image = Image.open(row["image"]).convert("RGB")
        rel_phrase = relation_phrase(row["relation"])
        obj, raw = call_qwen(args, image, object_validation_prompt(row, object_name, attrs, rel_phrase))
        pred = parse_box(obj, raw)
        if pred is None:
            return {"status": "rejected", "prompt_id": row.get("prompt_id"), "reason": "no_pred_box", "raw_response": raw}
        ok, metrics = validate_geometry(row["box"], pred, row.get("points") or [], args)
        if not ok:
            return {
                "status": "rejected",
                "prompt_id": row.get("prompt_id"),
                "reason": "object_info_to_box_geometry",
                "validation": metrics,
                "raw_response": raw,
                "object_name": object_name,
                "relation": row.get("relation"),
            }
        return {
            "status": "accepted",
            "prompt_id": row["prompt_id"],
            "image": row["image"],
            "old_user": row.get("old_user", ""),
            "relation": row["relation"],
            "relation_phrase": rel_phrase,
            "box": row["box"],
            "points": row.get("points") or [],
            "point_count": len(row.get("points") or []),
            "box_area": box_area,
            "object_name": object_name,
            "attributes": attrs,
            "region_type": region_type,
            "confidence": confidence,
            "object_info_to_box_validation": {
                **metrics,
                "method": "qwen_original_image_no_overlay",
                "raw_response": raw,
            },
            "heldout_reason": "goal_eval_object_holdout_exclude_from_future_training",
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": "rejected", "prompt_id": row.get("prompt_id"), "reason": "exception", "error": repr(exc)}


def make_region_rows(record: dict[str, Any], dataset_name: str) -> list[dict[str, Any]]:
    box = record["box"]
    target = {
        "kind": "region",
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
                    {"from": "system", "value": REGION_SYSTEM},
                    {"from": "human", "value": prompt},
                    {"from": "gpt", "value": box_answer(box)},
                ],
                "metadata": {
                    "split": "eval",
                    "source": "robopoint_region_ref_solution_a_bidir_goal_eval",
                    "task_type": "region_box_grounding_eval",
                    "prompt_mode": mode,
                    "original_prompt_id": record["prompt_id"],
                    "relation": record["relation"],
                    "point_count": record["point_count"],
                    "box_area": record["box_area"],
                    "description_to_box_validation": record["description_to_box_validation"],
                    "heldout_reason": record.get("heldout_reason", "goal_eval_region_holdout"),
                    "quality": "bidir_validated",
                },
            }
        )
    return rows


def make_object_rows(record: dict[str, Any], dataset_name: str) -> list[dict[str, Any]]:
    box = record["box"]
    target = {
        "kind": "object",
        "object_name": record["object_name"],
        "attributes": record["attributes"],
        "relation": record["relation_phrase"],
        "anchor_object": "highlighted object",
        "box": box,
    }
    attr_text = ", ".join(record["attributes"][:4])
    name_with_attrs = record["object_name"] if not attr_text else f"{record['object_name']} ({attr_text})"
    prompts = [
        (
            "object_text",
            f"<image>\nFind the target object: {name_with_attrs}. It is {record['relation_phrase']}.\n"
            "Return only <box>[[x1,y1],[x2,y2]]</box> with integer coordinates from 0 to 1000.",
        ),
        (
            "object_json",
            "<image>\nTarget object information:\n"
            + json.dumps(
                {
                    "object_name": record["object_name"],
                    "attributes": record["attributes"][:6],
                    "relation": record["relation_phrase"],
                    "anchor_object": "highlighted object",
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\nReturn only the bounding box as <box>[[x1,y1],[x2,y2]]</box>.",
        ),
        (
            "object_relation_compact",
            f"<image>\nTarget object information: {record['object_name']}; relation: {record['relation_phrase']}; "
            "anchor_object: highlighted object.\nReturn only <box>[[x1,y1],[x2,y2]]</box>.",
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
                    {"from": "system", "value": OBJECT_SYSTEM},
                    {"from": "human", "value": prompt},
                    {"from": "gpt", "value": box_answer(box)},
                ],
                "metadata": {
                    "split": "eval",
                    "source": "robopoint_object_ref_goal_eval",
                    "task_type": "object_box_grounding_eval",
                    "prompt_mode": mode,
                    "original_prompt_id": record["prompt_id"],
                    "relation": record["relation"],
                    "point_count": record["point_count"],
                    "box_area": record["box_area"],
                    "object_info_to_box_validation": record["object_info_to_box_validation"],
                    "heldout_reason": record["heldout_reason"],
                    "quality": "object_info_validated",
                },
            }
        )
    return rows


def select_balanced(rows: list[dict[str, Any]], target_n: int, key: str, seed: int) -> list[dict[str, Any]]:
    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key[str(row.get(key, ""))].append(row)
    rng = random.Random(seed)
    for bucket in by_key.values():
        rng.shuffle(bucket)
    selected: list[dict[str, Any]] = []
    ordered_keys = sorted(by_key, key=lambda k: (-len(by_key[k]), k))
    cursor = 0
    while len(selected) < target_n and any(by_key.values()):
        k = ordered_keys[cursor % len(ordered_keys)]
        if by_key[k]:
            selected.append(by_key[k].pop())
        cursor += 1
        if cursor > target_n * max(10, len(ordered_keys) * 4):
            break
    return selected[:target_n]


def build_region_eval(args: argparse.Namespace, output_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base = read_jsonl(args.region_base_manifest)
    base = select_balanced(base, min(args.region_base_count, len(base)), "relation", args.seed)
    rows: list[dict[str, Any]] = []
    for rec in base:
        rows.extend(make_region_rows(rec, "semantic_nav_goal_reg_eval_v1"))
    write_jsonl(output_dir / "semantic_nav_goal_reg_eval_v1_base_manifest.jsonl", base)
    write_jsonl(output_dir / "semantic_nav_goal_reg_eval_v1.jsonl", rows)
    (output_dir / "semantic_nav_goal_reg_eval_v1_holdout_prompt_ids.txt").write_text(
        "\n".join(rec["prompt_id"] for rec in base) + "\n", encoding="utf-8"
    )
    draw_preview(base, output_dir / "semantic_nav_goal_reg_eval_v1_preview.png", "REG goal eval: target box")
    return base, rows


def build_object_eval(args: argparse.Namespace, output_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    accepted_path = output_dir / "semantic_nav_goal_obj_eval_v1_base_manifest.jsonl"
    rejected_path = output_dir / "semantic_nav_goal_obj_eval_v1_rejected.jsonl"
    if accepted_path.exists() and not args.force_rebuild_object:
        accepted = read_jsonl(accepted_path)
        rejected = read_jsonl(rejected_path) if rejected_path.exists() else []
    else:
        raw = read_jsonl(args.object_base_annotations)
        rng = random.Random(args.seed)
        by_relation: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in raw:
            if row.get("relation") in SUPPORTED_OBJECT_RELATIONS:
                ann = row.get("annotation") or {}
                if ann.get("region_type") == "object" and ann.get("confidence") == "high" and object_name_is_precise(str(ann.get("object_name") or "")):
                    by_relation[row["relation"]].append(row)
        for bucket in by_relation.values():
            rng.shuffle(bucket)
        candidates: list[dict[str, Any]] = []
        per_rel_target = max(args.object_base_count // max(1, len(SUPPORTED_OBJECT_RELATIONS)), 1)
        for rel in SUPPORTED_OBJECT_RELATIONS:
            candidates.extend(by_relation.get(rel, [])[: max(args.object_scan_per_relation, per_rel_target * 6)])
        rng.shuffle(candidates)

        accepted = []
        rejected = []
        seen = set()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = []
            for row in candidates:
                if row["prompt_id"] in seen:
                    continue
                seen.add(row["prompt_id"])
                futures.append(pool.submit(process_object_candidate, args, row))
            for fut in as_completed(futures):
                result = fut.result()
                if result.get("status") == "accepted":
                    accepted.append(result)
                else:
                    rejected.append(result)
                if len(accepted) >= args.object_base_count and len(rejected) > args.object_base_count:
                    # Keep remaining futures running briefly; this avoids cancelling urllib calls mid-flight.
                    pass

        accepted = select_balanced(accepted, min(args.object_base_count, len(accepted)), "relation", args.seed)
        write_jsonl(accepted_path, accepted)
        write_jsonl(rejected_path, rejected)

    rows: list[dict[str, Any]] = []
    for rec in accepted:
        rows.extend(make_object_rows(rec, "semantic_nav_goal_obj_eval_v1"))
    write_jsonl(output_dir / "semantic_nav_goal_obj_eval_v1.jsonl", rows)
    (output_dir / "semantic_nav_goal_obj_eval_v1_holdout_prompt_ids.txt").write_text(
        "\n".join(rec["prompt_id"] for rec in accepted) + "\n", encoding="utf-8"
    )
    draw_preview(accepted, output_dir / "semantic_nav_goal_obj_eval_v1_preview.png", "OBJ goal eval: target box")
    return accepted, rows, rejected


def draw_preview(records: list[dict[str, Any]], path: Path, title: str, limit: int = 24) -> None:
    if not records:
        return
    thumbs = []
    for rec in records[:limit]:
        try:
            img = Image.open(rec["image"]).convert("RGB")
            w, h = img.size
            scale = min(220 / w, 150 / h)
            thumb = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
            d = ImageDraw.Draw(thumb)
            x1 = int(rec["box"][0][0] / 1000 * thumb.size[0])
            y1 = int(rec["box"][0][1] / 1000 * thumb.size[1])
            x2 = int(rec["box"][1][0] / 1000 * thumb.size[0])
            y2 = int(rec["box"][1][1] / 1000 * thumb.size[1])
            d.rectangle([x1, y1, x2, y2], outline=(0, 122, 255), width=3)
            label = rec.get("object_name") or rec.get("unique_description", "")[:34]
            d.rectangle([0, 0, thumb.size[0], 22], fill=(255, 255, 255))
            d.text((4, 4), str(label)[:38], fill=(15, 23, 42))
            thumbs.append(thumb)
        except Exception:
            continue
    if not thumbs:
        return
    cols = 4
    cell_w, cell_h = 230, 178
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h + 34), (248, 250, 252))
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        font = None
    d.text((10, 8), title, fill=(15, 23, 42), font=font)
    for i, thumb in enumerate(thumbs):
        x = (i % cols) * cell_w + 5
        y = (i // cols) * cell_h + 34
        sheet.paste(thumb, (x, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def metric_means(records: list[dict[str, Any]], key: str) -> dict[str, float]:
    vals: dict[str, list[float]] = defaultdict(list)
    for rec in records:
        metrics = rec.get(key) or rec.get("description_to_box_validation") or {}
        for name in ["iou", "target_coverage", "pred_coverage", "area_ratio_pred_over_target", "center_distance", "point_recall"]:
            if name in metrics:
                vals[name].append(float(metrics[name]))
    return {k: round(sum(v) / len(v), 4) for k, v in vals.items() if v}


def validate_output_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    box_re = re.compile(r"<box>\[\[(\d+),(\d+)\],\[(\d+),(\d+)\]\]</box>")
    bad = 0
    modes = Counter()
    ids = Counter()
    for row in rows:
        modes[row["metadata"]["prompt_mode"]] += 1
        ids[row["metadata"]["original_prompt_id"]] += 1
        m = box_re.fullmatch(row["conversations"][-1]["value"])
        if not m:
            bad += 1
            continue
        got = [[int(m.group(1)), int(m.group(2))], [int(m.group(3)), int(m.group(4))]]
        if got != row["target"]["box"]:
            bad += 1
    return {
        "rows": len(rows),
        "base": len(ids),
        "prompt_modes": dict(modes),
        "bad_box_answers": bad,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--region-base-manifest", type=Path, required=True)
    parser.add_argument("--object-base-annotations", type=Path, required=True)
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", default="qwen-122b")
    parser.add_argument("--region-base-count", type=int, default=48)
    parser.add_argument("--object-base-count", type=int, default=48)
    parser.add_argument("--object-scan-per-relation", type=int, default=50)
    parser.add_argument("--min-points", type=int, default=4)
    parser.add_argument("--min-box-area", type=int, default=350)
    parser.add_argument("--max-box-area", type=int, default=260000)
    parser.add_argument("--min-target-coverage", type=float, default=0.50)
    parser.add_argument("--min-point-recall", type=float, default=0.75)
    parser.add_argument("--max-center-distance", type=float, default=140.0)
    parser.add_argument("--max-area-ratio", type=float, default=6.0)
    parser.add_argument("--validation-image-max-side", type=int, default=1000)
    parser.add_argument("--request-timeout", type=float, default=90.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=1.5)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--force-rebuild-object", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reg_base, reg_rows = build_region_eval(args, args.output_dir)
    obj_base, obj_rows, obj_rejected = build_object_eval(args, args.output_dir)

    mixed_rows = reg_rows + obj_rows
    write_jsonl(args.output_dir / "semantic_nav_goal_eval_v1.jsonl", mixed_rows)
    summary = {
        "dataset": "semantic_nav_goal_eval_v1",
        "output_dir": str(args.output_dir),
        "reg_eval_jsonl": str(args.output_dir / "semantic_nav_goal_reg_eval_v1.jsonl"),
        "obj_eval_jsonl": str(args.output_dir / "semantic_nav_goal_obj_eval_v1.jsonl"),
        "mixed_eval_jsonl": str(args.output_dir / "semantic_nav_goal_eval_v1.jsonl"),
        "reg_base_manifest": str(args.output_dir / "semantic_nav_goal_reg_eval_v1_base_manifest.jsonl"),
        "obj_base_manifest": str(args.output_dir / "semantic_nav_goal_obj_eval_v1_base_manifest.jsonl"),
        "reg_holdout_prompt_ids": str(args.output_dir / "semantic_nav_goal_reg_eval_v1_holdout_prompt_ids.txt"),
        "obj_holdout_prompt_ids": str(args.output_dir / "semantic_nav_goal_obj_eval_v1_holdout_prompt_ids.txt"),
        "design": {
            "reg": "unique navigation-region description/relation -> region box",
            "obj": "object_name + attributes + relation to highlighted anchor -> object box",
            "answer_format": "<box>[[x1,y1],[x2,y2]]</box>",
            "coordinates": "normalized integer coordinates from 0 to 1000",
        },
        "thresholds": {
            "min_target_coverage": args.min_target_coverage,
            "min_point_recall": args.min_point_recall,
            "max_center_distance": args.max_center_distance,
            "max_area_ratio": args.max_area_ratio,
        },
        "reg": {
            "base": len(reg_base),
            "rows": len(reg_rows),
            "by_relation": dict(Counter(x["relation"] for x in reg_base)),
            "by_region_category": dict(Counter(x["region_category"] for x in reg_base)),
            "row_validation": validate_output_rows(reg_rows),
            "validation_metrics_mean": metric_means(reg_base, "description_to_box_validation"),
        },
        "obj": {
            "base": len(obj_base),
            "rows": len(obj_rows),
            "rejected_scanned": len(obj_rejected),
            "by_relation": dict(Counter(x["relation"] for x in obj_base)),
            "by_object_name": dict(Counter(x["object_name"] for x in obj_base).most_common(30)),
            "row_validation": validate_output_rows(obj_rows),
            "validation_metrics_mean": metric_means(obj_base, "object_info_to_box_validation"),
        },
        "mixed": validate_output_rows(mixed_rows),
        "notes": [
            "Object eval samples are from full object_ref annotations and must be excluded from future object training mixes by prompt_id.",
            "Region eval samples reuse the existing Solution-A bidirectional holdout and are already excluded from the regenerated region train set.",
            "The object eval avoids generic labels such as target object and keeps only high-confidence object regions.",
        ],
    }
    write_json(args.output_dir / "semantic_nav_goal_eval_v1_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
