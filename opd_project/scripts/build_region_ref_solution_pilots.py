#!/usr/bin/env python3
"""Build 10-sample pilots for two RegionRef repair strategies.

Solution A keeps the current point-derived bbox and asks a VLM to write a
unique full-image reference expression.

Solution B first redefines the target region box with deterministic heuristics
using the red anchor rectangle, relation, and point seeds, then asks the same
VLM to write a unique reference expression for that redesigned box.
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
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


RELATION_PHRASES = {
    "on": "on the highlighted area or surface",
    "on-front": "on the front side of the highlighted area or surface",
    "on-left": "on the left side of the highlighted area or surface",
    "on-right": "on the right side of the highlighted area or surface",
    "on-back": "on the back side of the highlighted area or surface",
    "left": "to the left of the highlighted object",
    "right": "to the right of the highlighted object",
    "inside": "inside the highlighted container or area",
    "beside": "beside the highlighted object",
    "front": "in front of the highlighted object",
    "behind": "behind the highlighted object",
    "above": "above the highlighted object",
    "below": "below the highlighted object",
    "between": "between the highlighted objects",
}

ANCHOR_BY_RELATION = {
    "on": "highlighted area or surface",
    "on-front": "highlighted area or surface",
    "on-left": "highlighted area or surface",
    "on-right": "highlighted area or surface",
    "on-back": "highlighted area or surface",
    "inside": "highlighted container or area",
    "between": "highlighted objects",
}

SYSTEM_PROMPT = (
    "You are a semantic navigation region-grounding assistant. Given an image "
    "and a uniquely described target navigation region, return the target "
    "region's bounding box in coordinates from 0 to 1000. Return only "
    "<box>[[x1,y1],[x2,y2]]</box>."
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def relation_phrase(relation: str) -> str:
    return RELATION_PHRASES.get(relation, "matching the highlighted spatial relation")


def anchor_object(relation: str) -> str:
    return ANCHOR_BY_RELATION.get(relation, "highlighted object")


def box_answer(box: list[list[int]]) -> str:
    return "<box>" + json.dumps(box, separators=(",", ":")) + "</box>"


def remote_url(args: argparse.Namespace, remote_path: str) -> str:
    rel = remote_path
    if args.remote_image_prefix and rel.startswith(args.remote_image_prefix):
        rel = rel[len(args.remote_image_prefix) :]
    rel = rel.lstrip("/")
    return args.image_base_url.rstrip("/") + "/" + rel.replace("\\", "/")


def fetch_image(args: argparse.Namespace, remote_path: str) -> Image.Image:
    with urllib.request.urlopen(remote_url(args, remote_path), timeout=args.image_timeout) as response:
        data = response.read()
    return Image.open(io.BytesIO(data)).convert("RGB")


def norm_box_to_pixels(box: list[list[int]], width: int, height: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(width - 1, round(box[0][0] / 1000 * width)))
    y1 = max(0, min(height - 1, round(box[0][1] / 1000 * height)))
    x2 = max(x1 + 1, min(width, round(box[1][0] / 1000 * width)))
    y2 = max(y1 + 1, min(height, round(box[1][1] / 1000 * height)))
    return x1, y1, x2, y2


def pixels_to_norm_box(box: tuple[int, int, int, int], width: int, height: int) -> list[list[int]]:
    x1, y1, x2, y2 = box
    return [
        [max(0, min(1000, round(x1 / width * 1000))), max(0, min(1000, round(y1 / height * 1000)))],
        [max(0, min(1000, round(x2 / width * 1000))), max(0, min(1000, round(y2 / height * 1000)))],
    ]


def points_box(points: list[list[int]], margin: int = 12) -> list[list[int]]:
    xs = [int(p[0]) for p in points if isinstance(p, list) and len(p) >= 2]
    ys = [int(p[1]) for p in points if isinstance(p, list) and len(p) >= 2]
    if not xs or not ys:
        return [[0, 0], [1, 1]]
    return [
        [max(0, min(xs) - margin), max(0, min(ys) - margin)],
        [min(1000, max(xs) + margin), min(1000, max(ys) + margin)],
    ]


def expand_box(box: list[list[int]], factor_x: float, factor_y: float) -> list[list[int]]:
    x1, y1 = box[0]
    x2, y2 = box[1]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = max(24, (x2 - x1) * factor_x)
    h = max(24, (y2 - y1) * factor_y)
    return [
        [max(0, round(cx - w / 2)), max(0, round(cy - h / 2))],
        [min(1000, round(cx + w / 2)), min(1000, round(cy + h / 2))],
    ]


def clip_box_to_region(box: list[list[int]], *, x_min=0, x_max=1000, y_min=0, y_max=1000) -> list[list[int]]:
    x1, y1 = box[0]
    x2, y2 = box[1]
    x1 = max(x_min, min(x_max - 1, x1))
    x2 = max(x1 + 1, min(x_max, x2))
    y1 = max(y_min, min(y_max - 1, y1))
    y2 = max(y1 + 1, min(y_max, y2))
    return [[round(x1), round(y1)], [round(x2), round(y2)]]


def red_anchor_box(image: Image.Image, point_box: list[list[int]] | None = None) -> list[list[int]] | None:
    width, height = image.size
    scale = min(1.0, 720 / max(width, height))
    small = image.resize((round(width * scale), round(height * scale)), Image.Resampling.BILINEAR) if scale < 1 else image
    sw, sh = small.size
    pix = small.load()
    red: set[tuple[int, int]] = set()
    for y in range(sh):
        for x in range(sw):
            r, g, b = pix[x, y]
            if r > 145 and g < 105 and b < 105 and r > g * 1.6 and r > b * 1.6:
                red.add((x, y))
    if len(red) < 20:
        return None

    components: list[tuple[int, int, int, int, int]] = []
    while red:
        start = red.pop()
        stack = [start]
        xs = [start[0]]
        ys = [start[1]]
        count = 1
        while stack:
            cx, cy = stack.pop()
            for nx in (cx - 1, cx, cx + 1):
                for ny in (cy - 1, cy, cy + 1):
                    if (nx, ny) in red:
                        red.remove((nx, ny))
                        stack.append((nx, ny))
                        xs.append(nx)
                        ys.append(ny)
                        count += 1
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        bw = x2 - x1 + 1
        bh = y2 - y1 + 1
        if count >= 15 and bw >= 8 and bh >= 8:
            components.append((x1, y1, x2 + 1, y2 + 1, count))
    if not components:
        return None

    px = py = None
    if point_box is not None:
        px = ((point_box[0][0] + point_box[1][0]) / 2) / 1000 * sw
        py = ((point_box[0][1] + point_box[1][1]) / 2) / 1000 * sh

    def distance_to_bbox(component: tuple[int, int, int, int, int]) -> float:
        if px is None or py is None:
            # Prefer meaningful rectangle-sized anchors over tiny red texture.
            x1, y1, x2, y2, count = component
            return -((x2 - x1) * (y2 - y1)) + count * -0.01
        x1, y1, x2, y2, count = component
        dx = max(x1 - px, 0, px - x2)
        dy = max(y1 - py, 0, py - y2)
        contains = x1 <= px <= x2 and y1 <= py <= y2
        area = max(1, (x2 - x1) * (y2 - y1))
        # If the point seed is inside multiple red boxes, choose the smallest
        # containing box; otherwise choose the nearest red anchor.
        return (-1_000_000 + area * 0.01) if contains else dx * dx + dy * dy + area * 0.0001

    x1, y1, x2, y2, _ = min(components, key=distance_to_bbox)
    if scale < 1:
        x1 = round(x1 / scale)
        y1 = round(y1 / scale)
        x2 = round(x2 / scale)
        y2 = round(y2 / scale)
    return pixels_to_norm_box((x1, y1, x2, y2), width, height)


def box_width_height(box: list[list[int]]) -> tuple[int, int]:
    return box[1][0] - box[0][0], box[1][1] - box[0][1]


def clip_if_reasonable(
    box: list[list[int]],
    original: list[list[int]],
    *,
    x_min=0,
    x_max=1000,
    y_min=0,
    y_max=1000,
) -> list[list[int]]:
    clipped = clip_box_to_region(box, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
    width, height = box_width_height(clipped)
    if width < 24 or height < 24:
        return original
    # Keep the seed center inside the final box; otherwise the hard geometric
    # clipping probably interpreted front/behind/above/below incorrectly.
    cx = (original[0][0] + original[1][0]) / 2
    cy = (original[0][1] + original[1][1]) / 2
    if not (clipped[0][0] <= cx <= clipped[1][0] and clipped[0][1] <= cy <= clipped[1][1]):
        return original
    return clipped


def infer_side(anchor: list[list[int]] | None, point_box: list[list[int]]) -> str:
    if anchor is None:
        return "beside"
    ax = (anchor[0][0] + anchor[1][0]) / 2
    ay = (anchor[0][1] + anchor[1][1]) / 2
    px = (point_box[0][0] + point_box[1][0]) / 2
    py = (point_box[0][1] + point_box[1][1]) / 2
    dx = px - ax
    dy = py - ay
    if abs(dx) >= abs(dy):
        return "right" if dx >= 0 else "left"
    return "below" if dy >= 0 else "above"


def redefine_region_box(row: dict[str, Any], image: Image.Image) -> tuple[list[list[int]], str, dict[str, Any]]:
    relation = row["relation"]
    current = row["box"]
    pbox = points_box(row.get("points") or [])
    anchor = red_anchor_box(image, pbox)
    strategy = "expanded_point_seed"
    new_relation = relation

    if anchor and relation in {"on", "inside"}:
        # For on/inside, the point seed usually marks only a sampled portion of
        # the requested area. Use the full highlighted anchor as the semantic
        # region, lightly inset to avoid learning the red outline itself.
        x1, y1 = anchor[0]
        x2, y2 = anchor[1]
        inset_x = max(4, round((x2 - x1) * 0.03))
        inset_y = max(4, round((y2 - y1) * 0.03))
        box = [[x1 + inset_x, y1 + inset_y], [x2 - inset_x, y2 - inset_y]]
        strategy = "anchor_region_inset"
    elif anchor and relation in {"on-left", "on-right", "on-front", "on-back"}:
        x1, y1 = anchor[0]
        x2, y2 = anchor[1]
        if relation == "on-left":
            box = [[x1, y1], [round((x1 + x2) / 2), y2]]
        elif relation == "on-right":
            box = [[round((x1 + x2) / 2), y1], [x2, y2]]
        elif relation == "on-front":
            box = [[x1, round((y1 + y2) / 2)], [x2, y2]]
        else:
            box = [[x1, y1], [x2, round((y1 + y2) / 2)]]
        strategy = "anchor_subregion"
    else:
        box = expand_box(pbox or current, 2.4, 2.2)
        if anchor:
            side = relation
            if relation == "beside":
                side = infer_side(anchor, pbox)
                new_relation = side
            if side == "left":
                box = clip_if_reasonable(box, pbox, x_max=anchor[0][0])
            elif side == "right":
                box = clip_if_reasonable(box, pbox, x_min=anchor[1][0])
            elif side == "below":
                box = clip_if_reasonable(box, pbox, y_min=anchor[1][1])
            elif side == "above":
                box = clip_if_reasonable(box, pbox, y_max=anchor[0][1])
        strategy = "expanded_seed_clipped_to_relation"

    box = clip_box_to_region(box)
    return box, new_relation, {"anchor_box": anchor, "strategy": strategy, "current_box": current}


def draw_overlay(
    image: Image.Image,
    target_box: list[list[int]],
    *,
    points: list[list[int]] | None = None,
    box_color: tuple[int, int, int] = (0, 120, 255),
    label: str | None = None,
) -> Image.Image:
    out = image.copy()
    width, height = out.size
    x1, y1, x2, y2 = norm_box_to_pixels(target_box, width, height)
    draw = ImageDraw.Draw(out, "RGBA")
    draw.rectangle([x1, y1, x2, y2], fill=(*box_color, 45), outline=(*box_color, 255), width=max(3, width // 250))
    if points:
        for px, py in points[:40]:
            x = round(px / 1000 * width)
            y = round(py / 1000 * height)
            draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(255, 40, 40, 220))
    if label:
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except Exception:
            font = ImageFont.load_default()
        draw.rectangle([x1, max(0, y1 - 28), x1 + 220, y1], fill=(*box_color, 210))
        draw.text((x1 + 6, max(0, y1 - 26)), label, fill=(255, 255, 255, 255), font=font)
    return out


def image_data_url(image: Image.Image, max_side: int = 1100) -> str:
    img = image.copy()
    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(out.getvalue()).decode("ascii")


def parse_jsonish(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        text = match.group(0)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def call_qwen(args: argparse.Namespace, data_url: str, prompt: str) -> tuple[dict[str, Any], str]:
    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
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


def normalize_annotation(obj: dict[str, Any], relation: str, solution: str) -> dict[str, Any]:
    def clean(value: Any, fallback: str) -> str:
        text = str(value or "").strip().strip("\"'")
        text = re.sub(r"\s+", " ", text)
        return text[:240] or fallback

    region_category = clean(obj.get("region_category") or obj.get("region_type"), "free_space").lower()
    if region_category not in {"free_space", "surface", "container", "object", "unclear"}:
        region_category = "unclear"
    anchor = clean(obj.get("anchor_phrase") or obj.get("anchor_object"), anchor_object(relation))
    relation_to_anchor = clean(obj.get("relation_to_anchor") or obj.get("relation"), relation_phrase(relation))
    desc = clean(obj.get("unique_description") or obj.get("description"), f"target region {relation_phrase(relation)}")
    desc = remove_overlay_artifacts(desc)
    confidence = clean(obj.get("confidence") or obj.get("localization_confidence"), "medium").lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "medium"
    return {
        "region_category": region_category,
        "anchor_phrase": anchor,
        "relation_to_anchor": relation_to_anchor,
        "unique_description": desc,
        "confidence": confidence,
        "solution": solution,
        "raw_annotation": obj,
    }


def remove_overlay_artifacts(text: str) -> str:
    # The blue/green target boxes are temporary annotation aids and are not
    # present in training images. Keep references to red rectangles because
    # those are part of the RoboPoint source images.
    replacements = [
        (r"\b(?:blue|green)\s+rectangular\s+(?:patch|area|region)\b", "small patch"),
        (r"\b(?:blue|green)\s+(?:patch|area|region)\b", "small patch"),
        (r"\b(?:blue|green)\s+rectangle\b", "target region"),
        (r"\b(?:blue|green)\s+box\b", "target region"),
        (r"\b(?:blue|green)-highlighted\s+", ""),
        (r"\b(?:blue|green)\s+highlighted\s+", ""),
        (r"\b(?:blue|green)\s+outlined\s+area\b", "target area"),
        (r"\b(?:blue|green)\s+highlighted\s+area\b", "target area"),
        (r"\b(?:blue|green)\s+highlight\b", "target area"),
        (r"\b(?:blue|green)\s+border\b", "target boundary"),
    ]
    out = text
    for pattern, repl in replacements:
        out = re.sub(pattern, repl, out, flags=re.I)
    out = re.sub(r"\bsmall\s+small\b", "small", out, flags=re.I)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def annotation_prompt(row: dict[str, Any], box: list[list[int]], solution: str, relation: str) -> str:
    original = row.get("old_user", "").replace("<image>", "").strip()
    relation_text = relation_phrase(relation)
    if solution == "A":
        title = "The blue rectangle marks the existing point-derived training box."
        box_note = (
            "This box may be only a small sampled patch of a larger navigation region. "
            "Describe the target so that a model could recover this specific patch."
        )
    else:
        title = "The blue rectangle marks a redesigned semantic navigation region box."
        box_note = (
            "Describe the target as a complete navigable region, not as a random point cloud. "
            "If the target is a surface/container/adjacent area, make the reference unique."
        )
    return (
        "You are creating training annotations for semantic navigation region grounding.\n"
        f"{title}\n"
        "The source image may already contain one or more red rectangles; those red rectangles "
        "are the highlighted anchor object/surface/container from the original RoboPoint task. "
        "The blue or green target overlay is only an annotation aid and is not part of the source image. "
        "Do not mention blue, green, target overlays, rectangles, boxes, coordinates, pixels, or annotation marks "
        "in the final description. You may mention red rectangles because they are part of the source image.\n"
        f"{box_note}\n\n"
        f"Original RoboPoint instruction:\n{original}\n\n"
        f"Known relation label: {relation_text}\n"
        f"Target normalized box for your reference only: {box}\n\n"
        "Return strict JSON only with these keys:\n"
        "{\n"
        '  "region_category": "free_space|surface|container|object|unclear",\n'
        '  "anchor_phrase": "short phrase naming the red-highlighted anchor",\n'
        '  "relation_to_anchor": "spatial relation to the anchor",\n'
        '  "unique_description": "one sentence, no coordinates, uniquely locating the target region in the original image",\n'
        '  "confidence": "high|medium|low"\n'
        "}"
    )


def training_record(
    row: dict[str, Any],
    annotation: dict[str, Any],
    *,
    box: list[list[int]],
    relation: str,
    solution_name: str,
    extra_metadata: dict[str, Any],
) -> dict[str, Any]:
    target = {
        "region_category": annotation["region_category"],
        "description": annotation["unique_description"],
        "relation": annotation["relation_to_anchor"],
        "anchor_object": annotation["anchor_phrase"],
        "box": box,
    }
    prompt = (
        "<image>\nNavigation target information:\n"
        + json.dumps(target | {"box": None}, ensure_ascii=False, separators=(",", ":"))
        + "\nReturn only the bounding box as <box>[[x1,y1],[x2,y2]]</box>."
    )
    return {
        "dataset": f"semantic_nav_region_box_grounding_{solution_name}_pilot10",
        "image": [row["image"]],
        "video": [],
        "target": target,
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": box_answer(box)},
        ],
        "metadata": {
            "source": "robopoint_region_ref_repair_pilot",
            "task_type": "region_box_grounding",
            "solution": solution_name,
            "original_prompt_id": row["prompt_id"],
            "original_relation": row["relation"],
            "normalized_relation": relation,
            "old_label": row.get("annotation", {}).get("object_name"),
            "old_region_type": row.get("annotation", {}).get("region_type"),
            "point_count": len(row.get("points") or []),
            "qwen_annotation": annotation,
            **extra_metadata,
        },
    }


def choose_rows(rows: list[dict[str, Any]], sample_size: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    wanted = ["on", "inside", "beside", "left", "right", "front", "behind", "between", "above", "below"]
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    for rel in wanted:
        candidates = [
            r
            for r in rows
            if r.get("relation") == rel
            and r.get("quality") == "high_quality"
            and len(r.get("points") or []) >= 4
            and r.get("prompt_id") not in used
        ]
        if candidates:
            r = rng.choice(candidates)
            selected.append(r)
            used.add(r["prompt_id"])
    if len(selected) < sample_size:
        candidates = [
            r
            for r in rows
            if r.get("quality") == "high_quality"
            and len(r.get("points") or []) >= 4
            and r.get("prompt_id") not in used
        ]
        rng.shuffle(candidates)
        selected.extend(candidates[: sample_size - len(selected)])
    return selected[:sample_size]


def make_preview(path: Path, records_a: list[dict[str, Any]], records_b: list[dict[str, Any]], images: dict[str, Image.Image]) -> None:
    try:
        title_font = ImageFont.truetype("arial.ttf", 15)
        small_font = ImageFont.truetype("arial.ttf", 11)
    except Exception:
        title_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    cards: list[Image.Image] = []
    for idx, (a, b) in enumerate(zip(records_a, records_b), start=1):
        image_path = a["image"][0]
        image = images[image_path].copy()
        width, height = image.size
        image.thumbnail((520, 280), Image.Resampling.LANCZOS)
        dw, dh = image.size
        sx, sy = dw / width, dh / height
        draw = ImageDraw.Draw(image, "RGBA")

        def draw_norm(box: list[list[int]], color: tuple[int, int, int], width_px: int) -> None:
            x1 = round(box[0][0] / 1000 * width * sx)
            y1 = round(box[0][1] / 1000 * height * sy)
            x2 = round(box[1][0] / 1000 * width * sx)
            y2 = round(box[1][1] / 1000 * height * sy)
            draw.rectangle([x1, y1, x2, y2], fill=(*color, 35), outline=(*color, 255), width=width_px)

        draw_norm(a["target"]["box"], (0, 110, 255), 3)
        draw_norm(b["target"]["box"], (0, 180, 90), 4)
        source_points = b["metadata"].get("source_points") or []
        for px, py in source_points[:40]:
            x = round(px / 1000 * width * sx)
            y = round(py / 1000 * height * sy)
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(255, 40, 40, 220))

        card = Image.new("RGB", (560, 470), (248, 250, 252))
        card.paste(image, ((560 - dw) // 2, 8))
        d = ImageDraw.Draw(card)
        y = dh + 18
        d.text((10, y), f"{idx}. rel={a['metadata']['original_relation']}  blue=A current bbox, green=B reboxed", fill=(15, 23, 42), font=title_font)
        y += 22
        a_desc = "A: " + a["target"]["description"]
        b_desc = "B: " + b["target"]["description"]
        for line in wrap_text(a_desc, 82)[:3]:
            d.text((10, y), line, fill=(30, 64, 175), font=small_font)
            y += 15
        for line in wrap_text(b_desc, 82)[:3]:
            d.text((10, y), line, fill=(21, 128, 61), font=small_font)
            y += 15
        d.text((10, y), f"A {a['target']['box']}   B {b['target']['box']}", fill=(71, 85, 105), font=small_font)
        cards.append(card)

    cols = 2
    rows_n = (len(cards) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 560, rows_n * 470), (226, 232, 240))
    for i, card in enumerate(cards):
        sheet.paste(card, ((i % cols) * 560, (i // cols) * 470))
    sheet.save(path)


def make_single_preview(path: Path, records: list[dict[str, Any]], images: dict[str, Image.Image], color: tuple[int, int, int]) -> None:
    try:
        title_font = ImageFont.truetype("arial.ttf", 14)
        small_font = ImageFont.truetype("arial.ttf", 11)
    except Exception:
        title_font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    cards: list[Image.Image] = []
    for idx, rec in enumerate(records, start=1):
        image = images[rec["image"][0]].copy()
        width, height = image.size
        image.thumbnail((430, 260), Image.Resampling.LANCZOS)
        dw, dh = image.size
        sx, sy = dw / width, dh / height
        draw = ImageDraw.Draw(image, "RGBA")
        box = rec["target"]["box"]
        x1 = round(box[0][0] / 1000 * width * sx)
        y1 = round(box[0][1] / 1000 * height * sy)
        x2 = round(box[1][0] / 1000 * width * sx)
        y2 = round(box[1][1] / 1000 * height * sy)
        draw.rectangle([x1, y1, x2, y2], fill=(*color, 42), outline=(*color, 255), width=4)
        for px, py in rec["metadata"].get("source_points") or []:
            x = round(px / 1000 * width * sx)
            y = round(py / 1000 * height * sy)
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(255, 40, 40, 220))
        card = Image.new("RGB", (460, 430), (248, 250, 252))
        card.paste(image, ((460 - dw) // 2, 8))
        d = ImageDraw.Draw(card)
        y = dh + 18
        d.text((10, y), f"{idx}. {rec['target']['region_category']} | rel={rec['metadata']['normalized_relation']}", fill=(15, 23, 42), font=title_font)
        y += 20
        for line in wrap_text(rec["target"]["description"], 62)[:5]:
            d.text((10, y), line, fill=(15, 23, 42), font=small_font)
            y += 15
        d.text((10, y), box_answer(rec["target"]["box"]), fill=(2, 132, 199), font=small_font)
        cards.append(card)
    cols = 2
    rows_n = (len(cards) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 460, rows_n * 430), (226, 232, 240))
    for i, card in enumerate(cards):
        sheet.paste(card, ((i % cols) * 460, (i // cols) * 430))
    sheet.save(path)


def wrap_text(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        if sum(len(w) + 1 for w in current) + len(word) > width and current:
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
    parser.add_argument("--image-base-url", required=True)
    parser.add_argument("--remote-image-prefix", default="/data/msz/dataset/RoboPoint/images")
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", default="qwen-122b")
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=260)
    parser.add_argument("--request-timeout", type=int, default=120)
    parser.add_argument("--image-timeout", type=int, default=30)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=1.5)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.annotations)
    selected = choose_rows(rows, args.sample_size, args.seed)
    images: dict[str, Image.Image] = {}
    records_a: list[dict[str, Any]] = []
    records_b: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []

    for idx, row in enumerate(selected, start=1):
        image = fetch_image(args, row["image"])
        images[row["image"]] = image

        a_box = row["box"]
        a_overlay = draw_overlay(image, a_box, box_color=(0, 110, 255))
        a_obj, a_raw = call_qwen(args, image_data_url(a_overlay), annotation_prompt(row, a_box, "A", row["relation"]))
        a_ann = normalize_annotation(a_obj, row["relation"], "A_current_bbox_unique_desc")
        a_record = training_record(
            row,
            a_ann,
            box=a_box,
            relation=row["relation"],
            solution_name="solution_a_current_bbox_unique_desc",
            extra_metadata={"source_points": row.get("points", []), "raw_qwen_response": a_raw},
        )
        records_a.append(a_record)

        b_box, b_relation, b_meta = redefine_region_box(row, image)
        b_overlay = draw_overlay(image, b_box, box_color=(0, 110, 255))
        b_obj, b_raw = call_qwen(args, image_data_url(b_overlay), annotation_prompt(row, b_box, "B", b_relation))
        b_ann = normalize_annotation(b_obj, b_relation, "B_reboxed_unique_desc")
        b_record = training_record(
            row,
            b_ann,
            box=b_box,
            relation=b_relation,
            solution_name="solution_b_reboxed_unique_desc",
            extra_metadata={**b_meta, "source_points": row.get("points", []), "raw_qwen_response": b_raw},
        )
        records_b.append(b_record)

        raw_rows.append(
            {
                "idx": idx,
                "prompt_id": row["prompt_id"],
                "image": row["image"],
                "original_relation": row["relation"],
                "old_label": row.get("annotation", {}).get("object_name"),
                "old_region_type": row.get("annotation", {}).get("region_type"),
                "solution_a": {"box": a_box, "annotation": a_ann},
                "solution_b": {"box": b_box, "relation": b_relation, "box_metadata": b_meta, "annotation": b_ann},
            }
        )
        print(
            f"[{idx}/{len(selected)}] {row['relation']} old={row.get('annotation', {}).get('object_name')} "
            f"A={a_ann['unique_description'][:70]} B={b_ann['unique_description'][:70]}",
            flush=True,
        )

    write_jsonl(args.output_dir / "solution_a_unique_desc_10.jsonl", records_a)
    write_jsonl(args.output_dir / "solution_b_reboxed_unique_desc_10.jsonl", records_b)
    write_json(args.output_dir / "solution_ab_raw_annotations_10.json", raw_rows)
    make_single_preview(args.output_dir / "solution_a_unique_desc_10_preview.png", records_a, images, (0, 110, 255))
    make_single_preview(args.output_dir / "solution_b_reboxed_unique_desc_10_preview.png", records_b, images, (0, 180, 90))
    make_preview(args.output_dir / "solution_ab_comparison_10.png", records_a, records_b, images)
    print(
        "[done] "
        + json.dumps(
            {
                "solution_a_jsonl": str(args.output_dir / "solution_a_unique_desc_10.jsonl"),
                "solution_b_jsonl": str(args.output_dir / "solution_b_reboxed_unique_desc_10.jsonl"),
                "raw_json": str(args.output_dir / "solution_ab_raw_annotations_10.json"),
                "preview_a": str(args.output_dir / "solution_a_unique_desc_10_preview.png"),
                "preview_b": str(args.output_dir / "solution_b_reboxed_unique_desc_10_preview.png"),
                "comparison": str(args.output_dir / "solution_ab_comparison_10.png"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
