#!/usr/bin/env python3
"""Build semantic-navigation box-grounding data from RoboPoint point samples.

The conversion pipeline is intentionally simple and auditable:
1. read cleaned RoboPoint point samples;
2. convert each point cloud to a bounding box with a small margin;
3. crop that box with ffmpeg;
4. ask a VLM labeler for the cropped target name/type;
5. expand each base sample into obj-only / relation-only / obj+relation prompts.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = (
    "You are a semantic navigation grounding assistant. Given an image and "
    "target object information, return the target object's bounding box in "
    "coordinates from 0 to 1000. Return only <box>[[x1,y1],[x2,y2]]</box>."
)

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

GENERIC_OBJECT_NAMES = {
    "object",
    "item",
    "thing",
    "target object",
    "unknown object",
    "unidentified object",
}


def log(message: str) -> None:
    print(message, flush=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def user_text(item: dict[str, Any]) -> str:
    for msg in item.get("messages", []):
        if isinstance(msg, dict) and msg.get("role") == "user":
            return str(msg.get("content", ""))
    for msg in item.get("conversations", []):
        if isinstance(msg, dict) and msg.get("from") in {"human", "user"}:
            return str(msg.get("value", msg.get("content", "")))
    return ""


def relation_from_prompt_id(prompt_id: str) -> str:
    for rel in sorted(RELATION_PHRASES, key=len, reverse=True):
        if prompt_id.endswith(f"_{rel}"):
            return rel
    tail = prompt_id.rsplit("_", 1)[-1] if "_" in prompt_id else ""
    return tail if tail in RELATION_PHRASES else "related"


def make_box(points: list[list[int]], margin_ratio: float, min_margin: int) -> list[list[int]]:
    xs = [int(round(float(p[0]))) for p in points if isinstance(p, (list, tuple)) and len(p) >= 2]
    ys = [int(round(float(p[1]))) for p in points if isinstance(p, (list, tuple)) and len(p) >= 2]
    if not xs or not ys:
        raise ValueError("empty point set")
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    margin_x = max(min_margin, round(width * margin_ratio))
    margin_y = max(min_margin, round(height * margin_ratio))
    return [
        [max(0, x1 - margin_x), max(0, y1 - margin_y)],
        [min(1000, x2 + margin_x), min(1000, y2 + margin_y)],
    ]


def box_stats(box: list[list[int]]) -> tuple[int, int, int]:
    width = int(box[1][0]) - int(box[0][0])
    height = int(box[1][1]) - int(box[0][1])
    return width, height, width * height


def image_size(path: str) -> tuple[int, int] | None:
    ffprobe = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=s=x:p=0",
        path,
    ]
    try:
        out = subprocess.check_output(ffprobe, text=True, stderr=subprocess.STDOUT, timeout=20).strip()
        if "x" in out:
            w_str, h_str = out.split("x", 1)
            return int(w_str), int(h_str)
    except Exception:
        pass

    # Some MACA training images live in a minimal runtime that has ffmpeg but
    # not ffprobe. Parse the input stream line from ffmpeg stderr as fallback.
    ffmpeg = ["ffmpeg", "-hide_banner", "-i", path]
    try:
        proc = subprocess.run(ffmpeg, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
    except Exception:
        return None
    text = proc.stderr + "\n" + proc.stdout
    matches = re.findall(r"(?<![0-9])([1-9][0-9]{1,5})x([1-9][0-9]{1,5})(?![0-9])", text)
    for w_str, h_str in matches:
        width, height = int(w_str), int(h_str)
        if width >= 16 and height >= 16:
            return width, height
    return None


def crop_box_to_jpeg(image_path: str, box: list[list[int]], output_path: Path, max_side: int = 512) -> bool:
    size = image_size(image_path)
    if size is None:
        return False
    width, height = size
    x1 = max(0, min(width - 1, round(box[0][0] / 1000 * width)))
    y1 = max(0, min(height - 1, round(box[0][1] / 1000 * height)))
    x2 = max(x1 + 1, min(width, round(box[1][0] / 1000 * width)))
    y2 = max(y1 + 1, min(height, round(box[1][1] / 1000 * height)))
    crop_w = max(1, x2 - x1)
    crop_h = max(1, y2 - y1)
    scale_filter = f"scale='if(gt(iw,ih),min({max_side},iw),-2)':'if(gt(ih,iw),min({max_side},ih),-2)'"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        image_path,
        "-vf",
        f"crop={crop_w}:{crop_h}:{x1}:{y1},{scale_filter}",
        "-frames:v",
        "1",
        "-q:v",
        "3",
        str(output_path),
    ]
    try:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
    except Exception:
        return False
    return output_path.exists() and output_path.stat().st_size > 0


def data_url_from_file(path: Path) -> str:
    data = path.read_bytes()
    suffix = path.suffix.lower()
    mime = "image/jpeg"
    if suffix == ".png":
        mime = "image/png"
    elif suffix == ".webp":
        mime = "image/webp"
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


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
        return {"object_name": text[:80].strip() or "target object"}


def clean_label(label: Any) -> str:
    text = str(label or "").strip().strip("\"'")
    text = re.sub(r"^(a|an|the)\s+", "", text, flags=re.I)
    text = re.sub(r"[^A-Za-z0-9 ,/_-]+", "", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text[:80] or "target object"


def normalize_annotation(obj: dict[str, Any], raw_response: str) -> dict[str, Any]:
    object_name = clean_label(obj.get("object_name") or obj.get("label") or obj.get("name"))
    attributes = obj.get("attributes") or []
    if isinstance(attributes, str):
        attributes = [attributes]
    if not isinstance(attributes, list):
        attributes = []
    attributes = [clean_label(x) for x in attributes if clean_label(x)]
    region_type = str(obj.get("region_type") or obj.get("type") or "unclear").strip().lower()
    if region_type not in {"object", "surface", "free_space", "container", "unclear"}:
        region_type = "unclear"
    confidence = str(obj.get("confidence") or "medium").strip().lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "medium"
    return {
        "object_name": object_name,
        "attributes": attributes[:5],
        "region_type": region_type,
        "confidence": confidence,
        "raw_response": raw_response,
    }


def label_crop(
    *,
    api_url: str,
    api_key: str,
    model: str,
    data_url: str,
    old_user: str,
    temperature: float,
    max_tokens: int,
    retries: int,
    retry_sleep: float,
) -> dict[str, Any]:
    prompt = (
        "You are labeling a cropped target region for semantic navigation "
        "box-grounding data.\n"
        "The crop comes from a RoboPoint sample whose original instruction was:\n"
        f"{old_user}\n\n"
        "Return strict JSON only:\n"
        "{\n"
        '  "object_name": "short English noun phrase for the main target in the crop, 1-5 words",\n'
        '  "attributes": ["optional visual attributes"],\n'
        '  "region_type": "object|surface|free_space|container|unclear",\n'
        '  "confidence": "high|medium|low"\n'
        "}\n"
        "If the crop is mostly empty/free space, use a navigational region label "
        'such as "empty space", "countertop area", or "container interior".'
    )
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = json.dumps(payload).encode("utf-8")

    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(api_url, data=body, headers=headers)
            with urllib.request.urlopen(req, timeout=90) as response:
                resp = json.loads(response.read().decode("utf-8"))
            raw = resp["choices"][0]["message"]["content"]
            return normalize_annotation(parse_jsonish(raw), raw)
        except Exception as exc:  # noqa: BLE001 - keep robust for long batch jobs
            last_error = repr(exc)
            if attempt < retries:
                time.sleep(retry_sleep * attempt)
    return {
        "object_name": "target object",
        "attributes": [],
        "region_type": "unclear",
        "confidence": "low",
        "raw_response": last_error,
        "error": last_error,
    }


def label_full_image_box(
    *,
    api_url: str,
    api_key: str,
    model: str,
    data_url: str,
    old_user: str,
    box: list[list[int]],
    relation: str,
    temperature: float,
    max_tokens: int,
    retries: int,
    retry_sleep: float,
) -> dict[str, Any]:
    prompt = (
        "You are labeling a target region for semantic navigation box-grounding data.\n"
        "The image is the original full image. Focus only on the normalized target box "
        f"{box}, where coordinates range from 0 to 1000 and the two points are "
        "[[left, top], [right, bottom]].\n"
        f"The relation label is: {relation_phrase(relation)}.\n"
        "The original RoboPoint instruction was:\n"
        f"{old_user}\n\n"
        "Return strict JSON only:\n"
        "{\n"
        '  "object_name": "short English noun phrase for the main target inside the box, 1-5 words",\n'
        '  "attributes": ["optional visual attributes"],\n'
        '  "region_type": "object|surface|free_space|container|unclear",\n'
        '  "confidence": "high|medium|low"\n'
        "}\n"
        "If the box is mostly empty/free space, use a navigational region label "
        'such as "empty space", "countertop area", or "container interior".'
    )
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = json.dumps(payload).encode("utf-8")

    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(api_url, data=body, headers=headers)
            with urllib.request.urlopen(req, timeout=90) as response:
                resp = json.loads(response.read().decode("utf-8"))
            raw = resp["choices"][0]["message"]["content"]
            return normalize_annotation(parse_jsonish(raw), raw)
        except Exception as exc:  # noqa: BLE001
            last_error = repr(exc)
            if attempt < retries:
                time.sleep(retry_sleep * attempt)
    return {
        "object_name": "target object",
        "attributes": [],
        "region_type": "unclear",
        "confidence": "low",
        "raw_response": last_error,
        "error": last_error,
    }


def relation_phrase(relation: str) -> str:
    return RELATION_PHRASES.get(relation, "matching the highlighted spatial relation")


def anchor_object(relation: str) -> str:
    return ANCHOR_BY_RELATION.get(relation, "highlighted object")


def box_answer(box: list[list[int]]) -> str:
    return "<box>" + json.dumps(box, separators=(",", ":")) + "</box>"


def variant_rows(base: dict[str, Any], annotation: dict[str, Any], *, source_image: str) -> list[dict[str, Any]]:
    object_name = annotation["object_name"]
    relation = base["relation"]
    rel_phrase = relation_phrase(relation)
    answer = box_answer(base["box"])
    anchor = anchor_object(relation)
    prompts = [
        (
            "obj_only",
            f"<image>\nFind the {object_name}. Return only <box>[[x1,y1],[x2,y2]]</box> "
            "with integer coordinates from 0 to 1000.",
            {"object_name": object_name, "relation": None, "anchor_object": None},
        ),
        (
            "relation_only",
            f"<image>\nFind the target region {rel_phrase}. Return only <box>[[x1,y1],[x2,y2]]</box> "
            "with integer coordinates from 0 to 1000.",
            {"object_name": None, "relation": rel_phrase, "anchor_object": anchor},
        ),
        (
            "obj_relation",
            "<image>\nTarget object information:\n"
            + json.dumps(
                {"object_name": object_name, "relation": rel_phrase, "anchor_object": anchor},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\nReturn only the bounding box as <box>[[x1,y1],[x2,y2]]</box>.",
            {"object_name": object_name, "relation": rel_phrase, "anchor_object": anchor},
        ),
    ]
    rows = []
    for mode, prompt, target in prompts:
        target = {**target, "box": base["box"], "attributes": annotation.get("attributes", [])}
        rows.append(
            {
                "dataset": "semantic_nav_box_grounding_v1",
                "image": [source_image],
                "video": [],
                "target": target,
                "conversations": [
                    {"from": "system", "value": SYSTEM_PROMPT},
                    {"from": "human", "value": prompt},
                    {"from": "gpt", "value": answer},
                ],
                "metadata": {
                    "source": "robopoint_box_grounding_proxy",
                    "task_type": "box_grounding",
                    "prompt_mode": mode,
                    "label_source": "qwen_crop_annotation",
                    "label_image_mode": annotation.get("label_image_mode"),
                    "region_type": annotation.get("region_type"),
                    "confidence": annotation.get("confidence"),
                    "conversion": "points_bbox_margin_v1",
                    "base_source": "robopoint",
                    "original_prompt_id": base["prompt_id"],
                    "relation": relation,
                    "point_count": len(base["points"]),
                },
            }
        )
    return rows


def build_candidates(args: argparse.Namespace) -> tuple[list[dict[str, Any]], Counter]:
    rng = random.Random(args.seed)
    raw_rows = read_jsonl(args.input)
    candidates: list[dict[str, Any]] = []
    dropped: Counter = Counter()
    allowed_relations = set(args.relations.split(",")) if args.relations else set(RELATION_PHRASES)
    allowed_prefixes = set(args.prefixes.split(",")) if args.prefixes else {"object_ref"}

    for item in raw_rows:
        meta = item.get("metadata") or {}
        if meta.get("source") != "robopoint":
            dropped["not_robopoint"] += 1
            continue
        prompt_id = str(item.get("prompt_id") or "")
        prefix = prompt_id.split("/", 1)[0] if "/" in prompt_id else ""
        if prefix not in allowed_prefixes:
            dropped["prefix"] += 1
            continue
        relation = relation_from_prompt_id(prompt_id)
        if relation not in allowed_relations:
            dropped["relation"] += 1
            continue
        points = item.get("gt_points") or []
        if len(points) < args.min_points:
            dropped["point_count"] += 1
            continue
        images = item.get("images") or item.get("image") or []
        if isinstance(images, str):
            images = [images]
        image_path = str(images[0]) if images else ""
        if not image_path or not Path(image_path).exists():
            dropped["missing_image"] += 1
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
        candidates.append(
            {
                "prompt_id": prompt_id,
                "image": image_path,
                "old_user": user_text(item),
                "points": points,
                "box": box,
                "relation": relation,
                "prefix": prefix,
            }
        )

    rng.shuffle(candidates)
    return candidates[: args.max_base_samples], dropped


def should_keep_annotation(annotation: dict[str, Any], args: argparse.Namespace) -> tuple[bool, str]:
    if annotation.get("confidence") == "low":
        return False, "low_confidence"
    if annotation.get("region_type") == "unclear":
        return False, "unclear_region"
    object_name = annotation.get("object_name", "")
    if object_name in GENERIC_OBJECT_NAMES:
        return False, "generic_object_name"
    if len(object_name) < 3:
        return False, "bad_object_name"
    if args.object_only and annotation.get("region_type") != "object":
        return False, "not_object"
    return True, "ok"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("/data/msz/opd_project/data/prompt_pool_clean.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("/data/msz/opd_project/data/semantic_nav_box_v1"))
    parser.add_argument("--name", default="semantic_nav_box_grounding_pilot_1k")
    parser.add_argument("--max-base-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--api-key", default=os.environ.get("QWEN_API_KEY", ""))
    parser.add_argument("--model", default="qwen-122b")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=1.5)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--label-image-mode", choices=["auto", "crop", "full"], default="auto")
    parser.add_argument("--margin-ratio", type=float, default=0.15)
    parser.add_argument("--min-margin", type=int, default=10)
    parser.add_argument("--min-points", type=int, default=4)
    parser.add_argument("--min-box-side", type=int, default=24)
    parser.add_argument("--min-box-area", type=int, default=1200)
    parser.add_argument("--max-box-area", type=int, default=400000)
    parser.add_argument("--relations", default="on,left,right,inside,beside,front,behind,between")
    parser.add_argument("--prefixes", default="object_ref")
    parser.add_argument("--object-only", action="store_true")
    parser.add_argument("--keep-rejected", action="store_true")
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected, dropped_candidates = build_candidates(args)
    log(f"[candidates] selected={len(selected)} dropped={dict(dropped_candidates)}")
    if not selected:
        raise RuntimeError("no candidates selected")
    if args.manifest_only:
        manifest_jsonl = args.output_dir / f"{args.name}_manifest.jsonl"
        image_list = args.output_dir / f"{args.name}_images.txt"
        write_jsonl(manifest_jsonl, selected)
        image_list.write_text("\n".join(row["image"] for row in selected) + "\n", encoding="utf-8")
        summary = {
            "input": str(args.input),
            "manifest_jsonl": str(manifest_jsonl),
            "image_list": str(image_list),
            "base_requested": args.max_base_samples,
            "base_selected": len(selected),
            "candidate_dropped": dict(dropped_candidates),
            "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k != "api_key"},
        }
        write_json(args.output_dir / f"{args.name}_manifest_summary.json", summary)
        log("[manifest] " + json.dumps(summary, ensure_ascii=False, separators=(",", ":")))
        return

    annotations: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    kept_counter: Counter = Counter()
    reject_counter: Counter = Counter()
    relation_counter: Counter = Counter()
    region_counter: Counter = Counter()

    with tempfile.TemporaryDirectory(prefix="semantic_nav_box_") as tmp_dir:
        tmp = Path(tmp_dir)
        for index, base in enumerate(selected, start=1):
            crop_path = tmp / f"crop_{index:05d}.jpg"
            used_image_mode = args.label_image_mode
            crop_ok = False
            if args.label_image_mode in {"auto", "crop"}:
                crop_ok = crop_box_to_jpeg(base["image"], base["box"], crop_path)
            if crop_ok:
                annotation = label_crop(
                    api_url=args.api_url,
                    api_key=args.api_key,
                    model=args.model,
                    data_url=data_url_from_file(crop_path),
                    old_user=base["old_user"],
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    retries=args.retries,
                    retry_sleep=args.retry_sleep,
                )
                used_image_mode = "crop"
            elif args.label_image_mode == "crop":
                reject_counter["crop_failed"] += 1
                rejected_rows.append({"reason": "crop_failed", **base})
                continue
            else:
                annotation = label_full_image_box(
                    api_url=args.api_url,
                    api_key=args.api_key,
                    model=args.model,
                    data_url=data_url_from_file(Path(base["image"])),
                    old_user=base["old_user"],
                    box=base["box"],
                    relation=base["relation"],
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    retries=args.retries,
                    retry_sleep=args.retry_sleep,
                )
                used_image_mode = "full"
            annotation["label_image_mode"] = used_image_mode
            keep, reason = should_keep_annotation(annotation, args)
            record = {
                "idx": index,
                **base,
                "annotation": annotation,
                "keep": keep,
                "reject_reason": None if keep else reason,
            }
            annotations.append(record)
            if keep:
                rows = variant_rows(base, annotation, source_image=base["image"])
                final_rows.extend(rows)
                summary_rows.append(record)
                kept_counter["base"] += 1
                relation_counter[base["relation"]] += 1
                region_counter[annotation.get("region_type", "?")] += 1
            else:
                reject_counter[reason] += 1
                if args.keep_rejected:
                    rows = variant_rows(base, annotation, source_image=base["image"])
                    for row in rows:
                        row["metadata"]["quality"] = "rejected"
                    final_rows.extend(rows)
                rejected_rows.append(record)

            if index % 25 == 0 or index == len(selected):
                log(
                    f"[label] {index}/{len(selected)} kept_base={kept_counter['base']} "
                    f"rejected={sum(reject_counter.values())} last={annotation.get('object_name')} "
                    f"{annotation.get('region_type')}/{annotation.get('confidence')}"
                )
            if args.sleep > 0:
                time.sleep(args.sleep)

    output_jsonl = args.output_dir / f"{args.name}.jsonl"
    annotations_jsonl = args.output_dir / f"{args.name}_annotations.jsonl"
    rejected_jsonl = args.output_dir / f"{args.name}_rejected.jsonl"
    summary_json = args.output_dir / f"{args.name}_summary.json"
    write_jsonl(output_jsonl, final_rows)
    write_jsonl(annotations_jsonl, annotations)
    write_jsonl(rejected_jsonl, rejected_rows)
    summary = {
        "input": str(args.input),
        "output_jsonl": str(output_jsonl),
        "base_requested": args.max_base_samples,
        "base_selected": len(selected),
        "base_kept": kept_counter["base"],
        "train_rows": len(final_rows),
        "variants_per_kept_base": 3,
        "candidate_dropped": dict(dropped_candidates),
        "rejected": dict(reject_counter),
        "relation_counts": dict(relation_counter),
        "region_type_counts": dict(region_counter),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k != "api_key"},
    }
    write_json(summary_json, summary)
    log("[done] " + json.dumps(summary, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
