#!/usr/bin/env python3
"""Label semantic-navigation box data without storing source images locally.

For each manifest row, the script reads the remote image through SSH as base64,
decodes it in memory, crops the synthesized target box with PIL, sends only the
small crop as a data URL to the VLM labeler, and writes JSONL checkpoints.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import random
import re
import shlex
import subprocess
import threading
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


SYSTEM_PROMPT = (
    "You are a semantic navigation grounding assistant. Given an image and "
    "target object information, return the target object's bounding box in "
    "coordinates from 0 to 1000. Return only <box>[[x1,y1],[x2,y2]]</box>."
)

REGION_SYSTEM_PROMPT = (
    "You are a semantic navigation region-grounding assistant. Given an image "
    "and target navigation region information, return the target region's "
    "bounding box in coordinates from 0 to 1000. Return only "
    "<box>[[x1,y1],[x2,y2]]</box>."
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
    "target region",
    "region",
    "area",
    "space",
    "target area",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def append_jsonl(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    line = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
    with lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(line)


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def relation_phrase(relation: str) -> str:
    return RELATION_PHRASES.get(relation, "matching the highlighted spatial relation")


def anchor_object(relation: str) -> str:
    return ANCHOR_BY_RELATION.get(relation, "highlighted object")


def remote_image_bytes(ssh_host: str, remote_path: str, timeout: int) -> bytes:
    cmd = ["ssh", ssh_host, f"base64 -w 0 {shlex.quote(remote_path)}"]
    b64 = subprocess.check_output(cmd, text=True, timeout=timeout)
    return base64.b64decode(b64)


def image_url_from_remote_path(args: argparse.Namespace, remote_path: str) -> str:
    rel = remote_path
    if args.remote_image_prefix and rel.startswith(args.remote_image_prefix):
        rel = rel[len(args.remote_image_prefix) :]
    rel = rel.lstrip("/")
    return args.image_base_url.rstrip("/") + "/" + rel.replace("\\", "/")


def image_bytes(args: argparse.Namespace, remote_path: str) -> bytes:
    if args.image_base_url:
        url = image_url_from_remote_path(args, remote_path)
        with urllib.request.urlopen(url, timeout=args.image_fetch_timeout) as response:
            return response.read()
    return remote_image_bytes(args.ssh_host, remote_path, args.ssh_timeout)


def crop_data_url_from_remote(args: argparse.Namespace, row: dict[str, Any]) -> tuple[str, tuple[int, int], float]:
    start = time.time()
    data = image_bytes(args, row["image"])
    image = Image.open(io.BytesIO(data)).convert("RGB")
    width, height = image.size
    box = row["box"]
    x1 = max(0, min(width - 1, round(box[0][0] / 1000 * width)))
    y1 = max(0, min(height - 1, round(box[0][1] / 1000 * height)))
    x2 = max(x1 + 1, min(width, round(box[1][0] / 1000 * width)))
    y2 = max(y1 + 1, min(height, round(box[1][1] / 1000 * height)))
    crop = image.crop((x1, y1, x2, y2))
    crop.thumbnail((args.crop_max_side, args.crop_max_side), Image.Resampling.LANCZOS)
    out = io.BytesIO()
    crop.save(out, format="JPEG", quality=args.jpeg_quality)
    crop_b64 = base64.b64encode(out.getvalue()).decode("ascii")
    return "data:image/jpeg;base64," + crop_b64, crop.size, time.time() - start


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


def clean_label(value: Any) -> str:
    text = str(value or "").strip().strip("\"'")
    text = re.sub(r"^(a|an|the)\s+", "", text, flags=re.I)
    text = re.sub(r"[^A-Za-z0-9 ,/_-]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()[:80] or "target object"


def normalize_annotation(obj: dict[str, Any], raw: str) -> dict[str, Any]:
    object_name = clean_label(obj.get("object_name") or obj.get("label") or obj.get("name"))
    attrs = obj.get("attributes") or []
    if isinstance(attrs, str):
        attrs = [attrs]
    if not isinstance(attrs, list):
        attrs = []
    attrs = [clean_label(x) for x in attrs if clean_label(x)][:5]
    region_type = str(obj.get("region_type") or obj.get("type") or "unclear").strip().lower()
    if region_type not in {"object", "surface", "free_space", "container", "unclear"}:
        region_type = "unclear"
    confidence = str(obj.get("confidence") or "medium").strip().lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "medium"
    return {
        "object_name": object_name,
        "attributes": attrs,
        "region_type": region_type,
        "confidence": confidence,
        "raw_response": raw,
    }


def quality(annotation: dict[str, Any]) -> str:
    if annotation.get("confidence") == "low":
        return "weak"
    if annotation.get("region_type") == "unclear":
        return "weak"
    if annotation.get("object_name") in GENERIC_OBJECT_NAMES:
        return "weak"
    return "high_quality"


def label_crop(args: argparse.Namespace, row: dict[str, Any]) -> dict[str, Any]:
    data_url, crop_size, fetch_crop_sec = crop_data_url_from_remote(args, row)
    if args.task_kind == "region":
        prompt = (
            "You are labeling a cropped target navigation region for semantic "
            "navigation box-grounding data.\n"
            "The crop is the target box synthesized from RoboPoint region points; "
            "it may be an empty navigable area, a surface area, or a container "
            "interior rather than a physical object.\n"
            f"The relation label is: {relation_phrase(row['relation'])}.\n"
            "The original RoboPoint instruction was:\n"
            f"{row['old_user']}\n\n"
            "Return strict JSON only:\n"
            "{\n"
            '  "object_name": "short English noun phrase for the target navigation region, 1-6 words",\n'
            '  "attributes": ["optional visual attributes"],\n'
            '  "region_type": "surface|free_space|container|object|unclear",\n'
            '  "confidence": "high|medium|low"\n'
            "}\n"
            "Use concrete region names such as empty floor space, countertop area, "
            "cabinet interior, surface area, or space beside the object. Avoid "
            "generic labels like region, area, object, or item."
        )
    else:
        prompt = (
            "You are labeling a cropped target region for semantic navigation "
            "box-grounding data.\n"
            "The crop is the target box synthesized from RoboPoint points.\n"
            f"The relation label is: {relation_phrase(row['relation'])}.\n"
            "The original RoboPoint instruction was:\n"
            f"{row['old_user']}\n\n"
            "Return strict JSON only:\n"
            "{\n"
            '  "object_name": "short English noun phrase for the main target in the crop, 1-5 words",\n'
            '  "attributes": ["optional visual attributes"],\n'
            '  "region_type": "object|surface|free_space|container|unclear",\n'
            '  "confidence": "high|medium|low"\n'
            "}\n"
            "Avoid generic labels like object or item. If the crop is mostly empty/free "
            "space, use a navigational region label such as empty space, countertop area, "
            "or container interior."
        )
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
        api_start = time.time()
        try:
            req = urllib.request.Request(args.api_url, data=body, headers=headers)
            with urllib.request.urlopen(req, timeout=args.request_timeout) as response:
                resp = json.loads(response.read().decode("utf-8"))
            raw = resp["choices"][0]["message"]["content"]
            ann = normalize_annotation(parse_jsonish(raw), raw)
            ann["crop_size"] = list(crop_size)
            ann["fetch_crop_sec"] = round(fetch_crop_sec, 3)
            ann["api_sec"] = round(time.time() - api_start, 3)
            return ann
        except Exception as exc:  # noqa: BLE001
            last_error = repr(exc)
            if attempt < args.retries:
                time.sleep(args.retry_sleep * attempt)
    return {
        "object_name": "target object",
        "attributes": [],
        "region_type": "unclear",
        "confidence": "low",
        "raw_response": last_error,
        "error": last_error,
        "crop_size": list(crop_size),
        "fetch_crop_sec": round(fetch_crop_sec, 3),
    }


def box_answer(box: list[list[int]]) -> str:
    return "<box>" + json.dumps(box, separators=(",", ":")) + "</box>"


def variant_rows(row: dict[str, Any], annotation: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    obj = annotation["object_name"]
    relation = row["relation"]
    rel = relation_phrase(relation)
    anchor = anchor_object(relation)
    answer = box_answer(row["box"])
    if args.task_kind == "region":
        region_type = annotation.get("region_type")
        prompts = [
            (
                "region_only",
                f"<image>\nFind the {obj}. Return only <box>[[x1,y1],[x2,y2]]</box> "
                "with integer coordinates from 0 to 1000.",
                {"region_name": obj, "region_type": region_type, "relation": None, "anchor_object": None},
            ),
            (
                "relation_only",
                f"<image>\nFind the navigable target region {rel}. Return only "
                "<box>[[x1,y1],[x2,y2]]</box> with integer coordinates from 0 to 1000.",
                {"region_name": None, "region_type": region_type, "relation": rel, "anchor_object": anchor},
            ),
            (
                "region_relation",
                "<image>\nNavigation target information:\n"
                + json.dumps(
                    {
                        "region_name": obj,
                        "region_type": region_type,
                        "relation": rel,
                        "anchor_object": anchor,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\nReturn only the bounding box as <box>[[x1,y1],[x2,y2]]</box>.",
                {"region_name": obj, "region_type": region_type, "relation": rel, "anchor_object": anchor},
            ),
        ]
        system_prompt = REGION_SYSTEM_PROMPT
        source = "robopoint_region_box_grounding_proxy"
        task_type = "region_box_grounding"
    else:
        prompts = [
            (
                "obj_only",
                f"<image>\nFind the {obj}. Return only <box>[[x1,y1],[x2,y2]]</box> "
                "with integer coordinates from 0 to 1000.",
                {"object_name": obj, "relation": None, "anchor_object": None},
            ),
            (
                "relation_only",
                f"<image>\nFind the target region {rel}. Return only <box>[[x1,y1],[x2,y2]]</box> "
                "with integer coordinates from 0 to 1000.",
                {"object_name": None, "relation": rel, "anchor_object": anchor},
            ),
            (
                "obj_relation",
                "<image>\nTarget object information:\n"
                + json.dumps(
                    {"object_name": obj, "relation": rel, "anchor_object": anchor},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\nReturn only the bounding box as <box>[[x1,y1],[x2,y2]]</box>.",
                {"object_name": obj, "relation": rel, "anchor_object": anchor},
            ),
        ]
        system_prompt = SYSTEM_PROMPT
        source = "robopoint_box_grounding_proxy"
        task_type = "box_grounding"
    rows = []
    q = quality(annotation)
    for mode, prompt, target in prompts:
        rows.append(
            {
                "dataset": args.output_name,
                "image": [row["image"]],
                "video": [],
                "target": {**target, "attributes": annotation.get("attributes", []), "box": row["box"]},
                "conversations": [
                    {"from": "system", "value": system_prompt},
                    {"from": "human", "value": prompt},
                    {"from": "gpt", "value": answer},
                ],
                "metadata": {
                    "source": source,
                    "task_type": task_type,
                    "prompt_mode": mode,
                    "quality": q,
                    "label_source": "qwen-122b_remote_crop_base64",
                    "region_type": annotation.get("region_type"),
                    "confidence": annotation.get("confidence"),
                    "conversion": "points_bbox_margin_v1",
                    "original_prompt_id": row["prompt_id"],
                    "relation": relation,
                    "point_count": len(row["points"]),
                },
            }
        )
    return rows


def process_one(args: argparse.Namespace, row: dict[str, Any]) -> dict[str, Any]:
    try:
        annotation = label_crop(args, row)
    except Exception as exc:  # noqa: BLE001
        annotation = {
            "object_name": "target object",
            "attributes": [],
            "region_type": "unclear",
            "confidence": "low",
            "raw_response": repr(exc),
            "error": repr(exc),
        }
    return {
        "idx": row.get("idx"),
        **row,
        "annotation": annotation,
        "quality": quality(annotation),
    }


def render_preview(path: Path, annotations: list[dict[str, Any]], args: argparse.Namespace) -> None:
    rows = annotations[:]
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.preview_size]
    try:
        title_font = ImageFont.truetype("arial.ttf", 16)
        small_font = ImageFont.truetype("arial.ttf", 12)
    except Exception:
        title_font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    cards: list[Image.Image] = []
    for row in rows:
        try:
            data = image_bytes(args, row["image"])
            image = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            continue
        width, height = image.size
        image.thumbnail((360, 260), Image.Resampling.LANCZOS)
        dw, dh = image.size
        sx, sy = dw / width, dh / height
        box = row["box"]
        x1 = round(box[0][0] / 1000 * width * sx)
        y1 = round(box[0][1] / 1000 * height * sy)
        x2 = round(box[1][0] / 1000 * width * sx)
        y2 = round(box[1][1] / 1000 * height * sy)
        draw = ImageDraw.Draw(image, "RGBA")
        draw.rectangle([x1, y1, x2, y2], fill=(0, 210, 255, 45), outline=(0, 210, 255, 255), width=3)
        for px, py in row.get("points", [])[:30]:
            x = round(px / 1000 * width * sx)
            y = round(py / 1000 * height * sy)
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(255, 40, 40, 230))
        card = Image.new("RGB", (380, 380), (248, 250, 252))
        card.paste(image, ((380 - dw) // 2, 8))
        d = ImageDraw.Draw(card)
        y = dh + 18
        ann = row["annotation"]
        d.text((10, y), f"{ann['object_name']} | {row['relation']} | {row['quality']}", fill=(15, 23, 42), font=title_font)
        y += 22
        d.text((10, y), f"{ann['region_type']} / {ann['confidence']}  {box_answer(box)}", fill=(2, 132, 199), font=small_font)
        cards.append(card)
    if not cards:
        return
    cols = 4
    rows_n = (len(cards) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 380, rows_n * 380), (226, 232, 240))
    for i, card in enumerate(cards):
        sheet.paste(card, ((i % cols) * 380, (i // cols) * 380))
    sheet.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ssh-host", required=True)
    parser.add_argument("--ssh-timeout", type=int, default=90)
    parser.add_argument("--image-base-url", default="")
    parser.add_argument("--remote-image-prefix", default="/data/msz/dataset/RoboPoint/images")
    parser.add_argument("--image-fetch-timeout", type=int, default=60)
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", default="qwen-122b")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--request-timeout", type=int, default=90)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=1.5)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--crop-max-side", type=int, default=512)
    parser.add_argument("--jpeg-quality", type=int, default=88)
    parser.add_argument("--preview-size", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument("--task-kind", choices=["object", "region"], default="object")
    parser.add_argument("--output-name", default="semantic_nav_box_grounding_full_object_ref_v1")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_jsonl(args.manifest)
    for idx, row in enumerate(manifest, start=1):
        row["idx"] = idx
    annotations_path = args.output_dir / f"{args.output_name}_base_annotations.jsonl"
    done: dict[str, dict[str, Any]] = {}
    if annotations_path.exists():
        for row in read_jsonl(annotations_path):
            done[row["prompt_id"]] = row
    pending = [row for row in manifest if row["prompt_id"] not in done]
    print(f"[start] manifest={len(manifest)} done={len(done)} pending={len(pending)} workers={args.workers}", flush=True)

    lock = threading.Lock()
    completed = 0
    if pending:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_one, args, row): row for row in pending}
            for future in as_completed(futures):
                record = future.result()
                append_jsonl(annotations_path, record, lock)
                done[record["prompt_id"]] = record
                completed += 1
                if completed % 100 == 0 or completed == len(pending):
                    ann = record["annotation"]
                    print(
                        f"[label] {completed}/{len(pending)} total={len(done)}/{len(manifest)} "
                        f"last={ann.get('object_name')} {ann.get('region_type')}/{ann.get('confidence')} "
                        f"{record['quality']} fetch={ann.get('fetch_crop_sec')} api={ann.get('api_sec')}",
                        flush=True,
                    )

    annotations = [done[row["prompt_id"]] for row in manifest if row["prompt_id"] in done]
    all_rows: list[dict[str, Any]] = []
    hq_rows: list[dict[str, Any]] = []
    for record in annotations:
        rows = variant_rows(record, record["annotation"], args)
        all_rows.extend(rows)
        if record["quality"] == "high_quality":
            hq_rows.extend(rows)

    all_jsonl = args.output_dir / f"{args.output_name}_all.jsonl"
    hq_jsonl = args.output_dir / f"{args.output_name}_high_quality.jsonl"
    summary_json = args.output_dir / f"{args.output_name}_summary.json"
    preview_png = args.output_dir / f"{args.output_name}_preview.png"
    write_jsonl(all_jsonl, all_rows)
    write_jsonl(hq_jsonl, hq_rows)

    counters = {
        "quality": Counter(row["quality"] for row in annotations),
        "region_type": Counter(row["annotation"].get("region_type") for row in annotations),
        "confidence": Counter(row["annotation"].get("confidence") for row in annotations),
        "relation": Counter(row.get("relation") for row in annotations),
        "object_name_top100": Counter(row["annotation"].get("object_name") for row in annotations).most_common(100),
    }
    summary = {
        "base_manifest": len(manifest),
        "base_annotated": len(annotations),
        "base_high_quality": counters["quality"].get("high_quality", 0),
        "base_weak": counters["quality"].get("weak", 0),
        "train_rows_all": len(all_rows),
        "train_rows_high_quality": len(hq_rows),
        "outputs": {
            "all_jsonl": str(all_jsonl),
            "high_quality_jsonl": str(hq_jsonl),
            "base_annotations_jsonl": str(annotations_path),
            "preview_png": str(preview_png),
        },
        "counters": {k: dict(v) if isinstance(v, Counter) else v for k, v in counters.items()},
    }
    write_json(summary_json, summary)
    render_preview(preview_png, annotations, args)
    print("[done] " + json.dumps(summary, ensure_ascii=False, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
