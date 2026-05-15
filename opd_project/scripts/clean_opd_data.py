#!/usr/bin/env python3
"""Clean OPD prompt-pool JSONL files into a strict, reusable schema.

This follows the local AceBrain data-line habit: normalize schema first,
filter incomplete media, keep deterministic stats, and write clean artifacts
instead of silently mutating raw inputs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


POINTING_SYSTEM_PROMPT = (
    "You are a helpful vision-language assistant. When the user asks for a "
    "location, answer with coordinates in the range 0 to 1000. Your answer "
    "must be formatted as \"<point>[[x1,y1],[x2,y2],...]</point>\". "
    "Return only the point tag."
)

POINTING_USER_SUFFIX = (
    "Return only <point>[[x,y],...]</point> with integer coordinates from 0 to 1000."
)

OLD_FORMAT_PATTERNS = [
    re.compile(
        r"\s*Your answer should be formatted as a list of tuples,?\s*i\.e\.\s*"
        r"\[\(x1,\s*y1\),\s*\(x2,\s*y2\),\s*\.\.\.\],?\s*where each tuple "
        r"contains the x and y coordinates of a point satisfying the conditions above\.",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\s*The coordinates should be between 0 and 1, indicating the normalized "
        r"pixel locations of the points in the image\.",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\s*Your answer should be formatted as .*?coordinates should be between "
        r"0 and 1.*?(?:\.|$)",
        flags=re.IGNORECASE,
    ),
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_no}: expected object, got {type(item).__name__}")
            rows.append(item)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def normalize_images(item: dict[str, Any]) -> list[str]:
    images = item.get("images", item.get("image", []))
    if isinstance(images, str):
        images = [images]
    if not isinstance(images, list):
        return []
    return [str(p) for p in images if isinstance(p, str) and p.strip()]


def normalize_messages(item: dict[str, Any]) -> list[dict[str, str]]:
    raw_messages = item.get("messages", item.get("conversations", []))
    if not isinstance(raw_messages, list):
        return []
    messages: list[dict[str, str]] = []
    for msg in raw_messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", msg.get("from"))
        content = msg.get("content", msg.get("value", ""))
        if role == "human":
            role = "user"
        elif role == "gpt":
            role = "assistant"
        if role not in {"system", "user", "assistant"}:
            continue
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(str(part.get("text", "")))
                elif isinstance(part, str):
                    parts.append(part)
            content = "\n".join(parts)
        messages.append({"role": str(role), "content": str(content)})
    return messages


def clean_user_text(text: str, *, has_image: bool) -> str:
    text = text.strip()
    text = text.replace("<image>\n<image>", "<image>")
    for pattern in OLD_FORMAT_PATTERNS:
        text = pattern.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("<image> ", "<image>\n")
    if has_image and "<image>" not in text:
        text = "<image>\n" + text
    if "0 and 1" in text or "list of tuples" in text:
        text = re.sub(r"(?i)coordinates? should be between 0 and 1[^.]*\.", "", text)
        text = re.sub(r"(?i)formatted as a list of tuples[^.]*\.", "", text)
        text = re.sub(r"\s+", " ", text).strip()
    if POINTING_USER_SUFFIX not in text:
        text = f"{text.rstrip()} {POINTING_USER_SUFFIX}".strip()
    return text


def normalize_points(points: Any) -> list[list[int]]:
    if not isinstance(points, list):
        return []
    clean: list[list[int]] = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            continue
        try:
            x = int(round(float(point[0])))
            y = int(round(float(point[1])))
        except (TypeError, ValueError):
            continue
        x = min(1000, max(0, x))
        y = min(1000, max(0, y))
        clean.append([x, y])
    return clean


def is_pointing(item: dict[str, Any]) -> bool:
    meta = item.get("metadata", {})
    if not isinstance(meta, dict):
        return False
    task_type = str(meta.get("task_type", "")).lower()
    source = str(meta.get("source", "")).lower()
    return "point" in task_type or source in {
        "robopoint",
        "sharerobot_affordance",
        "pixmo_points",
        "grasp_anything",
        "pacolvis",
        "refspatial_sim",
        "refspatial_3d",
    }


def has_missing_media(images: list[str]) -> bool:
    for image in images:
        if image.startswith("/") and not os.path.exists(image):
            return True
    return False


def clean_item(item: dict[str, Any], *, require_existing_media: bool) -> tuple[dict[str, Any] | None, str | None]:
    images = normalize_images(item)
    if not images:
        return None, "no_image"
    if require_existing_media and has_missing_media(images):
        return None, "missing_image"

    messages = normalize_messages(item)
    if not messages:
        return None, "no_messages"

    meta = item.get("metadata", {})
    if not isinstance(meta, dict):
        meta = {}
    pointing = is_pointing(item)

    user_text = next((m["content"] for m in messages if m["role"] == "user"), "")
    if not user_text:
        return None, "no_user"

    if pointing:
        cleaned_messages = [
            {"role": "system", "content": POINTING_SYSTEM_PROMPT},
            {"role": "user", "content": clean_user_text(user_text, has_image=bool(images))},
        ]
    else:
        # Replay/general data should keep its natural answer style and should not be
        # forced into point tags.
        cleaned_messages = [{"role": "user", "content": user_text.strip()}]

    prompt_id = str(item.get("prompt_id", item.get("id", ""))).strip()
    if not prompt_id:
        prompt_id = f"{meta.get('source', 'unknown')}_{abs(hash(json.dumps(item, sort_keys=True))) % 10**12}"

    cleaned: dict[str, Any] = {
        "prompt_id": prompt_id,
        "images": images,
        "messages": cleaned_messages,
        "metadata": {
            **meta,
            "task_type": "pointing" if pointing else meta.get("task_type", "general_qa"),
            "cleaning_version": "opd_v1_20260515",
        },
    }
    if "gt_points" in item:
        cleaned["gt_points"] = normalize_points(item.get("gt_points"))
        if pointing and not cleaned["gt_points"]:
            return None, "bad_gt_points"
    return cleaned, None


def clean_file(input_path: Path, output_path: Path, *, require_existing_media: bool) -> dict[str, Any]:
    rows = read_jsonl(input_path)
    clean_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    dropped = Counter()
    source_before = Counter()
    source_after = Counter()
    conflict_before = 0

    for item in rows:
        meta = item.get("metadata", {})
        source_before[str(meta.get("source", "?")) if isinstance(meta, dict) else "?"] += 1
        text = json.dumps(item.get("messages", item.get("conversations", [])), ensure_ascii=False)
        if "between 0 and 1" in text or "list of tuples" in text:
            conflict_before += 1

        cleaned, reason = clean_item(item, require_existing_media=require_existing_media)
        if cleaned is None:
            dropped[reason or "unknown"] += 1
            continue
        if cleaned["prompt_id"] in seen:
            dropped["duplicate_prompt_id"] += 1
            continue
        seen.add(cleaned["prompt_id"])
        source_after[str(cleaned.get("metadata", {}).get("source", "?"))] += 1
        clean_rows.append(cleaned)

    write_jsonl(output_path, clean_rows)
    return {
        "input": str(input_path),
        "output": str(output_path),
        "input_rows": len(rows),
        "output_rows": len(clean_rows),
        "dropped": dict(dropped),
        "source_before": dict(source_before),
        "source_after": dict(source_after),
        "old_format_conflicts_before": conflict_before,
        "require_existing_media": require_existing_media,
    }


def write_report(report_path: Path, stats: list[dict[str, Any]]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# OPD Data Cleaning Report",
        "",
        "Cleaning policy:",
        "- normalize all pointing prompts to strict `<point>[[x,y],...]</point>` output;",
        "- use integer coordinates in `[0, 1000]`;",
        "- remove old `(x, y)` / `0..1` tuple instructions;",
        "- filter samples whose local absolute image files are missing;",
        "- keep general replay prompts out of pointing format forcing.",
        "",
    ]
    for item in stats:
        lines.extend(
            [
                f"## {Path(item['output']).name}",
                "",
                f"- input rows: `{item['input_rows']}`",
                f"- output rows: `{item['output_rows']}`",
                f"- old-format conflicts before cleaning: `{item['old_format_conflicts_before']}`",
                f"- dropped: `{json.dumps(item['dropped'], ensure_ascii=False)}`",
                f"- source before: `{json.dumps(item['source_before'], ensure_ascii=False)}`",
                f"- source after: `{json.dumps(item['source_after'], ensure_ascii=False)}`",
                "",
            ]
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-in", default="/data/msz/opd_project/data/prompt_pool.jsonl")
    parser.add_argument("--prompt-out", default="/data/msz/opd_project/data/prompt_pool_clean.jsonl")
    parser.add_argument("--eval-in", default="/data/msz/opd_project/data/eval_robopoint_500.jsonl")
    parser.add_argument("--eval-out", default="/data/msz/opd_project/data/eval_robopoint_500_clean.jsonl")
    parser.add_argument("--stats-out", default="/data/msz/opd_project/data/cleaning_stats.json")
    parser.add_argument("--report-out", default="/data/msz/opd_project/data/cleaning_report.md")
    parser.add_argument("--keep-missing-media", action="store_true")
    args = parser.parse_args()

    require_existing_media = not args.keep_missing_media
    stats = [
        clean_file(Path(args.prompt_in), Path(args.prompt_out), require_existing_media=require_existing_media),
        clean_file(Path(args.eval_in), Path(args.eval_out), require_existing_media=require_existing_media),
    ]
    Path(args.stats_out).write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(Path(args.report_out), stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
