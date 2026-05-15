#!/usr/bin/env python3
"""Build a tiny OPD SFT-style JSONL for DeepSpeed ZeRO-2 smoke tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SYSTEM_FALLBACK = (
    "You are a helpful vision-language assistant. When the user asks for a "
    "location, answer with coordinates in the range 0 to 1000."
)


def target_from_gt(item: dict[str, Any], max_points: int) -> str:
    points = item.get("gt_points") or [[500, 500]]
    out: list[list[int]] = []
    for point in points[:max_points]:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        x = max(0, min(1000, int(round(float(point[0])))))
        y = max(0, min(1000, int(round(float(point[1])))))
        out.append([x, y])
    if not out:
        out = [[500, 500]]
    return "<point>" + json.dumps(out, separators=(",", ":")) + "</point>"


def user_text(item: dict[str, Any]) -> str:
    messages = item.get("messages") or []
    text = next((m.get("content") for m in messages if m.get("role") == "user"), "")
    text = str(text).replace("<image>\n", "").replace("<image>", "").strip()
    if not text:
        text = "Point to the target object."
    return "<image>\n" + text


def system_text(item: dict[str, Any]) -> str:
    messages = item.get("messages") or []
    text = next((m.get("content") for m in messages if m.get("role") == "system"), "")
    return str(text or SYSTEM_FALLBACK)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/data/msz/opd_project/data/prompt_pool_clean.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--max-points", type=int, default=8)
    args = parser.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    stats = {"read": 0, "skipped_schema": 0, "skipped_image": 0, "skipped_gt": 0}
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            if kept >= args.limit:
                break
            stats["read"] += 1
            item = json.loads(line)
            meta = item.get("metadata") or {}
            images = item.get("images") or []
            if meta.get("task_type") != "pointing" or not images:
                stats["skipped_schema"] += 1
                continue
            image = str(images[0]).replace("file://", "")
            if not Path(image).exists():
                stats["skipped_image"] += 1
                continue
            if not item.get("gt_points"):
                stats["skipped_gt"] += 1
                continue
            row = {
                "image": [image],
                "video": [],
                "conversations": [
                    {"from": "system", "value": system_text(item)},
                    {"from": "human", "value": user_text(item)},
                    {"from": "gpt", "value": target_from_gt(item, args.max_points)},
                ],
                "metadata": {
                    "source_prompt_id": item.get("prompt_id"),
                    "source": meta.get("source"),
                    "task_type": meta.get("task_type"),
                    "opd_zero2_smoke": True,
                },
            }
            fout.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            kept += 1

    stats["kept"] = kept
    if kept == 0:
        raise SystemExit(f"no rows written from {src}: {stats}")
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
