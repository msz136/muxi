#!/usr/bin/env python3
"""Filter JSONL rows to locally available media and normalize relative paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def normalize_items(items: list[Any], data_path: str | None, stats: dict[str, int]) -> list[str]:
    fixed: list[str] = []
    base = Path(data_path) if data_path else None
    for item in items or []:
        text = str(item)
        if text.startswith(("http://", "https://")):
            stats["drop_url_media_items"] += 1
            continue
        path = Path(text)
        if path.exists():
            fixed.append(str(path))
            continue
        if base is not None:
            candidate = base / text
            if candidate.exists():
                fixed.append(str(candidate))
                continue
        stats["drop_missing_media_items"] += 1
    return fixed


def has_placeholder(row: dict[str, Any], placeholder: str) -> bool:
    for turn in row.get("conversations") or []:
        if placeholder in str(turn.get("value", "")):
            return True
    return False


def convert(in_path: Path, out_path: Path) -> dict[str, int | str]:
    stats: dict[str, int | str] = {
        "input": str(in_path),
        "output": str(out_path),
        "input_rows": 0,
        "output_rows": 0,
        "drop_bad_json": 0,
        "drop_missing_required_image": 0,
        "drop_missing_required_video": 0,
        "drop_url_media_items": 0,
        "drop_missing_media_items": 0,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with in_path.open("r", encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as dst:
        for line in src:
            stats["input_rows"] = int(stats["input_rows"]) + 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                stats["drop_bad_json"] = int(stats["drop_bad_json"]) + 1
                continue
            row = dict(row)
            data_path = row.get("data_path")
            row["image"] = normalize_items(row.get("image") or [], data_path, stats)  # type: ignore[arg-type]
            row["video"] = normalize_items(row.get("video") or [], data_path, stats)  # type: ignore[arg-type]
            if has_placeholder(row, "<image>") and not row["image"]:
                stats["drop_missing_required_image"] = int(stats["drop_missing_required_image"]) + 1
                continue
            if has_placeholder(row, "<video>") and not row["video"]:
                stats["drop_missing_required_video"] = int(stats["drop_missing_required_video"]) + 1
                continue
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats["output_rows"] = int(stats["output_rows"]) + 1
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--suffix", default="_mediaok")
    parser.add_argument("--summary", default=None)
    args = parser.parse_args()

    summaries = []
    for item in args.inputs:
        in_path = Path(item)
        out_path = in_path.with_name(in_path.stem + args.suffix + in_path.suffix)
        summaries.append(convert(in_path, out_path))
    if args.summary:
        with Path(args.summary).open("w", encoding="utf-8") as f:
            json.dump(summaries, f, ensure_ascii=False, indent=2)
    print(json.dumps(summaries, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
