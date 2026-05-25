#!/usr/bin/env python3
"""Audit OPD student JSONL data for length, format, coordinate, and image anomalies."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


COORD_RE = re.compile(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]")


def now() -> str:
    return time.strftime("%F %T")


def log(obj: dict[str, Any]) -> None:
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def conv_stats(row: dict[str, Any]) -> tuple[int, int, int, str, str, int, int]:
    total = human = gpt = 0
    first_human = ""
    answer = ""
    human_turns = gpt_turns = 0
    for turn in row.get("conversations") or []:
        value = str(turn.get("value", ""))
        total += len(value)
        role = str(turn.get("from", "")).lower()
        if role in {"human", "user"}:
            human += len(value)
            human_turns += 1
            if not first_human:
                first_human = value
        elif role in {"gpt", "assistant"}:
            gpt += len(value)
            gpt_turns += 1
            if not answer:
                answer = value
    return total, human, gpt, first_human, answer, human_turns, gpt_turns


def expected_format(answer: str) -> str:
    has_point = "<point>" in answer
    has_box = "<box>" in answer
    if has_point and not has_box:
        return "point"
    if has_box and not has_point:
        return "box"
    if has_point and has_box:
        return "mixed_grounding"
    return "text"


def percentile(values: list[int | float], q: float) -> int | float | None:
    if not values:
        return None
    vals = sorted(values)
    idx = min(len(vals) - 1, max(0, math.ceil(q * len(vals)) - 1))
    return vals[idx]


def basic_stats(values: list[int | float]) -> dict[str, int | float | None]:
    if not values:
        return {"min": None, "p50": None, "p90": None, "p95": None, "p99": None, "p999": None, "max": None}
    vals = sorted(values)
    return {
        "min": vals[0],
        "p50": percentile(vals, 0.50),
        "p90": percentile(vals, 0.90),
        "p95": percentile(vals, 0.95),
        "p99": percentile(vals, 0.99),
        "p999": percentile(vals, 0.999),
        "max": vals[-1],
    }


def add_example(examples: dict[str, list[dict[str, Any]]], key: str, item: dict[str, Any], limit: int) -> None:
    if len(examples[key]) < limit:
        examples[key].append(item)


def first_image(row: dict[str, Any]) -> str:
    images = row.get("image") or []
    return str(images[0]) if images else ""


def audit_jsonl(path: Path, split: str, example_limit: int) -> dict[str, Any]:
    counters: Counter[str] = Counter()
    category: Counter[str] = Counter()
    subtype: Counter[str] = Counter()
    target_expert: Counter[str] = Counter()
    fmt_counter: Counter[str] = Counter()
    expected_meta_fmt: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    total_chars: list[int] = []
    human_chars: list[int] = []
    gpt_chars: list[int] = []
    point_counts: list[int] = []
    box_pair_counts: list[int] = []
    box_areas: list[int] = []
    image_counts: Counter[int] = Counter()
    fingerprints: set[str] = set()
    dup_fingerprints = 0
    images: set[str] = set()

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                counters["blank_line"] += 1
                continue
            try:
                row = json.loads(line)
            except Exception:
                counters["bad_json"] += 1
                add_example(examples, "bad_json", {"line": line_no, "head": line[:160]}, example_limit)
                continue

            counters["rows"] += 1
            total, human, gpt, prompt, answer, human_turns, gpt_turns = conv_stats(row)
            total_chars.append(total)
            human_chars.append(human)
            gpt_chars.append(gpt)
            fmt = expected_format(answer)
            fmt_counter[fmt] += 1

            images_list = row.get("image")
            img_count = len(images_list) if isinstance(images_list, list) else 0
            image_counts[img_count] += 1
            img = first_image(row)
            if img:
                images.add(img)

            opd = (row.get("metadata") or {}).get("opd") or {}
            category[str(opd.get("sample_category", "missing"))] += 1
            subtype[str(opd.get("sample_subtype", "missing"))] += 1
            target_expert[str(opd.get("target_expert", "missing"))] += 1
            expected_meta_fmt[str(opd.get("expected_format", "missing"))] += 1
            fp = opd.get("fingerprint")
            if not fp:
                counters["missing_fingerprint"] += 1
                add_example(examples, "missing_fingerprint", {"line": line_no}, example_limit)
            elif fp in fingerprints:
                dup_fingerprints += 1
                add_example(examples, "duplicate_fingerprint", {"line": line_no, "fingerprint": fp}, example_limit)
            else:
                fingerprints.add(str(fp))

            base_ex = {
                "line": line_no,
                "category": opd.get("sample_category"),
                "subtype": opd.get("sample_subtype"),
                "target_expert": opd.get("target_expert"),
                "source_file": opd.get("source_file"),
                "source_line": opd.get("source_line"),
                "fingerprint": fp,
                "total_chars": total,
                "human_chars": human,
                "gpt_chars": gpt,
                "prompt_head": prompt[:160],
                "answer_head": answer[:160],
            }

            if not row.get("conversations"):
                counters["missing_conversations"] += 1
                add_example(examples, "missing_conversations", base_ex, example_limit)
            if not answer:
                counters["missing_answer"] += 1
                add_example(examples, "missing_answer", base_ex, example_limit)
            if human_turns == 0:
                counters["missing_human_turn"] += 1
                add_example(examples, "missing_human_turn", base_ex, example_limit)
            if gpt_turns == 0:
                counters["missing_gpt_turn"] += 1
                add_example(examples, "missing_gpt_turn", base_ex, example_limit)
            if img_count == 0:
                counters["missing_image"] += 1
                add_example(examples, "missing_image", base_ex, example_limit)
            if img_count > 1:
                counters["multi_image"] += 1
                add_example(examples, "multi_image", {**base_ex, "image_count": img_count}, example_limit)
            if row.get("video"):
                counters["nonempty_video"] += 1
                add_example(examples, "nonempty_video", base_ex, example_limit)
            if total > 900:
                counters["total_chars_gt_900"] += 1
                add_example(examples, "total_chars_gt_900", base_ex, example_limit)
            if gpt > 500:
                counters["gpt_chars_gt_500"] += 1
                add_example(examples, "gpt_chars_gt_500", base_ex, example_limit)
            if human > 850:
                counters["human_chars_gt_850"] += 1
                add_example(examples, "human_chars_gt_850", base_ex, example_limit)
            if opd.get("expected_format") and opd.get("expected_format") != fmt:
                counters["metadata_expected_format_mismatch"] += 1
                add_example(examples, "metadata_expected_format_mismatch", {**base_ex, "actual_format": fmt, "metadata_format": opd.get("expected_format")}, example_limit)
            if fmt == "mixed_grounding":
                counters["mixed_point_box_answer"] += 1
                add_example(examples, "mixed_point_box_answer", base_ex, example_limit)

            coords = [(float(x), float(y)) for x, y in COORD_RE.findall(answer)]
            if fmt == "point":
                point_counts.append(len(coords))
                if len(coords) == 0:
                    counters["point_tag_without_coords"] += 1
                    add_example(examples, "point_tag_without_coords", base_ex, example_limit)
                if len(coords) > 50:
                    counters["point_count_gt_50"] += 1
                    add_example(examples, "point_count_gt_50", {**base_ex, "point_count": len(coords)}, example_limit)
                if any(x < 0 or x > 1000 or y < 0 or y > 1000 for x, y in coords):
                    counters["point_coord_out_of_range"] += 1
                    add_example(examples, "point_coord_out_of_range", {**base_ex, "coords_head": coords[:5]}, example_limit)
                stripped = answer.strip()
                if not (stripped.startswith("<point>") and stripped.endswith("</point>")):
                    counters["point_answer_extra_text"] += 1
                    add_example(examples, "point_answer_extra_text", base_ex, example_limit)
            elif fmt == "box":
                box_pair_counts.append(len(coords))
                if len(coords) < 2:
                    counters["box_tag_with_lt_2_pairs"] += 1
                    add_example(examples, "box_tag_with_lt_2_pairs", {**base_ex, "pair_count": len(coords)}, example_limit)
                if any(x < 0 or x > 1000 or y < 0 or y > 1000 for x, y in coords):
                    counters["box_coord_out_of_range"] += 1
                    add_example(examples, "box_coord_out_of_range", {**base_ex, "coords_head": coords[:5]}, example_limit)
                if len(coords) >= 2:
                    x1, y1 = coords[0]
                    x2, y2 = coords[1]
                    w, h = x2 - x1, y2 - y1
                    area = int(max(0.0, w) * max(0.0, h))
                    box_areas.append(area)
                    if w <= 0 or h <= 0:
                        counters["box_degenerate_or_inverted"] += 1
                        add_example(examples, "box_degenerate_or_inverted", {**base_ex, "coords_head": coords[:3]}, example_limit)
                    if 0 < area < 4:
                        counters["box_area_lt_4"] += 1
                        add_example(examples, "box_area_lt_4", {**base_ex, "area": area, "coords_head": coords[:2]}, example_limit)
                    if area > 950000:
                        counters["box_area_gt_950k"] += 1
                        add_example(examples, "box_area_gt_950k", {**base_ex, "area": area, "coords_head": coords[:2]}, example_limit)
                    if w > 0 and h > 0 and max(w / h, h / w) > 20:
                        counters["box_aspect_gt_20"] += 1
                        add_example(examples, "box_aspect_gt_20", {**base_ex, "aspect": round(max(w / h, h / w), 3), "coords_head": coords[:2]}, example_limit)
                stripped = answer.strip()
                if not (stripped.startswith("<box>") and stripped.endswith("</box>")):
                    counters["box_answer_extra_text"] += 1
                    add_example(examples, "box_answer_extra_text", base_ex, example_limit)
            elif fmt == "text" and ("<point>" in prompt or "<box>" in prompt):
                counters["text_answer_grounding_prompt"] += 1
                add_example(examples, "text_answer_grounding_prompt", base_ex, example_limit)

            if counters["rows"] % 100_000 == 0:
                log({"stage": "json_progress", "split": split, "rows": counters["rows"], "time": now()})

    return {
        "path": str(path),
        "rows": counters["rows"],
        "bad_json": counters["bad_json"],
        "counters": dict(counters.most_common()),
        "category_counts": dict(category.most_common()),
        "subtype_counts": dict(subtype.most_common()),
        "target_expert_counts": dict(target_expert.most_common()),
        "format_counts": dict(fmt_counter.most_common()),
        "metadata_expected_format_counts": dict(expected_meta_fmt.most_common()),
        "image_count_distribution": dict(image_counts.most_common()),
        "unique_fingerprints": len(fingerprints),
        "duplicate_fingerprints": dup_fingerprints,
        "unique_images": len(images),
        "images": images,
        "text_length_stats": {
            "total_chars": basic_stats(total_chars),
            "human_chars": basic_stats(human_chars),
            "gpt_chars": basic_stats(gpt_chars),
        },
        "point_count_stats": basic_stats(point_counts),
        "box_pair_count_stats": basic_stats(box_pair_counts),
        "box_area_stats": basic_stats(box_areas),
        "examples": dict(examples),
    }


def resolve_image(path_text: str, bases: list[Path]) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    for base in bases:
        candidate = base / path
        if candidate.exists():
            return candidate
    return bases[0] / path


def audit_images(images: set[str], *, bases: list[Path], example_limit: int) -> dict[str, Any]:
    counters: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    widths: list[int] = []
    heights: list[int] = []
    areas: list[int] = []
    aspects: list[float] = []
    pil_available = True
    try:
        from PIL import Image
    except Exception:
        pil_available = False
        Image = None  # type: ignore

    for idx, img in enumerate(sorted(images), start=1):
        resolved = resolve_image(img, bases)
        if not resolved.exists():
            counters["missing_image_file"] += 1
            add_example(examples, "missing_image_file", {"image": img, "resolved": str(resolved)}, example_limit)
            continue
        counters["existing_image_file"] += 1
        if pil_available:
            try:
                with Image.open(resolved) as im:  # type: ignore[union-attr]
                    w, h = im.size
            except Exception as exc:
                counters["image_open_error"] += 1
                add_example(examples, "image_open_error", {"image": img, "resolved": str(resolved), "error": repr(exc)}, example_limit)
                continue
            widths.append(w)
            heights.append(h)
            area = w * h
            areas.append(area)
            aspect = max(w / h, h / w) if w > 0 and h > 0 else float("inf")
            aspects.append(aspect)
            if w <= 0 or h <= 0:
                counters["image_nonpositive_dimension"] += 1
                add_example(examples, "image_nonpositive_dimension", {"image": img, "size": [w, h]}, example_limit)
            if min(w, h) < 16:
                counters["image_min_dim_lt_16"] += 1
                add_example(examples, "image_min_dim_lt_16", {"image": img, "size": [w, h]}, example_limit)
            if max(w, h) > 8000:
                counters["image_max_dim_gt_8000"] += 1
                add_example(examples, "image_max_dim_gt_8000", {"image": img, "size": [w, h]}, example_limit)
            if area > 80_000_000:
                counters["image_area_gt_80mp"] += 1
                add_example(examples, "image_area_gt_80mp", {"image": img, "size": [w, h], "area": area}, example_limit)
            if aspect > 20:
                counters["image_aspect_gt_20"] += 1
                add_example(examples, "image_aspect_gt_20", {"image": img, "size": [w, h], "aspect": round(aspect, 3)}, example_limit)

        if idx % 50_000 == 0:
            log({"stage": "image_progress", "checked": idx, "total_unique_images": len(images), "time": now()})

    return {
        "total_unique_images": len(images),
        "pil_available": pil_available,
        "counters": dict(counters.most_common()),
        "width_stats": basic_stats(widths),
        "height_stats": basic_stats(heights),
        "area_stats": basic_stats(areas),
        "aspect_stats": basic_stats(aspects),
        "examples": dict(examples),
    }


def strip_images(report: dict[str, Any]) -> dict[str, Any]:
    out = dict(report)
    out.pop("images", None)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default="/data/msz/point/opd_student_v1")
    parser.add_argument("--report", default="/data/msz/point/opd_student_v1/anomaly_audit_report.json")
    parser.add_argument("--example-limit", type=int, default=20)
    parser.add_argument("--skip-images", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    train = audit_jsonl(out_root / "train_prompts.jsonl", "train", args.example_limit)
    eval_ = audit_jsonl(out_root / "eval_prompts.jsonl", "eval", args.example_limit)

    train_fps = set()
    eval_fps = set()
    # The detailed duplicate check is already inside each split; keep overlap scan lightweight.
    for path, target in [(out_root / "train_prompts.jsonl", train_fps), (out_root / "eval_prompts.jsonl", eval_fps)]:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                fp = ((row.get("metadata") or {}).get("opd") or {}).get("fingerprint")
                if fp:
                    target.add(str(fp))

    train_images = train["images"]
    eval_images = eval_["images"]
    overlap = {
        "fingerprint_overlap": len(train_fps & eval_fps),
        "image_overlap": len(train_images & eval_images),
    }
    all_images = set(train_images) | set(eval_images)
    image_report = None
    if not args.skip_images:
        image_report = audit_images(all_images, bases=[Path("/data/msz/point"), Path("/data/msz"), out_root], example_limit=args.example_limit)

    report = {
        "generated_at": now(),
        "out_root": str(out_root),
        "thresholds": {
            "total_chars_max": 900,
            "gpt_chars_watch": 500,
            "point_count_max": 50,
            "coord_range": [0, 1000],
            "image_large_watch": {"max_dim_gt": 8000, "area_gt_pixels": 80_000_000, "aspect_gt": 20},
        },
        "train": strip_images(train),
        "eval": strip_images(eval_),
        "overlap": overlap,
        "images": image_report,
    }
    report_path = Path(args.report)
    tmp = report_path.with_suffix(report_path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, report_path)
    log({"stage": "done", "report": str(report_path), "train_rows": train["rows"], "eval_rows": eval_["rows"], "overlap": overlap, "time": now()})


if __name__ == "__main__":
    main()
