#!/usr/bin/env python3
"""Build the first mixed OPD manifest for object/region/general routing."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


OBJ_PATH = "/data/msz/opd_project/data/semantic_nav_box_v1/semantic_nav_box_grounding_full_object_ref_v1_high_quality.jsonl"
REG_PREDBOX_PATH = "/data/msz/opd_project/data/semantic_nav_region_box_v1/predbox_label_v1/semantic_nav_region_predbox_label_v1_high_quality_no_eval_holdout.jsonl"
REG_SOLUTION_PATH = "/data/msz/opd_project/data/semantic_nav_region_box_v1/solution_a_bidir_train_v1/semantic_nav_region_solution_a_bidir_train_v1_high_quality.jsonl"
GENERAL_PATH = "/data/msz/point/data_expert/expert_grounding_mix.jsonl"


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def has_turns(row: dict[str, Any]) -> bool:
    user = False
    assistant = False
    for turn in row.get("conversations", []):
        role = str(turn.get("from", "")).lower()
        value = str(turn.get("value", ""))
        user = user or (role in {"human", "user"} and bool(value.strip()))
        assistant = assistant or (role in {"gpt", "assistant"} and bool(value.strip()))
    return user and assistant


def has_media(row: dict[str, Any]) -> bool:
    for field in ("image", "video"):
        for item in row.get(field) or []:
            text = str(item)
            if text.startswith(("http://", "https://")):
                return True
            if Path(text).exists():
                return True
    return False


def normalize_media_paths(row: dict[str, Any]) -> dict[str, Any] | None:
    row = dict(row)
    data_path = row.get("data_path")
    for field in ("image", "video"):
        fixed: list[str] = []
        for item in row.get(field) or []:
            text = str(item)
            if text.startswith(("http://", "https://", "/")):
                fixed.append(text)
                continue
            if data_path:
                candidate = Path(str(data_path)) / text
                if candidate.exists():
                    fixed.append(str(candidate))
                    continue
            fixed.append(text)
        row[field] = fixed
    if not has_turns(row):
        return None
    return row


def sample_rows(
    rows: list[dict[str, Any]],
    count: int,
    rng: random.Random,
    bucket: str,
    teacher: str,
    source_name: str,
    dataset_filter: set[str] | None = None,
    require_media: bool = False,
) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    usable: list[dict[str, Any]] = []
    for row in rows:
        if dataset_filter is not None and str(row.get("dataset")) not in dataset_filter:
            continue
        fixed = normalize_media_paths(row)
        if fixed is None:
            continue
        if require_media and not has_media(fixed):
            continue
        usable.append(fixed)
    if not usable:
        raise ValueError(f"no usable rows for {source_name}")

    picked: list[dict[str, Any]] = []
    for idx in range(count):
        row = dict(rng.choice(usable) if count > len(usable) else usable[idx])
        if count <= len(usable):
            # Rows are shuffled below before slicing, so this is still random.
            pass
        row["opd"] = {
            "bucket": bucket,
            "teacher": teacher,
            "source": source_name,
            "source_dataset": row.get("dataset"),
        }
        picked.append(row)
    return picked


def sample_without_replacement(
    rows: list[dict[str, Any]],
    count: int,
    rng: random.Random,
    bucket: str,
    teacher: str,
    source_name: str,
    dataset_filter: set[str] | None = None,
    require_media: bool = False,
) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    usable: list[dict[str, Any]] = []
    order = list(range(len(rows)))
    rng.shuffle(order)
    for row_idx in order:
        row = rows[row_idx]
        if dataset_filter is not None and str(row.get("dataset")) not in dataset_filter:
            continue
        fixed = normalize_media_paths(row)
        if fixed is None:
            continue
        if require_media and not has_media(fixed):
            continue
        usable.append(fixed)
        if len(usable) >= count:
            break
    if not usable:
        raise ValueError(f"no usable rows for {source_name}")

    if count <= len(usable):
        chosen = usable[:count]
    else:
        chosen = list(usable)
        while len(chosen) < count:
            chosen.append(rng.choice(usable))

    picked: list[dict[str, Any]] = []
    for row in chosen:
        out = dict(row)
        out["opd"] = {
            "bucket": bucket,
            "teacher": teacher,
            "source": source_name,
            "source_dataset": row.get("dataset"),
        }
        picked.append(out)
    return picked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--total", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--obj-path", default=OBJ_PATH)
    parser.add_argument("--reg-predbox-path", default=REG_PREDBOX_PATH)
    parser.add_argument("--reg-solution-path", default=REG_SOLUTION_PATH)
    parser.add_argument("--general-path", default=GENERAL_PATH)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    obj_n = round(args.total * 0.35)
    reg_n = round(args.total * 0.35)
    general_n = args.total - obj_n - reg_n
    reg_predbox_n = round(reg_n * 0.70)
    reg_solution_n = reg_n - reg_predbox_n
    general_keepalive_n = general_n
    general_grounding_n = 0

    obj_rows = sample_without_replacement(
        load_jsonl(args.obj_path),
        obj_n,
        rng,
        bucket="obj",
        teacher="obj",
        source_name="object_ref_high_quality",
        require_media=True,
    )
    reg_rows = sample_without_replacement(
        load_jsonl(args.reg_predbox_path),
        reg_predbox_n,
        rng,
        bucket="reg",
        teacher="reg",
        source_name="region_predbox_label_v1",
        require_media=True,
    )
    reg_rows += sample_without_replacement(
        load_jsonl(args.reg_solution_path),
        reg_solution_n,
        rng,
        bucket="reg",
        teacher="reg",
        source_name="region_solution_a_bidir",
        require_media=True,
    )

    general_all = load_jsonl(args.general_path)
    general_rows = sample_without_replacement(
        general_all,
        general_keepalive_n,
        rng,
        bucket="general",
        teacher="both",
        source_name="general_robo2vlm_keepalive",
        dataset_filter={"robo2vlm-1"},
        require_media=True,
    )
    general_rows += sample_without_replacement(
        general_all,
        general_grounding_n,
        rng,
        bucket="general",
        teacher="both",
        source_name="general_grounding_local",
        dataset_filter={"sharerobot_affordance"},
        require_media=True,
    )

    rows = obj_rows + reg_rows + general_rows
    rng.shuffle(rows)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for idx, row in enumerate(rows):
            row.setdefault("opd", {})["mix_index"] = idx
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "output": str(out_path),
        "total": len(rows),
        "seed": args.seed,
        "ratios": {"obj": obj_n, "reg": reg_n, "general": general_n},
        "reg_sources": {"predbox": reg_predbox_n, "solution_a": reg_solution_n},
        "general_sources": {"robo2vlm_keepalive": general_keepalive_n, "grounding_local": general_grounding_n},
        "sources": {
            "obj": args.obj_path,
            "reg_predbox": args.reg_predbox_path,
            "reg_solution": args.reg_solution_path,
            "general": args.general_path,
        },
    }
    with open(out_path.with_suffix(".summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
