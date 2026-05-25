#!/usr/bin/env python3
"""Build deterministic full shuffled train files for seed0 expert SFT."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


EXPERTS = [
    "general_obj_expert",
    "general_reasoning_expert",
    "region_expert",
    "robopoint_expert",
    "spatial_rel_expert",
]


def dataset_name(row: dict[str, Any]) -> str:
    return str(
        row.get("dataset")
        or row.get("source")
        or row.get("data_source")
        or row.get("metadata", {}).get("source")
        or "unknown"
    )


def load_rows(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line for line in f if line.strip()]


def summarize_file(path: Path, head_n: int) -> dict[str, Any]:
    total = 0
    all_counts: Counter[str] = Counter()
    head_counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as f:
        for total, line in enumerate(f, start=1):
            row = json.loads(line)
            name = dataset_name(row)
            all_counts[name] += 1
            if total <= head_n:
                head_counts[name] += 1
    return {
        "rows": total,
        "head_rows": min(total, head_n),
        "all_source_counts": dict(all_counts.most_common()),
        "head_source_counts": dict(head_counts.most_common()),
    }


def shuffle_one(src: Path, dst: Path, seed: int, head_n: int) -> dict[str, Any]:
    rows = load_rows(src)
    rng = random.Random(seed)
    rng.shuffle(rows)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        f.writelines(rows)

    summary = summarize_file(dst, head_n=head_n)
    summary.update({"src": str(src), "dst": str(dst), "seed": seed})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="/data/msz/point/data_expert_seed0_v1")
    parser.add_argument("--output-root", default="/data/msz/point/data_expert_seed0_v1_shuffled")
    parser.add_argument("--head-samples", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    summary: dict[str, Any] = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "head_samples": args.head_samples,
        "seed": args.seed,
        "experts": {},
    }
    for expert_idx, expert in enumerate(EXPERTS):
        src = input_root / expert / "train.jsonl"
        dst = output_root / expert / "train_shuffled_seed20260520.jsonl"
        if dst.exists() and not args.force:
            expert_summary = summarize_file(dst, head_n=args.head_samples)
            expert_summary.update({"src": str(src), "dst": str(dst), "seed": args.seed + expert_idx, "reused": True})
        else:
            expert_summary = shuffle_one(
                src=src,
                dst=dst,
                seed=args.seed + expert_idx,
                head_n=args.head_samples,
            )
            expert_summary["reused"] = False
        summary["experts"][expert] = expert_summary
        print(json.dumps({expert: expert_summary}, ensure_ascii=False), flush=True)

    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({"summary": str(output_root / "summary.json")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
