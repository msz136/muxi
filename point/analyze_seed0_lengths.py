#!/usr/bin/env python3
from __future__ import annotations

import argparse
import heapq
import json
import math
from typing import Any

import torch
from transformers import AutoProcessor


DEFAULT_SYSTEM_PROMPT = (
    "You are a semantic navigation grounding assistant. Given an image and target "
    "object information, return the target object's bounding box in coordinates "
    "from 0 to 1000. Return only <box>[[x1,y1],[x2,y2]]</box>."
)


def extract_turns(example: dict[str, Any]) -> tuple[str, str, str]:
    system_text = DEFAULT_SYSTEM_PROMPT
    user_text = ""
    answer_text = ""
    for turn in example.get("conversations", []):
        role = str(turn.get("from", "")).lower()
        value = str(turn.get("value", ""))
        if role == "system":
            system_text = value or system_text
        elif role in {"human", "user"} and not user_text:
            user_text = value
        elif role in {"gpt", "assistant"} and not answer_text:
            answer_text = value
    return system_text, user_text, answer_text


def content_from_sample(example: dict[str, Any], user_text: str) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for image_path in example.get("image") or []:
        content.append(
            {
                "type": "image",
                "image": str(image_path),
                "min_pixels": 50176,
                "max_pixels": 50176,
            }
        )
    for video_path in example.get("video") or []:
        content.append(
            {
                "type": "video",
                "video": str(video_path),
                "min_pixels": 50176,
                "max_pixels": 50176,
                "min_frames": 32,
                "max_frames": 32,
                "fps": 1.0,
            }
        )
    text = user_text.replace("<image>", "").replace("<video>", "").strip()
    if text:
        content.append({"type": "text", "text": text})
    return content


def messages(example: dict[str, Any], include_answer: bool = True) -> list[dict[str, Any]]:
    system_text, user_text, answer_text = extract_turns(example)
    out = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": content_from_sample(example, user_text)},
    ]
    if include_answer:
        out.append({"role": "assistant", "content": answer_text})
    return out


def percentile(sorted_values: list[int], p: float) -> int:
    pos = int(math.ceil(p / 100.0 * len(sorted_values))) - 1
    return sorted_values[max(0, min(pos, len(sorted_values) - 1))]


def make_candidates(limit: int, seed: int, world_size: int, grad_accum: int, step: int) -> dict[str, Any]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    perm = torch.randperm(limit, generator=generator).tolist()
    global_start = step * world_size * grad_accum
    global_count = world_size * grad_accum * 2

    generator2 = torch.Generator()
    generator2.manual_seed(seed)
    dist_perm = torch.randperm(limit, generator=generator2).tolist()
    total_size = math.ceil(limit / world_size) * world_size
    if len(dist_perm) < total_size:
        dist_perm += dist_perm[: total_size - len(dist_perm)]
    rank0 = dist_perm[:total_size:world_size]
    rank0_start = step * grad_accum
    rank0_count = grad_accum * 4

    return {
        "global_positions_checked": [global_start, global_start + global_count - 1],
        "cand_global_indices": perm[global_start : global_start + global_count],
        "rank0_positions_checked": [rank0_start, rank0_start + rank0_count - 1],
        "cand_rank0_indices": rank0[rank0_start : rank0_start + rank0_count],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="/data/msz/models/8b_base")
    parser.add_argument("--limit", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--oom-step", type=int, default=516)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=True,
        min_pixels=50176,
        max_pixels=50176,
    )
    tokenizer = getattr(processor, "tokenizer", processor)

    candidates = make_candidates(args.limit, args.seed, args.world_size, args.grad_accum, args.oom_step)
    watch = set(candidates["cand_global_indices"]) | set(candidates["cand_rank0_indices"])

    lengths: list[int] = []
    top: list[tuple[int, int, int, str | None, str | None, str | None]] = []
    watch_rows: dict[int, dict[str, Any]] = {}
    batch_texts: list[str] = []
    batch_meta: list[tuple[int, dict[str, Any], int]] = []

    def flush_batch() -> None:
        if not batch_texts:
            return
        enc = tokenizer(batch_texts, add_special_tokens=False, padding=False, truncation=False)
        for (idx, row, chars), ids in zip(batch_meta, enc["input_ids"]):
            token_count = len(ids)
            lengths.append(token_count)
            item = (
                token_count,
                idx,
                chars,
                row.get("dataset"),
                (row.get("metadata") or {}).get("task_type"),
                (row.get("image") or [None])[0],
            )
            if len(top) < 50:
                heapq.heappush(top, item)
            elif token_count > top[0][0]:
                heapq.heapreplace(top, item)
            if idx in watch:
                _system, user_text, answer_text = extract_turns(row)
                watch_rows[idx] = {
                    "idx": idx,
                    "text_tokens_no_image_expand": token_count,
                    "chars": chars,
                    "user_chars": len(user_text),
                    "answer_chars": len(answer_text),
                    "dataset": row.get("dataset"),
                    "metadata": row.get("metadata"),
                    "image": (row.get("image") or [None])[0],
                }
        batch_texts.clear()
        batch_meta.clear()

    with open(args.data, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= args.limit:
                break
            row = json.loads(line)
            text = processor.apply_chat_template(messages(row), tokenize=False, add_generation_prompt=False)
            batch_texts.append(text)
            batch_meta.append((idx, row, len(text)))
            if len(batch_texts) >= args.batch_size:
                flush_batch()
    flush_batch()

    sorted_lengths = sorted(lengths)
    thresholds = [512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 10000, 12000, 14000, 16000]
    summary = {
        "n": len(lengths),
        "min": min(lengths),
        "mean": round(sum(lengths) / len(lengths), 2),
        "p50": percentile(sorted_lengths, 50),
        "p90": percentile(sorted_lengths, 90),
        "p95": percentile(sorted_lengths, 95),
        "p99": percentile(sorted_lengths, 99),
        "p99_5": percentile(sorted_lengths, 99.5),
        "p99_9": percentile(sorted_lengths, 99.9),
        "max": max(lengths),
        "counts_gt": {str(t): sum(1 for value in lengths if value > t) for t in thresholds},
        "sampler_candidates": candidates,
        "watch_rows_sorted": sorted(
            watch_rows.values(),
            key=lambda row: row["text_tokens_no_image_expand"],
            reverse=True,
        ),
        "top50_text_tokens": [
            {
                "tokens": token_count,
                "idx": idx,
                "chars": chars,
                "dataset": dataset,
                "task_type": task_type,
                "image": image,
            }
            for token_count, idx, chars, dataset, task_type, image in sorted(top, reverse=True)
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
