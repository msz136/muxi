#!/usr/bin/env python3
"""Attach routed teacher top1 token ids for OPD distillation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import AutoConfig, AutoProcessor, set_seed

from expert_sft import VisionConversationCollator


def log(msg: str) -> None:
    print(msg, flush=True)


def get_model_cls():
    for name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq", "Qwen3VLForConditionalGeneration"):
        cls = getattr(transformers, name, None)
        if cls is not None:
            log(f"[model] loader={name}")
            return cls
    raise RuntimeError("no Qwen3-VL compatible model loader found")


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_teacher(model_path: str, device: torch.device):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if hasattr(config, "use_cache"):
        config.use_cache = False
    if hasattr(config, "_attn_implementation"):
        config._attn_implementation = "eager"
    if hasattr(config, "attn_implementation"):
        config.attn_implementation = "eager"
    model = get_model_cls().from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    return model


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if key == "labels":
            continue
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def fill_teacher_top1(
    rows: list[dict[str, Any]],
    teacher_key: str,
    teacher_path: str,
    processor: Any,
    collator: VisionConversationCollator,
    args: argparse.Namespace,
) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    target_field = f"{teacher_key}_top1_ids"
    needed = [
        idx
        for idx, row in enumerate(rows)
        if row.get("opd", {}).get("teacher") in {teacher_key, "both"}
        and target_field not in row.get("opd", {})
    ]
    log(f"[prepare] teacher={teacher_key} needed={len(needed)} model={teacher_path}")
    if not needed:
        return

    model = load_teacher(teacher_path, device)
    done = 0
    for start in range(0, len(needed), args.batch_size):
        idxs = needed[start : start + args.batch_size]
        examples = [rows[idx] for idx in idxs]
        batch = collator(examples)
        labels = batch["labels"]
        valid_positions = [(labels[i] != -100).nonzero(as_tuple=False).flatten() for i in range(labels.shape[0])]
        with torch.inference_mode():
            outputs = model(**move_batch(batch, device), use_cache=False)
            top1 = outputs.logits[:, :-1, :].argmax(dim=-1).detach().cpu()
        for batch_idx, row_idx in enumerate(idxs):
            ids: list[int] = []
            for pos in valid_positions[batch_idx].tolist():
                if pos <= 0:
                    continue
                ids.append(int(top1[batch_idx, pos - 1].item()))
            if not ids:
                raise ValueError(f"no valid teacher ids row={row_idx}")
            rows[row_idx].setdefault("opd", {})[target_field] = ids
            rows[row_idx]["opd"][f"{teacher_key}_top1_count"] = len(ids)
        done += len(idxs)
        if done % max(args.log_every, args.batch_size) == 0 or done == len(needed):
            log(f"[prepare] teacher={teacher_key} done={done}/{len(needed)}")
        if args.save_every and done % args.save_every == 0:
            save_rows(args.output, rows)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--obj-teacher", required=True)
    parser.add_argument("--reg-teacher", required=True)
    parser.add_argument("--processor-model", default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--model-max-length", dest="model_max_length", type=int, default=16384)
    parser.add_argument("--min-pixels", dest="min_pixels", type=int, default=50176)
    parser.add_argument("--max-pixels", dest="max_pixels", type=int, default=50176)
    parser.add_argument("--video-min-frames", dest="video_min_frames", type=int, default=32)
    parser.add_argument("--video-max-frames", dest="video_max_frames", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=1000)
    args = parser.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    set_seed(args.seed)

    rows = load_rows(args.output if Path(args.output).exists() else args.input)
    processor_model = args.processor_model or args.obj_teacher
    processor = AutoProcessor.from_pretrained(
        processor_model,
        trust_remote_code=True,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    collator = VisionConversationCollator(processor, args)

    fill_teacher_top1(rows, "obj", args.obj_teacher, processor, collator, args)
    save_rows(args.output, rows)
    fill_teacher_top1(rows, "reg", args.reg_teacher, processor, collator, args)
    save_rows(args.output, rows)

    counts = {
        "rows": len(rows),
        "obj_top1": sum("obj_top1_ids" in r.get("opd", {}) for r in rows),
        "reg_top1": sum("reg_top1_ids" in r.get("opd", {}) for r in rows),
        "general_both": sum(
            "obj_top1_ids" in r.get("opd", {}) and "reg_top1_ids" in r.get("opd", {})
            for r in rows
            if r.get("opd", {}).get("teacher") == "both"
        ),
        "output": args.output,
    }
    with open(Path(args.output).with_suffix(".summary.json"), "w", encoding="utf-8") as f:
        json.dump(counts, f, ensure_ascii=False, indent=2)
    log("[prepare] complete " + json.dumps(counts, ensure_ascii=False))


if __name__ == "__main__":
    main()
