#!/usr/bin/env python3
"""Lightweight generation eval for semantic-nav <box> grounding data."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Any

import torch
import transformers
from qwen_vl_utils import process_vision_info
from transformers import AutoConfig, AutoProcessor, set_seed


BOX_RE = re.compile(
    r"<box>\s*\[\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]\s*,\s*"
    r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]\s*\]\s*</box>",
    re.IGNORECASE,
)


def get_model_cls():
    for name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq", "Qwen3VLForConditionalGeneration"):
        cls = getattr(transformers, name, None)
        if cls is not None:
            return cls
    raise RuntimeError("no Qwen3-VL compatible model loader found")


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sample_rows(rows: list[dict[str, Any]], num_samples: int, seed: int) -> list[tuple[int, dict[str, Any]]]:
    rng = random.Random(seed)
    idxs = list(range(len(rows)))
    rng.shuffle(idxs)
    idxs = idxs[: min(num_samples, len(rows))]
    return [(idx, rows[idx]) for idx in idxs]


def gt_box(row: dict[str, Any]) -> list[float]:
    box = row.get("target", {}).get("box")
    if not box:
        text = row.get("conversations", [{}])[-1].get("value", "")
        parsed = parse_box(text)
        if parsed is None:
            raise ValueError("row has no parseable target box")
        return parsed
    return normalize_box([float(box[0][0]), float(box[0][1]), float(box[1][0]), float(box[1][1])])


def parse_box(text: str) -> list[float] | None:
    m = BOX_RE.search(text)
    if not m:
        return None
    return normalize_box([float(v) for v in m.groups()])


def normalize_box(box: list[float]) -> list[float]:
    x1, y1, x2, y2 = box
    x1, x2 = sorted((max(0.0, min(1000.0, x1)), max(0.0, min(1000.0, x2))))
    y1, y2 = sorted((max(0.0, min(1000.0, y1)), max(0.0, min(1000.0, y2))))
    return [x1, y1, x2, y2]


def box_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def center_error(a: list[float], b: list[float]) -> float:
    acx, acy = (a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0
    bcx, bcy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
    return math.sqrt((acx - bcx) ** 2 + (acy - bcy) ** 2)


def build_prompt_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    system_text = ""
    user_text = ""
    for turn in row.get("conversations", []):
        role = str(turn.get("from", "")).lower()
        value = str(turn.get("value", ""))
        if role == "system":
            system_text = value
        elif role in {"human", "user"} and not user_text:
            user_text = value
    content: list[dict[str, Any]] = []
    for image_path in row.get("image") or []:
        content.append({"type": "image", "image": str(image_path)})
    text = user_text.replace("<image>", "").replace("<video>", "").strip()
    if text:
        content.append({"type": "text", "text": text})
    messages = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": content})
    return messages


def generate_one(model, processor, row: dict[str, Any], device: torch.device, args: argparse.Namespace) -> str:
    messages = build_prompt_messages(row)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info([messages])
    inputs = processor(
        text=[text],
        images=images if images else None,
        videos=videos if videos else None,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            num_beams=1,
        )
    new_tokens = generated[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    set_seed(args.seed)
    rows = load_rows(args.data)
    picked = sample_rows(rows, args.num_samples, args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True, min_pixels=50176, max_pixels=50176)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    if hasattr(config, "use_cache"):
        config.use_cache = True
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = get_model_cls().from_pretrained(
        args.model,
        config=config,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        attn_implementation="eager",
    ).to(device)
    model.eval()

    results: list[dict[str, Any]] = []
    for n, (idx, row) in enumerate(picked, start=1):
        target = gt_box(row)
        pred_text = generate_one(model, processor, row, device, args)
        pred_box = parse_box(pred_text)
        result = {
            "sample_no": n,
            "row_index": idx,
            "prompt_mode": row.get("metadata", {}).get("prompt_mode"),
            "relation": row.get("metadata", {}).get("relation"),
            "target_box": target,
            "prediction": pred_text,
            "pred_box": pred_box,
            "format_ok": pred_box is not None,
        }
        if pred_box is not None:
            result["iou"] = box_iou(pred_box, target)
            result["center_error"] = center_error(pred_box, target)
            result["coord_mae"] = sum(abs(pred_box[i] - target[i]) for i in range(4)) / 4.0
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    ok = [r for r in results if r["format_ok"]]
    ious = [float(r["iou"]) for r in ok]
    center_errors = [float(r["center_error"]) for r in ok]
    coord_maes = [float(r["coord_mae"]) for r in ok]
    summary = {
        "model": args.model,
        "data": args.data,
        "num_samples": len(results),
        "format_ok": len(ok),
        "format_rate": len(ok) / max(len(results), 1),
        "mean_iou": sum(ious) / len(ious) if ious else 0.0,
        "iou_at_0_3": sum(v >= 0.3 for v in ious) / len(ious) if ious else 0.0,
        "iou_at_0_5": sum(v >= 0.5 for v in ious) / len(ious) if ious else 0.0,
        "mean_center_error": sum(center_errors) / len(center_errors) if center_errors else 0.0,
        "mean_coord_mae": sum(coord_maes) / len(coord_maes) if coord_maes else 0.0,
        "results_path": str(out_path),
    }
    summary_path = out_path.with_suffix(".summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("[summary] " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
