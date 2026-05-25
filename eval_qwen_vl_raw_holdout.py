#!/usr/bin/env python3
"""Batch-generate and score Qwen3-VL models on raw_holdout_eval_v1."""

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

import torch
import transformers
from transformers import AutoConfig, AutoProcessor


COORD_RE = re.compile(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]")
BOX_FLAT_RE = re.compile(
    r"<box>\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]\s*</box>",
    re.I | re.S,
)
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def now() -> str:
    return time.strftime("%F %T")


def log(obj: dict[str, Any]) -> None:
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def get_model_cls():
    for name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq", "Qwen3VLForConditionalGeneration"):
        cls = getattr(transformers, name, None)
        if cls is not None:
            return cls
    raise RuntimeError("no Qwen3-VL compatible model loader found in transformers")


def answer_of(row: dict[str, Any]) -> str:
    if row.get("gold"):
        return str(row.get("gold"))
    for turn in row.get("conversations") or []:
        if str(turn.get("from", "")).lower() in {"gpt", "assistant"}:
            return str(turn.get("value", ""))
    return ""


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


def first_user_text(row: dict[str, Any]) -> str:
    for turn in row.get("conversations") or []:
        if str(turn.get("from", "")).lower() in {"human", "user"}:
            return str(turn.get("value", ""))
    return ""


def system_text(row: dict[str, Any]) -> str:
    for turn in row.get("conversations") or []:
        if str(turn.get("from", "")).lower() == "system":
            return str(turn.get("value", ""))
    return "You are a helpful vision-language assistant."


def content_from_row(row: dict[str, Any], min_pixels: int, max_pixels: int) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for image_path in row.get("image") or []:
        content.append({"type": "image", "image": str(image_path), "min_pixels": min_pixels, "max_pixels": max_pixels})
    text = first_user_text(row).replace("<image>", "").replace("<video>", "").strip()
    if text:
        content.append({"type": "text", "text": text})
    return content


def messages_from_row(row: dict[str, Any], min_pixels: int, max_pixels: int) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": system_text(row)},
        {"role": "user", "content": content_from_row(row, min_pixels, max_pixels)},
    ]


def parse_points(text: str) -> list[tuple[float, float]]:
    return [(float(x), float(y)) for x, y in COORD_RE.findall(text)]


def parse_box(text: str) -> tuple[float, float, float, float] | None:
    flat = BOX_FLAT_RE.search(text)
    if flat:
        x1, y1, x2, y2 = [float(v) for v in flat.groups()]
        return x1, y1, x2, y2
    box_match = re.search(r"<box>(.*?)</box>", text, re.I | re.S)
    if box_match:
        nums = [float(v) for v in NUMBER_RE.findall(box_match.group(1))]
        if len(nums) >= 4:
            return nums[0], nums[1], nums[2], nums[3]
    coords = parse_points(text)
    if len(coords) >= 2:
        (x1, y1), (x2, y2) = coords[:2]
        return x1, y1, x2, y2
    return None


def valid_box(box: tuple[float, float, float, float] | None) -> bool:
    if box is None:
        return False
    x1, y1, x2, y2 = box
    return 0 <= x1 <= 1000 and 0 <= y1 <= 1000 and 0 <= x2 <= 1000 and 0 <= y2 <= 1000 and x2 > x1 and y2 > y1


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-9)


def center_dist(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    acx, acy = (a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0
    bcx, bcy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
    return math.hypot(acx - bcx, acy - bcy)


def norm_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[\"'`]+|[\"'`.]+$", "", text)
    return text


def text_score(pred: str, gold: str) -> dict[str, float | bool | str]:
    p = norm_text(pred)
    g = norm_text(gold)
    exact = p == g
    p_delettered = re.sub(r"^[a-e]\s*[\.\):：、-]\s*", "", p).strip()
    loose = exact or p_delettered == g or (len(g) >= 2 and re.search(rf"(^|\b){re.escape(g)}($|\b)", p) is not None)
    bool_acc = None
    if g in {"true", "false", "yes", "no"}:
        bool_acc = exact or p_delettered == g
    mc_acc = None
    if len(g) == 1 and g in "abcde":
        mc_acc = p[:1] == g or p.startswith({"a": "a.", "b": "b.", "c": "c.", "d": "d.", "e": "e."}[g])
    return {"text_exact": exact, "text_loose": loose, "bool_acc": bool_acc, "mc_acc": mc_acc, "pred_norm": p, "gold_norm": g}


def score_row(pred: str, gold: str, expected: str) -> dict[str, Any]:
    out: dict[str, Any] = {"expected_format": expected}
    if expected == "box":
        gold_box = parse_box(gold)
        pred_box = parse_box(pred)
        out["format_pass"] = "<box>" in pred and valid_box(pred_box)
        out["coord_valid"] = valid_box(pred_box)
        if valid_box(gold_box) and valid_box(pred_box):
            assert gold_box is not None and pred_box is not None
            iou = box_iou(pred_box, gold_box)
            out.update({
                "iou": iou,
                "acc_iou_0_3": iou >= 0.3,
                "acc_iou_0_5": iou >= 0.5,
                "acc_iou_0_75": iou >= 0.75,
                "center_dist": center_dist(pred_box, gold_box),
            })
        else:
            out.update({"iou": None, "acc_iou_0_3": False, "acc_iou_0_5": False, "acc_iou_0_75": False, "center_dist": None})
        return out
    if expected == "point":
        gold_pts = parse_points(gold)
        pred_pts = parse_points(pred)
        coord_valid = bool(pred_pts) and all(0 <= x <= 1000 and 0 <= y <= 1000 for x, y in pred_pts)
        out["format_pass"] = "<point>" in pred and coord_valid
        out["coord_valid"] = coord_valid
        out["pred_point_count"] = len(pred_pts)
        out["gold_point_count"] = len(gold_pts)
        if pred_pts and gold_pts:
            pred_to_gold = [min(math.hypot(px - gx, py - gy) for gx, gy in gold_pts) for px, py in pred_pts]
            gold_to_pred = [min(math.hypot(px - gx, py - gy) for px, py in pred_pts) for gx, gy in gold_pts]
            min_dist = min(pred_to_gold)
            out.update({
                "min_point_dist": min_dist,
                "mean_pred_to_gold_dist": sum(pred_to_gold) / len(pred_to_gold),
                "mean_gold_to_pred_dist": sum(gold_to_pred) / len(gold_to_pred),
                "hit_at_50": min_dist <= 50,
                "hit_at_100": min_dist <= 100,
                "point_count_abs_diff": abs(len(pred_pts) - len(gold_pts)),
            })
        else:
            out.update({
                "min_point_dist": None,
                "mean_pred_to_gold_dist": None,
                "mean_gold_to_pred_dist": None,
                "hit_at_50": False,
                "hit_at_100": False,
                "point_count_abs_diff": abs(len(pred_pts) - len(gold_pts)),
            })
        return out
    text = text_score(pred, gold)
    out["format_pass"] = True
    out.update(text)
    return out


class Metrics:
    def __init__(self) -> None:
        self.n = 0
        self.c = Counter()
        self.s = Counter()

    def add(self, score: dict[str, Any]) -> None:
        self.n += 1
        for key, value in score.items():
            if isinstance(value, bool):
                self.c[key] += int(value)
            elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                self.s[key] += float(value)
                self.c[f"{key}__count"] += 1

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {"n": self.n}
        for key, val in sorted(self.c.items()):
            if key.endswith("__count"):
                continue
            out[key] = val / max(self.n, 1)
        for key, total in sorted(self.s.items()):
            denom = self.c.get(f"{key}__count", self.n)
            out[f"{key}_mean"] = total / max(denom, 1)
        return out


def load_rows(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--eval-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--min-pixels", type=int, default=50176)
    parser.add_argument("--max-pixels", type=int, default=50176)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--flush-every", type=int, default=50)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    metrics_path = out_dir / "metrics.json"
    rows = load_rows(Path(args.eval_path), args.limit)

    from qwen_vl_utils import process_vision_info

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    if hasattr(config, "_attn_implementation"):
        config._attn_implementation = "eager"
    if hasattr(config, "attn_implementation"):
        config.attn_implementation = "eager"
    model = get_model_cls().from_pretrained(
        args.model_path,
        config=config,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype="auto",
        attn_implementation="eager",
    )
    model.eval()
    model.to("cuda")
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True

    overall = Metrics()
    by_pool: dict[str, Metrics] = defaultdict(Metrics)
    by_format: dict[str, Metrics] = defaultdict(Metrics)
    by_group: dict[str, Metrics] = defaultdict(Metrics)
    start = time.time()
    processed = 0

    with pred_path.open("w", encoding="utf-8") as wf, torch.inference_mode():
        for start_idx in range(0, len(rows), args.batch_size):
            batch = rows[start_idx : start_idx + args.batch_size]
            messages = [messages_from_row(row, args.min_pixels, args.max_pixels) for row in batch]
            texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
            image_inputs, video_inputs = process_vision_info(messages)
            enc = processor(
                text=texts,
                images=image_inputs if image_inputs else None,
                videos=video_inputs if video_inputs else None,
                padding=True,
                truncation=True,
                max_length=4096,
                return_tensors="pt",
            )
            enc = {k: v.to("cuda") if torch.is_tensor(v) else v for k, v in enc.items()}
            generated = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                pad_token_id=getattr(tokenizer, "pad_token_id", None),
                eos_token_id=getattr(tokenizer, "eos_token_id", None),
            )
            new_tokens = generated[:, enc["input_ids"].shape[1] :]
            preds = processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for row, pred in zip(batch, preds):
                meta = ((row.get("metadata") or {}).get("raw_holdout_eval") or {})
                gold = answer_of(row)
                expected = str(meta.get("expected_format") or expected_format(gold))
                score = score_row(pred, gold, expected)
                pool = str(meta.get("source_pool") or row.get("dataset"))
                group = str(meta.get("group") or "unknown")
                overall.add(score)
                by_pool[pool].add(score)
                by_format[expected].add(score)
                by_group[f"{pool}/{group}"].add(score)
                wf.write(json.dumps({
                    "model": args.model_name,
                    "eval_index": meta.get("eval_index"),
                    "source_pool": pool,
                    "group": group,
                    "expected_format": expected,
                    "gold": gold,
                    "prediction": pred,
                    "score": score,
                    "metadata": meta,
                }, ensure_ascii=False) + "\n")
            processed += len(batch)
            if processed % (args.flush_every * args.batch_size) == 0 or processed == len(rows):
                wf.flush()
                elapsed = time.time() - start
                log({
                    "stage": "progress",
                    "model": args.model_name,
                    "processed": processed,
                    "total": len(rows),
                    "samples_per_sec": round(processed / max(elapsed, 1e-6), 4),
                    "time": now(),
                })

    summary = {
        "model": args.model_name,
        "model_path": args.model_path,
        "eval_path": args.eval_path,
        "rows": len(rows),
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "seconds": round(time.time() - start, 3),
        "overall": overall.summary(),
        "by_pool": {k: v.summary() for k, v in sorted(by_pool.items())},
        "by_format": {k: v.summary() for k, v in sorted(by_format.items())},
        "by_group": {k: v.summary() for k, v in sorted(by_group.items())},
        "prediction_file": str(pred_path),
    }
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log({"stage": "done", "model": args.model_name, "metrics": str(metrics_path), "seconds": summary["seconds"], "time": now()})


if __name__ == "__main__":
    main()
