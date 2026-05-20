#!/usr/bin/env python3
"""General image-text generation eval for Robo2VLM style keepalive samples."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import string
from pathlib import Path
from typing import Any


_EVAL_DEPS: tuple[Any, Any, Any, Any, Any, Any] | None = None


def load_eval_deps() -> tuple[Any, Any, Any, Any, Any, Any]:
    global _EVAL_DEPS
    if _EVAL_DEPS is None:
        import torch
        import transformers
        from qwen_vl_utils import process_vision_info
        from transformers import AutoConfig, AutoProcessor, set_seed

        _EVAL_DEPS = (torch, transformers, AutoConfig, AutoProcessor, set_seed, process_vision_info)
    return _EVAL_DEPS


def get_model_cls():
    _, transformers, _, _, _, _ = load_eval_deps()
    for name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq", "Qwen3VLForConditionalGeneration"):
        cls = getattr(transformers, name, None)
        if cls is not None:
            return cls
    raise RuntimeError("no Qwen3-VL compatible model loader found")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def first_turn(row: dict[str, Any], roles: set[str]) -> str:
    for turn in row.get("conversations", []):
        if str(turn.get("from", "")).lower() in roles:
            return str(turn.get("value", ""))
    return ""


def resolvable_media(row: dict[str, Any]) -> bool:
    for field in ("image", "video"):
        for item in row.get(field) or []:
            text = str(item)
            if text.startswith(("http://", "https://")) or os.path.exists(text):
                return True
    return False


def resolve_media_paths(row: dict[str, Any]) -> dict[str, Any]:
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
    return row


def media_key(row: dict[str, Any]) -> str:
    media = (row.get("image") or []) + (row.get("video") or [])
    return str(media[0]) if media else ""


def parse_dataset_counts(text: str | None) -> dict[str, int]:
    if not text:
        return {}
    counts: dict[str, int] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            name, count = part.split("=", 1)
        elif ":" in part:
            name, count = part.split(":", 1)
        else:
            raise ValueError(f"dataset count must use name=count: {part}")
        counts[name.strip()] = int(count.strip())
    return counts


def build_eval_set(args: argparse.Namespace) -> list[tuple[int, dict[str, Any]]]:
    used_media = set()
    if args.exclude_data and Path(args.exclude_data).exists():
        for row in load_jsonl(args.exclude_data):
            key = media_key(resolve_media_paths(row))
            if key:
                used_media.add(key)

    dataset_counts = parse_dataset_counts(args.dataset_counts)
    candidate_limits = {
        dataset: max(count, count * max(int(args.candidate_factor), 1))
        for dataset, count in dataset_counts.items()
    }
    candidates: list[tuple[int, dict[str, Any]]] = []
    candidates_by_dataset: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for idx, row in enumerate(load_jsonl(args.data)):
        fixed = resolve_media_paths(row)
        dataset = str(fixed.get("dataset"))
        if not dataset_counts and args.dataset != "all" and dataset != args.dataset:
            continue
        if dataset_counts and dataset not in dataset_counts:
            continue
        if dataset_counts and len(candidates_by_dataset.get(dataset, [])) >= candidate_limits[dataset]:
            if all(len(candidates_by_dataset.get(name, [])) >= limit for name, limit in candidate_limits.items()):
                break
            continue
        if media_key(fixed) in used_media:
            continue
        if not resolvable_media(fixed):
            continue
        answer = first_turn(fixed, {"gpt", "assistant"}).strip()
        if not answer or "<point>" in answer.lower() or "<box>" in answer.lower():
            continue
        if not first_turn(fixed, {"human", "user"}).strip():
            continue
        item = (idx, fixed)
        candidates.append(item)
        candidates_by_dataset.setdefault(dataset, []).append(item)

    rng = random.Random(args.seed)
    if dataset_counts:
        picked: list[tuple[int, dict[str, Any]]] = []
        for dataset, count in dataset_counts.items():
            bucket = candidates_by_dataset.get(dataset, [])
            if len(bucket) < count:
                raise ValueError(f"only {len(bucket)} candidates found for {dataset}, need {count}")
            rng.shuffle(bucket)
            picked.extend(bucket[:count])
        rng.shuffle(picked)
        if len(picked) != args.num_samples:
            raise ValueError(f"dataset counts sum to {len(picked)}, expected --num-samples {args.num_samples}")
        return picked
    if len(candidates) < args.num_samples:
        raise ValueError(f"only {len(candidates)} candidates found, need {args.num_samples}")
    rng.shuffle(candidates)
    return candidates[: args.num_samples]


def save_eval_set(args: argparse.Namespace) -> None:
    picked = build_eval_set(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for original_index, row in picked:
            out_row = dict(row)
            out_row["general_eval"] = {
                "source_data": args.data,
                "original_index": original_index,
                "seed": args.seed,
            }
            f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
    print(json.dumps({"out": str(out), "rows": len(picked)}, ensure_ascii=False), flush=True)


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("\n", " ")
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = 0
    pred_counts: dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in ref_tokens:
        if pred_counts.get(token, 0) > 0:
            common += 1
            pred_counts[token] -= 1
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def parse_options(prompt: str) -> dict[str, str]:
    if "Options:" not in prompt:
        return {}
    tail = prompt.split("Options:", 1)[1]
    matches = list(re.finditer(r"\b([A-D])\.\s*", tail))
    options: dict[str, str] = {}
    for idx, match in enumerate(matches):
        label = match.group(1).lower()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(tail)
        value = tail[start:end].strip()
        options[label] = value
    return options


def answer_candidates(prompt: str, reference: str) -> set[str]:
    candidates = {normalize_text(reference)}
    options = parse_options(prompt)
    ref_norm = normalize_text(reference)
    if ref_norm in options:
        candidates.add(normalize_text(options[ref_norm]))
    for label, value in options.items():
        if normalize_text(value) == ref_norm:
            candidates.add(label)
    return {c for c in candidates if c}


def relaxed_match(prompt: str, prediction: str, reference: str) -> bool:
    pred = normalize_text(prediction)
    candidates = answer_candidates(prompt, reference)
    if pred in candidates:
        return True
    pred_tokens = set(pred.split())
    for cand in candidates:
        cand_tokens = cand.split()
        if not cand_tokens:
            continue
        if len(cand_tokens) == 1:
            if cand_tokens[0] in pred_tokens:
                return True
        elif cand in pred:
            return True
    ref_norm = normalize_text(reference)
    if ref_norm in {"yes", "no"}:
        return bool(pred.split()[:1] == [ref_norm])
    return False


def build_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
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
    for video_path in row.get("video") or []:
        content.append({"type": "video", "video": str(video_path)})
    text = user_text.replace("<image>", "").replace("<video>", "").strip()
    if text:
        content.append({"type": "text", "text": text})
    messages = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": content})
    return messages


def generate_one(model, processor, row: dict[str, Any], device: torch.device, args: argparse.Namespace) -> str:
    torch, _, _, _, _, process_vision_info = load_eval_deps()
    messages = build_messages(row)
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


def run_eval(args: argparse.Namespace) -> None:
    torch, _, AutoConfig, AutoProcessor, _, _ = load_eval_deps()
    rows = load_jsonl(args.eval_set)
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
    for n, row in enumerate(rows, start=1):
        prompt = first_turn(row, {"human", "user"})
        reference = first_turn(row, {"gpt", "assistant"}).strip()
        prediction = generate_one(model, processor, row, device, args)
        exact = normalize_text(prediction) == normalize_text(reference)
        relaxed = relaxed_match(prompt, prediction, reference)
        result = {
            "sample_no": n,
            "source_index": row.get("general_eval", {}).get("original_index"),
            "dataset": row.get("dataset"),
            "image": row.get("image"),
            "prompt": prompt,
            "reference": reference,
            "prediction": prediction,
            "normalized_exact": exact,
            "relaxed_match": relaxed,
            "token_f1": token_f1(prediction, reference),
            "has_options": "Options:" in prompt,
        }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    summary = {
        "model": args.model,
        "eval_set": args.eval_set,
        "num_samples": len(results),
        "normalized_exact": sum(r["normalized_exact"] for r in results) / max(len(results), 1),
        "relaxed_match": sum(r["relaxed_match"] for r in results) / max(len(results), 1),
        "mean_token_f1": sum(float(r["token_f1"]) for r in results) / max(len(results), 1),
        "option_samples": sum(r["has_options"] for r in results),
        "results_path": str(out_path),
    }
    with open(out_path.with_suffix(".summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("[summary] " + json.dumps(summary, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    build = sub.add_parser("build")
    build.add_argument("--data", required=True)
    build.add_argument("--out", required=True)
    build.add_argument("--exclude-data", default=None)
    build.add_argument("--dataset", default="robo2vlm-1")
    build.add_argument("--dataset-counts", default=None)
    build.add_argument("--candidate-factor", type=int, default=10)
    build.add_argument("--num-samples", type=int, default=200)
    build.add_argument("--seed", type=int, default=20260520)

    ev = sub.add_parser("eval")
    ev.add_argument("--model", required=True)
    ev.add_argument("--eval-set", required=True)
    ev.add_argument("--out", required=True)
    ev.add_argument("--max-new-tokens", type=int, default=64)
    ev.add_argument("--device", default="cuda:0")
    ev.add_argument("--seed", type=int, default=20260520)

    args = parser.parse_args()
    if args.cmd == "build":
        save_eval_set(args)
    else:
        _, _, _, _, set_seed, _ = load_eval_deps()
        set_seed(args.seed)
        run_eval(args)


if __name__ == "__main__":
    main()
