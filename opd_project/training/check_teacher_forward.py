#!/usr/bin/env python3
"""Run a tiny teacher-model forward/generation sanity check.

This is not an eval script. It loads one or a few prompt-pool samples and checks
that the teacher can preprocess images and generate without crashing.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


def load_samples(path: str, limit: int) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            images = item.get("images", [])
            if images and all((not p.startswith("/")) or os.path.exists(p) for p in images):
                rows.append(item)
            if len(rows) >= limit:
                break
    if not rows:
        raise RuntimeError(f"No usable samples found in {path}")
    return rows


def to_qwen_messages(item: dict) -> list[dict]:
    sys_msg = next((m["content"] for m in item["messages"] if m["role"] == "system"), None)
    user_msg = next((m["content"] for m in item["messages"] if m["role"] == "user"), "")
    user_msg = user_msg.replace("<image>\n", "").replace("<image>", "").strip()
    messages: list[dict] = []
    if sys_msg:
        messages.append({"role": "system", "content": sys_msg})
    content = [{"type": "image", "image": f"file://{item['images'][0]}"}]
    content.append({"type": "text", "text": user_msg})
    messages.append({"role": "user", "content": content})
    return messages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", default="/data/msz/opd_project/data/prompt_pool_clean.jsonl")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    print(f"[teacher-forward] model={args.model}")
    print(f"[teacher-forward] data={args.data}")
    samples = load_samples(args.data, args.limit)

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.eval()

    for idx, sample in enumerate(samples):
        messages = to_qwen_messages(sample)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        ).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        response = processor.decode(generated, skip_special_tokens=True)
        print(json.dumps({
            "sample_index": idx,
            "prompt_id": sample.get("prompt_id"),
            "response": response,
        }, ensure_ascii=False))

    print("[teacher-forward] OK")


if __name__ == "__main__":
    main()
