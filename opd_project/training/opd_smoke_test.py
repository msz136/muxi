#!/usr/bin/env python3
"""Minimal 8B student forward/backward smoke test for OPD readiness.

This does not run evaluation and does not require slime/sglang. It verifies the
student can consume the cleaned OPD prompt format, compute a response-token
loss, backpropagate finite gradients, and optionally take one tiny optimizer
step on trainable non-vision parameters.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


def load_first_pointing(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            meta = item.get("metadata", {})
            images = item.get("images", [])
            if meta.get("task_type") != "pointing":
                continue
            if not images or any(p.startswith("/") and not os.path.exists(p) for p in images):
                continue
            if item.get("gt_points"):
                return item
    raise RuntimeError(f"No usable pointing sample with gt_points found in {path}")


def target_from_gt(item: dict) -> str:
    points = item.get("gt_points") or [[500, 500]]
    return "<point>" + json.dumps(points[:8], separators=(",", ":")) + "</point>"


def to_qwen_messages(item: dict, *, include_answer: bool) -> list[dict]:
    sys_msg = next((m["content"] for m in item["messages"] if m["role"] == "system"), None)
    user_msg = next((m["content"] for m in item["messages"] if m["role"] == "user"), "")
    user_msg = user_msg.replace("<image>\n", "").replace("<image>", "").strip()
    messages: list[dict] = []
    if sys_msg:
        messages.append({"role": "system", "content": sys_msg})
    messages.append({
        "role": "user",
        "content": [
            {"type": "image", "image": f"file://{item['images'][0]}"},
            {"type": "text", "text": user_msg},
        ],
    })
    if include_answer:
        messages.append({"role": "assistant", "content": target_from_gt(item)})
    return messages


def freeze_vision(model: torch.nn.Module) -> None:
    for name, param in model.named_parameters():
        if ".visual." in name or name.startswith("visual.") or name.startswith("model.visual."):
            param.requires_grad_(False)


def finite_grad_stats(model: torch.nn.Module) -> tuple[int, float]:
    count = 0
    max_abs = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        if not torch.isfinite(grad).all():
            raise RuntimeError("Non-finite gradient detected")
        count += 1
        max_abs = max(max_abs, float(grad.abs().max().item()))
    return count, max_abs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/data/msz/models/8b_base")
    parser.add_argument("--data", default="/data/msz/opd_project/data/prompt_pool_clean.jsonl")
    parser.add_argument("--lr", type=float, default=1e-8)
    parser.add_argument("--optimizer-step", action="store_true")
    parser.add_argument("--freeze-vision", action="store_true", default=True)
    args = parser.parse_args()

    print(f"[opd-smoke] model={args.model}")
    print(f"[opd-smoke] data={args.data}")
    sample = load_first_pointing(args.data)
    print(f"[opd-smoke] prompt_id={sample.get('prompt_id')}")
    print(f"[opd-smoke] target={target_from_gt(sample)}")

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    if args.freeze_vision:
        freeze_vision(model)
    model.train()
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    prompt_messages = to_qwen_messages(sample, include_answer=False)
    full_messages = to_qwen_messages(sample, include_answer=True)
    prompt_text = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    full_text = processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)

    image_inputs, video_inputs = process_vision_info(full_messages)
    full_inputs = processor(
        text=[full_text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
    ).to(model.device)
    prompt_inputs = processor(
        text=[prompt_text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    labels = full_inputs["input_ids"].clone()
    prompt_len = min(prompt_inputs["input_ids"].shape[1], labels.shape[1] - 1)
    labels[:, :prompt_len] = -100
    full_inputs["labels"] = labels

    optimizer = None
    if args.optimizer_step:
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=args.lr)
        optimizer.zero_grad(set_to_none=True)

    outputs = model(**full_inputs)
    loss = outputs.loss
    if loss is None or not torch.isfinite(loss):
        raise RuntimeError(f"Invalid loss: {loss}")
    print(f"[opd-smoke] loss={float(loss.detach().cpu()):.6f}")
    loss.backward()
    grad_count, grad_max_abs = finite_grad_stats(model)
    print(f"[opd-smoke] grad_tensors={grad_count} grad_max_abs={grad_max_abs:.6e}")
    if optimizer is not None:
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        print("[opd-smoke] optimizer_step=ok")
    print("[opd-smoke] OK")


if __name__ == "__main__":
    main()
