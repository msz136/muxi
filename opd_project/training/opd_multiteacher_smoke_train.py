#!/usr/bin/env python3
"""50-step multi-teacher OPD smoke trainer for Qwen3-VL.

This is intentionally small and defensive. It verifies the data path, student
forward/backward/update path, expert3/expert4 teacher forward path, 5:5 teacher
loss blend, logging, and checkpointing without touching evaluation jobs.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import shutil
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


SYSTEM_FALLBACK = (
    "You are a helpful vision-language assistant. When the user asks for a "
    "location, answer with coordinates in the range 0 to 1000."
)


@dataclass
class StepResult:
    step: int
    prompt_id: str
    loss: float
    distill_loss: float
    hard_ce: float
    grad_norm: float
    active_tokens: int
    elapsed_sec: float


def log(msg: str) -> None:
    print(msg, flush=True)


def jdump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def write_jsonl(path: Path, obj: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(jdump(obj) + "\n")


def require_path(path: str | Path, label: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{label} not found: {p}")
    return p


def is_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "oom" in text or isinstance(exc, torch.cuda.OutOfMemoryError)


def canonical_image_path(path: str) -> str:
    if path.startswith("file://"):
        path = path[len("file://") :]
    return path


def load_clean_pointing_rows(path: Path, limit: int | None, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dropped = {"json": 0, "schema": 0, "missing_image": 0, "no_gt": 0}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                dropped["json"] += 1
                continue

            meta = item.get("metadata") or {}
            images = item.get("images") or []
            messages = item.get("messages") or []
            gt_points = item.get("gt_points") or []
            if meta.get("task_type") != "pointing" or not images or not messages:
                dropped["schema"] += 1
                continue
            if not gt_points:
                dropped["no_gt"] += 1
                continue
            image_path = canonical_image_path(str(images[0]))
            if not Path(image_path).exists():
                dropped["missing_image"] += 1
                continue
            item["_line_no"] = line_no
            rows.append(item)

    if not rows:
        raise RuntimeError(f"No usable pointing rows found in {path}; dropped={dropped}")
    rng = random.Random(seed)
    rng.shuffle(rows)
    if limit is not None and limit > 0:
        rows = rows[:limit]
    log(f"[data] usable_pointing_rows={len(rows)} dropped={dropped}")
    return rows


def target_from_gt(item: dict[str, Any], max_points: int) -> str:
    points = item.get("gt_points") or [[500, 500]]
    clean_points: list[list[int]] = []
    for point in points[:max_points]:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        x = max(0, min(1000, int(round(float(point[0])))))
        y = max(0, min(1000, int(round(float(point[1])))))
        clean_points.append([x, y])
    if not clean_points:
        clean_points = [[500, 500]]
    return "<point>" + json.dumps(clean_points, separators=(",", ":")) + "</point>"


def to_qwen_messages(item: dict[str, Any], include_answer: bool, max_points: int, min_pixels: int, max_pixels: int) -> list[dict[str, Any]]:
    messages = item.get("messages") or []
    sys_msg = next((m.get("content") for m in messages if m.get("role") == "system"), SYSTEM_FALLBACK)
    user_msg = next((m.get("content") for m in messages if m.get("role") == "user"), "")
    user_msg = str(user_msg).replace("<image>\n", "").replace("<image>", "").strip()
    image_path = canonical_image_path(str((item.get("images") or [])[0]))

    out: list[dict[str, Any]] = []
    if sys_msg:
        out.append({"role": "system", "content": str(sys_msg)})
    out.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"file://{image_path}",
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                },
                {"type": "text", "text": user_msg},
            ],
        }
    )
    if include_answer:
        out.append({"role": "assistant", "content": target_from_gt(item, max_points)})
    return out


def encode_sample(
    processor: Any,
    item: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    prompt_messages = to_qwen_messages(
        item,
        include_answer=False,
        max_points=args.max_points_per_target,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    full_messages = to_qwen_messages(
        item,
        include_answer=True,
        max_points=args.max_points_per_target,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    prompt_text = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    full_text = processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
    image_inputs, video_inputs = process_vision_info(full_messages)

    full_inputs = processor(
        text=[full_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    prompt_inputs = processor(
        text=[prompt_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )

    if "input_ids" not in full_inputs or full_inputs["input_ids"].shape[1] < 2:
        raise RuntimeError("encoded sample is too short")
    pixel_values = full_inputs.get("pixel_values")
    if pixel_values is not None and not torch.isfinite(pixel_values).all():
        raise RuntimeError("non-finite pixel_values")
    image_grid = full_inputs.get("image_grid_thw")
    if image_grid is not None and (image_grid <= 0).any():
        raise RuntimeError(f"invalid image_grid_thw={image_grid.tolist()}")

    labels = full_inputs["input_ids"].clone()
    prompt_len = int(prompt_inputs["attention_mask"].sum(dim=1).item())
    prompt_len = min(prompt_len, labels.shape[1] - 1)
    labels[:, :prompt_len] = -100
    labels[full_inputs["attention_mask"] == 0] = -100
    if int((labels[:, 1:] != -100).sum().item()) == 0:
        raise RuntimeError("sample has no active assistant target tokens")
    full_inputs["labels"] = labels
    return dict(full_inputs)


def move_inputs(inputs: dict[str, torch.Tensor], device: torch.device, include_labels: bool) -> dict[str, torch.Tensor]:
    moved: dict[str, torch.Tensor] = {}
    for key, value in inputs.items():
        if key == "labels" and not include_labels:
            continue
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def load_vl_model(path: str, device_index: int, *, train: bool) -> Qwen3VLForConditionalGeneration:
    log(f"[model] loading path={path} device=cuda:{device_index} train={train}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        path,
        dtype=torch.bfloat16,
        device_map={"": device_index},
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if train:
        model.train()
    else:
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
    return model


def configure_trainable(model: torch.nn.Module, scope: str) -> list[torch.nn.Parameter]:
    for param in model.parameters():
        param.requires_grad_(False)

    selected: list[tuple[str, torch.nn.Parameter]] = []
    for name, param in model.named_parameters():
        is_lm_head = name == "lm_head.weight" or name == "lm_head.bias" or name.startswith("lm_head.")
        is_final_norm = (
            name.endswith("language_model.norm.weight")
            or name.endswith("language_model.norm.bias")
            or name.endswith("model.norm.weight")
            or name.endswith("model.norm.bias")
        )
        is_nonvision = "visual" not in name and ".vision" not in name
        train_this = False
        if scope == "lm_head":
            train_this = is_lm_head
        elif scope == "head_norm":
            train_this = is_lm_head or is_final_norm
        elif scope == "all_nonvision":
            train_this = is_nonvision
        else:
            raise ValueError(f"unknown train_scope={scope}")
        if train_this:
            param.requires_grad_(True)
            selected.append((name, param))

    if not selected:
        raise RuntimeError(f"No trainable parameters selected for scope={scope}")
    total = sum(p.numel() for _, p in selected)
    log(f"[trainable] scope={scope} tensors={len(selected)} params={total:,}")
    log("[trainable] names=" + ",".join(name for name, _ in selected[:20]))
    return [p for _, p in selected]


def sanitize_trainable_params(model: torch.nn.Module, bad_log: Path, step: int) -> int:
    fixed = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            finite = torch.isfinite(param.data)
            if not finite.all():
                count = int((~finite).sum().item())
                param.data[~finite] = 0.0
                fixed += count
                write_jsonl(bad_log, {"event": "param_nan", "step": step, "name": name, "count": count})
    if fixed:
        log(f"[param_nan] sanitized_values={fixed} step={step}")
    return fixed


def sanitize_grads(model: torch.nn.Module, bad_log: Path, step: int) -> int:
    fixed = 0
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        finite = torch.isfinite(param.grad)
        if not finite.all():
            count = int((~finite).sum().item())
            param.grad.data[~finite] = 0.0
            fixed += count
            write_jsonl(bad_log, {"event": "grad_nan", "step": step, "name": name, "count": count})
    if fixed:
        log(f"[grad_nan] sanitized_values={fixed} step={step}")
    return fixed


def finite_scalar(value: Any) -> bool:
    try:
        if torch.is_tensor(value):
            value = float(value.detach().cpu().item())
        else:
            value = float(value)
        return math.isfinite(value)
    except Exception:
        return False


def optimizer_state_dict_cpu_copy(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    """Build a CPU checkpoint copy without mutating the live optimizer state."""
    state_dict = optimizer.state_dict()
    cpu_state: dict[Any, dict[str, Any]] = {}
    for state_id, state_values in state_dict.get("state", {}).items():
        copied_values: dict[str, Any] = {}
        for key, value in state_values.items():
            if torch.is_tensor(value):
                copied_values[key] = value.detach().cpu().clone()
            else:
                copied_values[key] = copy.deepcopy(value)
        cpu_state[state_id] = copied_values
    return {
        "state": cpu_state,
        "param_groups": copy.deepcopy(state_dict.get("param_groups", [])),
    }


def save_checkpoint(
    output_dir: Path,
    step: int,
    model: torch.nn.Module,
    processor: Any,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    result: StepResult,
) -> None:
    ckpt = output_dir / f"checkpoint-{step}"
    tmp = output_dir / f".checkpoint-{step}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    model.save_pretrained(
        tmp / "model",
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    processor.save_pretrained(tmp / "processor")

    torch.save(optimizer_state_dict_cpu_copy(optimizer), tmp / "optimizer.pt")

    if args.save_trainable_copy:
        trainable_state = {
            name: param.detach().cpu()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        torch.save(trainable_state, tmp / "trainable_state.pt")

    state = {
        "step": step,
        "save_type": "full_model_with_optimizer",
        "student": args.student,
        "teacher3": args.teacher3,
        "teacher4": args.teacher4,
        "train_scope": args.train_scope,
        "model_dir": "model",
        "processor_dir": "processor",
        "optimizer_state": "optimizer.pt",
        "loss": result.loss,
        "distill_loss": result.distill_loss,
        "hard_ce": result.hard_ce,
        "grad_norm": result.grad_norm,
        "prompt_id": result.prompt_id,
        "note": "Full student model plus optimizer state. Teachers are referenced by path, not copied.",
    }
    (tmp / "training_state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    if ckpt.exists():
        shutil.rmtree(ckpt)
    os.replace(tmp, ckpt)
    log(f"[checkpoint] saved={ckpt}")


def compute_losses(
    student: torch.nn.Module,
    teacher3: torch.nn.Module,
    teacher4: torch.nn.Module,
    encoded_cpu: dict[str, torch.Tensor],
    student_device: torch.device,
    teacher3_device: torch.device,
    teacher4_device: torch.device,
    hard_ce_coeff: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    student_inputs = move_inputs(encoded_cpu, student_device, include_labels=True)
    teacher3_inputs = move_inputs(encoded_cpu, teacher3_device, include_labels=False)
    teacher4_inputs = move_inputs(encoded_cpu, teacher4_device, include_labels=False)

    labels = student_inputs["labels"]
    student_outputs = student(**student_inputs)
    student_logits = student_outputs.logits
    shift_student = student_logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    active = shift_labels.ne(-100)
    active_tokens = int(active.sum().item())
    if active_tokens <= 0:
        raise RuntimeError("no active tokens after shift")

    active_student_logits = shift_student[active].float()
    active_labels = shift_labels[active]
    student_logp = F.log_softmax(active_student_logits, dim=-1)

    with torch.no_grad():
        teacher3_logits = teacher3(**teacher3_inputs).logits[:, :-1, :]
        active_t3 = active.to(teacher3_device)
        teacher3_probs = F.softmax(teacher3_logits[active_t3].float(), dim=-1).to(student_device)
        del teacher3_logits

        teacher4_logits = teacher4(**teacher4_inputs).logits[:, :-1, :]
        active_t4 = active.to(teacher4_device)
        teacher4_probs = F.softmax(teacher4_logits[active_t4].float(), dim=-1).to(student_device)
        del teacher4_logits

    if teacher3_probs.shape != teacher4_probs.shape or teacher3_probs.shape != student_logp.shape:
        raise RuntimeError(
            "vocab/shape mismatch: "
            f"student={tuple(student_logp.shape)} "
            f"teacher3={tuple(teacher3_probs.shape)} teacher4={tuple(teacher4_probs.shape)}"
        )
    teacher_probs = 0.5 * teacher3_probs + 0.5 * teacher4_probs
    distill_loss = -(teacher_probs * student_logp).sum(dim=-1).mean()
    hard_ce = F.cross_entropy(active_student_logits, active_labels)
    loss = distill_loss + hard_ce_coeff * hard_ce
    return loss, distill_loss.detach(), hard_ce.detach(), active_tokens


def train_one_step(
    step: int,
    item: dict[str, Any],
    processor: Any,
    student: torch.nn.Module,
    teacher3: torch.nn.Module,
    teacher4: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    devices: tuple[torch.device, torch.device, torch.device],
    args: argparse.Namespace,
    bad_log: Path,
) -> StepResult:
    start = time.time()
    student_device, teacher3_device, teacher4_device = devices
    encoded_cpu = encode_sample(processor, item, args)
    prompt_id = str(item.get("prompt_id") or f"line-{item.get('_line_no')}")

    sanitize_trainable_params(student, bad_log, step)
    optimizer.zero_grad(set_to_none=True)
    loss, distill_loss, hard_ce, active_tokens = compute_losses(
        student,
        teacher3,
        teacher4,
        encoded_cpu,
        student_device,
        teacher3_device,
        teacher4_device,
        args.hard_ce_coeff,
    )
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite loss={loss}")
    loss.backward()
    fixed_grads = sanitize_grads(student, bad_log, step)
    grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
        [p for p in student.parameters() if p.requires_grad and p.grad is not None],
        args.max_grad_norm,
    )
    grad_norm = float(grad_norm_tensor.detach().cpu().item())
    if fixed_grads or not finite_scalar(grad_norm):
        optimizer.zero_grad(set_to_none=True)
        write_jsonl(
            bad_log,
            {"event": "skip_optimizer", "step": step, "reason": "bad_grad", "grad_norm": grad_norm},
        )
        raise RuntimeError(f"bad_grad fixed={fixed_grads} grad_norm={grad_norm}")
    optimizer.step()
    sanitize_trainable_params(student, bad_log, step)
    return StepResult(
        step=step,
        prompt_id=prompt_id,
        loss=float(loss.detach().cpu().item()),
        distill_loss=float(distill_loss.cpu().item()),
        hard_ce=float(hard_ce.cpu().item()),
        grad_norm=grad_norm,
        active_tokens=active_tokens,
        elapsed_sec=time.time() - start,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", default="/data/msz/models/8b_base")
    parser.add_argument("--teacher3", default="/data/msz/models/expert3")
    parser.add_argument("--teacher4", default="/data/msz/models/expert4")
    parser.add_argument("--data", default="/data/msz/opd_project/data/prompt_pool_clean.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-limit", type=int, default=512)
    parser.add_argument("--max-retry-per-step", type=int, default=3)
    parser.add_argument("--max-bad-steps", type=int, default=20)
    parser.add_argument("--student-device", type=int, default=0)
    parser.add_argument("--teacher3-device", type=int, default=1)
    parser.add_argument("--teacher4-device", type=int, default=2)
    parser.add_argument("--train-scope", choices=["lm_head", "head_norm", "all_nonvision"], default="head_norm")
    parser.add_argument("--learning-rate", type=float, default=5e-7)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--hard-ce-coeff", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--min-pixels", type=int, default=50176)
    parser.add_argument("--max-pixels", type=int, default=50176)
    parser.add_argument("--max-points-per-target", type=int, default=8)
    parser.add_argument("--max-shard-size", default="4GB")
    parser.add_argument("--save-trainable-copy", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/MACA device is required for this smoke test")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_log = output_dir / "train_metrics.jsonl"
    bad_log = output_dir / "bad_batches.jsonl"
    run_state = {
        "event": "start",
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    write_jsonl(metrics_log, run_state)
    log("[start] " + jdump(run_state))

    require_path(args.student, "student")
    require_path(args.teacher3, "teacher3")
    require_path(args.teacher4, "teacher4")
    data_path = require_path(args.data, "data")
    rows = load_clean_pointing_rows(data_path, args.sample_limit, args.seed)

    processor = AutoProcessor.from_pretrained(args.student, trust_remote_code=True)
    student = load_vl_model(args.student, args.student_device, train=True)
    if args.gradient_checkpointing:
        student.config.use_cache = False
        if hasattr(student, "gradient_checkpointing_enable"):
            student.gradient_checkpointing_enable()
        if hasattr(student, "enable_input_require_grads"):
            student.enable_input_require_grads()
        log("[model] gradient_checkpointing=enabled")
    teacher3 = load_vl_model(args.teacher3, args.teacher3_device, train=False)
    teacher4 = load_vl_model(args.teacher4, args.teacher4_device, train=False)
    trainable = configure_trainable(student, args.train_scope)
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)

    devices = (
        torch.device(f"cuda:{args.student_device}"),
        torch.device(f"cuda:{args.teacher3_device}"),
        torch.device(f"cuda:{args.teacher4_device}"),
    )

    completed_steps = 0
    bad_steps = 0
    cursor = 0
    last_result: StepResult | None = None
    while completed_steps < args.max_steps:
        item = rows[cursor % len(rows)]
        cursor += 1
        next_step = completed_steps + 1
        ok = False
        last_error = ""
        for attempt in range(1, args.max_retry_per_step + 1):
            try:
                result = train_one_step(
                    next_step,
                    item,
                    processor,
                    student,
                    teacher3,
                    teacher4,
                    optimizer,
                    devices,
                    args,
                    bad_log,
                )
                ok = True
                break
            except Exception as exc:
                last_error = repr(exc)
                bad_steps += 1
                write_jsonl(
                    bad_log,
                    {
                        "event": "step_error",
                        "step": next_step,
                        "attempt": attempt,
                        "prompt_id": item.get("prompt_id"),
                        "error": last_error,
                        "traceback": traceback.format_exc(limit=8),
                    },
                )
                log(f"[bad_step] step={next_step} attempt={attempt} error={last_error}")
                optimizer.zero_grad(set_to_none=True)
                if is_oom(exc) and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if attempt >= args.max_retry_per_step:
                    break

        if not ok:
            if bad_steps > args.max_bad_steps:
                raise RuntimeError(f"too many bad steps: {bad_steps}; last_error={last_error}")
            continue

        completed_steps += 1
        last_result = result
        metric = {
            "event": "step",
            "step": result.step,
            "prompt_id": result.prompt_id,
            "loss": result.loss,
            "distill_loss": result.distill_loss,
            "hard_ce": result.hard_ce,
            "grad_norm": result.grad_norm,
            "active_tokens": result.active_tokens,
            "elapsed_sec": result.elapsed_sec,
            "bad_steps": bad_steps,
        }
        write_jsonl(metrics_log, metric)
        if completed_steps % args.log_every == 0:
            log(
                "[step] "
                f"{completed_steps}/{args.max_steps} "
                f"loss={result.loss:.6f} distill={result.distill_loss:.6f} "
                f"hard_ce={result.hard_ce:.6f} grad_norm={result.grad_norm:.6f} "
                f"tokens={result.active_tokens} sec={result.elapsed_sec:.2f} "
                f"id={result.prompt_id}"
            )
        if completed_steps % args.save_steps == 0:
            save_checkpoint(output_dir, completed_steps, student, processor, optimizer, args, result)

    if last_result is None:
        raise RuntimeError("no completed steps")
    if completed_steps % args.save_steps != 0:
        save_checkpoint(output_dir, completed_steps, student, processor, optimizer, args, last_result)
    final = {
        "event": "complete",
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "completed_steps": completed_steps,
        "bad_steps": bad_steps,
        "output_dir": str(output_dir),
    }
    write_jsonl(metrics_log, final)
    log("[complete] " + jdump(final))


if __name__ == "__main__":
    main()
