#!/usr/bin/env python3
"""Final-only expert SFT for Qwen3-VL semantic navigation box grounding."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
import transformers
from torch.utils.data import Dataset
from transformers import AutoConfig, AutoProcessor, Trainer, TrainerCallback, TrainingArguments, set_seed

try:
    from transformers.integrations import HfDeepSpeedConfig
except Exception:  # pragma: no cover - depends on installed transformers build
    HfDeepSpeedConfig = None


DEFAULT_SYSTEM_PROMPT = (
    "You are a semantic navigation grounding assistant. Given an image and target "
    "object information, return the target object's bounding box in coordinates "
    "from 0 to 1000. Return only <box>[[x1,y1],[x2,y2]]</box>."
)


def is_rank0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def log(msg: str) -> None:
    if is_rank0():
        print(msg, flush=True)


class JsonlConversationDataset(Dataset):
    def __init__(self, path: str | Path, limit_samples: int | None = None):
        self.path = str(path)
        self.rows: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.rows.append(json.loads(line))
                if limit_samples and len(self.rows) >= limit_samples:
                    break
        if not self.rows:
            raise ValueError(f"no usable JSONL rows found in {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


class VisionConversationCollator:
    def __init__(self, processor: Any, args: argparse.Namespace):
        from qwen_vl_utils import process_vision_info

        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.args = args
        self.process_vision_info = process_vision_info
        self.empty_label_fixes = 0

    def _extract_turns(self, example: dict[str, Any]) -> tuple[str, str, str]:
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
        if not user_text:
            raise ValueError("sample has no user turn")
        if not answer_text:
            raise ValueError("sample has no assistant answer")
        return system_text, user_text, answer_text

    def _content_from_sample(self, example: dict[str, Any], user_text: str) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        for image_path in example.get("image") or []:
            content.append(
                {
                    "type": "image",
                    "image": str(image_path),
                    "min_pixels": self.args.min_pixels,
                    "max_pixels": self.args.max_pixels,
                }
            )
        for video_path in example.get("video") or []:
            content.append(
                {
                    "type": "video",
                    "video": str(video_path),
                    "min_pixels": self.args.min_pixels,
                    "max_pixels": self.args.max_pixels,
                    "min_frames": self.args.video_min_frames,
                    "max_frames": self.args.video_max_frames,
                    "fps": 1.0,
                }
            )
        text = user_text.replace("<image>", "").replace("<video>", "").strip()
        if text:
            content.append({"type": "text", "text": text})
        return content

    def _messages(self, example: dict[str, Any], include_answer: bool) -> list[dict[str, Any]]:
        system_text, user_text, answer_text = self._extract_turns(example)
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": self._content_from_sample(example, user_text)},
        ]
        if include_answer:
            messages.append({"role": "assistant", "content": answer_text})
        return messages

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        full_messages = [self._messages(example, include_answer=True) for example in batch]
        prompt_messages = [self._messages(example, include_answer=False) for example in batch]

        texts_full = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
            for msg in full_messages
        ]
        texts_prompt = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in prompt_messages
        ]

        image_inputs, video_inputs = self.process_vision_info(full_messages)
        image_inputs = image_inputs if image_inputs else None
        video_inputs = video_inputs if video_inputs else None

        enc = self.processor(
            text=texts_full,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            truncation=True,
            max_length=self.args.model_max_length,
            return_tensors="pt",
        )
        prompt_enc = self.processor(
            text=texts_prompt,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            truncation=True,
            max_length=self.args.model_max_length,
            return_tensors="pt",
        )

        labels = enc["input_ids"].clone()
        prompt_lens = prompt_enc["attention_mask"].sum(dim=1)
        attention_mask = enc["attention_mask"]
        for row_idx in range(labels.shape[0]):
            prompt_len = min(int(prompt_lens[row_idx].item()), labels.shape[1])
            labels[row_idx, :prompt_len] = -100
            labels[row_idx, attention_mask[row_idx] == 0] = -100
            if not torch.any(labels[row_idx] != -100):
                valid_positions = torch.nonzero(attention_mask[row_idx], as_tuple=False).flatten()
                if len(valid_positions) == 0:
                    raise ValueError("sample produced no valid tokens")
                pos = int(valid_positions[-1].item())
                labels[row_idx, pos] = enc["input_ids"][row_idx, pos]
                self.empty_label_fixes += 1

        for key, value in enc.items():
            if torch.is_tensor(value) and value.is_floating_point() and not torch.isfinite(value).all():
                raise FloatingPointError(f"non-finite tensor produced by processor: {key}")

        enc["labels"] = labels
        return dict(enc)


class AbortOnNonFiniteLog(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
        for key in ("loss", "grad_norm", "learning_rate"):
            if not logs or key not in logs:
                continue
            value = logs[key]
            try:
                finite = math.isfinite(float(value))
            except Exception:
                finite = True
            if not finite:
                raise FloatingPointError(f"non-finite metric at step {state.global_step}: {key}={value}")


def get_model_cls():
    for name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq", "Qwen3VLForConditionalGeneration"):
        cls = getattr(transformers, name, None)
        if cls is not None:
            log(f"[model] loader={name}")
            return cls
    raise RuntimeError("no Qwen3-VL compatible model loader found in transformers")


def freeze_parameters(model: torch.nn.Module, args: argparse.Namespace) -> None:
    for name, param in model.named_parameters():
        lower = name.lower()
        is_vision = any(token in lower for token in ("visual", "vision"))
        is_projector = any(token in lower for token in ("merger", "mm_projector", "multi_modal_projector"))
        if is_vision and not args.tune_mm_vision and not is_projector:
            param.requires_grad = False
        if is_projector and not args.tune_mm_mlp:
            param.requires_grad = False

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100.0 * trainable / max(total, 1)
    log(f"[model] trainable={trainable:,} / total={total:,} ({pct:.2f}%)")


def load_model_and_processor(args: argparse.Namespace):
    if args.deepspeed and HfDeepSpeedConfig is not None:
        HfDeepSpeedConfig(args.deepspeed)
        log(f"[deepspeed] config={args.deepspeed}")

    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if hasattr(config, "use_cache"):
        config.use_cache = False
    if hasattr(config, "_attn_implementation"):
        config._attn_implementation = "eager"
    if hasattr(config, "attn_implementation"):
        config.attn_implementation = "eager"

    model = get_model_cls().from_pretrained(
        args.model_name_or_path,
        config=config,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype="auto",
        attn_implementation="eager",
    )
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
        log("[model] gradient_checkpointing=enabled")
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    freeze_parameters(model, args)
    return model, processor, tokenizer


def build_training_args(args: argparse.Namespace) -> TrainingArguments:
    kwargs: dict[str, Any] = {}
    if args.gradient_checkpointing:
        kwargs["gradient_checkpointing"] = True
        kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

    return TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_strategy="no",
        save_only_model=True,
        save_safetensors=True,
        bf16=args.bf16,
        deepspeed=args.deepspeed,
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,
        report_to=[],
        optim=args.optim,
        disable_tqdm=False,
        seed=args.seed,
        data_seed=args.seed,
        **kwargs,
    )


def save_final_model(trainer: Trainer, processor: Any, args: argparse.Namespace) -> None:
    trainer.accelerator.wait_for_everyone()
    if not trainer.is_world_process_zero():
        trainer.accelerator.wait_for_everyone()
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_to_save = trainer.model
    if hasattr(trainer.model_wrapped, "module"):
        model_to_save = trainer.model_wrapped.module
    elif hasattr(model_to_save, "module"):
        model_to_save = model_to_save.module

    log(f"[save] writing final model only to {output_dir}")
    model_to_save.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    processor.save_pretrained(output_dir)
    trainer.state.save_to_json(str(output_dir / "trainer_state.json"))
    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "data_path": args.data_path,
                "model_name_or_path": args.model_name_or_path,
                "num_train_epochs": args.num_train_epochs,
                "per_device_train_batch_size": args.per_device_train_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "learning_rate": args.learning_rate,
                "save_policy": "final_model_only_no_intermediate_checkpoints",
                "transformers_version": transformers.__version__,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    trainer.accelerator.wait_for_everyone()


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-name-or-path", "--model_name_or_path", dest="model_name_or_path", required=True)
    parser.add_argument("--data-path", "--data_path", dest="data_path", required=True)
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", required=True)
    parser.add_argument("--deepspeed", "--deepspeed_config", dest="deepspeed", default=None)
    parser.add_argument("--num-train-epochs", "--num_train_epochs", dest="num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", "--max_steps", dest="max_steps", type=int, default=-1)
    parser.add_argument("--per-device-train-batch-size", "--per_device_train_batch_size", dest="per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", "--gradient_accumulation_steps", dest="gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning-rate", "--learning_rate", dest="learning_rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", "--weight_decay", dest="weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", "--warmup_ratio", dest="warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", "--max_grad_norm", dest="max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lr-scheduler-type", "--lr_scheduler_type", dest="lr_scheduler_type", default="cosine")
    parser.add_argument("--logging-steps", "--logging_steps", dest="logging_steps", type=int, default=1)
    parser.add_argument("--model-max-length", "--model_max_length", "--max_seq_length", dest="model_max_length", type=int, default=16384)
    parser.add_argument("--min-pixels", "--min_pixels", dest="min_pixels", type=int, default=50176)
    parser.add_argument("--max-pixels", "--max_pixels", dest="max_pixels", type=int, default=50176)
    parser.add_argument("--video-min-frames", "--video_min_frames", dest="video_min_frames", type=int, default=32)
    parser.add_argument("--video-max-frames", "--video_max_frames", dest="video_max_frames", type=int, default=32)
    parser.add_argument("--dataloader-num-workers", "--dataloader_num_workers", dest="dataloader_num_workers", type=int, default=4)
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument("--max-shard-size", "--max_shard_size", dest="max_shard_size", default="4GB")
    parser.add_argument("--limit-samples", "--limit_samples", dest="limit_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient-checkpointing", "--gradient_checkpointing", dest="gradient_checkpointing", action="store_true")
    parser.add_argument("--tune-mm-vision", "--tune_mm_vision", dest="tune_mm_vision", action="store_true")
    parser.add_argument("--tune-mm-mlp", "--tune_mm_mlp", dest="tune_mm_mlp", action=argparse.BooleanOptionalAction, default=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    train_parser = sub.add_parser("train")
    add_common_args(train_parser)
    inspect_parser = sub.add_parser("inspect")
    add_common_args(inspect_parser)
    args, _ = parser.parse_known_args()
    return args


def inspect_data(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    _model_unused, processor, _tokenizer_unused = None, None, None
    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    dataset = JsonlConversationDataset(args.data_path, limit_samples=args.per_device_train_batch_size)
    collator = VisionConversationCollator(processor, args)
    batch = collator([dataset[i] for i in range(min(len(dataset), args.per_device_train_batch_size))])
    shapes = {k: tuple(v.shape) for k, v in batch.items() if torch.is_tensor(v)}
    valid_labels = int((batch["labels"] != -100).sum().item())
    log(f"[inspect] samples={len(dataset)} shapes={shapes} valid_labels={valid_labels}")


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    log(f"[env] transformers={transformers.__version__}")
    log(f"[data] path={args.data_path}")
    log(f"[output] dir={args.output_dir}")

    model, processor, _tokenizer = load_model_and_processor(args)
    dataset = JsonlConversationDataset(args.data_path, limit_samples=args.limit_samples)
    collator = VisionConversationCollator(processor, args)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    effective_bs = world_size * args.per_device_train_batch_size * args.gradient_accumulation_steps
    expected_steps = math.ceil(len(dataset) / max(effective_bs, 1))
    log(f"[train] samples={len(dataset)} world_size={world_size} effective_batch={effective_bs} expected_steps={expected_steps}")
    log("[train] save_strategy=no; final model will be saved after trainer.train()")

    trainer = Trainer(
        model=model,
        args=build_training_args(args),
        train_dataset=dataset,
        data_collator=collator,
        processing_class=processor,
        callbacks=[AbortOnNonFiniteLog()],
    )
    trainer.train()
    log(f"[train] finished global_step={trainer.state.global_step}")
    if collator.empty_label_fixes:
        log(f"[data] empty_label_fixes={collator.empty_label_fixes}")
    save_final_model(trainer, processor, args)
    log("[done] final-only expert SFT complete")


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    if args.command == "inspect":
        inspect_data(args)
    elif args.command == "train":
        train(args)
    else:  # pragma: no cover
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
