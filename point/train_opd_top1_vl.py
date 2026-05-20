#!/usr/bin/env python3
"""Train Qwen3-VL with routed top1 OPD losses from object/region experts."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import Trainer, TrainerCallback, set_seed

from expert_sft import (
    AbortOnNonFiniteLog,
    VisionConversationCollator,
    build_training_args,
    load_model_and_processor,
    log,
    save_final_model,
)


class OPDTop1Dataset(Dataset):
    def __init__(self, path: str | Path, limit_samples: int | None = None):
        self.rows: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                self._validate(row)
                self.rows.append(row)
                if limit_samples and len(self.rows) >= limit_samples:
                    break
        if not self.rows:
            raise ValueError(f"no OPD rows found in {path}")

    @staticmethod
    def _validate(row: dict[str, Any]) -> None:
        opd = row.get("opd") or {}
        teacher = opd.get("teacher")
        if teacher in {"obj", "both"} and "obj_top1_ids" not in opd:
            raise ValueError("missing obj_top1_ids")
        if teacher in {"reg", "both"} and "reg_top1_ids" not in opd:
            raise ValueError("missing reg_top1_ids")
        if teacher not in {"obj", "reg", "both"}:
            raise ValueError(f"bad teacher route: {teacher}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


class OPDTop1Collator(VisionConversationCollator):
    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        enc = super().__call__(batch)
        labels = enc["labels"]
        obj_top1 = torch.full_like(labels, -100)
        reg_top1 = torch.full_like(labels, -100)
        loss_weights = torch.zeros((labels.shape[0], 2), dtype=torch.float32)

        for row_idx, example in enumerate(batch):
            opd = example.get("opd") or {}
            positions = (labels[row_idx] != -100).nonzero(as_tuple=False).flatten().tolist()
            teacher = opd.get("teacher")
            if teacher in {"obj", "both"}:
                ids = [int(v) for v in opd.get("obj_top1_ids", [])]
                if len(ids) != len(positions):
                    raise ValueError(f"obj top1 length mismatch row={row_idx}: {len(ids)} != {len(positions)}")
                for pos, token_id in zip(positions, ids):
                    obj_top1[row_idx, pos] = token_id
            if teacher in {"reg", "both"}:
                ids = [int(v) for v in opd.get("reg_top1_ids", [])]
                if len(ids) != len(positions):
                    raise ValueError(f"reg top1 length mismatch row={row_idx}: {len(ids)} != {len(positions)}")
                for pos, token_id in zip(positions, ids):
                    reg_top1[row_idx, pos] = token_id

            if teacher == "obj":
                loss_weights[row_idx, 0] = 1.0
            elif teacher == "reg":
                loss_weights[row_idx, 1] = 1.0
            elif teacher == "both":
                loss_weights[row_idx, 0] = 0.5
                loss_weights[row_idx, 1] = 0.5

        enc["obj_top1_labels"] = obj_top1
        enc["reg_top1_labels"] = reg_top1
        enc["opd_loss_weights"] = loss_weights
        return enc


class OPDMetricCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, model=None, **kwargs):  # type: ignore[override]
        trainer = kwargs.get("trainer")
        if logs is None or trainer is None:
            return


class OPDTop1Trainer(Trainer):
    def _sample_ce(self, logits: torch.Tensor, target_labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[:, :-1, :]
        shift_targets = target_labels[:, 1:].to(logits.device)
        losses: list[torch.Tensor] = []
        for row_idx in range(shift_targets.shape[0]):
            mask = shift_targets[row_idx] != -100
            if not torch.any(mask):
                losses.append(shift_logits[row_idx, :1, :1].sum() * 0.0)
                continue
            row_logits = shift_logits[row_idx][mask]
            row_targets = shift_targets[row_idx][mask]
            losses.append(F.cross_entropy(row_logits.float(), row_targets, reduction="mean"))
        return torch.stack(losses)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        obj_top1 = inputs.pop("obj_top1_labels")
        reg_top1 = inputs.pop("reg_top1_labels")
        weights = inputs.pop("opd_loss_weights")
        inputs.pop("labels", None)

        outputs = model(**inputs, use_cache=False)
        obj_loss = self._sample_ce(outputs.logits, obj_top1)
        reg_loss = self._sample_ce(outputs.logits, reg_top1)
        sample_loss = weights[:, 0].to(obj_loss.device) * obj_loss + weights[:, 1].to(reg_loss.device) * reg_loss
        loss = sample_loss.mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite OPD loss: {float(loss.detach().cpu())}")

        with torch.no_grad():
            obj_mask = weights[:, 0] > 0
            reg_mask = weights[:, 1] > 0
            self._last_obj_loss = float(obj_loss[obj_mask.to(obj_loss.device)].mean().detach().cpu()) if obj_mask.any() else None
            self._last_reg_loss = float(reg_loss[reg_mask.to(reg_loss.device)].mean().detach().cpu()) if reg_mask.any() else None
            self._last_opd_loss = float(loss.detach().cpu())
        return (loss, outputs) if return_outputs else loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("train", nargs="?")
    parser.add_argument("--model-name-or-path", "--model_name_or_path", dest="model_name_or_path", required=True)
    parser.add_argument("--data-path", "--data_path", dest="data_path", required=True)
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", required=True)
    parser.add_argument("--deepspeed", "--deepspeed_config", dest="deepspeed", default=None)
    parser.add_argument("--num-train-epochs", "--num_train_epochs", dest="num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", "--max_steps", dest="max_steps", type=int, default=-1)
    parser.add_argument("--per-device-train-batch-size", "--per_device_train_batch_size", dest="per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", "--gradient_accumulation_steps", dest="gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning-rate", "--learning_rate", dest="learning_rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", "--weight_decay", dest="weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", "--warmup_ratio", dest="warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", "--max_grad_norm", dest="max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lr-scheduler-type", "--lr_scheduler_type", dest="lr_scheduler_type", default="cosine")
    parser.add_argument("--logging-steps", "--logging_steps", dest="logging_steps", type=int, default=1)
    parser.add_argument("--model-max-length", "--model_max_length", dest="model_max_length", type=int, default=16384)
    parser.add_argument("--min-pixels", "--min_pixels", dest="min_pixels", type=int, default=50176)
    parser.add_argument("--max-pixels", "--max_pixels", dest="max_pixels", type=int, default=50176)
    parser.add_argument("--video-min-frames", "--video_min_frames", dest="video_min_frames", type=int, default=32)
    parser.add_argument("--video-max-frames", "--video_max_frames", dest="video_max_frames", type=int, default=32)
    parser.add_argument("--dataloader-num-workers", "--dataloader_num_workers", dest="dataloader_num_workers", type=int, default=4)
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument("--max-shard-size", "--max_shard_size", dest="max_shard_size", default="4GB")
    parser.add_argument("--limit-samples", "--limit_samples", dest="limit_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient-checkpointing", "--gradient_checkpointing", dest="gradient_checkpointing", action="store_true")
    parser.add_argument("--tune-mm-vision", "--tune_mm_vision", dest="tune_mm_vision", action="store_true")
    parser.add_argument("--tune-mm-mlp", "--tune_mm_mlp", dest="tune_mm_mlp", action=argparse.BooleanOptionalAction, default=True)
    args, _ = parser.parse_known_args()
    return args


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    set_seed(args.seed)
    log(f"[opd] data={args.data_path}")
    log(f"[opd] student={args.model_name_or_path}")
    log(f"[opd] output={args.output_dir}")

    model, processor, _tokenizer = load_model_and_processor(args)
    dataset = OPDTop1Dataset(args.data_path, limit_samples=args.limit_samples)
    collator = OPDTop1Collator(processor, args)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    effective_bs = world_size * args.per_device_train_batch_size * args.gradient_accumulation_steps
    expected_steps = math.ceil(len(dataset) / max(effective_bs, 1))
    log(f"[opd] samples={len(dataset)} world_size={world_size} effective_batch={effective_bs} expected_steps={expected_steps}")
    log("[opd] save_strategy=no; final model only")

    trainer = OPDTop1Trainer(
        model=model,
        args=build_training_args(args),
        train_dataset=dataset,
        data_collator=collator,
        processing_class=processor,
        callbacks=[AbortOnNonFiniteLog()],
    )
    result = trainer.train()
    log(f"[opd] finished global_step={trainer.state.global_step} train_loss={result.training_loss}")
    save_final_model(trainer, processor, args)
    if trainer.is_world_process_zero():
        extra = {
            "opd_data_path": args.data_path,
            "student_model_name_or_path": args.model_name_or_path,
            "train_loss": float(result.training_loss),
            "last_obj_loss": getattr(trainer, "_last_obj_loss", None),
            "last_reg_loss": getattr(trainer, "_last_reg_loss", None),
            "last_opd_loss": getattr(trainer, "_last_opd_loss", None),
        }
        with open(Path(args.output_dir) / "opd_run_summary.json", "w", encoding="utf-8") as f:
            json.dump(extra, f, ensure_ascii=False, indent=2)
    log("[done] OPD top1 training complete")


if __name__ == "__main__":
    main()
