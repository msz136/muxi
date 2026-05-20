#!/usr/bin/env python3
"""Online routed OPD for Qwen3-VL using teacher rollouts and top1 logits."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import transformers
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoConfig, Trainer, set_seed

from expert_sft import (
    AbortOnNonFiniteLog,
    VisionConversationCollator,
    build_training_args,
    get_model_cls,
    load_model_and_processor,
    log,
    save_final_model,
)


ROUTE_TO_ID = {"obj": 0, "reg": 1}
ID_TO_ROUTE = {value: key for key, value in ROUTE_TO_ID.items()}


class OPDOnlineDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        world_size: int,
        per_device_batch_size: int,
        limit_samples: int | None = None,
    ):
        grouped: dict[str, list[dict[str, Any]]] = {"obj": [], "reg": []}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                opd = row.get("opd") or {}
                teacher = opd.get("teacher")
                if teacher == "both":
                    grouped["obj"].append(self._with_route(row, "obj", 0.5))
                    grouped["reg"].append(self._with_route(row, "reg", 0.5))
                elif teacher in {"obj", "reg"}:
                    grouped[teacher].append(self._with_route(row, teacher, 1.0))
                else:
                    raise ValueError(f"bad teacher route: {teacher}")

        rows: list[dict[str, Any]] = []
        pad_to = max(world_size * per_device_batch_size, 1)
        for route in ("obj", "reg"):
            route_rows = grouped[route]
            if not route_rows:
                continue
            while len(route_rows) % pad_to:
                route_rows.append(dict(route_rows[len(route_rows) % len(grouped[route])]))
            rows.extend(route_rows)

        if limit_samples:
            rows = rows[:limit_samples]
        if not rows:
            raise ValueError(f"no OPD rows found in {path}")
        self.rows = rows

    @staticmethod
    def _with_route(row: dict[str, Any], teacher_route: str, loss_weight: float) -> dict[str, Any]:
        out = dict(row)
        opd = dict(out.get("opd") or {})
        opd["teacher_route"] = teacher_route
        opd["loss_weight"] = loss_weight
        out["opd"] = opd
        return out

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


class OPDPromptCollator(VisionConversationCollator):
    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        prompt_messages = [self._messages(example, include_answer=False) for example in batch]
        texts_prompt = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in prompt_messages
        ]

        image_inputs, video_inputs = self.process_vision_info(prompt_messages)
        image_inputs = image_inputs if image_inputs else None
        video_inputs = video_inputs if video_inputs else None

        enc = self.processor(
            text=texts_prompt,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            truncation=True,
            max_length=self.args.model_max_length,
            return_tensors="pt",
        )
        for key, value in enc.items():
            if torch.is_tensor(value) and value.is_floating_point() and not torch.isfinite(value).all():
                raise FloatingPointError(f"non-finite tensor produced by processor: {key}")

        routes: list[int] = []
        weights: list[float] = []
        for example in batch:
            opd = example.get("opd") or {}
            route = str(opd.get("teacher_route"))
            if route not in ROUTE_TO_ID:
                raise ValueError(f"bad teacher_route: {route}")
            routes.append(ROUTE_TO_ID[route])
            weights.append(float(opd.get("loss_weight", 1.0)))

        enc["prompt_lens"] = enc["attention_mask"].sum(dim=1).to(torch.long)
        enc["opd_route_ids"] = torch.tensor(routes, dtype=torch.long)
        enc["opd_loss_weights"] = torch.tensor(weights, dtype=torch.float32)
        return dict(enc)


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
    for param in model.parameters():
        param.requires_grad_(False)
    return model


class OPDOnlineTrainer(Trainer):
    def __init__(self, *args, obj_teacher_path: str, reg_teacher_path: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_paths = {"obj": obj_teacher_path, "reg": reg_teacher_path}
        self._teacher_route: str | None = None
        self._teacher_model: torch.nn.Module | None = None

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        sampler = (
            DistributedSampler(self.train_dataset, shuffle=False)
            if self.args.world_size > 1
            else SequentialSampler(self.train_dataset)
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def _teacher(self, route: str, device: torch.device):
        if self._teacher_route == route and self._teacher_model is not None:
            return self._teacher_model
        if self._teacher_model is not None:
            del self._teacher_model
            self._teacher_model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if self.is_world_process_zero():
            print(f"[teacher] loading route={route} path={self.teacher_paths[route]}", flush=True)
        self._teacher_model = load_teacher(self.teacher_paths[route], device)
        self._teacher_route = route
        return self._teacher_model

    def unload_teacher(self) -> None:
        if self._teacher_model is not None:
            del self._teacher_model
            self._teacher_model = None
        self._teacher_route = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

    def _model_inputs_for_sequence(
        self,
        prompt_inputs: dict[str, torch.Tensor],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        out = {}
        for key, value in prompt_inputs.items():
            if key in {"input_ids", "attention_mask"}:
                continue
            out[key] = value
        out["input_ids"] = input_ids
        out["attention_mask"] = attention_mask
        return out

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        route_ids = inputs.pop("opd_route_ids")
        weights = inputs.pop("opd_loss_weights")
        prompt_lens = inputs.pop("prompt_lens")
        if route_ids.numel() == 0:
            raise ValueError("empty OPD route ids")
        if not torch.all(route_ids == route_ids[0]):
            raise ValueError(f"mixed teacher routes in one microbatch: {route_ids.detach().cpu().tolist()}")

        route = ID_TO_ROUTE[int(route_ids[0].detach().cpu().item())]
        device = inputs["input_ids"].device
        teacher = self._teacher(route, device)
        tokenizer = getattr(self.processing_class, "tokenizer", self.processing_class)

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.args.opd_max_new_tokens,
            "do_sample": self.args.opd_do_sample,
            "pad_token_id": getattr(tokenizer, "pad_token_id", None),
            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        }
        if self.args.opd_do_sample:
            gen_kwargs["temperature"] = self.args.opd_temperature
            gen_kwargs["top_p"] = self.args.opd_top_p

        with torch.inference_mode():
            generated_infer = teacher.generate(**inputs, **gen_kwargs)
            teacher_attention_mask = torch.ones_like(
                generated_infer,
                dtype=inputs["attention_mask"].dtype,
                device=device,
            )
            teacher_inputs = self._model_inputs_for_sequence(inputs, generated_infer, teacher_attention_mask)
            teacher_outputs = teacher(**teacher_inputs, use_cache=False)
            teacher_top1_infer = teacher_outputs.logits[:, :-1, :].argmax(dim=-1)

        generated = generated_infer.detach().clone()
        teacher_top1 = teacher_top1_infer.detach().clone()
        attention_mask = torch.ones_like(generated, dtype=inputs["attention_mask"].dtype, device=device)

        target_labels = torch.full_like(generated, -100)
        for row_idx in range(generated.shape[0]):
            start = int(prompt_lens[row_idx].detach().cpu().item())
            end = int(attention_mask[row_idx].sum().detach().cpu().item())
            if end <= start:
                continue
            target_labels[row_idx, start:end] = teacher_top1[row_idx, start - 1 : end - 1]

        student_inputs = self._model_inputs_for_sequence(inputs, generated, attention_mask)
        outputs = model(**student_inputs, use_cache=False)
        sample_loss = self._sample_ce(outputs.logits, target_labels)
        loss = (sample_loss * weights.to(sample_loss.device)).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite OPD online loss: {float(loss.detach().cpu())}")

        with torch.no_grad():
            self._last_route = route
            self._last_response_tokens = float((target_labels != -100).sum(dim=1).float().mean().detach().cpu())
            self._last_opd_loss = float(loss.detach().cpu())
        return (loss, outputs) if return_outputs else loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("train", nargs="?")
    parser.add_argument("--model-name-or-path", "--model_name_or_path", dest="model_name_or_path", required=True)
    parser.add_argument("--data-path", "--data_path", dest="data_path", required=True)
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", required=True)
    parser.add_argument("--obj-teacher", "--obj_teacher", dest="obj_teacher", required=True)
    parser.add_argument("--reg-teacher", "--reg_teacher", dest="reg_teacher", required=True)
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
    parser.add_argument("--dataloader-num-workers", "--dataloader_num_workers", dest="dataloader_num_workers", type=int, default=0)
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument("--max-shard-size", "--max_shard_size", dest="max_shard_size", default="4GB")
    parser.add_argument("--limit-samples", "--limit_samples", dest="limit_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--opd-max-new-tokens", "--opd_max_new_tokens", dest="opd_max_new_tokens", type=int, default=48)
    parser.add_argument("--opd-do-sample", "--opd_do_sample", dest="opd_do_sample", action="store_true")
    parser.add_argument("--opd-temperature", "--opd_temperature", dest="opd_temperature", type=float, default=0.7)
    parser.add_argument("--opd-top-p", "--opd_top_p", dest="opd_top_p", type=float, default=0.9)
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
    log(f"[opd-online] data={args.data_path}")
    log(f"[opd-online] student={args.model_name_or_path}")
    log(f"[opd-online] obj_teacher={args.obj_teacher}")
    log(f"[opd-online] reg_teacher={args.reg_teacher}")
    log(f"[opd-online] output={args.output_dir}")

    model, processor, _tokenizer = load_model_and_processor(args)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    dataset = OPDOnlineDataset(
        args.data_path,
        world_size=world_size,
        per_device_batch_size=args.per_device_train_batch_size,
        limit_samples=args.limit_samples,
    )
    collator = OPDPromptCollator(processor, args)

    effective_bs = world_size * args.per_device_train_batch_size * args.gradient_accumulation_steps
    expected_steps = math.ceil(len(dataset) / max(effective_bs, 1))
    log(f"[opd-online] expanded_samples={len(dataset)} world_size={world_size} effective_batch={effective_bs} expected_steps={expected_steps}")
    log("[opd-online] teacher rollout/logits are computed online; no precomputed logit cache")
    log("[opd-online] save_strategy=no; final model only")

    training_args = build_training_args(args)
    training_args.dataloader_drop_last = True
    training_args.opd_max_new_tokens = args.opd_max_new_tokens  # type: ignore[attr-defined]
    training_args.opd_do_sample = args.opd_do_sample  # type: ignore[attr-defined]
    training_args.opd_temperature = args.opd_temperature  # type: ignore[attr-defined]
    training_args.opd_top_p = args.opd_top_p  # type: ignore[attr-defined]

    trainer = OPDOnlineTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        processing_class=processor,
        callbacks=[AbortOnNonFiniteLog()],
        obj_teacher_path=args.obj_teacher,
        reg_teacher_path=args.reg_teacher,
    )
    result = trainer.train()
    log(f"[opd-online] finished global_step={trainer.state.global_step} train_loss={result.training_loss}")
    trainer.unload_teacher()
    save_final_model(trainer, processor, args)
    if trainer.is_world_process_zero():
        extra = {
            "opd_data_path": args.data_path,
            "student_model_name_or_path": args.model_name_or_path,
            "obj_teacher": args.obj_teacher,
            "reg_teacher": args.reg_teacher,
            "train_loss": float(result.training_loss),
            "last_route": getattr(trainer, "_last_route", None),
            "last_response_tokens": getattr(trainer, "_last_response_tokens", None),
            "last_opd_loss": getattr(trainer, "_last_opd_loss", None),
            "opd_mode": "online_teacher_rollout_top1",
        }
        with open(Path(args.output_dir) / "opd_online_run_summary.json", "w", encoding="utf-8") as f:
            json.dump(extra, f, ensure_ascii=False, indent=2)
    log("[done] OPD online training complete")


if __name__ == "__main__":
    main()
