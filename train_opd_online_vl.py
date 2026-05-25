#!/usr/bin/env python3
"""Online routed OPD for Qwen3-VL using teacher rollouts and top1 logits.

This script is intentionally close to the normal Trainer flow: the dataset and
collator build prompt-only batches, and the Trainer override only replaces the
loss with online teacher rollout + teacher-top1 distillation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
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


EXPERT_ROUTES = (
    "general_obj_expert",
    "region_expert",
    "robopoint_expert",
    "spatial_rel_expert",
    "general_reasoning_expert",
)
ROUTE_TO_ID = {name: idx for idx, name in enumerate(EXPERT_ROUTES)}
ID_TO_ROUTE = {value: key for key, value in ROUTE_TO_ID.items()}

ROUTE_ALIASES = {
    "obj": "general_obj_expert",
    "general_obj": "general_obj_expert",
    "reg": "region_expert",
    "region": "region_expert",
    "point": "robopoint_expert",
    "robopoint": "robopoint_expert",
    "spatial": "spatial_rel_expert",
    "spatial_rel": "spatial_rel_expert",
    "reasoning": "general_reasoning_expert",
    "general": "general_reasoning_expert",
    "general_reasoning": "general_reasoning_expert",
}

DEFAULT_TEACHER_PATHS = {
    "general_obj_expert": "/data/msz/models/seed0_general_obj_expert_100k_mb1_filtered_stdtrainer_maxnorm5_v1",
    "region_expert": "/data/msz/models/seed0_region_expert_100k_mb1_filtered_stdtrainer_maxnorm5_v1",
    "robopoint_expert": "/data/msz/models/seed0_robopoint_expert_100k_mb1_filtered_stdtrainer_maxnorm5_v1",
    "spatial_rel_expert": "/data/msz/models/seed0_spatial_rel_expert_100k_mb1_filtered_stdtrainer_maxnorm5_v1",
    "general_reasoning_expert": "/data/msz/models/seed0_general_reasoning_expert_100k_mb1_filtered_stdtrainer_maxnorm5_v1",
}


def canonical_route(route: Any) -> str:
    value = str(route or "").strip()
    if value in ROUTE_TO_ID:
        return value
    if value in ROUTE_ALIASES:
        return ROUTE_ALIASES[value]
    raise ValueError(f"unknown OPD route/expert: {route!r}")


def row_opd_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata") or {}
    if isinstance(metadata, dict) and isinstance(metadata.get("opd"), dict):
        return metadata["opd"]
    opd = row.get("opd") or {}
    return opd if isinstance(opd, dict) else {}


def route_sort_key(row: dict[str, Any]) -> tuple[int, str]:
    route = str(row.get("_opd_route", ""))
    return ROUTE_TO_ID.get(route, 999), route


def configure_left_padding(processor: Any) -> None:
    tokenizer = getattr(processor, "tokenizer", processor)
    for obj in (processor, tokenizer):
        if hasattr(obj, "padding_side"):
            obj.padding_side = "left"
    if hasattr(tokenizer, "pad_token_id") and getattr(tokenizer, "pad_token_id", None) is None:
        eos_token = getattr(tokenizer, "eos_token", None)
        if eos_token is not None and hasattr(tokenizer, "pad_token"):
            tokenizer.pad_token = eos_token


class OPDOnlineDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        world_size: int,
        per_device_batch_size: int,
        limit_samples: int | None = None,
        route_policy: str = "target",
        group_by_route: bool = True,
        shuffle_samples: bool = False,
        route_block_shuffle: bool = False,
        shuffle_seed: int = 20260520,
        pad_to_world_batch: bool = False,
    ):
        self.path = str(path)
        self.route_policy = route_policy
        self.group_by_route = group_by_route
        self.shuffle_samples = shuffle_samples
        self.route_block_shuffle = route_block_shuffle
        self.shuffle_seed = shuffle_seed
        self.padded_rows = 0
        self.route_counts: Counter[str] = Counter()
        self.schedule_route_counts: Counter[str] = Counter()
        self.category_counts: Counter[str] = Counter()
        self.format_counts: Counter[str] = Counter()
        self.candidate_len_counts: Counter[int] = Counter()
        self.raw_rows_seen = 0

        rows: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if limit_samples is not None and self.raw_rows_seen >= limit_samples:
                    break
                row = json.loads(line)
                self.raw_rows_seen += 1
                routed_rows = self._routed_rows(row, route_policy)
                rows.extend(routed_rows)

        if route_block_shuffle:
            rows = self._route_block_shuffle(rows, max(world_size * per_device_batch_size, 1), shuffle_seed)
        elif group_by_route:
            rows.sort(key=route_sort_key)
        elif shuffle_samples:
            rng = random.Random(shuffle_seed)
            rng.shuffle(rows)

        if pad_to_world_batch and rows:
            pad_to = max(world_size * per_device_batch_size, 1)
            original_len = len(rows)
            while len(rows) % pad_to:
                rows.append(dict(rows[(len(rows) - original_len) % original_len]))
                self.padded_rows += 1

        if not rows:
            raise ValueError(f"no OPD rows found in {path}")
        self.rows = rows
        self.schedule_route_counts.update(str(row.get("_opd_route")) for row in rows)

    def _routed_rows(self, row: dict[str, Any], route_policy: str) -> list[dict[str, Any]]:
        opd = row_opd_metadata(row)
        candidates_raw = opd.get("candidate_experts") or []
        candidates = [canonical_route(item) for item in candidates_raw] if candidates_raw else []
        target = opd.get("target_expert")

        if not target and opd.get("teacher"):
            teacher = str(opd["teacher"])
            if teacher == "both":
                candidates = [canonical_route("obj"), canonical_route("reg")]
                target = candidates[0]
            else:
                target = canonical_route(teacher)

        if target:
            target_route = canonical_route(target)
        elif candidates:
            target_route = candidates[0]
        else:
            raise ValueError(f"row has no target_expert/candidate_experts/teacher route: {opd}")

        if route_policy == "target":
            routes = [(target_route, 1.0)]
        elif route_policy == "candidates":
            active = candidates or [target_route]
            weight = 1.0 / len(active)
            routes = [(route, weight) for route in active]
        else:
            raise ValueError(f"bad route_policy={route_policy!r}")

        category = str(opd.get("sample_category") or "unknown")
        expected_format = str(opd.get("expected_format") or "unknown")
        self.category_counts[category] += 1
        self.format_counts[expected_format] += 1
        self.candidate_len_counts[len(candidates)] += 1

        out: list[dict[str, Any]] = []
        for route, weight in routes:
            self.route_counts[route] += 1
            cloned = dict(row)
            cloned["_opd_route"] = route
            cloned["_opd_loss_weight"] = float(weight)
            cloned["_opd_target_expert"] = target_route
            cloned["_opd_candidate_experts"] = candidates
            cloned["_opd_sample_category"] = category
            cloned["_opd_expected_format"] = expected_format
            out.append(cloned)
        return out

    def _route_block_shuffle(self, rows: list[dict[str, Any]], block_size: int, seed: int) -> list[dict[str, Any]]:
        if block_size <= 1:
            rng = random.Random(seed)
            out = list(rows)
            rng.shuffle(out)
            return out

        rng = random.Random(seed)
        grouped: dict[str, list[dict[str, Any]]] = {route: [] for route in EXPERT_ROUTES}
        for row in rows:
            grouped[canonical_route(row.get("_opd_route"))].append(row)

        blocks: list[list[dict[str, Any]]] = []
        for route in EXPERT_ROUTES:
            route_rows = grouped[route]
            if not route_rows:
                continue
            rng.shuffle(route_rows)
            original_len = len(route_rows)
            while len(route_rows) % block_size:
                route_rows.append(dict(route_rows[(len(route_rows) - original_len) % original_len]))
                self.padded_rows += 1
            blocks.extend(route_rows[idx : idx + block_size] for idx in range(0, len(route_rows), block_size))
        rng.shuffle(blocks)
        return [row for block in blocks for row in block]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]

    def summary(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "raw_rows_seen": self.raw_rows_seen,
            "expanded_rows": len(self.rows),
            "padded_rows": self.padded_rows,
            "route_policy": self.route_policy,
            "group_by_route": self.group_by_route,
            "shuffle_samples": self.shuffle_samples,
            "route_block_shuffle": self.route_block_shuffle,
            "shuffle_seed": self.shuffle_seed,
            "route_counts": dict(self.route_counts),
            "schedule_route_counts": dict(self.schedule_route_counts),
            "sample_category_counts": dict(self.category_counts),
            "expected_format_counts": dict(self.format_counts),
            "candidate_len_counts": {str(k): v for k, v in self.candidate_len_counts.items()},
        }


class OPDPromptCollator(VisionConversationCollator):
    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        configure_left_padding(self.processor)
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
            route = canonical_route(example.get("_opd_route"))
            routes.append(ROUTE_TO_ID[route])
            weights.append(float(example.get("_opd_loss_weight", 1.0)))

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

    hf_deepspeed_config = None
    hf_deepspeed_module = None
    try:
        import transformers.integrations.deepspeed as hf_deepspeed_module

        weak_ref = getattr(hf_deepspeed_module, "_hf_deepspeed_config_weak_ref", None)
        hf_deepspeed_config = weak_ref() if weak_ref is not None else None
        hf_deepspeed_module.unset_hf_deepspeed_config()
    except Exception:
        hf_deepspeed_module = None

    try:
        import deepspeed

        zero_init_context = deepspeed.zero.Init(enabled=False)
    except Exception:
        zero_init_context = nullcontext()

    try:
        with zero_init_context:
            model = get_model_cls().from_pretrained(
                model_path,
                config=config,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
                attn_implementation="eager",
            ).to(device)
    finally:
        if hf_deepspeed_module is not None and hf_deepspeed_config is not None:
            hf_deepspeed_module.set_hf_deepspeed_config(hf_deepspeed_config)

    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


@dataclass
class TeacherHandle:
    route: str
    path: str
    engine: Any

    @property
    def module(self):
        return getattr(self.engine, "module", self.engine)


def load_teacher_zero3(model_path: str, route: str, ds_config: str, dtype: torch.dtype) -> TeacherHandle:
    import deepspeed
    from transformers.integrations import HfDeepSpeedConfig

    teacher_ds_config = load_teacher_ds_config(ds_config)
    # Keep a strong reference long enough for Transformers' zero3-aware loader.
    hf_ds_config = HfDeepSpeedConfig(teacher_ds_config)
    _HF_DS_CONFIG_HOLDERS.append(hf_ds_config)

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
        torch_dtype=dtype,
        attn_implementation="eager",
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    engine, _optimizer, _training_dataloader, _lr_scheduler = deepspeed.initialize(
        model=model,
        model_parameters=[],
        config=teacher_ds_config,
        dist_init_required=False,
    )
    engine.eval()
    engine.module.eval()
    return TeacherHandle(route=route, path=model_path, engine=engine)


_HF_DS_CONFIG_HOLDERS: list[Any] = []


def load_teacher_ds_config(ds_config: str) -> dict[str, Any]:
    with open(ds_config, "r", encoding="utf-8") as f:
        config = json.load(f)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if config.get("train_batch_size") == "auto":
        config["train_batch_size"] = max(world_size, 1)
    if config.get("train_micro_batch_size_per_gpu") == "auto":
        config["train_micro_batch_size_per_gpu"] = 1
    if config.get("gradient_accumulation_steps") == "auto":
        config["gradient_accumulation_steps"] = 1
    return config


class OPDOnlineTrainer(Trainer):
    def __init__(self, *args, teacher_paths: dict[str, str], teacher_load_mode: str, teacher_deepspeed: str | None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_paths = {canonical_route(k): v for k, v in teacher_paths.items()}
        self.teacher_load_mode = teacher_load_mode
        self.teacher_deepspeed = teacher_deepspeed
        self._teacher_route: str | None = None
        self._teacher_model: torch.nn.Module | None = None
        self._teacher_handles: dict[str, TeacherHandle] = {}
        self._teacher_load_counts: Counter[str] = Counter()
        self._route_loss_counts: Counter[str] = Counter()

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

    def _preload_zero3_teachers(self) -> None:
        if self._teacher_handles:
            return
        if not self.teacher_deepspeed:
            raise ValueError("--teacher-deepspeed is required for preloaded_zero3 teacher mode")
        dtype = torch.bfloat16 if self.args.bf16 else torch.float32
        for route in EXPERT_ROUTES:
            model_path = self.teacher_paths[route]
            if self.is_world_process_zero():
                print(f"[teacher-zero3] preloading route={route} path={model_path}", flush=True)
            handle = load_teacher_zero3(model_path, route, self.teacher_deepspeed, dtype)
            self._teacher_handles[route] = handle
            self._teacher_load_counts[route] += 1
        if self.is_world_process_zero():
            print("[teacher-zero3] all 5 teachers are resident and ZeRO-3 sharded", flush=True)

    def _teacher(self, route: str, device: torch.device):
        if self.teacher_load_mode == "preloaded_zero3":
            self._preload_zero3_teachers()
            return self._teacher_handles[route].module

        if self._teacher_route == route and self._teacher_model is not None:
            return self._teacher_model
        if self._teacher_model is not None:
            del self._teacher_model
            self._teacher_model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        model_path = self.teacher_paths[route]
        if self.is_world_process_zero():
            print(f"[teacher] loading route={route} path={model_path}", flush=True)
        self._teacher_model = load_teacher(model_path, device)
        self._teacher_route = route
        self._teacher_load_counts[route] += 1
        return self._teacher_model

    def unload_teacher(self) -> None:
        if self._teacher_handles:
            for handle in self._teacher_handles.values():
                del handle.engine
            self._teacher_handles.clear()
        if self._teacher_model is not None:
            del self._teacher_model
            self._teacher_model = None
        self._teacher_route = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _sample_ce_and_entropy(self, logits: torch.Tensor, target_labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shift_logits = logits[:, :-1, :]
        shift_targets = target_labels[:, 1:].to(logits.device)
        losses: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        for row_idx in range(shift_targets.shape[0]):
            mask = shift_targets[row_idx] != -100
            if not torch.any(mask):
                zero = shift_logits[row_idx, :1, :1].sum() * 0.0
                losses.append(zero)
                entropies.append(zero)
                continue
            row_logits = shift_logits[row_idx][mask]
            row_targets = shift_targets[row_idx][mask]
            log_probs = F.log_softmax(row_logits.float(), dim=-1)
            losses.append(F.nll_loss(log_probs, row_targets, reduction="mean"))
            entropies.append((-(log_probs.exp() * log_probs).sum(dim=-1)).mean())
        return torch.stack(losses), torch.stack(entropies)

    def _attention_mask_for_generated(
        self,
        prompt_attention_mask: torch.Tensor,
        generated: torch.Tensor,
    ) -> torch.Tensor:
        prompt_width = prompt_attention_mask.shape[1]
        generated_width = generated.shape[1]
        if generated_width <= prompt_width:
            return prompt_attention_mask[:, :generated_width].to(generated.device)
        suffix = torch.ones(
            (generated.shape[0], generated_width - prompt_width),
            dtype=prompt_attention_mask.dtype,
            device=generated.device,
        )
        return torch.cat([prompt_attention_mask.to(generated.device), suffix], dim=1)

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

    def log(self, logs: dict[str, float], *args, **kwargs) -> None:
        enriched = dict(logs)
        if hasattr(self, "_last_route_id"):
            enriched["opd_route_id"] = self._last_route_id
        if hasattr(self, "_last_response_tokens"):
            enriched["opd_response_tokens"] = self._last_response_tokens
        if hasattr(self, "_last_opd_loss"):
            enriched["opd_loss"] = self._last_opd_loss
        if hasattr(self, "_last_entropy"):
            enriched["entropy"] = self._last_entropy
            enriched["opd_entropy"] = self._last_entropy
        super().log(enriched, *args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        route_ids = inputs.pop("opd_route_ids")
        weights = inputs.pop("opd_loss_weights")
        _prompt_lens = inputs.pop("prompt_lens")
        if route_ids.numel() == 0:
            raise ValueError("empty OPD route ids")
        if not torch.all(route_ids == route_ids[0]):
            raise ValueError(
                "mixed teacher routes in one microbatch; keep --per-device-train-batch-size=1 "
                f"or bucket by route: {route_ids.detach().cpu().tolist()}"
            )

        route_id = int(route_ids[0].detach().cpu().item())
        if self.teacher_load_mode == "preloaded_zero3" and torch.distributed.is_available() and torch.distributed.is_initialized():
            route_minmax = torch.tensor([route_id, -route_id], device=inputs["input_ids"].device, dtype=torch.long)
            torch.distributed.all_reduce(route_minmax, op=torch.distributed.ReduceOp.MIN)
            min_route = int(route_minmax[0].detach().cpu().item())
            max_route = int(-route_minmax[1].detach().cpu().item())
            if min_route != max_route:
                raise RuntimeError(
                    "ZeRO-3 sharded teachers require every rank to enter the same teacher per microstep; "
                    f"got route range {min_route}..{max_route}. Use --route-block-shuffle."
                )
        route = ID_TO_ROUTE[route_id]
        device = inputs["input_ids"].device
        teacher = self._teacher(route, device)
        tokenizer = getattr(self.processing_class, "tokenizer", self.processing_class)

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.args.opd_max_new_tokens,
            "do_sample": self.args.opd_do_sample,
            "pad_token_id": getattr(tokenizer, "pad_token_id", None),
            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        }
        if self.teacher_load_mode == "preloaded_zero3":
            gen_kwargs["synced_gpus"] = True
        if self.args.opd_do_sample:
            gen_kwargs["temperature"] = self.args.opd_temperature
            gen_kwargs["top_p"] = self.args.opd_top_p

        with torch.inference_mode():
            generated_infer = teacher.generate(**inputs, **gen_kwargs)
            teacher_attention_mask = self._attention_mask_for_generated(inputs["attention_mask"], generated_infer)
            teacher_inputs = self._model_inputs_for_sequence(inputs, generated_infer, teacher_attention_mask)
            teacher_outputs = teacher(**teacher_inputs, use_cache=False)
            teacher_top1_infer = teacher_outputs.logits[:, :-1, :].argmax(dim=-1)

        generated = generated_infer.detach().clone()
        teacher_top1 = teacher_top1_infer.detach().clone()
        attention_mask = self._attention_mask_for_generated(inputs["attention_mask"], generated)

        target_labels = torch.full_like(generated, -100)
        prompt_width = inputs["input_ids"].shape[1]
        end = generated.shape[1]
        for row_idx in range(generated.shape[0]):
            start = prompt_width
            if end <= start:
                continue
            target_labels[row_idx, start:end] = teacher_top1[row_idx, start - 1 : end - 1]

        student_inputs = self._model_inputs_for_sequence(inputs, generated, attention_mask)
        outputs = model(**student_inputs, use_cache=False)
        sample_loss, sample_entropy = self._sample_ce_and_entropy(outputs.logits, target_labels)
        loss = (sample_loss * weights.to(sample_loss.device)).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite OPD online loss: {float(loss.detach().cpu())}")

        with torch.no_grad():
            self._last_route_id = float(route_id)
            self._last_response_tokens = float((target_labels != -100).sum(dim=1).float().mean().detach().cpu())
            self._last_opd_loss = float(loss.detach().cpu())
            self._last_entropy = float(sample_entropy.mean().detach().cpu())
            self._route_loss_counts[route] += int(route_ids.numel())
        return (loss, outputs) if return_outputs else loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("train", nargs="?")
    parser.add_argument("--model-name-or-path", "--model_name_or_path", dest="model_name_or_path", required=True)
    parser.add_argument("--data-path", "--data_path", dest="data_path", required=True)
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", required=True)
    parser.add_argument("--general-obj-teacher", "--general_obj_teacher", dest="general_obj_teacher", default=None)
    parser.add_argument("--region-teacher", "--region_teacher", dest="region_teacher", default=None)
    parser.add_argument("--robopoint-teacher", "--robopoint_teacher", dest="robopoint_teacher", default=None)
    parser.add_argument("--spatial-rel-teacher", "--spatial_rel_teacher", dest="spatial_rel_teacher", default=None)
    parser.add_argument(
        "--general-reasoning-teacher",
        "--general_reasoning_teacher",
        dest="general_reasoning_teacher",
        default=None,
    )
    parser.add_argument("--obj-teacher", "--obj_teacher", dest="obj_teacher", default=None)
    parser.add_argument("--reg-teacher", "--reg_teacher", dest="reg_teacher", default=None)
    parser.add_argument("--route-policy", "--route_policy", dest="route_policy", choices=["target", "candidates"], default="target")
    parser.add_argument("--group-by-route", "--group_by_route", dest="group_by_route", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shuffle-samples", "--shuffle_samples", dest="shuffle_samples", action="store_true")
    parser.add_argument("--route-block-shuffle", "--route_block_shuffle", dest="route_block_shuffle", action="store_true")
    parser.add_argument("--pad-to-world-batch", "--pad_to_world_batch", dest="pad_to_world_batch", action="store_true")
    parser.add_argument("--sequential-sampling", "--sequential_sampling", dest="sequential_sampling", action="store_true")
    parser.add_argument("--teacher-load-mode", "--teacher_load_mode", dest="teacher_load_mode", choices=["lazy_full", "preloaded_zero3"], default="lazy_full")
    parser.add_argument("--teacher-deepspeed", "--teacher_deepspeed", dest="teacher_deepspeed", default=None)
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
    parser.add_argument("--save-steps", "--save_steps", dest="save_steps", type=int, default=0)
    parser.add_argument("--save-total-limit", "--save_total_limit", dest="save_total_limit", type=int, default=None)
    parser.add_argument("--resume-from-checkpoint", "--resume_from_checkpoint", dest="resume_from_checkpoint", default=None)
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


def build_teacher_paths(args: argparse.Namespace) -> dict[str, str]:
    paths = dict(DEFAULT_TEACHER_PATHS)
    overrides = {
        "general_obj_expert": args.general_obj_teacher,
        "region_expert": args.region_teacher,
        "robopoint_expert": args.robopoint_teacher,
        "spatial_rel_expert": args.spatial_rel_teacher,
        "general_reasoning_expert": args.general_reasoning_teacher,
    }
    for route, value in overrides.items():
        if value:
            paths[route] = value
    if args.obj_teacher:
        paths["general_obj_expert"] = args.obj_teacher
    if args.reg_teacher:
        paths["region_expert"] = args.reg_teacher

    missing = [route for route, path in paths.items() if not Path(path).exists()]
    if missing:
        details = {route: paths[route] for route in missing}
        raise FileNotFoundError(f"missing OPD teacher paths: {details}")
    return paths


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    set_seed(args.seed)
    teacher_paths = build_teacher_paths(args)

    log(f"[opd-online] data={args.data_path}")
    log(f"[opd-online] student={args.model_name_or_path}")
    log(f"[opd-online] output={args.output_dir}")
    if args.teacher_load_mode == "preloaded_zero3" and not args.teacher_deepspeed:
        args.teacher_deepspeed = args.deepspeed
    log(
        f"[opd-online] route_policy={args.route_policy} group_by_route={args.group_by_route} "
        f"shuffle_samples={args.shuffle_samples} route_block_shuffle={args.route_block_shuffle}"
    )
    log(f"[opd-online] teacher_load_mode={args.teacher_load_mode} teacher_deepspeed={args.teacher_deepspeed}")
    for route in EXPERT_ROUTES:
        log(f"[opd-online] teacher[{route}]={teacher_paths[route]}")

    model, processor, _tokenizer = load_model_and_processor(args)
    configure_left_padding(processor)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    dataset = OPDOnlineDataset(
        args.data_path,
        world_size=world_size,
        per_device_batch_size=args.per_device_train_batch_size,
        limit_samples=args.limit_samples,
        route_policy=args.route_policy,
        group_by_route=args.group_by_route,
        shuffle_samples=args.shuffle_samples,
        route_block_shuffle=args.route_block_shuffle,
        shuffle_seed=args.seed,
        pad_to_world_batch=args.pad_to_world_batch,
    )
    collator = OPDPromptCollator(processor, args)

    effective_bs = world_size * args.per_device_train_batch_size * args.gradient_accumulation_steps
    expected_steps = math.ceil(len(dataset) / max(effective_bs, 1))
    log(
        "[opd-online] "
        f"raw_samples={dataset.raw_rows_seen} expanded_samples={len(dataset)} "
        f"world_size={world_size} effective_batch={effective_bs} expected_steps={expected_steps}"
    )
    log(f"[opd-online] dataset_summary={json.dumps(dataset.summary(), ensure_ascii=False)}")
    log("[opd-online] teacher rollout/logits are computed online; no precomputed logit cache")
    if args.save_steps > 0:
        log(
            "[opd-online] "
            f"save_strategy=steps save_steps={args.save_steps} "
            f"save_total_limit={args.save_total_limit}; final model also saved after train"
        )
    else:
        log("[opd-online] save_strategy=no; final model only")

    training_args = build_training_args(args)
    if args.save_steps > 0:
        training_args.save_strategy = "steps"
        training_args.save_steps = args.save_steps
        training_args.save_total_limit = args.save_total_limit
        training_args.save_only_model = False
    training_args.dataloader_drop_last = False
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
        teacher_paths=teacher_paths,
        teacher_load_mode=args.teacher_load_mode,
        teacher_deepspeed=args.teacher_deepspeed,
    )
    if args.resume_from_checkpoint:
        log(f"[opd-online] resume_from_checkpoint={args.resume_from_checkpoint}")
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    log(f"[opd-online] finished global_step={trainer.state.global_step} train_loss={result.training_loss}")
    trainer.unload_teacher()
    save_final_model(trainer, processor, args)
    if trainer.is_world_process_zero():
        extra = {
            "opd_data_path": args.data_path,
            "student_model_name_or_path": args.model_name_or_path,
            "teacher_paths": teacher_paths,
            "teacher_load_mode": args.teacher_load_mode,
            "teacher_deepspeed": args.teacher_deepspeed,
            "dataset_summary": dataset.summary(),
            "train_loss": float(result.training_loss),
            "last_route_id": getattr(trainer, "_last_route_id", None),
            "last_response_tokens": getattr(trainer, "_last_response_tokens", None),
            "last_opd_loss": getattr(trainer, "_last_opd_loss", None),
            "last_entropy": getattr(trainer, "_last_entropy", None),
            "save_steps": args.save_steps,
            "save_total_limit": args.save_total_limit,
            "save_only_model": bool(training_args.save_only_model),
            "teacher_load_counts": dict(trainer._teacher_load_counts),
            "route_loss_counts": dict(trainer._route_loss_counts),
            "opd_mode": "five_expert_online_teacher_rollout_top1",
        }
        with open(Path(args.output_dir) / "opd_online_run_summary.json", "w", encoding="utf-8") as f:
            json.dump(extra, f, ensure_ascii=False, indent=2)
    log("[done] OPD online training complete")


if __name__ == "__main__":
    main()
