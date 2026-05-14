#!/usr/bin/env python3
"""Expert SFT training for Qwen3-VL-8B-Instruct with multi-dataset grounding + VQA data.

Features:
- Per-batch OOM handling: skips bad batches instead of crashing
- Batch timeout: skips batches that take too long (hung I/O)
- Auto batch-size probing: falls back 8 -> 6 -> 4 -> 2 -> 1
- Smoke mode: runs first N batches to validate config
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

# ── MACA NCCL workaround: disable all_gather at module level ──────────
# Must happen BEFORE any Trainer import — trainer.py has module-level
# `from trainer_pt_utils import distributed_broadcast_scalars` caching.
def _patch_maca_nccl():
    import torch
    def _safe_broadcast_scalars(scalars, num_total_examples=None, use_sum=False, device=None):
        """No-op replacement: return tensor with scalar values so callers
        that do .sum().item() get a valid result."""
        if isinstance(scalars, (float, int)):
            return torch.tensor(float(scalars), device=device)
        if hasattr(scalars, 'sum'):
            return scalars
        return torch.tensor([float(s) for s in scalars], device=device)

    from transformers import trainer_pt_utils as _tpu
    _tpu.distributed_broadcast_scalars = _safe_broadcast_scalars
    import transformers.trainer as _tr
    if hasattr(_tr, 'distributed_broadcast_scalars'):
        _tr.distributed_broadcast_scalars = _safe_broadcast_scalars
_patch_maca_nccl()


def jdump(x: object) -> str:
    return json.dumps(x, ensure_ascii=False)


def log(msg: str) -> None:
    print(msg, flush=True)


# ── constants ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a helpful vision-language assistant. "
    "When the user asks for a location, answer with coordinates "
    "in the range 0 to 1000."
)

# ── dataset ────────────────────────────────────────────────────────────


class JsonlDataset:
    def __init__(self, path: Path | str):
        self.rows: list[dict] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    self.rows.append(json.loads(line))
                except Exception:
                    pass
        if not self.rows:
            raise SystemExit(f"no usable samples in {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        return self.rows[i]


# ── collator ───────────────────────────────────────────────────────────


class SafeCollator:
    """OOM-safe collator: processes full batch through processor at once for correct vision padding."""

    def __init__(self, processor, tok, bad_samples_path: str, args):
        self.processor = processor
        self.tok = tok
        self.args = args
        self.bad_path = bad_samples_path
        self.bad = open(bad_samples_path, "a", encoding="utf-8")
        self.skipped = 0
        self.timed_out = 0
        self.batch_timeout = int(getattr(args, "batch_timeout", 0) or 0)

    def _build_messages(self, ex: dict, with_answer: bool) -> dict | None:
        try:
            cs = ex["conversations"]
            sys_val = cs[0]["value"]
            usr = cs[1]["value"]
            ans = cs[2]["value"] if len(cs) > 2 else ""
        except (KeyError, IndexError):
            return None
        content: list[dict] = []
        for p in ex.get("image", []):
            content.append({"type": "image", "image": p})
        for p in ex.get("video", []):
            content.append({
                "type": "video", "video": p,
                "max_pixels": self.args.max_pixels,
                "min_pixels": self.args.min_pixels,
                "max_frames": self.args.video_max_frames,
                "min_frames": self.args.video_min_frames,
                "fps": 1.0,
            })
        content.append({"type": "text", "text": usr.replace("<image>", "").replace("<video>", "").strip()})
        msgs = [
            {"role": "system", "content": sys_val},
            {"role": "user", "content": content},
        ]
        if with_answer:
            msgs.append({"role": "assistant", "content": ans})
        return msgs

    def _try_load_images(self, messages: dict) -> list | None:
        """Try to load all images for a sample. Returns list of PIL Images,
        or None if any image failed to load (sample should be skipped)."""
        import concurrent.futures
        def _load():
            try:
                from qwen_vl_utils import process_vision_info
                imgs, _vids = process_vision_info([messages])
                if imgs:
                    return list(imgs)
            except Exception:
                pass
            # Fallback: manual loading, return None if any fail
            from PIL import Image
            import requests as _req
            from io import BytesIO
            result = []
            for c in messages[1].get("content", []):
                val = c.get("image") if isinstance(c, dict) else None
                if not isinstance(val, str):
                    continue
                try:
                    if val.startswith("http://") or val.startswith("https://"):
                        resp = _req.get(val, timeout=2)
                        resp.raise_for_status()
                        result.append(Image.open(BytesIO(resp.content)).convert("RGB"))
                    else:
                        if not Path(val).exists() and not val.startswith("/"):
                            # RoboPoint relative paths → resolve against images dir
                            val = f"/data/msz/dataset/RoboPoint/images/{val}"
                        if Path(val).exists():
                            img = Image.open(val).convert("RGB")
                            # Check for NaN/Inf pixels (corrupted images)
                            import numpy as np
                            arr = np.array(img, dtype=np.float32)
                            if np.any(~np.isfinite(arr)):
                                return None  # corrupted image
                            result.append(img)
                        else:
                            return None  # local file not found
                except Exception:
                    return None  # download or load failed
            return result

        ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            return ex.submit(_load).result(timeout=15)
        except (concurrent.futures.TimeoutError, Exception):
            return None
        finally:
            ex.shutdown(wait=False)

    def _dummy_batch(self, n_skipped: int = 1) -> dict:
        """Minimal batch with one valid token — produces finite loss/gradients."""
        dummy = [[{"role": "user", "content": [{"type": "text", "text": "skip"}]},
                   {"role": "assistant", "content": [{"type": "text", "text": "skip"}]}]]
        dummy_texts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False) for m in dummy]
        enc = self.processor(text=dummy_texts, images=None, videos=None,
                             padding=True, truncation=True, max_length=64, return_tensors="pt")
        labels = enc["input_ids"].clone()
        labels[:] = -100
        # Keep ONE token valid (first non-pad after prompt) to get finite loss
        pad_id = getattr(self.tok, "pad_token_id", 0) or 0
        for i in range(labels.shape[0]):
            row = enc["input_ids"][i]
            valid_mask = row != pad_id
            valid_mask[:2] = False  # skip BOS + first tokens (prompt)
            idx = valid_mask.nonzero(as_tuple=True)[0]
            if len(idx) > 0:
                labels[i, idx[0].item()] = row[idx[0].item()]
        out = dict(enc)
        out["labels"] = labels
        self.skipped += n_skipped
        return out

    def _collate_impl(self, batch: list[dict]) -> dict:
        """Core collation: build messages, load images, tokenize."""
        import torch

        all_imgs_flat = []
        full_msgs, prompt_msgs = [], []

        for ex in batch:
            try:
                mf = self._build_messages(ex, True)
                mp = self._build_messages(ex, False)
                if mf is None or mp is None:
                    continue
                img_slots = [c for c in mf[1].get("content", [])
                            if isinstance(c, dict) and c.get("type") == "image"]
                if img_slots:
                    loaded = self._try_load_images(mf)
                    if loaded is None or len(loaded) != len(img_slots):
                        self.skipped += 1
                        continue
                    all_imgs_flat.extend(loaded)
                full_msgs.append(mf)
                prompt_msgs.append(mp)
            except Exception as e:
                self.skipped += 1
                self.bad.write(jdump({"err": repr(e), "sample": ex}) + "\n")
                self.bad.flush()

        if not full_msgs:
            return self._dummy_batch(len(batch))

        texts_full = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False) for m in full_msgs]
        texts_prompt = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in prompt_msgs]

        enc = self.processor(
            text=texts_full, images=all_imgs_flat or None, videos=None,
            padding=True, truncation=True,
            max_length=self.args.model_max_length, return_tensors="pt",
        )
        # Check for NaN/Inf in image tensors (corrupted images / bad transforms → skip batch)
        import torch
        pv = enc.get("pixel_values")
        if pv is not None and not torch.isfinite(pv).all():
            return self._dummy_batch(len(batch))
        # Check image_grid_thw for zero dimensions (can cause division by zero in vision encoder)
        igt = enc.get("image_grid_thw")
        if pv is not None and igt is not None and (igt <= 0).any():
            return self._dummy_batch(len(batch))

        encp = self.processor(
            text=texts_prompt, images=all_imgs_flat or None, videos=None,
            padding=True, truncation=True,
            max_length=self.args.model_max_length, return_tensors="pt",
        )

        prompt_lens = encp["attention_mask"].sum(dim=1)
        labels = enc["input_ids"].clone()
        B = labels.shape[0]
        for i in range(B):
            pl = int(prompt_lens[i].item())
            labels[i, :pl] = -100
            pad_mask = enc["attention_mask"][i] == 0
            labels[i, pad_mask] = -100

        out = dict(enc)
        out["labels"] = labels
        return out

    def __call__(self, batch: list[dict]) -> dict:
        if not self.batch_timeout:
            return self._collate_impl(batch)
        import concurrent.futures
        ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            return ex.submit(self._collate_impl, batch).result(timeout=self.batch_timeout)
        except concurrent.futures.TimeoutError:
            self.timed_out += 1
            log(f"[collator] batch timed out after {self.batch_timeout}s "
                f"(total_timeouts={self.timed_out}, skipped={self.skipped})")
            return self._dummy_batch(len(batch))
        finally:
            ex.shutdown(wait=False)

    def close(self) -> None:
        self.bad.close()


# ── model loader ──────────────────────────────────────────────────────


def load_model_and_processor(args):
    import torch
    from transformers import AutoConfig, AutoProcessor, set_seed

    set_seed(42)

    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path, trust_remote_code=True,
        min_pixels=args.min_pixels, max_pixels=args.max_pixels,
    )
    tok = getattr(processor, "tokenizer", processor)
    if getattr(tok, "pad_token_id", None) is None:
        tok.pad_token = tok.eos_token

    cfg = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if hasattr(cfg, "use_cache"):
        cfg.use_cache = False

    # AutoModel detection
    try:
        from transformers import AutoModelForImageTextToText as AutoModel
    except Exception:
        try:
            from transformers import AutoModelForVision2Seq as AutoModel
        except Exception:
            from transformers import AutoModelForCausalLM as AutoModel

    model = AutoModel.from_pretrained(
        args.model_name_or_path, config=cfg, trust_remote_code=True,
        torch_dtype="auto", low_cpu_mem_usage=True, attn_implementation="eager",
    )
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable") and args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # Parameter freezing based on flags
    for n, p in model.named_parameters():
        ln = n.lower()
        if "visual" in ln or "vision" in ln:
            if not args.tune_mm_vision and "merger" not in ln and "mlp" not in ln:
                p.requires_grad = False
        if "merger" in ln or ("visual" in ln and "mlp" in ln):
            p.requires_grad = args.tune_mm_mlp

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(f"[model] trainable={trainable:,} / total={total:,} ({100 * trainable / total:.1f}%)")

    return model, processor, tok


# ── oom-safe trainer ──────────────────────────────────────────────────


def build_trainer(model, processor, tok, dataset, collator, args):
    from transformers import Trainer, TrainingArguments

    class OomSafeTrainer(Trainer):
        """Trainer that catches OOM and NaN gradients, skips bad batches."""

        oom_count = 0
        nan_count = 0
        grad_nan_count = 0
        param_nan_count = 0

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._patch_dist_ops()

        def _nested_gather(self, tensors, name=None):
            """Override to skip dist.all_gather which hangs on MACA NCCL."""
            return tensors

        @staticmethod
        def _patch_dist_ops():
            """Patch unstable NCCL ops for MACA platform."""
            import torch
            import torch.distributed as dist
            if not dist.is_initialized():
                return

            # ── barrier patch ──────────────────────────────────────────
            _orig_barrier = dist.barrier
            def _safe_barrier(group=None, async_op=False, device_ids=None):
                try:
                    return _orig_barrier(group=group, async_op=async_op, device_ids=device_ids)
                except Exception:
                    pass
                try:
                    t = torch.zeros(1)
                    dist.all_reduce(t, group=group, async_op=async_op)
                except Exception:
                    pass
            dist.barrier = _safe_barrier

        def _has_nan_grad(self, model) -> bool:
            import torch
            # DeepSpeed ZeRO-2: check wrapped model params for NaN grads
            m = model.module if hasattr(model, 'module') else model
            for p in m.parameters():
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all():
                        return True
            return False

        def _sanitize_params(self, model):
            """Replace NaN/Inf parameter values with 0.0 before forward pass."""
            import torch
            m = model.module if hasattr(model, 'module') else model
            fixed = 0
            with torch.no_grad():
                for p in m.parameters():
                    if p.requires_grad and not torch.isfinite(p).all():
                        nan_mask = ~torch.isfinite(p.data)
                        p.data[nan_mask] = 0.0
                        fixed += 1
            if fixed > 0:
                self.param_nan_count += 1
                log(f"[param_nan] sanitized {fixed} NaN params "
                    f"step={self.state.global_step} total_events={self.param_nan_count}")

        def _sanitize_grads(self, model):
            """Replace NaN/Inf gradient values with 0.0 after backward pass."""
            import torch
            m = model.module if hasattr(model, 'module') else model
            fixed = 0
            for p in m.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    nan_mask = ~torch.isfinite(p.grad)
                    p.grad[nan_mask] = 0.0
                    fixed += 1
            if fixed > 0:
                self.grad_nan_count += 1
                log(f"[grad_nan] sanitized {fixed} NaN grads "
                    f"step={self.state.global_step} total_events={self.grad_nan_count}")
                if self.is_world_process_zero():
                    with open(args.bad_batches, "a", encoding="utf-8") as fb:
                        fb.write(f"[grad_nan] step={self.state.global_step} "
                                 f"fixed={fixed} grads total={self.grad_nan_count}\n")

        def _compute_loss_safe(self, model, inputs):
            """Compute loss forward pass. Returns (loss, loss_val)."""
            model.train()
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()
            loss_val = loss.item() if loss.dim() == 0 else loss.detach().mean().item()
            return loss, loss_val

        def training_step(self, model, inputs, num_items_in_batch=None):
            import torch
            import torch.distributed as dist
            denom = self.args.gradient_accumulation_steps

            # ── Fix 2: sanitize parameters before forward (catch NaN that leaked from prior micro-batches) ──
            self._sanitize_params(model)

            for attempt in range(args.max_retry_per_batch + 1):
                try:
                    loss, loss_val = self._compute_loss_safe(model, inputs)

                    # Check for NaN/zero loss BEFORE backward to prevent NaN gradient sync
                    if loss_val == 0.0 or not bool(torch.isfinite(loss)):
                        self.nan_count += 1
                        # Cross-rank coordination: if ANY rank has NaN, ALL skip backward
                        if dist.is_initialized() and dist.get_world_size() > 1:
                            nan_flag = torch.tensor([1.0], device=loss.device)
                            try:
                                dist.all_reduce(nan_flag, op=dist.ReduceOp.MAX)
                            except Exception:
                                pass
                        # Zero any accumulated gradients from previous micro-batches
                        try:
                            model.zero_grad(set_to_none=True)
                        except (TypeError, Exception):
                            pass
                        log(f"[bad_batch] loss={loss_val} step={self.state.global_step} "
                            f"nan_count={self.nan_count}")
                        if self.is_world_process_zero():
                            with open(args.bad_batches, "a", encoding="utf-8") as fb:
                                fb.write(f"[bad_batch] step={self.state.global_step} "
                                         f"loss={loss_val} nan_count={self.nan_count}\n")
                        return loss.detach() / denom

                    # Clean loss — safe to backward (gradients sync via DeepSpeed)
                    self.accelerator.backward(loss)

                    # ── Fix 1: sanitize NaN gradients after backward (catches NaN that forward missed) ──
                    self._sanitize_grads(model)

                    return loss.detach() / denom

                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    msg = str(e)
                    if "out of memory" not in msg.lower() and "oom" not in msg.lower():
                        raise
                    self.oom_count += 1
                    log(f"[oom] batch failed (attempt {attempt + 1}/{args.max_retry_per_batch + 1})")
                    try:
                        model.zero_grad(set_to_none=True)
                    except TypeError:
                        model.zero_grad()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if attempt < args.max_retry_per_batch:
                        continue
                    log(f"[skip] giving up on batch after {args.max_retry_per_batch + 1} attempts")
                    if self.is_world_process_zero():
                        with open(args.bad_batches, "a", encoding="utf-8") as fb:
                            fb.write(f"oom: {msg}\n")
                    try:
                        model.zero_grad(set_to_none=True)
                    except TypeError:
                        model.zero_grad()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    z = next(p for p in model.parameters() if p.requires_grad).sum() * 0.0
                    self.accelerator.backward(z)
                    return z.detach()

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        deepspeed=args.deepspeed,
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to=[],
        gradient_checkpointing=args.gradient_checkpointing,
        eval_strategy=args.eval_strategy,
    )
    trainer = OomSafeTrainer(
        model=model, args=targs, train_dataset=dataset,
        data_collator=collator, processing_class=processor,
    )
    return trainer


# ── batch size probing ───────────────────────────────────────────────


def probe_batch_size(args) -> int:
    """Try decreasing batch sizes to find one that fits in GPU memory."""
    import torch

    sizes = [8, 6, 4, 2, 1]
    for bs in sizes:
        log(f"[probe] trying per_device_batch_size={bs}")
        args.per_device_train_batch_size = bs
        try:
            model, processor, tok = load_model_and_processor(args)
            ds = JsonlDataset(args.data_path)
            # Use a tiny slice for probing
            ds.rows = ds.rows[:4]
            collator = SafeCollator(processor, tok, args.bad_samples, args)
            trainer = build_trainer(model, processor, tok, ds, collator, args)
            trainer.train()
            log(f"[probe] batch_size={bs} OK")
            del model, trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return bs
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            msg = str(e)
            if "out of memory" not in msg.lower() and "oom" not in msg.lower():
                log(f"[probe] unexpected error with bs={bs}: {msg[:200]}")
            else:
                log(f"[probe] bs={bs} OOM")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
    raise RuntimeError("all batch sizes OOM")


# ── smoke test ────────────────────────────────────────────────────────


def smoke(args) -> int:
    """Run first N batches and report results. Returns chosen batch size."""
    import torch

    log(f"[smoke] data={args.data_path}")
    log(f"[smoke] smoke_batches={args.smoke_batches}")

    # Probe batch size if not forced
    if args.per_device_train_batch_size == 0 or args.probe:
        bs = probe_batch_size(args)
        args.per_device_train_batch_size = bs
        log(f"[smoke] chosen batch_size={bs}")
    else:
        log(f"[smoke] using batch_size={args.per_device_train_batch_size}")

    model, processor, tok = load_model_and_processor(args)
    ds = JsonlDataset(args.data_path)
    # Limit to smoke batches
    samples_per_gpu = args.per_device_train_batch_size * args.gradient_accumulation_steps
    max_samples = args.smoke_batches * samples_per_gpu
    if args.smoke_batches > 0 and len(ds.rows) > max_samples:
        ds.rows = ds.rows[:max_samples]
    log(f"[smoke] smoke samples={len(ds.rows)}")

    collator = SafeCollator(processor, tok, args.bad_samples, args)
    trainer = build_trainer(model, processor, tok, ds, collator, args)
    resume = getattr(args, "resume_from_checkpoint", "") or None
    trainer.train(resume_from_checkpoint=resume)

    log(f"[smoke] done. oom_count={trainer.oom_count}, nan_loss_count={trainer.nan_count}, "
        f"grad_nan_count={trainer.grad_nan_count}, param_nan_count={trainer.param_nan_count}, "
        f"collator_skipped={collator.skipped}, collator_timed_out={collator.timed_out}")
    bad_batches = 0
    if Path(args.bad_batches).exists():
        bad_batches = len(Path(args.bad_batches).read_text(encoding="utf-8").splitlines())
    log(f"[smoke] bad_batches={bad_batches}")
    log(f"[smoke] batch_size={args.per_device_train_batch_size} OK for full training")

    # Save final model if full training (checkpoints already saved by trainer)
    if not args.save_strategy == "no":
        trainer.save_model(args.output_dir)
        processor.save_pretrained(args.output_dir)
        log(f"[smoke] final model saved to {args.output_dir}")

    return args.per_device_train_batch_size


# ── full train ────────────────────────────────────────────────────────


def train(args) -> None:
    import torch

    if args.per_device_train_batch_size == 0 or args.probe:
        bs = probe_batch_size(args)
        args.per_device_train_batch_size = bs
        log(f"[train] probed batch_size={bs}")

    model, processor, tok = load_model_and_processor(args)
    ds = JsonlDataset(args.data_path)
    log(f"[train] total samples={len(ds.rows)}")
    collator = SafeCollator(processor, tok, args.bad_samples, args)
    trainer = build_trainer(model, processor, tok, ds, collator, args)
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)


# ── cli ───────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(description="Expert SFT training")
    p.add_argument("--local_rank", type=int, default=0, help="deepspeed injected")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Shared args
    def add_common(s):
        s.add_argument("--model-name-or-path", type=str,
                       default="/data/msz/models/Qwen3-VL-8B-Instruct")
        s.add_argument("--data-path", type=str,
                       default="/data/msz/point/data_expert/expert_grounding_mix.jsonl")
        s.add_argument("--output-dir", type=str, default="/data/msz/point/outputs/expert_sft")
        s.add_argument("--deepspeed", type=str,
                       default="/data/msz/point/configs/zero2.json")
        s.add_argument("--per-device-train-batch-size", type=int, default=8)
        s.add_argument("--gradient-accumulation-steps", type=int, default=4)
        s.add_argument("--learning-rate", type=float, default=5e-6)
        s.add_argument("--num-train-epochs", type=float, default=1)
        s.add_argument("--max-steps", type=int, default=-1)
        s.add_argument("--model-max-length", type=int, default=16384)
        s.add_argument("--min-pixels", type=int, default=50176)
        s.add_argument("--max-pixels", type=int, default=50176)
        s.add_argument("--video-max-frames", type=int, default=32)
        s.add_argument("--video-min-frames", type=int, default=32)
        s.add_argument("--weight-decay", type=float, default=0)
        s.add_argument("--warmup-ratio", type=float, default=0.03)
        s.add_argument("--max-grad-norm", type=float, default=1.0)
        s.add_argument("--lr-scheduler-type", type=str, default="cosine")
        s.add_argument("--bf16", action="store_true", default=True)
        s.add_argument("--tune-mm-vision", action="store_true", default=False)
        s.add_argument("--tune-mm-mlp", action="store_true", default=True)
        s.add_argument("--tune-mm-llm", action="store_true", default=True)
        s.add_argument("--gradient-checkpointing", action="store_true", default=True)
        s.add_argument("--dataloader-num-workers", type=int, default=4)
        s.add_argument("--eval-strategy", type=str, default="no")
        s.add_argument("--save-strategy", type=str, default="steps")
        s.add_argument("--save-steps", type=int, default=1000)
        s.add_argument("--save-total-limit", type=int, default=1)
        s.add_argument("--batch-timeout", type=int, default=120,
                       help="max seconds per batch before skipping (0=no timeout)")
        s.add_argument("--logging-steps", type=int, default=1)
        s.add_argument("--max-retry-per-batch", type=int, default=3,
                       help="max retries per OOM batch before skipping")
        s.add_argument("--bad-samples", type=str,
                       default="/data/msz/point/bad/bad_samples.jsonl")
        s.add_argument("--bad-batches", type=str,
                       default="/data/msz/point/bad/bad_batches.log")
        s.add_argument("--resume-from-checkpoint", type=str, default="")
        s.add_argument("--probe", action="store_true",
                       help="auto-probe best batch size (8->6->4->2->1)")

    c_smoke = sub.add_parser("smoke", help="smoke test: first N batches")
    add_common(c_smoke)
    c_smoke.add_argument("--smoke-batches", type=int, default=100,
                         help="number of batches to run for smoke test")

    c_train = sub.add_parser("train", help="full SFT training")
    add_common(c_train)

    args = p.parse_args()

    # Ensure bad dir exists
    Path(args.bad_samples).parent.mkdir(parents=True, exist_ok=True)
    Path(args.bad_batches).parent.mkdir(parents=True, exist_ok=True)

    if args.cmd == "smoke":
        smoke(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
