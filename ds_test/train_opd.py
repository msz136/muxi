#!/usr/bin/env python3
import argparse
import json
import os
import random
import time

import torch
import torch.distributed as dist
import transformers
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

try:
    from transformers.integrations import HfDeepSpeedConfig
except Exception:
    HfDeepSpeedConfig = None

try:
    import deepspeed
except Exception:
    deepspeed = None


def get_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", "0"))


def get_world_size():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", "1"))


def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", os.environ.get("local_rank", "0")))


def ensure_distributed():
    if get_world_size() <= 1:
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        return
    if dist.is_available() and dist.is_initialized():
        torch.cuda.set_device(get_local_rank())
        return
    if deepspeed is not None:
        deepspeed.init_distributed()
    else:
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(get_local_rank())


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def build_prompt(example):
    instruction = example.get("instruction", "")
    inp = example.get("input", "")
    if inp:
        return f"### Instruction:\n{instruction}\n{inp}\n\n### Response:\n"
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def patch_qwen3_config(config):
    if hasattr(config, "_attn_implementation"):
        config._attn_implementation = "eager"
    if hasattr(config, "attn_implementation"):
        config.attn_implementation = "eager"
    if hasattr(config, "use_cache"):
        config.use_cache = False
    if hasattr(config, "sliding_window"):
        config.sliding_window = None
    if hasattr(config, "use_sliding_window"):
        config.use_sliding_window = False
    if hasattr(config, "max_window_layers"):
        config.max_window_layers = 0
    return config


def load_tokenizer(model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_inference_model(model_name_or_path):
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    patch_qwen3_config(config)
    kwargs = dict(
        config=config,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype="auto",
        attn_implementation="eager",
    )
    if get_world_size() > 1:
        kwargs["tp_plan"] = "auto"
    else:
        kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    model.eval()
    return model


def first_param_device(model):
    return next(model.parameters()).device


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def wait_for_marker(marker_path, timeout_sec):
    start = time.time()
    while not os.path.exists(marker_path):
        if time.time() - start > timeout_sec:
            raise TimeoutError(f"Timed out waiting for marker: {marker_path}")
        time.sleep(5)


def pad_to_length(tensor_1d, target_len, pad_value):
    length = tensor_1d.size(0)
    if length >= target_len:
        return tensor_1d[:target_len]
    pad = torch.full((target_len - length,), pad_value, dtype=tensor_1d.dtype)
    return torch.cat([tensor_1d, pad], dim=0)


def prepare_rollout_and_teacher_cache(args):
    os.makedirs(args.output_dir, exist_ok=True)
    prepared_path = args.prepared_data_path or os.path.join(args.output_dir, "opd_prepared.pt")
    marker_path = prepared_path + ".ready.json"
    lock_path = prepared_path + ".lock"

    if get_rank() == 0 and os.path.exists(prepared_path) and os.path.exists(marker_path) and not args.overwrite_prepared_cache:
        print("Reusing prepared OPD cache:", prepared_path, flush=True)
        barrier()
        return prepared_path
    if get_rank() != 0 and os.path.exists(prepared_path) and os.path.exists(marker_path) and not args.overwrite_prepared_cache:
        barrier()
        return prepared_path

    if get_rank() == 0:
        if os.path.exists(lock_path):
            os.remove(lock_path)
        open(lock_path, "w", encoding="utf-8").close()
    try:
        with open(args.data_path, "r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        rows = rows[: args.max_prompts]

        tokenizer = load_tokenizer(args.model_name_or_path)
        rollout_model = load_inference_model(args.model_name_or_path)
        rollout_device = first_param_device(rollout_model)

        prepared_samples = []
        summary_rows = []

        for idx, row in enumerate(rows):
            prompt = build_prompt(row)
            prompt_inputs = tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                max_length=args.max_seq_length - args.max_completion_length,
            )
            prompt_len = int(prompt_inputs["input_ids"].shape[1])
            prompt_inputs = {k: v.to(rollout_device) for k, v in prompt_inputs.items()}

            sequences = rollout_model.generate(
                **prompt_inputs,
                do_sample=True,
                temperature=args.rollout_temperature,
                top_p=args.rollout_top_p,
                max_new_tokens=args.max_completion_length,
                num_return_sequences=args.num_generations,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            padded_ids = []
            padded_mask = []
            completion_texts = []
            for seq in sequences:
                seq = seq.detach().to("cpu")
                attention_mask = torch.ones_like(seq, dtype=torch.long)
                padded_ids.append(pad_to_length(seq, args.max_seq_length, tokenizer.pad_token_id))
                padded_mask.append(pad_to_length(attention_mask, args.max_seq_length, 0))

                completion_ids = seq[prompt_len:]
                completion_texts.append(
                    tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
                )

            prepared_samples.append(
                {
                    "input_ids": torch.stack(padded_ids, dim=0),
                    "attention_mask": torch.stack(padded_mask, dim=0),
                    "prompt_len": torch.tensor(prompt_len, dtype=torch.long),
                    "teacher_topk_ids": None,
                    "teacher_topk_logps": None,
                    "prompt_text": prompt,
                    "completions": completion_texts,
                }
            )
            summary_rows.append(
                {
                    "index": idx,
                    "prompt_preview": prompt[:160],
                    "completions": completion_texts,
                }
            )

        if args.teacher_model_name_or_path == args.model_name_or_path:
            teacher_model = rollout_model
            teacher_device = rollout_device
            if get_rank() == 0:
                print("Teacher review reuses student rollout checkpoint", flush=True)
        else:
            del rollout_model
            torch.cuda.empty_cache()
            teacher_model = load_inference_model(args.teacher_model_name_or_path)
            teacher_device = first_param_device(teacher_model)
            if get_rank() == 0:
                print("Teacher review loaded separate checkpoint", flush=True)

        for sample in prepared_samples:
            input_ids = sample["input_ids"].to(teacher_device)
            attention_mask = sample["attention_mask"].to(teacher_device)
            with torch.no_grad():
                outputs = teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
            shift_log_probs = torch.log_softmax(outputs.logits[:, :-1, :].float(), dim=-1)
            topk_logps, topk_ids = torch.topk(
                shift_log_probs,
                k=args.teacher_topk,
                dim=-1,
            )
            sample["teacher_topk_ids"] = topk_ids.to("cpu")
            sample["teacher_topk_logps"] = topk_logps.to("cpu")

        for sample in prepared_samples:
            sample.pop("prompt_text", None)
            sample.pop("completions", None)

        if get_rank() == 0:
            torch.save(prepared_samples, prepared_path)
            save_json(
                marker_path,
                {
                    "status": "ready",
                    "prepared_path": prepared_path,
                    "num_samples": len(prepared_samples),
                    "teacher_model_name_or_path": args.teacher_model_name_or_path,
                    "student_model_name_or_path": args.model_name_or_path,
                    "num_generations": args.num_generations,
                    "teacher_topk": args.teacher_topk,
                    "rows": summary_rows,
                },
            )
            print("Prepared OPD cache:", prepared_path, flush=True)
        barrier()
        return prepared_path
    finally:
        if get_rank() == 0 and os.path.exists(lock_path):
            os.remove(lock_path)


class OPDPreparedDataset(Dataset):
    def __init__(self, prepared_path):
        self.samples = torch.load(prepared_path, map_location="cpu")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def opd_data_collator(features):
    batch = {}
    for key in features[0]:
        values = [feature[key] for feature in features]
        if torch.is_tensor(values[0]):
            batch[key] = torch.stack(values)
        else:
            batch[key] = values
    return batch


class OPDTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = inputs["input_ids"]                  # [B, G, L]
        attention_mask = inputs["attention_mask"]        # [B, G, L]
        prompt_len = inputs["prompt_len"]                # [B]
        teacher_topk_ids = inputs["teacher_topk_ids"]    # [B, G, L-1, K]
        teacher_topk_logps = inputs["teacher_topk_logps"]

        batch_size, num_generations, seq_len = input_ids.shape
        topk = teacher_topk_ids.shape[-1]

        flat_input_ids = input_ids.view(batch_size * num_generations, seq_len)
        flat_attention_mask = attention_mask.view(batch_size * num_generations, seq_len)
        flat_teacher_ids = teacher_topk_ids.view(batch_size * num_generations, seq_len - 1, topk)
        flat_teacher_logps = teacher_topk_logps.view(batch_size * num_generations, seq_len - 1, topk)

        outputs = model(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
            use_cache=False,
        )
        student_shift_logits = outputs.logits[:, :-1, :].float()
        student_shift_logps = torch.log_softmax(student_shift_logits, dim=-1)
        student_selected_logps = torch.gather(
            student_shift_logps,
            dim=-1,
            index=flat_teacher_ids,
        )

        teacher_support_logps = flat_teacher_logps - torch.logsumexp(
            flat_teacher_logps,
            dim=-1,
            keepdim=True,
        )
        teacher_support_probs = teacher_support_logps.exp()
        token_kl = (
            teacher_support_probs * (teacher_support_logps - student_selected_logps)
        ).sum(dim=-1)

        shift_mask = flat_attention_mask[:, 1:].float()
        response_mask = torch.zeros_like(token_kl, dtype=torch.float32)
        expanded_prompt_len = prompt_len.unsqueeze(1).repeat(1, num_generations).view(-1)
        for idx in range(response_mask.size(0)):
            start = max(int(expanded_prompt_len[idx].item()) - 1, 0)
            response_mask[idx, start:] = 1.0
        response_mask = response_mask * shift_mask

        denom = response_mask.sum(dim=-1).clamp_min(1.0)
        sample_loss = (token_kl * response_mask).sum(dim=-1) / denom
        loss = sample_loss.mean()

        self._last_token_kl = float(token_kl.mean().detach().cpu())
        self._last_loss = float(loss.detach().cpu())

        return (loss, outputs) if return_outputs else loss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--teacher_model_name_or_path", type=str, default=None)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--prepared_data_path", type=str, default=None)
    parser.add_argument("--overwrite_prepared_cache", action="store_true")
    parser.add_argument("--max_prompts", type=int, default=10)
    parser.add_argument("--num_generations", type=int, default=2)
    parser.add_argument("--teacher_topk", type=int, default=8)
    parser.add_argument("--max_completion_length", type=int, default=8)
    parser.add_argument("--max_seq_length", type=int, default=64)
    parser.add_argument("--rollout_temperature", type=float, default=0.7)
    parser.add_argument("--rollout_top_p", type=float, default=0.9)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_strategy", type=str, default="no")
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--deepspeed_config", type=str, default=None)
    parser.add_argument("--prepare_timeout_sec", type=int, default=7200)
    args, _ = parser.parse_known_args()
    if args.teacher_model_name_or_path is None:
        args.teacher_model_name_or_path = args.model_name_or_path
    return args


def main():
    args = parse_args()
    set_seed(42)
    random.seed(42)
    ensure_distributed()

    print("transformers =", transformers.__version__, flush=True)
    print("student_model =", args.model_name_or_path, flush=True)
    print("teacher_model =", args.teacher_model_name_or_path, flush=True)
    print("data =", args.data_path, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("deepspeed_config =", args.deepspeed_config, flush=True)

    prepared_path = prepare_rollout_and_teacher_cache(args)
    if get_rank() != 0:
        wait_for_marker(prepared_path + ".ready.json", args.prepare_timeout_sec)

    if args.deepspeed_config and HfDeepSpeedConfig is not None:
        HfDeepSpeedConfig(args.deepspeed_config)
        print("HfDeepSpeedConfig enabled", flush=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        bf16=args.bf16,
        deepspeed=args.deepspeed_config,
        remove_unused_columns=False,
        report_to=[],
        disable_tqdm=False,
    )

    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )
    patch_qwen3_config(config)
    print("config class =", type(config), flush=True)
    print("config.model_type =", getattr(config, "model_type", None), flush=True)
    print("config.architectures =", getattr(config, "architectures", None), flush=True)

    tokenizer = load_tokenizer(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=config,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype="auto",
        attn_implementation="eager",
    )
    print("model class =", type(model), flush=True)

    patch_qwen3_config(model.config)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("gradient checkpointing enabled", flush=True)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    train_dataset = OPDPreparedDataset(prepared_path)
    print("prepared_dataset_size =", len(train_dataset), flush=True)

    trainer = OPDTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=opd_data_collator,
        processing_class=tokenizer,
    )

    train_result = trainer.train()

    if trainer.is_world_process_zero():
        os.makedirs(args.output_dir, exist_ok=True)
        result = {
            "status": "passed",
            "loss": float(train_result.training_loss),
            "teacher_topk": args.teacher_topk,
            "num_generations": args.num_generations,
            "max_steps": args.max_steps,
            "dataset_size": len(train_dataset),
            "teacher_model_name_or_path": args.teacher_model_name_or_path,
            "student_model_name_or_path": args.model_name_or_path,
            "mean_token_kl": getattr(trainer, "_last_token_kl", None),
        }
        save_json(os.path.join(args.output_dir, "opd_smoke_result.json"), result)
        print("OPD loss:", result["loss"], flush=True)
        print("mean token KL:", result["mean_token_kl"], flush=True)
        print("OPD rollout/review/distill completed", flush=True)


if __name__ == "__main__":
    main()

