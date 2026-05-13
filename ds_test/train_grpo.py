#!/usr/bin/env python3
import argparse
import json
import os
import random

import torch
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


def build_prompt(example):
    instruction = example.get("instruction", "")
    inp = example.get("input", "")
    if inp:
        return f"### Instruction:\n{instruction}\n{inp}\n\n### Response:\n"
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def simple_reward(completion_text):
    text = completion_text.strip()
    score = 0.0
    if text:
        score += 0.2
    score += min(sum(ch.isalpha() for ch in text) / 30.0, 1.0)
    if "." in text:
        score += 0.2
    if "weather" in text.lower():
        score += 0.3
    return float(score)


class GRPOSmokeDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_length, max_completion_length, num_generations):
        self.samples = []

        with open(data_path, "r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]

        for raw in rows:
            prompt = build_prompt(raw)
            prompt_ids = tokenizer(
                prompt,
                add_special_tokens=False,
                truncation=True,
                max_length=max_seq_length - max_completion_length,
            )["input_ids"]
            prompt_len = len(prompt_ids)

            completions = [
                "The weather is really nice today.",
                "The weather is good today.",
            ][:num_generations]

            full_texts = [prompt + text for text in completions]
            tokenized = tokenizer(
                full_texts,
                add_special_tokens=False,
                padding="max_length",
                truncation=True,
                max_length=max_seq_length,
                return_tensors="pt",
            )

            rewards = torch.tensor([simple_reward(text) for text in completions], dtype=torch.float32)
            if rewards.numel() >= 2:
                advantages = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-6)
            else:
                advantages = torch.ones_like(rewards)
            if float(advantages.abs().sum()) == 0.0:
                advantages = torch.tensor([1.0, -1.0], dtype=torch.float32)[: rewards.numel()]

            self.samples.append(
                {
                    "input_ids": tokenized["input_ids"],
                    "attention_mask": tokenized["attention_mask"],
                    "advantages": advantages,
                    "rewards": rewards,
                    "prompt_len": torch.tensor(prompt_len, dtype=torch.long),
                    "prompt_text": prompt,
                    "completions": completions,
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = dict(self.samples[idx])
        sample.pop("prompt_text", None)
        sample.pop("completions", None)
        return sample


def grpo_data_collator(features):
    batch = {}
    for key in features[0]:
        values = [feature[key] for feature in features]
        if torch.is_tensor(values[0]):
            batch[key] = torch.stack(values)
        else:
            batch[key] = torch.tensor(values)
    return batch


class GRPOSmokeTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        advantages = inputs["advantages"]
        rewards = inputs["rewards"]
        prompt_len = inputs["prompt_len"]

        batch_size, num_generations, seq_len = input_ids.shape

        flat_input_ids = input_ids.view(batch_size * num_generations, seq_len)
        flat_attention_mask = attention_mask.view(batch_size * num_generations, seq_len)

        outputs = model(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
            use_cache=False,
        )
        logits = outputs.logits.float()
        shift_logits = logits[:, :-1, :]
        shift_labels = flat_input_ids[:, 1:]
        shift_mask = flat_attention_mask[:, 1:].float()

        log_probs = torch.log_softmax(shift_logits, dim=-1)
        token_logps = torch.gather(
            log_probs,
            dim=-1,
            index=shift_labels.unsqueeze(-1),
        ).squeeze(-1)

        token_logps = token_logps.view(batch_size, num_generations, -1)
        shift_mask = shift_mask.view(batch_size, num_generations, -1)
        comp_mask = torch.zeros_like(token_logps, dtype=torch.float32)

        for b in range(batch_size):
            start = max(int(prompt_len[b].item()) - 1, 0)
            comp_mask[b, :, start:] = 1.0

        comp_mask = comp_mask * shift_mask
        denom = comp_mask.sum(dim=-1).clamp_min(1.0)
        logps = (token_logps * comp_mask).sum(dim=-1) / denom

        loss = -(advantages.detach() * logps).mean()

        self._last_rewards = rewards.detach().cpu().tolist()
        self._last_advantages = advantages.detach().cpu().tolist()
        self._last_logps = logps.detach().cpu().tolist()

        return (loss, outputs) if return_outputs else loss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_generations", type=int, default=2)
    parser.add_argument("--max_completion_length", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--max_seq_length", type=int, default=64)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_strategy", type=str, default="no")
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--deepspeed_config", type=str, default=None)
    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_args()
    set_seed(42)
    random.seed(42)

    print("transformers =", transformers.__version__, flush=True)
    print("model =", args.model_name_or_path, flush=True)
    print("data =", args.data_path, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("deepspeed_config =", args.deepspeed_config, flush=True)

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
    print("config class =", type(config), flush=True)
    print("config.model_type =", getattr(config, "model_type", None), flush=True)
    print("config.architectures =", getattr(config, "architectures", None), flush=True)

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

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=config,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype="auto",
        attn_implementation="eager",
    )
    print("model class =", type(model), flush=True)

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = "eager"
    if hasattr(model.config, "attn_implementation"):
        model.config.attn_implementation = "eager"
    if hasattr(model.config, "sliding_window"):
        model.config.sliding_window = None
    if hasattr(model.config, "use_sliding_window"):
        model.config.use_sliding_window = False
    if hasattr(model.config, "max_window_layers"):
        model.config.max_window_layers = 0
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("gradient checkpointing enabled", flush=True)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    train_dataset = GRPOSmokeDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
    )
    print("dataset_size =", len(train_dataset), flush=True)

    trainer = GRPOSmokeTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=grpo_data_collator,
        processing_class=tokenizer,
    )

    train_result = trainer.train()

    if trainer.is_world_process_zero():
        os.makedirs(args.output_dir, exist_ok=True)
        result = {
            "status": "passed",
            "loss": float(train_result.training_loss),
            "rewards": getattr(trainer, "_last_rewards", []),
            "advantages": getattr(trainer, "_last_advantages", []),
            "logps": getattr(trainer, "_last_logps", []),
            "max_steps": args.max_steps,
            "dataset_size": len(train_dataset),
        }
        with open(
            os.path.join(args.output_dir, "grpo_smoke_result.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print("GRPO loss:", result["loss"], flush=True)
        print("rewards:", result["rewards"], flush=True)
        print("advantages:", result["advantages"], flush=True)
        print("GRPO forward/backward/step completed", flush=True)


if __name__ == "__main__":
    main()
