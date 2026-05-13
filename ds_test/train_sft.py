#!/usr/bin/env python3
import os
import json
import argparse

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    set_seed,
)
from torch.utils.data import Dataset

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


class SFTSmokeDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_length):
        self.samples = []

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)

                prompt = build_prompt(ex)
                answer = ex.get("output", "")

                prompt_ids = tokenizer(
                    prompt,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max_seq_length,
                )["input_ids"]

                full_ids = tokenizer(
                    prompt + answer,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max_seq_length,
                )["input_ids"]

                input_ids = list(full_ids)
                attention_mask = [1] * len(input_ids)
                labels = list(full_ids)

                prompt_len = min(len(prompt_ids), len(labels))
                for i in range(prompt_len):
                    labels[i] = -100

                pad_len = max_seq_length - len(input_ids)
                if pad_len > 0:
                    input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
                    attention_mask = attention_mask + [0] * pad_len
                    labels = labels + [-100] * pad_len

                self.samples.append(
                    {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": labels,
                    }
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_strategy", type=str, default="no")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--deepspeed_config", type=str, default=None)

    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_args()
    set_seed(42)

    print("transformers =", transformers.__version__, flush=True)
    print("model =", args.model_name_or_path, flush=True)
    print("data =", args.data_path, flush=True)
    print("output_dir =", args.output_dir, flush=True)
    print("deepspeed_config =", args.deepspeed_config, flush=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
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

    dschf = None
    if args.deepspeed_config and HfDeepSpeedConfig is not None:
        dschf = HfDeepSpeedConfig(args.deepspeed_config)
        print("HfDeepSpeedConfig enabled", flush=True)

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

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("gradient checkpointing enabled", flush=True)

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    train_dataset = SFTSmokeDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("REAL SFT smoke finished", flush=True)


if __name__ == "__main__":
    main()
