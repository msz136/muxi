#!/usr/bin/env python3
import argparse, json, os
import torch, transformers
from torch.utils.data import Dataset
from transformers import AutoConfig, AutoTokenizer, Trainer, TrainingArguments, set_seed
try:
    from transformers.integrations import HfDeepSpeedConfig
except Exception:
    HfDeepSpeedConfig = None

def build_prompt(ex):
    ins = ex.get("instruction", "")
    inp = ex.get("input", "")
    return f"### Instruction:\n{ins}\n{inp}\n\n### Response:\n" if inp else f"### Instruction:\n{ins}\n\n### Response:\n"

class SFTDataset(Dataset):
    def __init__(self, path, tokenizer, max_len):
        self.rows = []
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            ex = json.loads(line)
            prompt = build_prompt(ex)
            answer = ex.get("output", "")
            prompt_ids = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=max_len)["input_ids"]
            full = tokenizer(prompt + answer, add_special_tokens=False, truncation=True, max_length=max_len)["input_ids"]
            labels = list(full)
            for i in range(min(len(prompt_ids), len(labels))):
                labels[i] = -100
            pad = max_len - len(full)
            if pad > 0:
                full += [tokenizer.pad_token_id] * pad
                labels += [-100] * pad
            self.rows.append({
                "input_ids": torch.tensor(full, dtype=torch.long),
                "attention_mask": torch.tensor([1 if x != tokenizer.pad_token_id else 0 for x in full], dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            })
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]

def get_model_cls():
    for name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq", "Qwen3VLForConditionalGeneration"):
        cls = getattr(transformers, name, None)
        if cls is not None:
            print("model loader =", name, flush=True)
            return cls
    raise RuntimeError("No Qwen3-VL compatible model loader found in transformers")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--data_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--deepspeed_config", required=True)
    p.add_argument("--max_seq_length", type=int, default=64)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--bf16", action="store_true")
    args, _ = p.parse_known_args()

    set_seed(42)
    print("transformers =", transformers.__version__, flush=True)
    if HfDeepSpeedConfig:
        HfDeepSpeedConfig(args.deepspeed_config)

    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    cfg = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if hasattr(cfg, "use_cache"):
        cfg.use_cache = False
    for k in ("_attn_implementation", "attn_implementation"):
        if hasattr(cfg, k):
            setattr(cfg, k, "eager")

    model = get_model_cls().from_pretrained(
        args.model_name_or_path,
        config=cfg,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype="auto",
        attn_implementation="eager",
    )
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_strategy="no",
        bf16=args.bf16,
        deepspeed=args.deepspeed_config,
        remove_unused_columns=False,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=SFTDataset(args.data_path, tok, args.max_seq_length),
        processing_class=tok,
    )
    trainer.train()
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print("Qwen3-VL 8B SFT smoke finished:", args.output_dir, flush=True)

if __name__ == "__main__":
    main()
