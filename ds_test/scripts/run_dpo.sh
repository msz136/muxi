#!/bin/bash

echo "Starting DPO Smoke Test on 8xC500..."
echo "======================================"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO
export PYTHONPATH=/data/msz/models/venv/lib/python3.12/site-packages:$PYTHONPATH
export DS_ACCELERATOR="cuda"
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 正确写法：--module
deepspeed --num_gpus=8 \
    --module transformers.trainer \
    --model_name_or_path /data/msz/models/Qwen3.5-27B \
    --data_path /data/msz/ds_test/data/smoke_data.jsonl \
    --output_dir /data/msz/ds_test/logs/dpo_output \
    --beta 0.1 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-6 \
    --max_seq_length 512 \
    --max_prompt_length 256 \
    --deepspeed /data/msz/ds_test/configs/ds_config_zero2.json \
    --logging_steps 1 \
    --save_steps 10 \
    --save_total_limit 1 \
    --bf16 \
    --tf32 \
    --dataloader_num_workers 0 \
    --remove_unused_columns false \
    --overwrite_output_dir \
    --do_train

if [ $? -eq 0 ]; then
    echo "✓ DPO Smoke Test PASSED"
else
    echo "✗ DPO Smoke Test FAILED"
    exit 1
fi
