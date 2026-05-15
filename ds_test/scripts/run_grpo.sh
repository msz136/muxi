#!/usr/bin/env bash
set -euo pipefail

echo "Starting GRPO Smoke Test on 8xC500..."
echo "====================================="

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO
export DS_ACCELERATOR=cuda
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_PATH="/data/msz/models/qwen3-vl-32b"
DATA_PATH="/data/msz/ds_test/data/smoke_data.jsonl"
TRAIN_SCRIPT="/data/msz/ds_test/train_grpo.py"
OUTPUT_DIR="/data/msz/ds_test/logs/grpo_qwen3_vl_text_converted"
DS_CONFIG="/data/msz/ds_test/configs/ds_config_grpo_smoke_zero3_offload.json"

deepspeed --num_gpus=8 "${TRAIN_SCRIPT}" \
  --model_name_or_path "${MODEL_PATH}" \
  --data_path "${DATA_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_generations 2 \
  --max_completion_length 8 \
  --learning_rate 1e-6 \
  --max_seq_length 64 \
  --max_steps 10 \
  --logging_steps 1 \
  --bf16 \
  --deepspeed_config "${DS_CONFIG}" \
  2>&1 | tee "${OUTPUT_DIR}.log"
