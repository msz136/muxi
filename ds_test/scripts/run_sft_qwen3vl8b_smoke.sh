#!/usr/bin/env bash
set -euo pipefail

echo "Starting Qwen3-VL-8B-Instruct FULL SFT Smoke on 8xC500..."
echo "=========================================================="

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO
export DS_ACCELERATOR=cuda
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_PATH="/data/msz/models/Qwen3-VL-8B-Instruct"
DATA_PATH="/data/msz/ds_test/data/sft_qwen3vl8b_smoke.jsonl"
TRAIN_SCRIPT="/data/msz/ds_test/train_sft.py"
OUTPUT_DIR="/data/msz/ds_test/logs/sft_qwen3vl8b_instruct_smoke_$(date +%Y%m%d_%H%M%S)"
DS_CONFIG="/data/msz/ds_test/configs/ds_config_sft_smoke_zero3_offload.json"

test -f "${MODEL_PATH}/config.json"
test -f "${TRAIN_SCRIPT}"
test -f "${DS_CONFIG}"

mx-smi || true

deepspeed --num_gpus=8 "${TRAIN_SCRIPT}" \
  --model_name_or_path "${MODEL_PATH}" \
  --data_path "${DATA_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 2e-5 \
  --max_seq_length 64 \
  --logging_steps 1 \
  --save_strategy no \
  --bf16 \
  --deepspeed_config "${DS_CONFIG}"

echo "SFT smoke finished: ${OUTPUT_DIR}"
