#!/usr/bin/env bash
set -euo pipefail

echo "Starting OPD Smoke Test on 8xC500..."
echo "===================================="

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO
export DS_ACCELERATOR=cuda
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_PATH="/data/msz/models/qwen3-vl-32b-text-converted"
TEACHER_PATH="/data/msz/models/qwen3-vl-32b-text-converted"
DATA_PATH="/data/msz/ds_test/data/opd_random_10.jsonl"
DATA_SCRIPT="/data/msz/ds_test/scripts/make_random_grpo_data.py"
TRAIN_SCRIPT="/data/msz/ds_test/train_opd.py"
OUTPUT_DIR="/data/msz/ds_test/logs/opd_qwen3_vl_text_smoke"
DS_CONFIG="/data/msz/ds_test/configs/ds_config_grpo_smoke_zero3_offload.json"

python "${DATA_SCRIPT}" \
  --output "${DATA_PATH}" \
  --num_samples 10 \
  --seed 42

rm -f "${OUTPUT_DIR}/opd_prepared.pt" \
      "${OUTPUT_DIR}/opd_prepared.pt.ready.json" \
      "${OUTPUT_DIR}/opd_prepared.pt.lock"

deepspeed --num_gpus=8 "${TRAIN_SCRIPT}" \
  --model_name_or_path "${MODEL_PATH}" \
  --teacher_model_name_or_path "${TEACHER_PATH}" \
  --data_path "${DATA_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_prompts 10 \
  --num_generations 2 \
  --teacher_topk 8 \
  --max_completion_length 8 \
  --max_seq_length 64 \
  --learning_rate 1e-6 \
  --max_steps 10 \
  --logging_steps 1 \
  --bf16 \
  --deepspeed_config "${DS_CONFIG}" \
  2>&1 | tee "${OUTPUT_DIR}.log"
