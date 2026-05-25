#!/usr/bin/env bash
set -euo pipefail

export MACA_PATH=/opt/maca-3.5.3
export DS_ACCELERATOR=cuda
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=3600
export TOKENIZERS_PARALLELISM=false

PROJECT_DIR=/data/msz/point
BASE_MODEL=/data/msz/models/8b_base
DATA_PATH=${PROJECT_DIR}/data_expert_seed0_v1_shuffled/general_obj_expert/train_from_step1100_seed42_mb2.jsonl
OUTPUT_DIR=/data/msz/models/seed0_general_obj_expert_step1100_base_mb2_oomskip_retry_v1
LOG_PATH=${PROJECT_DIR}/logs/seed0_general_obj_expert_step1100_base_mb2_oomskip_retry_v1.log

mkdir -p "${PROJECT_DIR}/logs" "${OUTPUT_DIR}"

cd "${PROJECT_DIR}"
echo "[run] general_obj step1100 retry mb=2 base=${BASE_MODEL} data=${DATA_PATH} output=${OUTPUT_DIR}" > "${LOG_PATH}"

/opt/conda/bin/deepspeed --num_gpus=8 "${PROJECT_DIR}/expert_sft.py" train \
  --model-name-or-path "${BASE_MODEL}" \
  --data-path "${DATA_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --deepspeed "${PROJECT_DIR}/configs/zero2.json" \
  --num-train-epochs 1 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 4 \
  --learning-rate 5e-6 \
  --weight-decay 0 \
  --warmup-ratio 0.03 \
  --max-grad-norm 1.0 \
  --lr-scheduler-type cosine \
  --logging-steps 1 \
  --model-max-length 16384 \
  --min-pixels 50176 \
  --max-pixels 50176 \
  --dataloader-num-workers 4 \
  --bf16 \
  --sequential-sampling \
  >> "${LOG_PATH}" 2>&1
