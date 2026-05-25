#!/usr/bin/env bash
set -euo pipefail

cd /data/msz/point

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MACA_PATH=${MACA_PATH:-/opt/maca-3.5.3}
export PATH="${MACA_PATH}/bin:${MACA_PATH}/mxgpu_llvm/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH}"
export LD_LIBRARY_PATH="${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${MACA_PATH}/ompi/lib:${LD_LIBRARY_PATH:-}"
export DS_ACCELERATOR=cuda
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=3600
export TOKENIZERS_PARALLELISM=false

RUN_NAME=opd_seed0_five_online_300k_mb4_save500_zero3_leftpad_v1
LOG=/data/msz/point/logs/${RUN_NAME}.log
OUT=/data/msz/models/${RUN_NAME}

mkdir -p /data/msz/point/logs "${OUT}"

deepspeed --num_gpus=8 train_opd_online_vl.py train \
  --model-name-or-path /data/msz/models/8b_base \
  --data-path /data/msz/point/opd_student_v1/train_prompts.jsonl \
  --output-dir "${OUT}" \
  --deepspeed /data/msz/point/configs/zero3_opd_maca.json \
  --route-policy target \
  --no-group-by-route \
  --route-block-shuffle \
  --teacher-load-mode preloaded_zero3 \
  --teacher-deepspeed /data/msz/point/configs/zero3_opd_maca.json \
  --limit-samples 300000 \
  --save-steps 500 \
  --save-total-limit 5 \
  --per-device-train-batch-size 4 \
  --gradient-accumulation-steps 4 \
  --learning-rate 1e-6 \
  --warmup-ratio 0.03 \
  --max-grad-norm 1.0 \
  --num-train-epochs 1 \
  --logging-steps 1 \
  --model-max-length 16384 \
  --min-pixels 50176 \
  --max-pixels 50176 \
  --bf16 \
  --gradient-checkpointing \
  2>&1 | tee "${LOG}"
