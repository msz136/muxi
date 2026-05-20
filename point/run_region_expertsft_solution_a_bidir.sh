#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/msz/point
DATA_PATH=/data/msz/opd_project/data/semantic_nav_region_box_v1/solution_a_bidir_train_v1/semantic_nav_region_solution_a_bidir_train_v1_high_quality.jsonl
MODEL=/data/msz/models/Qwen3-VL-8B-Instruct
DS_CONFIG=${ROOT}/configs/zero2.json
LOG_DIR=${ROOT}/logs
RUN_ID=region_expertsft_solution_a_bidir_final_$(date +%Y%m%d_%H%M%S)
OUT_DIR=${ROOT}/outputs/${RUN_ID}
LOG_FILE=${LOG_DIR}/${RUN_ID}.log

NUM_GPUS=8
PER_DEVICE_BS=1
GRAD_ACC=4
LEARNING_RATE=${LEARNING_RATE:-5e-6}

mkdir -p "${LOG_DIR}" "${OUT_DIR}" "${ROOT}/bad"
exec > >(tee -a "${LOG_FILE}") 2>&1

export MACA_PATH=${MACA_PATH:-/opt/maca-3.5.3}
export PATH="${MACA_PATH}/bin:${MACA_PATH}/mxgpu_llvm/bin:/opt/conda/bin:${PATH}"
export LD_LIBRARY_PATH="${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${MACA_PATH}/ompi/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export DS_ACCELERATOR=cuda
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=3600
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false

test -f "${MODEL}/config.json"
test -f "${DATA_PATH}"
test -f "${DS_CONFIG}"
test -f "${ROOT}/expert_sft.py"

ROWS=$(wc -l < "${DATA_PATH}")
EFFECTIVE_BS=$((NUM_GPUS * PER_DEVICE_BS * GRAD_ACC))
EXPECTED_STEPS=$(((ROWS + EFFECTIVE_BS - 1) / EFFECTIVE_BS))

echo "[region-expertsft] start=$(date)"
echo "[region-expertsft] data=${DATA_PATH}"
echo "[region-expertsft] model=${MODEL}"
echo "[region-expertsft] output=${OUT_DIR}"
echo "[region-expertsft] log=${LOG_FILE}"
echo "[region-expertsft] rows=${ROWS}"
echo "[region-expertsft] per_device_batch_size=${PER_DEVICE_BS}"
echo "[region-expertsft] gradient_accumulation_steps=${GRAD_ACC}"
echo "[region-expertsft] learning_rate=${LEARNING_RATE}"
echo "[region-expertsft] effective_batch_size=${EFFECTIVE_BS}"
echo "[region-expertsft] expected_steps=${EXPECTED_STEPS}"
echo "[region-expertsft] save_policy=final_model_only_no_intermediate_checkpoints"
mx-smi || true

deepspeed --num_gpus="${NUM_GPUS}" "${ROOT}/expert_sft.py" train \
  --model-name-or-path "${MODEL}" \
  --data-path "${DATA_PATH}" \
  --output-dir "${OUT_DIR}" \
  --deepspeed "${DS_CONFIG}" \
  --num-train-epochs 1 \
  --per-device-train-batch-size "${PER_DEVICE_BS}" \
  --gradient-accumulation-steps "${GRAD_ACC}" \
  --learning-rate "${LEARNING_RATE}" \
  --model-max-length 16384 \
  --min-pixels 50176 \
  --max-pixels 50176 \
  --video-min-frames 32 \
  --video-max-frames 32 \
  --weight-decay 0 \
  --warmup-ratio 0.03 \
  --max-grad-norm 1 \
  --lr-scheduler-type cosine \
  --logging-steps 1 \
  --dataloader-num-workers 4 \
  --optim adamw_torch \
  --bf16 \
  --gradient-checkpointing

echo "[region-expertsft] done=$(date)"
echo "[region-expertsft] output=${OUT_DIR}"
