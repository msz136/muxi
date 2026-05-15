#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/msz/opd_project}"
POINT_ROOT="${POINT_ROOT:-/data/msz/point}"
RUN_ID="${RUN_ID:-zero2full_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${ROOT}/outputs/opd_zero2_full_${RUN_ID}}"
LOG_DIR="${ROOT}/logs"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/opd_zero2_full_${RUN_ID}.log}"
DATA_OUT="${DATA_OUT:-${ROOT}/data/opd_zero2_smoke_${RUN_ID}.jsonl}"
BAD_DIR="${OUT_DIR}/bad"

mkdir -p "${OUT_DIR}" "${LOG_DIR}" "${BAD_DIR}" "$(dirname "${DATA_OUT}")"
exec > >(tee -a "${LOG_FILE}") 2>&1

export MACA_HOME="${MACA_HOME:-/opt/maca}"
export MACA_PATH="${MACA_PATH:-/opt/maca}"
export LD_LIBRARY_PATH="${MACA_HOME}/lib:${MACA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export DS_ACCELERATOR="${DS_ACCELERATOR:-cuda}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"

cd "${ROOT}"

echo "[zero2-full] run_id=${RUN_ID}"
echo "[zero2-full] out_dir=${OUT_DIR}"
echo "[zero2-full] log_file=${LOG_FILE}"
echo "[zero2-full] data_out=${DATA_OUT}"
echo "[zero2-full] cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "[zero2-full] expert_loss_fixed=55.0"
date

/opt/conda/bin/python3 "${ROOT}/scripts/build_zero2_smoke_data.py" \
  --input "${ROOT}/data/prompt_pool_clean.jsonl" \
  --output "${DATA_OUT}" \
  --limit "${DATA_LIMIT:-128}"

wc -l "${DATA_OUT}"

deepspeed --num_gpus="${NUM_GPUS:-8}" "${POINT_ROOT}/expert_sft.py" smoke \
  --model-name-or-path /data/msz/models/8b_base \
  --data-path "${DATA_OUT}" \
  --output-dir "${OUT_DIR}" \
  --deepspeed "${ROOT}/configs/ds_zero2.json" \
  --per-device-train-batch-size "${PER_DEVICE_BS:-1}" \
  --gradient-accumulation-steps "${GRAD_ACCUM:-1}" \
  --learning-rate "${LR:-5e-7}" \
  --max-steps "${MAX_STEPS:-1}" \
  --smoke-batches "${SMOKE_BATCHES:-8}" \
  --model-max-length "${MODEL_MAX_LENGTH:-4096}" \
  --min-pixels "${MIN_PIXELS:-50176}" \
  --max-pixels "${MAX_PIXELS:-50176}" \
  --weight-decay "${WEIGHT_DECAY:-0.01}" \
  --warmup-ratio "${WARMUP_RATIO:-0.0}" \
  --max-grad-norm "${MAX_GRAD_NORM:-1.0}" \
  --lr-scheduler-type constant \
  --logging-steps 1 \
  --save-strategy steps \
  --save-steps "${SAVE_STEPS:-1}" \
  --save-total-limit 2 \
  --dataloader-num-workers "${DATALOADER_WORKERS:-0}" \
  --max-retry-per-batch "${MAX_RETRY_PER_BATCH:-2}" \
  --batch-timeout "${BATCH_TIMEOUT:-120}" \
  --bad-samples "${BAD_DIR}/bad_samples.jsonl" \
  --bad-batches "${BAD_DIR}/bad_batches.log" \
  --bf16 \
  --gradient-checkpointing

date
echo "[zero2-full] complete"
