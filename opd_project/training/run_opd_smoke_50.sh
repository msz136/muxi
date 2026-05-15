#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/msz/opd_project}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${ROOT}/outputs/opd_smoke_50_${RUN_ID}}"
LOG_DIR="${ROOT}/logs"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/opd_smoke_50_${RUN_ID}.log}"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

export MACA_HOME="${MACA_HOME:-/opt/maca}"
export MACA_PATH="${MACA_PATH:-/opt/maca}"
export LD_LIBRARY_PATH="${MACA_HOME}/lib:${MACA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export DS_ACCELERATOR="${DS_ACCELERATOR:-cuda}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"

cd "${ROOT}"

echo "[launcher] run_id=${RUN_ID}"
echo "[launcher] root=${ROOT}"
echo "[launcher] out_dir=${OUT_DIR}"
echo "[launcher] log_file=${LOG_FILE}"
echo "[launcher] cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
date

/opt/conda/bin/python3 "${ROOT}/training/opd_multiteacher_smoke_train.py" \
  --student /data/msz/models/8b_base \
  --teacher3 /data/msz/models/expert3 \
  --teacher4 /data/msz/models/expert4 \
  --data "${ROOT}/data/prompt_pool_clean.jsonl" \
  --output-dir "${OUT_DIR}" \
  --max-steps 50 \
  --save-steps 25 \
  --log-every 1 \
  --train-scope head_norm \
  --learning-rate 5e-7 \
  --weight-decay 0.01 \
  --hard-ce-coeff 0.05 \
  --max-grad-norm 1.0 \
  --max-length 4096 \
  --min-pixels 50176 \
  --max-pixels 50176 \
  --sample-limit 512 \
  --seed 42 \
  --max-retry-per-step 3 \
  --max-bad-steps 20

date
echo "[launcher] complete"
