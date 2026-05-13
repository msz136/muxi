#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/msz/point
DATA_PATH="${DATA_PATH:-${ROOT}/data_expert/expert_smoke_v1_local.jsonl}"
MODEL="${MODEL:-/data/msz/models/Qwen3-VL-8B-Instruct}"
LOG_DIR="${ROOT}/logs"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${ROOT}/outputs/smoke_${RUN_ID}"

mkdir -p "${ROOT}"/{configs,outputs,logs,bad} "${LOG_DIR}" "${OUT_DIR}"

exec > >(tee -a "${LOG_DIR}/smoke_${RUN_ID}.log") 2>&1

echo "[smoke] start: $(date)"
echo "[smoke] data=${DATA_PATH}"
echo "[smoke] model=${MODEL}"
echo "[smoke] run_id=${RUN_ID}"
hostname
uname -a

# Environment
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NCCL_DEBUG=INFO
export DS_ACCELERATOR=cuda
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Check preconditions
test -f "${MODEL}/config.json" || { echo "missing model: ${MODEL}/config.json"; exit 1; }
test -f "${DATA_PATH}" || { echo "missing data: ${DATA_PATH}"; exit 1; }
test -f "${ROOT}/configs/zero2.json" || { echo "missing deepspeed config"; exit 1; }

echo "[smoke] sample count: $(wc -l < "${DATA_PATH}")"

# Deepspeed config path
DS_CONFIG="${ROOT}/configs/zero2.json"

# Smoke test with auto batch size probing (8 -> 6 -> 4 -> 2 -> 1)
echo "[smoke] launching with probe (auto batch size fallback)"
deepspeed --num_gpus=8 "${ROOT}/expert_sft.py" smoke \
  --model-name-or-path "${MODEL}" \
  --data-path "${DATA_PATH}" \
  --output-dir "${OUT_DIR}" \
  --deepspeed "${DS_CONFIG}" \
  --per-device-train-batch-size 8 \
  --gradient-accumulation-steps 4 \
  --learning-rate 5e-6 \
  --num-train-epochs 1 \
  --smoke-batches 50 \
  --model-max-length 16384 \
  --min-pixels 50176 --max-pixels 50176 \
  --video-max-frames 32 --video-min-frames 32 \
  --weight-decay 0 \
  --warmup-ratio 0.03 \
  --max-grad-norm 1 \
  --lr-scheduler-type cosine \
  --logging-steps 1 \
  --save-strategy no \
  --dataloader-num-workers 0 \
  --max-retry-per-batch 3 \
  --bf16 \
  --gradient-checkpointing

echo "[smoke] finished: $(date)"
echo "[smoke] output: ${OUT_DIR}"
