#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/msz/point
DATA_PATH="${DATA_PATH:-${ROOT}/data_expert/expert_grounding_mix.jsonl}"
MODEL="${MODEL:-/data/msz/models/Qwen3-VL-8B-Instruct}"
LOG_DIR="${ROOT}/logs"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${ROOT}/outputs/expert_sft_${RUN_ID}"
DS_CONFIG="${ROOT}/configs/zero2.json"

mkdir -p "${ROOT}"/{configs,outputs,logs,bad} "${LOG_DIR}" "${OUT_DIR}"

exec > >(tee -a "${LOG_DIR}/full_sft_${RUN_ID}.log") 2>&1

echo "[full_sft] start: $(date)"
echo "[full_sft] data=${DATA_PATH}"
echo "[full_sft] model=${MODEL}"

# Environment
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NCCL_DEBUG=INFO
export DS_ACCELERATOR=cuda
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

test -f "${MODEL}/config.json" || { echo "missing model"; exit 1; }
test -f "${DATA_PATH}" || { echo "missing data: ${DATA_PATH}"; exit 1; }

echo "[full_sft] sample count: $(wc -l < "${DATA_PATH}")"

# ---- batch size probing ----
PROBE_OUT="${ROOT}/outputs/probe_${RUN_ID}"
CHOSEN_BS=""
for BS in 8 6 4 2 1; do
  echo "[full_sft] probe per_device_train_batch_size=${BS}"
  mkdir -p "${PROBE_OUT}_bs${BS}"
  if deepspeed --num_gpus=8 "${ROOT}/expert_sft.py" smoke \
    --model-name-or-path "${MODEL}" \
    --data-path "${DATA_PATH}" \
    --output-dir "${PROBE_OUT}_bs${BS}" \
    --deepspeed "${DS_CONFIG}" \
    --per-device-train-batch-size "${BS}" \
    --gradient-accumulation-steps 4 \
    --learning-rate 5e-6 \
    --smoke-batches 4 \
    --model-max-length 16384 \
    --min-pixels 50176 --max-pixels 50176 \
    --weight-decay 0 --warmup-ratio 0.03 --max-grad-norm 1 \
    --lr-scheduler-type cosine --logging-steps 1 \
    --save-strategy no --dataloader-num-workers 4 \
    --max-retry-per-batch 0 \
    --bf16 --gradient-checkpointing; then
    CHOSEN_BS="${BS}"; break
  fi
  echo "[full_sft] bs=${BS} OOM, trying smaller"
done
test -n "${CHOSEN_BS}" || { echo "all batch sizes OOM"; exit 1; }
echo "${CHOSEN_BS}" > "${ROOT}/configs/chosen_batch_size.txt"
echo "[full_sft] using per_device_batch_size=${CHOSEN_BS}"

# ---- full SFT ----
echo "[full_sft] launching training"
for TRY in 1 2 3; do
  echo "[full_sft] attempt ${TRY}/3"
  if deepspeed --num_gpus=8 "${ROOT}/expert_sft.py" train \
    --model-name-or-path "${MODEL}" \
    --data-path "${DATA_PATH}" \
    --output-dir "${OUT_DIR}" \
    --deepspeed "${DS_CONFIG}" \
    --per-device-train-batch-size "${CHOSEN_BS}" \
    --gradient-accumulation-steps 4 \
    --learning-rate 5e-6 \
    --num-train-epochs 1 \
    --model-max-length 16384 \
    --min-pixels 50176 --max-pixels 50176 \
    --video-max-frames 32 --video-min-frames 32 \
    --weight-decay 0 --warmup-ratio 0.03 --max-grad-norm 1 \
    --lr-scheduler-type cosine --logging-steps 1 \
    --save-strategy steps --save-steps 1000 --save-total-limit 1 \
    --dataloader-num-workers 4 --max-retry-per-batch 3 \
    --bf16 --gradient-checkpointing; then
    echo "[full_sft] done: ${OUT_DIR}"
    exit 0
  fi
  sleep $((TRY * 60))
done

echo "[full_sft] failed after 3 retries"
exit 1
