#!/usr/bin/env bash
# Full SFT training launch — tmux session "full"
# Usage: bash point/launch_full_sft.sh
set -euo pipefail

ROOT=/data/msz/point
DATA_PATH="${ROOT}/data_expert/expert_mix_v1_shuffled.jsonl"
MODEL=/data/msz/models/Qwen3-VL-8B-Instruct
DS_CONFIG="${ROOT}/configs/zero2.json"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${ROOT}/outputs/fullsft_v1_${RUN_ID}"

mkdir -p "${ROOT}"/{outputs,logs,bad} "${OUT_DIR}"

# Log to file + stdout
LOG_FILE="${ROOT}/logs/fullsft_${RUN_ID}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "============================================"
echo "[fullsft] start: $(date)"
echo "[fullsft] data=${DATA_PATH}  ($(wc -l < "${DATA_PATH}") lines)"
echo "[fullsft] model=${MODEL}"
echo "[fullsft] output=${OUT_DIR}"
echo "[fullsft] save-steps=600"
echo "============================================"

# Environment
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DS_ACCELERATOR=cuda
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# MACA NCCL workarounds
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=3600

test -f "${MODEL}/config.json" || { echo "FATAL: missing model"; exit 1; }
test -f "${DATA_PATH}" || { echo "FATAL: missing data: ${DATA_PATH}"; exit 1; }

# ---- batch size probing ----
PROBE_OUT="${ROOT}/outputs/probe_${RUN_ID}"
CHOSEN_BS=""
for BS in 8 6 4 2 1; do
  echo "[fullsft] probe per_device_train_batch_size=${BS}"
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
  echo "[fullsft] bs=${BS} OOM, trying smaller"
done
test -n "${CHOSEN_BS}" || { echo "[fullsft] FATAL: all batch sizes OOM"; exit 1; }
echo "${CHOSEN_BS}" > "${ROOT}/configs/chosen_batch_size.txt"
echo "[fullsft] using per_device_batch_size=${CHOSEN_BS}"

echo "Effective BS = 8 GPUs x ${CHOSEN_BS} x 4 GA steps = $((8 * CHOSEN_BS * 4))"

# ---- full SFT ----
echo "[fullsft] launching full SFT with save-steps=600"
deepspeed --num_gpus=8 "${ROOT}/expert_sft.py" train \
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
  --save-strategy steps --save-steps 600 --save-total-limit 2 \
  --dataloader-num-workers 4 --max-retry-per-batch 3 \
  --bf16 --gradient-checkpointing

echo "[fullsft] done: ${OUT_DIR}"
echo "[fullsft] end: $(date)"
