#!/usr/bin/env bash
set -euo pipefail

# Expert SFT on synthetic semantic-nav box-grounding data
# Data: semantic_nav_box_grounding_full_object_ref_v1_high_quality.jsonl (~41K rows)
# ~217 steps @ effective batch 192, 1 epoch, no intermediate checkpoints

ROOT=/data/msz/point
DATA_PATH="/data/msz/opd_project/data/semantic_nav_box_v1/semantic_nav_box_grounding_full_object_ref_v1_high_quality.jsonl"
MODEL="/data/msz/models/Qwen3-VL-8B-Instruct"
DS_CONFIG="${ROOT}/configs/zero2.json"
LOG_DIR="${ROOT}/logs"
RUN_ID="expertsft_semantic_nav_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${ROOT}/outputs/${RUN_ID}"

mkdir -p "${LOG_DIR}" "${OUT_DIR}" "${ROOT}/bad"

exec > >(tee -a "${LOG_DIR}/${RUN_ID}.log") 2>&1

echo "[expertsft] start: $(date)"
echo "[expertsft] data=${DATA_PATH}"
echo "[expertsft] model=${MODEL}"
echo "[expertsft] output=${OUT_DIR}"
echo "[expertsft] rows=$(wc -l < "${DATA_PATH}")"

# Preconditions
test -f "${MODEL}/config.json"  || { echo "ERROR: missing model ${MODEL}"; exit 1; }
test -f "${DATA_PATH}"          || { echo "ERROR: missing data ${DATA_PATH}"; exit 1; }
test -f "${DS_CONFIG}"          || { echo "ERROR: missing ds config ${DS_CONFIG}"; exit 1; }

# Environment
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export DS_ACCELERATOR=cuda
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=3600
export NCCL_DEBUG=WARN

# Batch size probe: 8 -> 6 -> 4 -> 2 -> 1
PROBE_OUT="${ROOT}/outputs/probe_${RUN_ID}"
CHOSEN_BS=""
for BS in 8 6 4 2 1; do
    echo "[expertsft] probing per_device_train_batch_size=${BS}"
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
        --save-strategy no \
        --dataloader-num-workers 4 \
        --max-retry-per-batch 0 \
        --bf16 --gradient-checkpointing 2>&1; then
        CHOSEN_BS="${BS}"
        break
    fi
    echo "[expertsft] bs=${BS} failed, trying smaller"
done
test -n "${CHOSEN_BS}" || { echo "ERROR: all batch sizes failed probe"; exit 1; }
echo "[expertsft] chosen per_device_batch_size=${CHOSEN_BS}"
echo "${CHOSEN_BS}" > "${ROOT}/configs/chosen_bs_semantic_nav.txt"

# Clean up probe dirs
rm -rf "${PROBE_OUT}_bs"*

# Full training — 1 epoch, no intermediate checkpoints, save final model only
echo "[expertsft] launching full training"
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
    --save-strategy no \
    --dataloader-num-workers 4 \
    --max-retry-per-batch 3 \
    --bf16 --gradient-checkpointing

echo "[expertsft] done: $(date)"
echo "[expertsft] output: ${OUT_DIR}"
