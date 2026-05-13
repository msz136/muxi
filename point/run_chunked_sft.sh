#!/usr/bin/env bash
set -euo pipefail

# Chunked full SFT: each chunk trains independently for 1 epoch.
# Model weights carry forward via HF format; optimizer/scheduler reset per chunk.
#
# Why independent (not resume_from_checkpoint):
#   HF Trainer's epoch counter blocks resume when dataset changes.
#   num_train_epochs=1 → epoch=1.0 after chunk1, range(1,1)=empty → skip on resume.
#   num_train_epochs>1 → Trainer loops same chunk data N times – undesirable.
#   Trade-off: ~2 warmup steps per chunk × ~104 chunks ≈ negligible overhead.

ROOT=/data/msz/point
FULL_DATA="${ROOT}/data_expert/expert_mix_v1.jsonl"
ORIGINAL_MODEL="/data/msz/models/Qwen3-VL-8B-Instruct"
DS_CONFIG="${ROOT}/configs/zero2.json"

BS=6
GA=4
EFFECTIVE_BATCH=$((8 * BS * GA))          # 192
SAMPLES_PER_BATCH=$((BS * GA))            # 24

CHUNK_ROWS=15000
TOTAL_ROWS=$(wc -l < "$FULL_DATA")
NUM_CHUNKS=$(( (TOTAL_ROWS + CHUNK_ROWS - 1) / CHUNK_ROWS ))
TOTAL_STEPS=$(( (TOTAL_ROWS + EFFECTIVE_BATCH - 1) / EFFECTIVE_BATCH ))

RUN_ID="$(date +%Y%m%d_%H%M%S)"
RESUME_CHUNK="${1:-1}"  # chunk number to start from (1-based)

BASE_DIR="${ROOT}/outputs/chunked_sft_v1_20260509_105943"
if [ "$RESUME_CHUNK" -gt 1 ]; then
    # Resume from existing output directory, using previous chunk's model
    PREV_CHUNK=$((RESUME_CHUNK - 1))
    MODEL_TO_CHECK="${BASE_DIR}/chunk_${PREV_CHUNK}"
else
    BASE_DIR="${ROOT}/outputs/chunked_sft_v1_${RUN_ID}"
    MODEL_TO_CHECK=""
fi
mkdir -p "${BASE_DIR}"
LOG_DIR="${ROOT}/logs"
mkdir -p "${LOG_DIR}"

exec > >(tee -a "${LOG_DIR}/chunked_sft_${RUN_ID}.log") 2>&1

echo "=== Chunked SFT v1 ==="
echo "start: $(date)"
echo "base dir: ${BASE_DIR}"
echo "total: ${TOTAL_ROWS} rows, ${TOTAL_STEPS} steps, ${NUM_CHUNKS} chunks"
echo "per chunk: ~${CHUNK_ROWS} rows, effective batch ${EFFECTIVE_BATCH}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=3600
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export DS_ACCELERATOR=cuda
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

START=$(( (RESUME_CHUNK - 1) * CHUNK_ROWS + 1 ))
CHUNK=$RESUME_CHUNK

if [ "$RESUME_CHUNK" -gt 1 ]; then
    MODEL_TO_LOAD="${BASE_DIR}/chunk_${PREV_CHUNK}"
    if [ ! -f "${MODEL_TO_LOAD}/config.json" ]; then
        echo "ERROR: no model found at ${MODEL_TO_LOAD}" >&2
        exit 1
    fi
else
    MODEL_TO_LOAD="$ORIGINAL_MODEL"
fi

echo "=== Starting from chunk ${RESUME_CHUNK} (row ${START}) ==="

while [ "$START" -le "$TOTAL_ROWS" ]; do
    END=$((START + CHUNK_ROWS - 1))
    [ "$END" -gt "$TOTAL_ROWS" ] && END=$TOTAL_ROWS

    ACTUAL_ROWS=$((END - START + 1))
    ACTUAL_BATCHES=$(( (ACTUAL_ROWS + SAMPLES_PER_BATCH - 1) / SAMPLES_PER_BATCH ))
    STEPS=$(( (ACTUAL_ROWS + EFFECTIVE_BATCH - 1) / EFFECTIVE_BATCH ))

    CHUNK_OUT="${BASE_DIR}/chunk_${CHUNK}"
    CHUNK_FILE="${BASE_DIR}/_chunk_${CHUNK}.jsonl"
    mkdir -p "$CHUNK_OUT"

    echo "[chunk ${CHUNK}/${NUM_CHUNKS}] rows ${START}-${END} (${ACTUAL_ROWS}r, ~${STEPS}s)"
    sed -n "${START},${END}p" "$FULL_DATA" > "$CHUNK_FILE"
    echo "[chunk ${CHUNK}] model: ${MODEL_TO_LOAD}"

    deepspeed --num_gpus=8 "${ROOT}/expert_sft.py" smoke \
        --model-name-or-path "${MODEL_TO_LOAD}" \
        --data-path "${CHUNK_FILE}" \
        --output-dir "${CHUNK_OUT}" \
        --deepspeed "${DS_CONFIG}" \
        --per-device-train-batch-size "$BS" \
        --gradient-accumulation-steps "$GA" \
        --learning-rate 5e-6 \
        --num-train-epochs 1 \
        --smoke-batches "$ACTUAL_BATCHES" \
        --model-max-length 16384 \
        --min-pixels 50176 --max-pixels 50176 \
        --video-max-frames 32 --video-min-frames 32 \
        --weight-decay 0 --warmup-ratio 0.03 --max-grad-norm 1 \
        --lr-scheduler-type cosine --logging-steps 1 \
        --save-strategy epoch \
        --dataloader-num-workers 0 --max-retry-per-batch 3 --batch-timeout 60 \
        --bf16 --gradient-checkpointing

    # After training, smoke() called trainer.save_model() → HF weights in CHUNK_OUT/
    MODEL_TO_LOAD="$CHUNK_OUT"
    # Verify model files exist
    if [ ! -f "${CHUNK_OUT}/config.json" ]; then
        echo "[chunk ${CHUNK}] ERROR: no config.json in ${CHUNK_OUT}" >&2
        exit 1
    fi

    rm -f "$CHUNK_FILE"
    echo "[chunk ${CHUNK}] done. saves: $(du -sh "$CHUNK_OUT" | cut -f1)"
    echo "[chunk ${CHUNK}] progress: $((END * 100 / TOTAL_ROWS))% (${END}/${TOTAL_ROWS})"

    START=$((END + 1))
    CHUNK=$((CHUNK + 1))
done

echo "[done] finished: $(date)"
echo "[done] final model: ${MODEL_TO_LOAD}"
echo "[done] output: ${BASE_DIR}"
