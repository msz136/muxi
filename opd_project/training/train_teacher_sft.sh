#!/usr/bin/env bash
# OPD Pointing Expert - 第一步: 训练 Pointing Teacher (SFT)
# 使用已有的 expert_sft.py 框架训练一个 pointing expert 作为 OPD 教师
#
# 运行: bash /data/msz/opd_project/training/train_teacher_sft.sh
set -euo pipefail

ROOT="/data/msz/opd_project"
POINT_ROOT="/data/msz/point"
MODEL="/data/msz/models/Qwen3-VL-8B-Instruct"
DS_CONFIG="${ROOT}/configs/ds_zero2.json"
RUN_ID="teacher_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${ROOT}/outputs/${RUN_ID}"
LOG_DIR="${ROOT}/logs"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

exec > >(tee -a "${LOG_DIR}/${RUN_ID}.log") 2>&1

echo "============================================"
echo " Training Pointing Teacher (SFT)"
echo " $(date)"
echo "============================================"
echo "Model: ${MODEL}"
echo "Output: ${OUT_DIR}"

# 使用已有的 grounding point 数据 (已经过验证)
# 优先使用 data_robopoint_new 中的数据 (861K robopoint samples)
DATA_PATH="${POINT_ROOT}/data_robopoint_new/grounding_point.jsonl"
if [ ! -f "${DATA_PATH}" ]; then
    DATA_PATH="${POINT_ROOT}/data_expert/grounding_point.jsonl"
fi

echo "Data: ${DATA_PATH}"
echo "Samples: $(wc -l < "${DATA_PATH}")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export DS_ACCELERATOR=cuda
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

test -f "${MODEL}/config.json" || { echo "ERROR: model not found"; exit 1; }
test -f "${DATA_PATH}" || { echo "ERROR: data not found"; exit 1; }

# Batch size probing
CHOSEN_BS=""
for BS in 4 2 1; do
    echo "[probe] trying batch_size=${BS}"
    PROBE_DIR="${ROOT}/outputs/probe_${RUN_ID}_bs${BS}"
    mkdir -p "${PROBE_DIR}"
    if deepspeed --num_gpus=8 "${POINT_ROOT}/expert_sft.py" smoke \
        --model-name-or-path "${MODEL}" \
        --data-path "${DATA_PATH}" \
        --output-dir "${PROBE_DIR}" \
        --deepspeed "${DS_CONFIG}" \
        --per-device-train-batch-size "${BS}" \
        --gradient-accumulation-steps 4 \
        --learning-rate 5e-6 \
        --smoke-batches 4 \
        --model-max-length 4096 \
        --min-pixels 50176 --max-pixels 50176 \
        --weight-decay 0 --warmup-ratio 0.03 --max-grad-norm 1 \
        --lr-scheduler-type cosine --logging-steps 1 \
        --save-strategy no --dataloader-num-workers 4 \
        --max-retry-per-batch 0 \
        --bf16 --gradient-checkpointing; then
        CHOSEN_BS="${BS}"
        break
    fi
    echo "[probe] bs=${BS} OOM"
done

test -n "${CHOSEN_BS}" || { echo "ERROR: all batch sizes OOM"; exit 1; }
echo "[train] using batch_size=${CHOSEN_BS}"

# Full training
deepspeed --num_gpus=8 "${POINT_ROOT}/expert_sft.py" train \
    --model-name-or-path "${MODEL}" \
    --data-path "${DATA_PATH}" \
    --output-dir "${OUT_DIR}" \
    --deepspeed "${DS_CONFIG}" \
    --per-device-train-batch-size "${CHOSEN_BS}" \
    --gradient-accumulation-steps 4 \
    --learning-rate 5e-6 \
    --num-train-epochs 1 \
    --model-max-length 4096 \
    --min-pixels 50176 --max-pixels 50176 \
    --weight-decay 0 --warmup-ratio 0.03 --max-grad-norm 1 \
    --lr-scheduler-type cosine --logging-steps 10 \
    --save-strategy steps --save-steps 2000 --save-total-limit 2 \
    --dataloader-num-workers 4 --max-retry-per-batch 3 \
    --bf16 --gradient-checkpointing

echo ""
echo "============================================"
echo " Teacher Training Complete"
echo " Checkpoint: ${OUT_DIR}"
echo "============================================"
echo ""
echo "Next: update teacher path in opd_pointing_cluster.yaml"
echo "  teacher.name_or_path: ${OUT_DIR}"
