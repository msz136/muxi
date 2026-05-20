#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/msz/point
DATA_DIR=${ROOT}/data_opd
LOG_DIR=${ROOT}/logs
TOTAL=${TOTAL:-2048}
SEED=${SEED:-20260520}
RUN_ID=${RUN_ID:-opd_online_mix_v1_${TOTAL}_final_$(date +%Y%m%d_%H%M%S)}
OUT_DIR=${ROOT}/outputs/${RUN_ID}
LOG_FILE=${LOG_DIR}/${RUN_ID}.log

BASE_MODEL=/data/msz/models/8b_base
OBJ_TEACHER=/data/msz/models/expert_obj_v1
REG_TEACHER=/data/msz/models/expert_reg_v1
DS_CONFIG=${ROOT}/configs/zero2.json

MIX_PATH=${DATA_DIR}/opd_mix_v1_${TOTAL}_mediaok_seed${SEED}.jsonl

NUM_GPUS=8
PER_DEVICE_BS=1
GRAD_ACC=4
LEARNING_RATE=${LEARNING_RATE:-1e-6}
OPD_MAX_NEW_TOKENS=${OPD_MAX_NEW_TOKENS:-32}

mkdir -p "${DATA_DIR}" "${LOG_DIR}" "${OUT_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

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
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false

echo "[opd-online-v1] start=$(date)"
echo "[opd-online-v1] run_id=${RUN_ID}"
echo "[opd-online-v1] base=${BASE_MODEL}"
echo "[opd-online-v1] obj_teacher=${OBJ_TEACHER}"
echo "[opd-online-v1] reg_teacher=${REG_TEACHER}"
echo "[opd-online-v1] mix=${MIX_PATH}"
echo "[opd-online-v1] output=${OUT_DIR}"
echo "[opd-online-v1] total=${TOTAL}"
echo "[opd-online-v1] ratios=obj35/reg35/general30"
echo "[opd-online-v1] routing=obj->obj teacher rollout/logit, reg->reg teacher rollout/logit, general duplicated with 0.5 obj + 0.5 reg loss"
echo "[opd-online-v1] no_precomputed_logits=true"
mx-smi || true

test -f "${BASE_MODEL}/config.json"
test -f "${OBJ_TEACHER}/config.json"
test -f "${REG_TEACHER}/config.json"
test -f "${DS_CONFIG}"

if [[ ! -f "${MIX_PATH}" ]]; then
  /opt/conda/bin/python3 "${ROOT}/build_opd_mix_v1.py" \
    --output "${MIX_PATH}" \
    --total "${TOTAL}" \
    --seed "${SEED}"
else
  echo "[opd-online-v1] reusing mix=${MIX_PATH}"
fi

ROWS=$(wc -l < "${MIX_PATH}")
EFFECTIVE_BS=$((NUM_GPUS * PER_DEVICE_BS * GRAD_ACC))
echo "[opd-online-v1] original_rows=${ROWS}"
echo "[opd-online-v1] per_device_batch_size=${PER_DEVICE_BS}"
echo "[opd-online-v1] gradient_accumulation_steps=${GRAD_ACC}"
echo "[opd-online-v1] learning_rate=${LEARNING_RATE}"
echo "[opd-online-v1] effective_batch_size=${EFFECTIVE_BS}"
echo "[opd-online-v1] opd_max_new_tokens=${OPD_MAX_NEW_TOKENS}"
echo "[opd-online-v1] save_policy=final_model_only_no_intermediate_checkpoints"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
deepspeed --num_gpus="${NUM_GPUS}" --master_port 29524 "${ROOT}/train_opd_online_vl.py" train \
  --model-name-or-path "${BASE_MODEL}" \
  --obj-teacher "${OBJ_TEACHER}" \
  --reg-teacher "${REG_TEACHER}" \
  --data-path "${MIX_PATH}" \
  --output-dir "${OUT_DIR}" \
  --deepspeed "${DS_CONFIG}" \
  --num-train-epochs 1 \
  --per-device-train-batch-size "${PER_DEVICE_BS}" \
  --gradient-accumulation-steps "${GRAD_ACC}" \
  --learning-rate "${LEARNING_RATE}" \
  --opd-max-new-tokens "${OPD_MAX_NEW_TOKENS}" \
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
  --dataloader-num-workers 0 \
  --optim adamw_torch \
  --bf16 \
  --gradient-checkpointing

echo "[opd-online-v1] done=$(date)"
echo "[opd-online-v1] output=${OUT_DIR}"
