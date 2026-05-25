#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/msz/point
DATA_ROOT=${ROOT}/data_expert_seed0_v1_shuffled
MODEL_BASE=/data/msz/models/8b_base
OUT_BASE=/data/msz/models
RUN_TAG=seed0_100k_mb1_v1
LIMIT_SAMPLES=${LIMIT_SAMPLES:-100000}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export MACA_PATH=${MACA_PATH:-/opt/maca-3.5.3}
export PATH="${MACA_PATH}/bin:${MACA_PATH}/mxgpu_llvm/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH}"
export LD_LIBRARY_PATH="${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${MACA_PATH}/ompi/lib:${LD_LIBRARY_PATH:-}"
export DS_ACCELERATOR=cuda
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=3600

mkdir -p "${ROOT}/logs" "${ROOT}/outputs"

EXPERTS=(
  general_obj_expert
  general_reasoning_expert
  region_expert
  robopoint_expert
  spatial_rel_expert
)

cat > "${ROOT}/outputs/${RUN_TAG}_manifest.json" <<EOF
{
  "run_tag": "${RUN_TAG}",
  "model_base": "${MODEL_BASE}",
  "data_root": "${DATA_ROOT}",
  "out_base": "${OUT_BASE}",
  "limit_samples": ${LIMIT_SAMPLES},
  "per_device_train_batch_size": 1,
  "gradient_accumulation_steps": 4,
  "effective_batch_size": 32,
  "expected_steps_per_expert": 3125,
  "save_policy": "final_model_only_no_intermediate_checkpoints",
  "nan_policy": "abort_on_any_non_finite_logged_metric",
  "oom_policy": "stop_and_report_if_microbatch_1_ooms"
}
EOF

run_one() {
  local expert=$1
  local data_path="${DATA_ROOT}/${expert}/train_shuffled_seed20260520.jsonl"
  local output_dir="${OUT_BASE}/seed0_${expert}_100k_mb1_v1"
  local log_path="${ROOT}/logs/seed0_${expert}_100k_mb1_v1.log"

  if [[ -f "${output_dir}/trainer_state.json" && -f "${output_dir}/config.json" ]]; then
    echo "[skip] ${expert} already has final output at ${output_dir}" | tee -a "${ROOT}/logs/${RUN_TAG}.log"
    return 0
  fi

  test -f "${data_path}"
  test -f "${MODEL_BASE}/config.json"

  echo "[start] $(date '+%F %T') expert=${expert} mb=1 data=${data_path} output=${output_dir}" | tee -a "${ROOT}/logs/${RUN_TAG}.log"
  deepspeed --num_gpus=8 "${ROOT}/expert_sft.py" train \
    --model-name-or-path "${MODEL_BASE}" \
    --data-path "${data_path}" \
    --output-dir "${output_dir}" \
    --deepspeed "${ROOT}/configs/zero2.json" \
    --num-train-epochs 1 \
    --limit-samples "${LIMIT_SAMPLES}" \
    --per-device-train-batch-size 1 \
    --gradient-accumulation-steps 4 \
    --learning-rate 5e-6 \
    --weight-decay 0 \
    --warmup-ratio 0.03 \
    --max-grad-norm 1.0 \
    --lr-scheduler-type cosine \
    --logging-steps 1 \
    --model-max-length 16384 \
    --min-pixels 50176 \
    --max-pixels 50176 \
    --dataloader-num-workers 4 \
    --bf16 2>&1 | tee "${log_path}"

  echo "[done] $(date '+%F %T') expert=${expert} mb=1 output=${output_dir}" | tee -a "${ROOT}/logs/${RUN_TAG}.log"
}

main() {
  echo "[run] tag=${RUN_TAG} limit=${LIMIT_SAMPLES} mb=1 base=${MODEL_BASE}" | tee -a "${ROOT}/logs/${RUN_TAG}.log"
  for expert in "${EXPERTS[@]}"; do
    run_one "${expert}"
  done
  echo "[all_done] $(date '+%F %T') ${RUN_TAG}" | tee -a "${ROOT}/logs/${RUN_TAG}.log"
}

main "$@"
