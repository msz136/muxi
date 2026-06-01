#!/usr/bin/env bash
set -euo pipefail

EVAL_PATH="${EVAL_PATH:-/data/msz/point/eval_raw_holdout_v1/raw_holdout_eval_v1_10k.jsonl}"
MODEL_ROOT="${MODEL_ROOT:-/data/msz/models/opd_seed0_five_online_300k_mb4_save500_zero3_leftpad_v1}"
OUT_ROOT="${OUT_ROOT:-/data/msz/point/eval_raw_holdout_v1/opd_ckpts_$(date +%Y%m%d_%H%M%S)}"
BATCH_SIZE="${BATCH_SIZE:-256}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
SCRIPT="${SCRIPT:-/data/msz/point/eval_qwen_vl_raw_holdout.py}"
SUMMARY_SCRIPT="${SUMMARY_SCRIPT:-/data/msz/point/summarize_raw_holdout_eval.py}"
PYTHON="${PYTHON:-/opt/conda/bin/python3}"

export MACA_PATH="${MACA_PATH:-/opt/maca-3.5.3}"
export PATH="${MACA_PATH}/bin:${MACA_PATH}/mxgpu_llvm/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH}"
export LD_LIBRARY_PATH="${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${MACA_PATH}/ompi/lib:${LD_LIBRARY_PATH:-}"
export DS_ACCELERATOR=cuda
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "${OUT_ROOT}/logs"
cat > "${OUT_ROOT}/models.tsv" <<EOF
0	opd_ckpt_500	${MODEL_ROOT}/checkpoint-500
1	opd_ckpt_1000	${MODEL_ROOT}/checkpoint-1000
2	opd_ckpt_1500	${MODEL_ROOT}/checkpoint-1500
3	opd_ckpt_2000	${MODEL_ROOT}/checkpoint-2000
EOF

echo "${OUT_ROOT}" > /data/msz/point/eval_raw_holdout_v1/latest_opd_ckpt_run_dir.txt
echo "Launching OPD checkpoint evals into ${OUT_ROOT}"

pids=()
names=()
while IFS=$'\t' read -r gpu name model_path; do
  model_out="${OUT_ROOT}/${name}"
  mkdir -p "${model_out}"
  echo "launch gpu=${gpu} model=${name} path=${model_path}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  DS_ACCELERATOR=cuda \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${PYTHON}" "${SCRIPT}" \
    --model-name "${name}" \
    --model-path "${model_path}" \
    --eval-path "${EVAL_PATH}" \
    --out-dir "${model_out}" \
    --batch-size "${BATCH_SIZE}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    > "${OUT_ROOT}/logs/${name}.log" 2>&1 &
  pid=$!
  echo "${pid}" > "${model_out}/pid.txt"
  pids+=("${pid}")
  names+=("${name}")
done < "${OUT_ROOT}/models.tsv"

echo "All launches submitted."
status=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "FAILED model=${names[$i]} pid=${pids[$i]}" | tee -a "${OUT_ROOT}/failed_models.txt"
    status=1
  fi
done
if [[ "${status}" -ne 0 ]]; then
  echo "Some evals failed; see ${OUT_ROOT}/failed_models.txt"
  exit "${status}"
fi

echo "All OPD checkpoint evals completed."
"${PYTHON}" "${SUMMARY_SCRIPT}" --run-dir "${OUT_ROOT}" > "${OUT_ROOT}/comparison_summary_stdout.json"
