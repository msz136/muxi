#!/usr/bin/env bash
set -euo pipefail
cd /data/msz/point
RUN_DIR="$1"
EVAL_PATH="/data/msz/point/eval_raw_holdout_v1/raw_holdout_eval_v1_10k.jsonl"
SCRIPT="/data/msz/point/eval_qwen_vl_raw_holdout.py"
SUMMARY_SCRIPT="/data/msz/point/summarize_raw_holdout_eval.py"
PYTHON="/opt/conda/bin/python3"
BATCH_SIZE=24
MAX_NEW_TOKENS=64
export MACA_PATH="${MACA_PATH:-/opt/maca-3.5.3}"
export PATH="${MACA_PATH}/bin:${MACA_PATH}/mxgpu_llvm/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH}"
export LD_LIBRARY_PATH="${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${MACA_PATH}/ompi/lib:${LD_LIBRARY_PATH:-}"
export DS_ACCELERATOR=cuda
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pids=()
names=()
while IFS=$ \t read -r gpu name model_path; do
  model_out="${RUN_DIR}/${name}"
  mkdir -p "${model_out}"
  echo "launch gpu=${gpu} model=${name} path=${model_path}"
  CUDA_VISIBLE_DEVICES="${gpu}" DS_ACCELERATOR=cuda PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${PYTHON}" "${SCRIPT}" \
      --model-name "${name}" \
      --model-path "${model_path}" \
      --eval-path "${EVAL_PATH}" \
      --out-dir "${model_out}" \
      --batch-size "${BATCH_SIZE}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      > "${RUN_DIR}/logs/${name}.log" 2>&1 &
  pid=$!
  echo "${pid}" > "${model_out}/pid.txt"
  pids+=("${pid}")
  names+=("${name}")
done < "${RUN_DIR}/models.tsv"
echo "All launches submitted."
status=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "FAILED model=${names[$i]} pid=${pids[$i]}" | tee -a "${RUN_DIR}/failed_models.txt"
    status=1
  fi
done
if [[ "${status}" -ne 0 ]]; then
  echo "Some evals failed; see ${RUN_DIR}/failed_models.txt"
  exit "${status}"
fi
echo "All checkpoint evals completed."
"${PYTHON}" "${SUMMARY_SCRIPT}" --run-dir "${RUN_DIR}" > "${RUN_DIR}/comparison_summary_stdout.json"
