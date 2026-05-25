#!/usr/bin/env bash
set -euo pipefail

EVAL_PATH="${EVAL_PATH:-/data/msz/point/eval_raw_holdout_v1/raw_holdout_eval_v1_10k.jsonl}"
OUT_ROOT="${OUT_ROOT:-/data/msz/point/eval_raw_holdout_v1/runs_8models_$(date +%Y%m%d_%H%M%S)}"
BATCH_SIZE="${BATCH_SIZE:-24}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
LIMIT="${LIMIT:-}"
SCRIPT="${SCRIPT:-/data/msz/point/eval_qwen_vl_raw_holdout.py}"
SUMMARY_SCRIPT="${SUMMARY_SCRIPT:-/data/msz/point/summarize_raw_holdout_eval.py}"
PYTHON="${PYTHON:-/opt/conda/bin/python3}"

export MACA_PATH="${MACA_PATH:-/opt/maca-3.5.3}"
export PATH="${MACA_PATH}/bin:${MACA_PATH}/mxgpu_llvm/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH}"
export LD_LIBRARY_PATH="${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${MACA_PATH}/ompi/lib:${LD_LIBRARY_PATH:-}"
export DS_ACCELERATOR=cuda
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "${OUT_ROOT}/logs"
cat > "${OUT_ROOT}/models.tsv" <<'EOF'
0	base	/data/msz/models/8b_base
1	qwen3_vl_8b_instruct	/data/msz/models/Qwen3-VL-8B-Instruct
2	general_reasoning_expert	/data/msz/models/seed0_general_reasoning_expert_100k_mb1_filtered_stdtrainer_maxnorm5_v1
3	robopoint_expert	/data/msz/models/seed0_robopoint_expert_100k_mb1_filtered_stdtrainer_maxnorm5_v1
4	general_obj_expert	/data/msz/models/seed0_general_obj_expert_100k_mb1_filtered_stdtrainer_maxnorm5_v1
5	region_expert	/data/msz/models/seed0_region_expert_100k_mb1_filtered_stdtrainer_maxnorm5_v1
6	spatial_rel_expert	/data/msz/models/seed0_spatial_rel_expert_100k_mb1_filtered_stdtrainer_maxnorm5_v1
7	opd_final	/data/msz/models/opd_seed0_five_online_300k_mb4_save500_zero3_leftpad_v1/checkpoint-2344
EOF

echo "${OUT_ROOT}" > /data/msz/point/eval_raw_holdout_v1/latest_run_dir.txt
echo "Launching evals into ${OUT_ROOT}"

pids=()
names=()
while IFS=$'\t' read -r gpu name model_path; do
  model_out="${OUT_ROOT}/${name}"
  mkdir -p "${model_out}"
  echo "launch gpu=${gpu} model=${name} path=${model_path}"
  limit_args=()
  if [[ -n "${LIMIT}" ]]; then
    limit_args=(--limit "${LIMIT}")
  fi
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
    "${limit_args[@]}" \
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
echo "All evals completed."
"${PYTHON}" "${SUMMARY_SCRIPT}" --run-dir "${OUT_ROOT}" > "${OUT_ROOT}/comparison_summary_stdout.json"
