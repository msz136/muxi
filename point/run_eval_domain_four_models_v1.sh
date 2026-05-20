#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/msz/point
EVAL_SCRIPT=${ROOT}/eval_semantic_nav_box.py
REGION_EVAL=/data/msz/opd_project/data/semantic_nav_goal_eval_v1/semantic_nav_goal_reg_eval_v1.jsonl
OBJECT_EVAL=/data/msz/opd_project/data/semantic_nav_box_v1/eval/semantic_nav_box_heldout_object_ref_eval_v1.jsonl
OUT_ROOT=${ROOT}/outputs/eval_domain_four_models_v1_$(date +%Y%m%d_%H%M%S)
SEED=${SEED:-20260520}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}

mkdir -p "${OUT_ROOT}"
echo "${OUT_ROOT}" > "${ROOT}/outputs/latest_domain_four_models_eval_dir.txt"

export MACA_PATH=${MACA_PATH:-/opt/maca-3.5.3}
export PATH="${MACA_PATH}/bin:${MACA_PATH}/mxgpu_llvm/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH}"
export LD_LIBRARY_PATH="${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${MACA_PATH}/ompi/lib:${LD_LIBRARY_PATH:-}"
export DS_ACCELERATOR=cuda
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

declare -A MODELS=(
  [base]="/data/msz/models/8b_base"
  [obj_expert]="/data/msz/models/expert_obj_v1"
  [reg_expert]="/data/msz/models/expert_reg_v1"
  [opd_v1]="/data/msz/point/outputs/opd_online_mix_v1_2048_final_20260520_142529"
)

cat > "${OUT_ROOT}/manifest.json" <<EOF
{
  "out_root": "${OUT_ROOT}",
  "seed": ${SEED},
  "max_new_tokens": ${MAX_NEW_TOKENS},
  "region_eval": "${REGION_EVAL}",
  "object_eval": "${OBJECT_EVAL}",
  "models": {
    "base": "${MODELS[base]}",
    "obj_expert": "${MODELS[obj_expert]}",
    "reg_expert": "${MODELS[reg_expert]}",
    "opd_v1": "${MODELS[opd_v1]}"
  }
}
EOF

run_one() {
  local name=$1
  local model=$2
  local split=$3
  local data=$4
  local samples=$5
  local gpu=$6
  local out="${OUT_ROOT}/${name}__${split}.jsonl"
  local log="${OUT_ROOT}/${name}__${split}.log"
  local rc_file="${OUT_ROOT}/${name}__${split}.rc"
  (
    set +e
    export CUDA_VISIBLE_DEVICES="${gpu}"
    echo "[eval] start name=${name} split=${split} gpu=${gpu} model=${model}" > "${log}"
    /opt/conda/bin/python3 "${EVAL_SCRIPT}" \
      --model "${model}" \
      --data "${data}" \
      --out "${out}" \
      --num-samples "${samples}" \
      --seed "${SEED}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --device cuda:0 >> "${log}" 2>&1
    rc=$?
    echo "${rc}" > "${rc_file}"
    echo "[eval] done name=${name} split=${split} rc=${rc}" >> "${log}"
    exit 0
  ) &
}

aggregate() {
  /opt/conda/bin/python3 - "${OUT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
rows = []
for summary_path in sorted(out_root.glob("*.summary.json")):
    stem = summary_path.name.removesuffix(".summary.json")
    if "__" not in stem:
        continue
    model_name, split = stem.split("__", 1)
    rc_path = out_root / f"{model_name}__{split}.rc"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows.append({
        "model_name": model_name,
        "split": split,
        "rc": rc_path.read_text(encoding="utf-8").strip() if rc_path.exists() else None,
        "model": summary.get("model"),
        "data": summary.get("data"),
        "num_samples": summary.get("num_samples"),
        "format_ok": summary.get("format_ok"),
        "format_rate": summary.get("format_rate"),
        "mean_iou_parseable": summary.get("mean_iou"),
        "mean_iou_all": (summary.get("mean_iou", 0.0) or 0.0) * (summary.get("format_rate", 0.0) or 0.0),
        "iou_at_0_3_parseable": summary.get("iou_at_0_3"),
        "iou_at_0_3_all": (summary.get("iou_at_0_3", 0.0) or 0.0) * (summary.get("format_rate", 0.0) or 0.0),
        "iou_at_0_5_parseable": summary.get("iou_at_0_5"),
        "iou_at_0_5_all": (summary.get("iou_at_0_5", 0.0) or 0.0) * (summary.get("format_rate", 0.0) or 0.0),
        "mean_center_error": summary.get("mean_center_error"),
        "mean_coord_mae": summary.get("mean_coord_mae"),
        "results_path": summary.get("results_path"),
    })

order = {"base": 0, "obj_expert": 1, "reg_expert": 2, "opd_v1": 3}
rows.sort(key=lambda row: (row["split"], order.get(row["model_name"], 99)))
(out_root / "summary_matrix.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

headers = [
    "split", "model_name", "rc", "num_samples", "format_ok", "format_rate",
    "mean_iou_parseable", "mean_iou_all", "iou_at_0_3_all", "iou_at_0_5_all",
    "mean_center_error", "mean_coord_mae",
]
lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
for row in rows:
    vals = []
    for key in headers:
        value = row.get(key)
        vals.append(f"{value:.6f}" if isinstance(value, float) else str(value))
    lines.append("| " + " | ".join(vals) + " |")
(out_root / "summary_matrix.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({"out_root": str(out_root), "rows": len(rows)}, ensure_ascii=False), flush=True)
PY
}

main() {
  test -f "${EVAL_SCRIPT}"
  test -f "${REGION_EVAL}"
  test -f "${OBJECT_EVAL}"
  for path in "${MODELS[@]}"; do
    test -f "${path}/config.json"
  done

  echo "[domain matrix] out_root=${OUT_ROOT}"
  run_one base "${MODELS[base]}" latest_region "${REGION_EVAL}" 144 0
  run_one obj_expert "${MODELS[obj_expert]}" latest_region "${REGION_EVAL}" 144 1
  run_one reg_expert "${MODELS[reg_expert]}" latest_region "${REGION_EVAL}" 144 2
  run_one opd_v1 "${MODELS[opd_v1]}" latest_region "${REGION_EVAL}" 144 3
  run_one base "${MODELS[base]}" latest_object "${OBJECT_EVAL}" 1251 4
  run_one obj_expert "${MODELS[obj_expert]}" latest_object "${OBJECT_EVAL}" 1251 5
  run_one reg_expert "${MODELS[reg_expert]}" latest_object "${OBJECT_EVAL}" 1251 6
  run_one opd_v1 "${MODELS[opd_v1]}" latest_object "${OBJECT_EVAL}" 1251 7
  wait
  aggregate
  echo "[domain matrix] done ${OUT_ROOT}"
}

main "$@"
