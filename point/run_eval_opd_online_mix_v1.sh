#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/msz/point
MODEL=/data/msz/point/outputs/opd_online_mix_v1_2048_final_20260520_142529
REGION=/data/msz/opd_project/data/semantic_nav_goal_eval_v1/semantic_nav_goal_reg_eval_v1.jsonl
OBJECT=/data/msz/opd_project/data/semantic_nav_box_v1/eval/semantic_nav_box_heldout_object_ref_eval_v1.jsonl
OUT=${ROOT}/outputs/eval_opd_online_mix_v1_2048_$(date +%Y%m%d_%H%M%S)
SEED=${SEED:-20260520}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}

mkdir -p "${OUT}"
echo "${OUT}" > "${ROOT}/outputs/latest_opd_eval_dir.txt"

export MACA_PATH=${MACA_PATH:-/opt/maca-3.5.3}
export PATH="${MACA_PATH}/bin:${MACA_PATH}/mxgpu_llvm/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH}"
export LD_LIBRARY_PATH="${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${MACA_PATH}/ompi/lib:${LD_LIBRARY_PATH:-}"
export DS_ACCELERATOR=cuda
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

cat > "${OUT}/manifest.json" <<EOF
{
  "model": "${MODEL}",
  "region_eval": "${REGION}",
  "object_eval": "${OBJECT}",
  "seed": ${SEED},
  "max_new_tokens": ${MAX_NEW_TOKENS}
}
EOF

run_one() {
  local split=$1
  local data=$2
  local samples=$3
  local gpu=$4
  local out_jsonl="${OUT}/opd_online_mix_v1_2048__${split}.jsonl"
  local log="${OUT}/opd_online_mix_v1_2048__${split}.log"
  (
    set +e
    export CUDA_VISIBLE_DEVICES="${gpu}"
    /opt/conda/bin/python3 "${ROOT}/eval_semantic_nav_box.py" \
      --model "${MODEL}" \
      --data "${data}" \
      --out "${out_jsonl}" \
      --num-samples "${samples}" \
      --seed "${SEED}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --device cuda:0 > "${log}" 2>&1
    echo $? > "${OUT}/${split}.rc"
  ) &
}

run_one latest_region "${REGION}" 144 0
run_one latest_object "${OBJECT}" 1251 1
wait

/opt/conda/bin/python3 - "${OUT}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
rows = []
for summary_path in sorted(out.glob("*.summary.json")):
    stem = summary_path.name.removesuffix(".summary.json")
    model_name, split = stem.split("__", 1)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows.append({
        "model_name": model_name,
        "split": split,
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

rows.sort(key=lambda r: r["split"])
(out / "summary_matrix.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
headers = [
    "split", "model_name", "num_samples", "format_ok", "format_rate",
    "mean_iou_all", "iou_at_0_3_all", "iou_at_0_5_all",
    "mean_center_error", "mean_coord_mae",
]
lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
for row in rows:
    vals = []
    for key in headers:
        value = row.get(key)
        vals.append(f"{value:.6f}" if isinstance(value, float) else str(value))
    lines.append("| " + " | ".join(vals) + " |")
(out / "summary_matrix.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({"out": str(out), "rows": len(rows)}, ensure_ascii=False))
PY

echo "[eval] done ${OUT}"
