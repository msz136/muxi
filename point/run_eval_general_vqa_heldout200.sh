#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/msz/point
EVAL_SET=${EVAL_SET:-${ROOT}/data_eval/general_mixed_heldout200_seed20260520.jsonl}
OUT=${OUT:-${ROOT}/outputs/eval_general_vqa_heldout200_$(date +%Y%m%d_%H%M%S)}
SEED=${SEED:-20260520}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}
GPUS_CSV=${GPUS_CSV:-2,3,4,5}

mkdir -p "${OUT}"
echo "${OUT}" > "${ROOT}/outputs/latest_general_eval_dir.txt"

export MACA_PATH=${MACA_PATH:-/opt/maca-3.5.3}
export PATH="${MACA_PATH}/bin:${MACA_PATH}/mxgpu_llvm/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH}"
export LD_LIBRARY_PATH="${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${MACA_PATH}/ompi/lib:${LD_LIBRARY_PATH:-}"
export DS_ACCELERATOR=cuda
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

cat > "${OUT}/manifest.json" <<EOF
{
  "eval_set": "${EVAL_SET}",
  "seed": ${SEED},
  "max_new_tokens": ${MAX_NEW_TOKENS},
  "models": {
    "base": "/data/msz/models/8b_base",
    "obj_expert": "/data/msz/models/expert_obj_v1",
    "reg_expert": "/data/msz/models/expert_reg_v1",
    "opd_v1": "/data/msz/point/outputs/opd_online_mix_v1_2048_final_20260520_142529"
  }
}
EOF

IFS=',' read -r -a GPUS <<< "${GPUS_CSV}"
if [[ ${#GPUS[@]} -lt 4 ]]; then
  echo "need at least four GPU ids in GPUS_CSV, got ${GPUS_CSV}" >&2
  exit 2
fi

run_one() {
  local name=$1
  local model=$2
  local gpu=$3
  local out_jsonl="${OUT}/${name}__general_heldout200.jsonl"
  local log="${OUT}/${name}__general_heldout200.log"
  (
    set +e
    export CUDA_VISIBLE_DEVICES="${gpu}"
    /opt/conda/bin/python3 "${ROOT}/eval_general_vqa.py" eval \
      --model "${model}" \
      --eval-set "${EVAL_SET}" \
      --out "${out_jsonl}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --device cuda:0 \
      --seed "${SEED}" > "${log}" 2>&1
    echo $? > "${OUT}/${name}.rc"
  ) &
}

run_one base /data/msz/models/8b_base "${GPUS[0]}"
run_one obj_expert /data/msz/models/expert_obj_v1 "${GPUS[1]}"
run_one reg_expert /data/msz/models/expert_reg_v1 "${GPUS[2]}"
run_one opd_v1 /data/msz/point/outputs/opd_online_mix_v1_2048_final_20260520_142529 "${GPUS[3]}"
wait

/opt/conda/bin/python3 - "${OUT}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
rows = []
for summary_path in sorted(out.glob("*__general_heldout200.summary.json")):
    name = summary_path.name.split("__", 1)[0]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rc_path = out / f"{name}.rc"
    rows.append({
        "model_name": name,
        "rc": rc_path.read_text(encoding="utf-8").strip() if rc_path.exists() else None,
        "num_samples": summary.get("num_samples"),
        "normalized_exact": summary.get("normalized_exact"),
        "relaxed_match": summary.get("relaxed_match"),
        "mean_token_f1": summary.get("mean_token_f1"),
        "option_samples": summary.get("option_samples"),
        "results_path": summary.get("results_path"),
    })

order = {"base": 0, "obj_expert": 1, "reg_expert": 2, "opd_v1": 3}
rows.sort(key=lambda row: order.get(row["model_name"], 99))
(out / "summary_matrix.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

headers = [
    "model_name", "rc", "num_samples", "normalized_exact", "relaxed_match",
    "mean_token_f1", "option_samples",
]
lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
for row in rows:
    vals = []
    for key in headers:
        value = row.get(key)
        vals.append(f"{value:.6f}" if isinstance(value, float) else str(value))
    lines.append("| " + " | ".join(vals) + " |")
(out / "summary_matrix.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({"out": str(out), "rows": len(rows)}, ensure_ascii=False), flush=True)
PY

echo "[general eval] done ${OUT}"
