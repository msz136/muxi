#!/usr/bin/env bash
set -euo pipefail

EVAL_PATH="${EVAL_PATH:-/data/msz/point/eval_raw_holdout_v1/raw_holdout_eval_v1_10k.jsonl}"
OUT_ROOT="${OUT_ROOT:-/data/msz/point/eval_raw_holdout_v1/openai_base64_qwen122_qwen35_nothink_$(date +%Y%m%d_%H%M%S)}"
SCRIPT="${SCRIPT:-/data/msz/point/eval_openai_vl_raw_holdout_base64.py}"
SUMMARY_SCRIPT="${SUMMARY_SCRIPT:-/data/msz/point/summarize_raw_holdout_eval.py}"
PYTHON="${PYTHON:-/opt/conda/bin/python3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
LIMIT="${LIMIT:-}"
WORKERS="${WORKERS:-4}"
API_TIMEOUT="${API_TIMEOUT:-300}"
API_RETRIES="${API_RETRIES:-3}"
FLUSH_EVERY="${FLUSH_EVERY:-50}"
EXPECTED_FORMAT_FILTER="${EXPECTED_FORMAT_FILTER:-}"
ENFORCE_POINT_PROTOCOL="${ENFORCE_POINT_PROTOCOL:-0}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is required" >&2
  exit 2
fi

mkdir -p "${OUT_ROOT}/logs"
cat > "${OUT_ROOT}/models.tsv" <<'EOF'
qwen-122b	qwen-122b	http://127.0.0.1:13001/v1
qwen3-35b-vl	qwen-35b	http://127.0.0.1:13002/v1
EOF

echo "${OUT_ROOT}" > /data/msz/point/eval_raw_holdout_v1/latest_openai_base64_run_dir.txt
echo "Launching OpenAI-compatible base64 evals into ${OUT_ROOT}"

pids=()
names=()
while IFS=$'\t' read -r name model_id api_base; do
  model_out="${OUT_ROOT}/${name}"
  mkdir -p "${model_out}"
  echo "launch model=${name} model_id=${model_id} api_base=${api_base}"
  limit_args=()
  if [[ -n "${LIMIT}" ]]; then
    limit_args=(--limit "${LIMIT}")
  fi
  extra_args=()
  if [[ -n "${EXPECTED_FORMAT_FILTER}" ]]; then
    extra_args+=(--expected-format-filter "${EXPECTED_FORMAT_FILTER}")
  fi
  if [[ "${ENFORCE_POINT_PROTOCOL}" == "1" || "${ENFORCE_POINT_PROTOCOL}" == "true" ]]; then
    extra_args+=(--enforce-point-protocol)
  fi
  "${PYTHON}" "${SCRIPT}" \
    --model-name "${name}" \
    --model-id "${model_id}" \
    --api-base "${api_base}" \
    --eval-path "${EVAL_PATH}" \
    --out-dir "${model_out}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --workers "${WORKERS}" \
    --timeout "${API_TIMEOUT}" \
    --retries "${API_RETRIES}" \
    --flush-every "${FLUSH_EVERY}" \
    "${limit_args[@]}" \
    "${extra_args[@]}" \
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

echo "All OpenAI-compatible base64 evals completed."
"${PYTHON}" "${SUMMARY_SCRIPT}" --run-dir "${OUT_ROOT}" > "${OUT_ROOT}/comparison_summary_stdout.json"

"${PYTHON}" - "${OUT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
summary = json.loads((run_dir / "comparison_summary.json").read_text(encoding="utf-8"))

def fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)

headers = [
    "model",
    "rows",
    "api_errors",
    "sec",
    "overall_format",
    "box_n",
    "box_iou",
    "box_acc05",
    "point_n",
    "point_hit100",
    "text_n",
    "text_loose",
]
rows = []
for item in summary.get("models", []):
    name = item["model"]
    metrics_path = run_dir / name / "metrics.json"
    raw = json.loads(metrics_path.read_text(encoding="utf-8"))
    by_format = raw.get("by_format") or {}
    box = by_format.get("box") or {}
    point = by_format.get("point") or {}
    text = by_format.get("text") or {}
    rows.append([
        name,
        raw.get("rows"),
        raw.get("api_errors"),
        raw.get("seconds"),
        (raw.get("overall") or {}).get("format_pass"),
        box.get("n"),
        box.get("iou_mean"),
        box.get("acc_iou_0_5"),
        point.get("n"),
        point.get("hit_at_100"),
        text.get("n"),
        text.get("text_loose"),
    ])

tsv_lines = ["\t".join(headers)]
tsv_lines += ["\t".join(fmt(v) for v in row) for row in rows]
(run_dir / "comparison_summary_table.tsv").write_text("\n".join(tsv_lines) + "\n", encoding="utf-8")

md = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
for row in rows:
    md.append("| " + " | ".join(fmt(v) for v in row) + " |")
(run_dir / "comparison_summary_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print(run_dir / "comparison_summary_table.md")
PY
