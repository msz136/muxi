#!/usr/bin/env bash
set -euo pipefail

ROOT="${POINT_ROOT:-/data/msz/point}"
DATA_ROOT="${DATA_ROOT:-/data/msz/dataset}"
OUT_DIR="${OUT_DIR:-${ROOT}/data_expert}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "${ROOT}" "${OUT_DIR}" "${ROOT}/logs"
LOG="${ROOT}/logs/data_only_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "${LOG}") 2>&1

echo "[data-only] start: $(date)"
echo "[data-only] root=${ROOT}"
echo "[data-only] data_root=${DATA_ROOT}"
echo "[data-only] out_dir=${OUT_DIR}"
echo "[data-only] log=${LOG}"

if pgrep -f "point_sft.py convert" >/dev/null 2>&1; then
  echo "[data-only] warning: old point_sft.py convert process is still running"
  echo "[data-only] stop it with: pkill -f '[p]oint_sft.py convert'"
fi

python "${SCRIPT_DIR}/point_data_only.py" \
  --data-root "${DATA_ROOT}" \
  --out-dir "${OUT_DIR}" \
  "$@"

echo "[data-only] outputs:"
ls -lh "${OUT_DIR}" || true
wc -l "${OUT_DIR}"/*.jsonl 2>/dev/null || true
echo "[data-only] finished: $(date)"

