#!/usr/bin/env bash
set -euo pipefail

cd /data/msz/point

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
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
export TOKENIZERS_PARALLELISM=false

COLDSTART_MODEL=$(tr -d '\r' < /data/msz/point/latest_opd_offpolicy_coldstart500_newexperts_p0p1_output.txt)
DATA_SKIP_EXPANDED=64000
RUN_NAME="opd_p0p1_studentrollout_full5000_skip${DATA_SKIP_EXPANDED}_save1000_zero3_mb16_accum1_from_coldstart500_newexperts_maxnew64_prefix64_veto1to0s500_maxnorm5_$(date +%Y%m%d_%H%M%S)"
LOG="/data/msz/point/logs/${RUN_NAME}.log"
PEAK_LOG="/data/msz/point/logs/${RUN_NAME}_peakmem.tsv"
PEAK_SUMMARY="/data/msz/point/logs/${RUN_NAME}_peakmem_summary.tsv"
OUT="/data/msz/models/${RUN_NAME}"

mkdir -p /data/msz/point/logs "${OUT}"
: > "${PEAK_LOG}"
echo "${OUT}" > /data/msz/point/latest_opd_p0p1_full5000_from_coldstart500_newexperts_output.txt
echo "${LOG}" > /data/msz/point/latest_opd_p0p1_full5000_from_coldstart500_newexperts_log.txt

monitor_peak_mem() {
  while true; do
    ts=$(date +%s)
    mx-smi 2>/dev/null | awk -v ts="${ts}" '
      /^\|[[:space:]]*[0-9]+[[:space:]]+MetaX/ { gpu=$2; next }
      gpu != "" && /MiB/ {
        if (match($0, /[0-9]+\/[0-9]+ MiB/)) {
          s=substr($0, RSTART, RLENGTH)
          split(s, a, "/")
          print ts "\t" gpu "\t" a[1]
          gpu=""
        }
      }
    ' >> "${PEAK_LOG}" || true
    sleep 5
  done
}

summarize_peak_mem() {
  /opt/conda/bin/python3 - "${PEAK_LOG}" "${PEAK_SUMMARY}" <<'PY_SUMMARY'
import sys
from pathlib import Path
src = Path(sys.argv[1])
dst = Path(sys.argv[2])
peaks = {}
if src.exists():
    for line in src.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        _, gpu, mem = parts
        try:
            mem_i = int(mem)
        except ValueError:
            continue
        peaks[gpu] = max(peaks.get(gpu, 0), mem_i)
all_peak = max(peaks.values()) if peaks else 0
with dst.open("w", encoding="utf-8") as f:
    f.write("gpu\tpeak_mib\n")
    for gpu in sorted(peaks, key=lambda x: int(x) if x.isdigit() else x):
        f.write(f"{gpu}\t{peaks[gpu]}\n")
    f.write(f"all\t{all_peak}\n")
PY_SUMMARY
}

monitor_peak_mem &
MON_PID=$!
trap 'kill "${MON_PID}" 2>/dev/null || true' EXIT

set +e
deepspeed --num_gpus=8 train_opd_online_vl.py train \
  --model-name-or-path "${COLDSTART_MODEL}" \
  --data-path /data/msz/point/opd_student_v1/train_prompts.jsonl \
  --output-dir "${OUT}" \
  --deepspeed /data/msz/point/configs/zero3_opd_maca_gradclip5.json \
  --route-policy target \
  --no-group-by-route \
  --route-block-shuffle \
  --skip-expanded-samples "${DATA_SKIP_EXPANDED}" \
  --teacher-load-mode preloaded_zero3 \
  --teacher-deepspeed /data/msz/point/configs/zero3_opd_maca_gradclip5.json \
  --general-obj-teacher /data/msz/models/seed0_general_obj_expert_200k_from100k_mb1_filtered_stdtrainer_maxnorm5_v1 \
  --region-teacher /data/msz/models/seed0_region_expert_200k_from100k_mb1_filtered_stdtrainer_maxnorm5_v1 \
  --robopoint-teacher /data/msz/models/seed0_robopoint_expert_200k_from100k_mb1_filtered_stdtrainer_maxnorm5_v1 \
  --spatial-rel-teacher /data/msz/models/seed0_spatial_rel_expert_200k_from100k_mb1_filtered_stdtrainer_maxnorm5_v1 \
  --general-reasoning-teacher /data/msz/models/seed0_general_reasoning_expert_200k_from100k_mb1_filtered_stdtrainer_maxnorm5_v1 \
  --opd-rollout-source student \
  --opd-max-new-tokens 64 \
  --opd-prefix-loss-tokens 64 \
  --opd-veto-beta-start 1.0 \
  --opd-veto-beta-end 0.0 \
  --opd-veto-beta-steps 500 \
  --opd-overlap-top-k 10 \
  --max-steps 5000 \
  --save-steps 1000 \
  --save-total-limit 5 \
  --per-device-train-batch-size 16 \
  --gradient-accumulation-steps 1 \
  --learning-rate 1e-6 \
  --warmup-ratio 0.03 \
  --max-grad-norm 5.0 \
  --num-train-epochs 1 \
  --logging-steps 1 \
  --model-max-length 16384 \
  --min-pixels 50176 \
  --max-pixels 50176 \
  --bf16 \
  --gradient-checkpointing \
  2>&1 | tee "${LOG}"
STATUS=${PIPESTATUS[0]}
set -e

kill "${MON_PID}" 2>/dev/null || true
wait "${MON_PID}" 2>/dev/null || true
summarize_peak_mem
{
  echo "[peakmem] samples=${PEAK_LOG}"
  echo "[peakmem] summary=${PEAK_SUMMARY}"
  cat "${PEAK_SUMMARY}"
  echo "[exit] status=${STATUS}"
} | tee -a "${LOG}"
exit "${STATUS}"
