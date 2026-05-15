#!/usr/bin/env bash
# 数据集下载脚本 - 使用 hf-mirror.com 镜像下载公开数据集
# 在 conda 环境中运行: source /opt/conda/bin/activate && bash scripts/download_datasets.sh
set -euo pipefail

export HF_ENDPOINT="https://hf-mirror.com"
DATA_ROOT="/data/msz/dataset"
EVAL_ROOT="/data/msz/opd_project/evaluation/benchmarks"

mkdir -p "${DATA_ROOT}" "${EVAL_ROOT}"

echo "============================================"
echo " OPD Pointing Expert - Dataset Download"
echo " Mirror: ${HF_ENDPOINT}"
echo " Target: ${DATA_ROOT}"
echo "============================================"

# ─── 1. PixMo-Points (allenai/pixmo-points) ───
echo ""
echo "[1/5] Downloading PixMo-Points..."
PIXMO_DIR="${DATA_ROOT}/PixMo-Points"
if [ -d "${PIXMO_DIR}" ] && [ "$(find "${PIXMO_DIR}" -name '*.json*' | wc -l)" -gt 0 ]; then
    echo "  SKIP: already exists at ${PIXMO_DIR}"
else
    mkdir -p "${PIXMO_DIR}"
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='allenai/pixmo-points',
    repo_type='dataset',
    local_dir='${PIXMO_DIR}',
    resume_download=True,
)
print('  PixMo-Points download complete')
"
fi

# ─── 2. PACO-LVIS (subset for pointing) ───
echo ""
echo "[2/5] Downloading PACO-LVIS..."
PACO_DIR="${DATA_ROOT}/PACO-LVIS"
if [ -d "${PACO_DIR}" ] && [ "$(find "${PACO_DIR}" -name '*.json*' | wc -l)" -gt 0 ]; then
    echo "  SKIP: already exists at ${PACO_DIR}"
else
    mkdir -p "${PACO_DIR}"
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='PACO-dataset/PACO-LVIS',
    repo_type='dataset',
    local_dir='${PACO_DIR}',
    resume_download=True,
)
print('  PACO-LVIS download complete')
" 2>&1 || echo "  WARNING: PACO-LVIS download failed, will try alternative"
fi

# ─── 3. Grasp-Anything ───
echo ""
echo "[3/5] Downloading Grasp-Anything..."
GRASP_DIR="${DATA_ROOT}/Grasp-Anything"
if [ -d "${GRASP_DIR}" ] && [ "$(find "${GRASP_DIR}" -name '*.json*' | wc -l)" -gt 0 ]; then
    echo "  SKIP: already exists at ${GRASP_DIR}"
else
    mkdir -p "${GRASP_DIR}"
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='airvlab/Grasp-Anything',
    repo_type='dataset',
    local_dir='${GRASP_DIR}',
    resume_download=True,
)
print('  Grasp-Anything download complete')
" 2>&1 || echo "  WARNING: Grasp-Anything download failed, will try alternative"
fi

# ─── 4. RefSpatial-Bench (评估集) ───
echo ""
echo "[4/5] Downloading RefSpatial-Bench (evaluation)..."
REFSPATIAL_DIR="${EVAL_ROOT}/RefSpatial-Bench"
if [ -d "${REFSPATIAL_DIR}" ] && [ "$(find "${REFSPATIAL_DIR}" -name '*.json*' | wc -l)" -gt 0 ]; then
    echo "  SKIP: already exists at ${REFSPATIAL_DIR}"
else
    mkdir -p "${REFSPATIAL_DIR}"
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='BAAI/RefSpatial-Bench',
    repo_type='dataset',
    local_dir='${REFSPATIAL_DIR}',
    resume_download=True,
)
print('  RefSpatial-Bench download complete')
"
fi

# ��── 5. ViewSpatial-Bench (评估集) ───
echo ""
echo "[5/5] Downloading ViewSpatial-Bench (evaluation)..."
VIEWSPATIAL_DIR="${EVAL_ROOT}/ViewSpatial-Bench"
if [ -d "${VIEWSPATIAL_DIR}" ] && [ "$(find "${VIEWSPATIAL_DIR}" -name '*.json*' | wc -l)" -gt 0 ]; then
    echo "  SKIP: already exists at ${VIEWSPATIAL_DIR}"
else
    mkdir -p "${VIEWSPATIAL_DIR}"
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='lidingm/ViewSpatial-Bench',
    repo_type='dataset',
    local_dir='${VIEWSPATIAL_DIR}',
    resume_download=True,
)
print('  ViewSpatial-Bench download complete')
"
fi

echo ""
echo "============================================"
echo " Download Summary"
echo "============================================"
echo "Checking downloaded datasets:"
for d in "${PIXMO_DIR}" "${PACO_DIR}" "${GRASP_DIR}" "${REFSPATIAL_DIR}" "${VIEWSPATIAL_DIR}"; do
    if [ -d "$d" ]; then
        count=$(find "$d" -type f | wc -l)
        echo "  $(basename $d): ${count} files"
    else
        echo "  $(basename $d): NOT FOUND"
    fi
done
echo ""
echo "Pre-existing datasets:"
echo "  RoboPoint: $(find ${DATA_ROOT}/RoboPoint -type f | wc -l) files"
echo "  ShareRobot: $(find ${DATA_ROOT}/ShareRobot -type f | wc -l) files"
echo "  EmbSpatial: $(find ${DATA_ROOT}/EmbSpatial -type f | wc -l) files"
echo ""
echo "Done!"
