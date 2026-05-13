#!/bin/bash
# ============================================================
# Grounding/Point Expert — 数据集 & 模型一键部署脚本
# 含 3 次重试 + 多次失败跳过机制
# ============================================================
set -euo pipefail

# ===================== 可配置变量 =====================
MAX_RETRIES=3
RETRY_DELAY_BASE=30
HF_MIRROR="${HF_ENDPOINT:-https://hf-mirror.com}"

# 目标路径（按需修改）
MODEL_DIR="${MODEL_DIR:-/data/msz/models}"
DATASET_BASE="${DATASET_BASE:-/data/msz/dataset}"
EMBODIED_DIR="${EMBODIED_DIR:-/data/msz/dataset/embodied_jsons}"

LOG_FILE="$(cd "$(dirname "$0")" && pwd)/deploy_$(date +%Y%m%d_%H%M%S).log"
SUCCESS_ITEMS=()
FAILED_ITEMS=()

# ===================== 工具函数 =====================
log()  { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"; }
warn() { echo "[$(date '+%H:%M:%S')] [WARN] $*" | tee -a "$LOG_FILE" >&2; }

retry() {
    local name="$1"; shift
    local attempt=0

    while [ $attempt -lt $MAX_RETRIES ]; do
        attempt=$((attempt + 1))
        log "  [${name}] 第 ${attempt}/${MAX_RETRIES} 次尝试..."
        if eval "$*"; then
            log "  [${name}] ✓ 成功"
            return 0
        fi
        if [ $attempt -lt $MAX_RETRIES ]; then
            local delay=$((RETRY_DELAY_BASE * attempt))
            log "  [${name}] 失败，${delay}s 后重试..."
            sleep $delay
        fi
    done
    warn "  [${name}] ✗ 已失败 ${MAX_RETRIES} 次，跳过"
    return 1
}

ensure_git_lfs() {
    if ! command -v git-lfs &>/dev/null; then
        log "安装 git-lfs ..."
        (sudo apt-get update -qq && sudo apt-get install -y -qq git-lfs) 2>/dev/null || \
        (sudo yum install -y git-lfs) 2>/dev/null || \
        (conda install -y -c conda-forge git-lfs) 2>/dev/null || true
        git lfs install 2>/dev/null || true
    fi
}

ensure_hf_cli() {
    command -v huggingface-cli &>/dev/null && return 0
    log "安装 huggingface_hub[cli] ..."
    pip install "huggingface_hub[cli]" -q 2>/dev/null || true
}

# 已存在则跳过
already_exists() {
    local path="$1"
    [ -d "$path" ] && [ -n "$(ls -A "$path" 2>/dev/null)" ]
}

# ===================== 预检 =====================
log "============================================"
log "Grounding 数据集 & 模型部署"
log "时间: $(date '+%Y-%m-%d %H:%M:%S')"
log "日志: ${LOG_FILE}"
log "============================================"
log ""

ensure_git_lfs
ensure_hf_cli
mkdir -p "${MODEL_DIR}" "${DATASET_BASE}" "${EMBODIED_DIR}"

log "目标路径:"
log "  模型: ${MODEL_DIR}"
log "  数据集: ${DATASET_BASE}"
log "  Embodied JSON: ${EMBODIED_DIR}"
log ""

# ===================== 模型 =====================
log "========== 模型 =========="

MODEL_TARGET="${MODEL_DIR}/Qwen3-VL-8B-Instruct"

if [ -f "${MODEL_TARGET}/config.json" ]; then
    log "Qwen3-VL-8B-Instruct 已存在，跳过"
    SUCCESS_ITEMS+=("Qwen3-VL-8B-Instruct")
else
    log "下载 Qwen3-VL-8B-Instruct (约 17GB) ..."

    if retry "Qwen3-VL-8B-Instruct (ModelScope)" \
        "GIT_LFS_SKIP_SMUDGE=1 git clone --depth=1 https://www.modelscope.cn/Qwen/Qwen3-VL-8B-Instruct.git '${MODEL_TARGET}' && cd '${MODEL_TARGET}' && git lfs pull"; then
        SUCCESS_ITEMS+=("Qwen3-VL-8B-Instruct")
    elif retry "Qwen3-VL-8B-Instruct (HF Mirror)" \
        "export HF_ENDPOINT='${HF_MIRROR}' && huggingface-cli download Qwen/Qwen3-VL-8B-Instruct --local-dir '${MODEL_TARGET}' --resume-download"; then
        SUCCESS_ITEMS+=("Qwen3-VL-8B-Instruct")
    else
        FAILED_ITEMS+=("Qwen3-VL-8B-Instruct")
    fi
fi

log ""

# ===================== 数据集 =====================
log "========== 数据集 =========="

# --- RoboPoint (wentao-yuan/robopoint-data, HuggingFace) ---
RP_TARGET="${DATASET_BASE}/RoboPoint"
if already_exists "${RP_TARGET}"; then
    log "RoboPoint 已存在，跳过"
    SUCCESS_ITEMS+=("RoboPoint")
else
    log "下载 RoboPoint (wentao-yuan/robopoint-data, ~1.43M 样本) ..."
    if retry "RoboPoint" \
        "export HF_ENDPOINT='${HF_MIRROR}' && mkdir -p '${RP_TARGET}' && huggingface-cli download wentao-yuan/robopoint-data --local-dir '${RP_TARGET}' --repo-type dataset --resume-download"; then
        # 解压分卷图片包
        if ls "${RP_TARGET}"/region_ref.tar.gz.part_* &>/dev/null 2>&1; then
            log "  解压 RoboPoint 图片..."
            cat "${RP_TARGET}"/region_ref.tar.gz.part_* > "${RP_TARGET}"/region_ref.tar.gz
            mkdir -p "${RP_TARGET}/images"
            tar -xzf "${RP_TARGET}"/region_ref.tar.gz -C "${RP_TARGET}/images/" 2>/dev/null || \
                tar -xzf "${RP_TARGET}"/region_ref.tar.gz -C "${RP_TARGET}/"
        fi
        SUCCESS_ITEMS+=("RoboPoint")
    else
        FAILED_ITEMS+=("RoboPoint")
    fi
fi

# --- PixMo-Points (allenai/pixmo-points, HuggingFace) ---
PP_TARGET="${DATASET_BASE}/pixmo-points"
if already_exists "${PP_TARGET}"; then
    log "PixMo-Points 已存在，跳过"
    SUCCESS_ITEMS+=("PixMo-Points")
else
    log "下载 PixMo-Points (allenai/pixmo-points, ~198MB 标注, ~2.37M 样本) ..."
    log "  注意: 图片通过 URL 引用，需另外下载图片文件"
    if retry "PixMo-Points" \
        "export HF_ENDPOINT='${HF_MIRROR}' && mkdir -p '${PP_TARGET}' && huggingface-cli download allenai/pixmo-points --local-dir '${PP_TARGET}' --repo-type dataset --resume-download"; then
        SUCCESS_ITEMS+=("PixMo-Points")
    else
        FAILED_ITEMS+=("PixMo-Points")
    fi
fi

# --- Struct2D-Set (github.com/neu-vi/struct2d) ---
SS_TARGET="${DATASET_BASE}/Struct2D-Set"
if already_exists "${SS_TARGET}"; then
    log "Struct2D-Set 已存在，跳过"
    SUCCESS_ITEMS+=("Struct2D-Set")
else
    log "下载 Struct2D-Set (github.com/neu-vi/struct2d, ~21k 样本) ..."
    log "  注意: 代码仓库已公开，数据集需从仓库获取"
    if retry "Struct2D-Set" \
        "mkdir -p '${SS_TARGET}' && git clone --depth=1 https://github.com/neu-vi/struct2d.git /tmp/struct2d_clone && (cp -r /tmp/struct2d_clone/data/* '${SS_TARGET}/' 2>/dev/null || warn '仓库中暂无 data/ 目录，数据集可能尚未正式发布，请关注仓库 Releases'); rm -rf /tmp/struct2d_clone"; then
        SUCCESS_ITEMS+=("Struct2D-Set")
    else
        FAILED_ITEMS+=("Struct2D-Set")
    fi
fi

# --- ShareRobot (BAAI/ShareRobot, HuggingFace) ---
SR_TARGET="${DATASET_BASE}/ShareRobot"
if already_exists "${SR_TARGET}"; then
    log "ShareRobot 已存在，跳过"
    SUCCESS_ITEMS+=("ShareRobot")
else
    log "下载 ShareRobot (BAAI/ShareRobot, trajectory + affordance) ..."
    if retry "ShareRobot (HF)" \
        "export HF_ENDPOINT='${HF_MIRROR}' && mkdir -p '${SR_TARGET}' && huggingface-cli download BAAI/ShareRobot --local-dir '${SR_TARGET}' --repo-type dataset --resume-download"; then
        SUCCESS_ITEMS+=("ShareRobot")
    elif retry "ShareRobot (GitHub)" \
        "git clone --depth=1 https://github.com/FlagOpen/ShareRobot.git '${SR_TARGET}'"; then
        SUCCESS_ITEMS+=("ShareRobot")
    else
        FAILED_ITEMS+=("ShareRobot")
    fi
fi

# --- Robo2VLM-1 (keplerccc/Robo2VLM-1, HuggingFace) ---
R2_TARGET="${DATASET_BASE}/Robo2VLM-1"
if already_exists "${R2_TARGET}"; then
    log "Robo2VLM-1 已存在，跳过"
    SUCCESS_ITEMS+=("Robo2VLM-1")
else
    log "下载 Robo2VLM-1 (keplerccc/Robo2VLM-1) ..."
    if retry "Robo2VLM-1" \
        "export HF_ENDPOINT='${HF_MIRROR}' && mkdir -p '${R2_TARGET}' && huggingface-cli download keplerccc/Robo2VLM-1 --local-dir '${R2_TARGET}' --repo-type dataset --resume-download"; then
        SUCCESS_ITEMS+=("Robo2VLM-1")
    else
        FAILED_ITEMS+=("Robo2VLM-1")
    fi
fi

# --- RoboVQA (ModelScope: nv-community/Cosmos-Reason1-SFT-Dataset) ---
RV_TARGET="${DATASET_BASE}/robovqa"
if already_exists "${RV_TARGET}"; then
    log "RoboVQA 已存在，跳过"
    SUCCESS_ITEMS+=("RoboVQA")
else
    log "下载 RoboVQA (ModelScope: nv-community/Cosmos-Reason1-SFT-Dataset) ..."
    if retry "RoboVQA" \
        "pip install modelscope -q 2>/dev/null; python -c \"
from modelscope import snapshot_download
snapshot_download('nv-community/Cosmos-Reason1-SFT-Dataset', cache_dir='${RV_TARGET}')
print('下载完成')
\""; then
        SUCCESS_ITEMS+=("RoboVQA")
    else
        FAILED_ITEMS+=("RoboVQA")
    fi
fi

# --- EmbSpatial (thanhqt2002/embodied-spatial-reasoning, HuggingFace) ---
ES_TARGET="${DATASET_BASE}/EmbSpatial"
if already_exists "${ES_TARGET}"; then
    log "EmbSpatial 已存在，跳过"
    SUCCESS_ITEMS+=("EmbSpatial")
else
    log "下载 EmbSpatial (thanhqt2002/embodied-spatial-reasoning) ..."
    if retry "EmbSpatial" \
        "export HF_ENDPOINT='${HF_MIRROR}' && mkdir -p '${ES_TARGET}' && huggingface-cli download thanhqt2002/embodied-spatial-reasoning --local-dir '${ES_TARGET}' --repo-type dataset --resume-download"; then
        SUCCESS_ITEMS+=("EmbSpatial")
    else
        FAILED_ITEMS+=("EmbSpatial")
    fi
fi

# --- Phys100k (unira-zwj/Phys100k-physqa, HuggingFace) ---
PH_TARGET="${DATASET_BASE}/Phys100k"
if already_exists "${PH_TARGET}"; then
    log "Phys100k 已存在，跳过"
    SUCCESS_ITEMS+=("Phys100k")
else
    log "下载 Phys100k (unira-zwj/Phys100k-physqa) ..."
    if retry "Phys100k" \
        "export HF_ENDPOINT='${HF_MIRROR}' && mkdir -p '${PH_TARGET}' && huggingface-cli download unira-zwj/Phys100k-physqa --local-dir '${PH_TARGET}' --repo-type dataset --resume-download"; then
        SUCCESS_ITEMS+=("Phys100k")
    else
        FAILED_ITEMS+=("Phys100k")
    fi
fi

# --- RoboRefIt (via robot_sugar GitHub) ---
RR_TARGET="${DATASET_BASE}/RoboRefIt"
if already_exists "${RR_TARGET}"; then
    log "RoboRefIt 已存在，跳过"
    SUCCESS_ITEMS+=("RoboRefIt")
else
    log "下载 RoboRefIt (via robot_sugar) ..."
    log "  注意: 需从 https://github.com/vlc-robot/robot_sugar/blob/main/DATASET.md 获取数据下载链接"
    if retry "RoboRefIt" \
        "git clone --depth=1 https://github.com/vlc-robot/robot_sugar.git /tmp/robot_sugar && mkdir -p '${RR_TARGET}' && (cd /tmp/robot_sugar && cat DATASET.md | grep -i download); rm -rf /tmp/robot_sugar"; then
        SUCCESS_ITEMS+=("RoboRefIt")
    else
        FAILED_ITEMS+=("RoboRefIt")
    fi
fi

log ""

# ===================== 汇总 =====================
log "============================================"
log "部署完成"
log "============================================"
log ""

log "--- 模型 ---"
log "  Qwen3-VL-8B-Instruct: ${MODEL_DIR}/Qwen3-VL-8B-Instruct"

log ""
log "--- 数据集 ---"
log "  RoboPoint:      ${DATASET_BASE}/RoboPoint"
log "  PixMo-Points:   ${DATASET_BASE}/pixmo-points"
log "  Struct2D-Set:   ${DATASET_BASE}/Struct2D-Set"
log "  ShareRobot:     ${DATASET_BASE}/ShareRobot"
log "  Robo2VLM-1:     ${DATASET_BASE}/Robo2VLM-1"
log "  RoboVQA:        ${DATASET_BASE}/robovqa"
log "  EmbSpatial:     ${DATASET_BASE}/EmbSpatial"
log "  Phys100k:       ${DATASET_BASE}/Phys100k"
log "  RoboRefIt:      ${DATASET_BASE}/RoboRefIt"

log ""
if [ ${#SUCCESS_ITEMS[@]} -gt 0 ]; then
    log "✓ 成功 (${#SUCCESS_ITEMS[@]}):"
    for it in "${SUCCESS_ITEMS[@]}"; do log "    ${it}"; done
fi
if [ ${#FAILED_ITEMS[@]} -gt 0 ]; then
    log "✗ 失败需手动处理 (${#FAILED_ITEMS[@]}):"
    for it in "${FAILED_ITEMS[@]}"; do log "    ${it}"; done
fi

log ""
log "日志: ${LOG_FILE}"

