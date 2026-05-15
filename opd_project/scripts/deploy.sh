#!/usr/bin/env bash
# OPD Pointing Expert - 一键部署脚本
# 在集群上运行: bash /data/msz/opd_project/scripts/deploy.sh
set -euo pipefail

PROJECT_ROOT="/data/msz/opd_project"
DATA_ROOT="/data/msz/dataset"

echo "============================================"
echo " OPD Pointing Expert - Deployment"
echo " $(date)"
echo "============================================"

# ─── Step 1: 确认目录结构 ───
echo ""
echo "[Step 1] Creating directory structure..."
mkdir -p "${PROJECT_ROOT}"/{configs,data,training,evaluation/benchmarks,merge,scripts,outputs,logs}
echo "  Done."

# ─── Step 2: 下载缺失数据集 ───
echo ""
echo "[Step 2] Downloading missing datasets..."
echo "  Using conda env for huggingface_hub downloads..."

# 检查哪些数据集需要下载
NEED_DOWNLOAD=0
for ds in PixMo-Points PACO-LVIS Grasp-Anything; do
    if [ ! -d "${DATA_ROOT}/${ds}" ] || [ "$(find "${DATA_ROOT}/${ds}" -type f 2>/dev/null | wc -l)" -eq 0 ]; then
        echo "  MISSING: ${ds}"
        NEED_DOWNLOAD=1
    else
        echo "  OK: ${ds}"
    fi
done

# 评估集
for ds in RefSpatial-Bench ViewSpatial-Bench; do
    if [ ! -d "${PROJECT_ROOT}/evaluation/benchmarks/${ds}" ] || [ "$(find "${PROJECT_ROOT}/evaluation/benchmarks/${ds}" -type f 2>/dev/null | wc -l)" -eq 0 ]; then
        echo "  MISSING (eval): ${ds}"
        NEED_DOWNLOAD=1
    else
        echo "  OK (eval): ${ds}"
    fi
done

if [ "${NEED_DOWNLOAD}" -eq 1 ]; then
    echo ""
    echo "  Starting downloads (using hf-mirror.com)..."
    source /opt/conda/bin/activate
    bash "${PROJECT_ROOT}/scripts/download_datasets.sh"
else
    echo "  All datasets present, skipping download."
fi

# ─── Step 3: 构建 Prompt Pool ───
echo ""
echo "[Step 3] Building prompt pool..."
if [ -f "${PROJECT_ROOT}/data/prompt_pool.jsonl" ]; then
    LINES=$(wc -l < "${PROJECT_ROOT}/data/prompt_pool.jsonl")
    echo "  Prompt pool exists: ${LINES} prompts"
    echo "  To rebuild: python3 ${PROJECT_ROOT}/scripts/build_prompt_pool.py"
else
    echo "  Building from available datasets..."
    python3 "${PROJECT_ROOT}/scripts/build_prompt_pool.py" \
        --output "${PROJECT_ROOT}/data/prompt_pool.jsonl" \
        --target-size 50000 \
        --seed 42
fi

# ─── Step 4: 验证环境 ───
echo ""
echo "[Step 4] Verifying training environment..."
python3 -c "
import torch
import transformers
import deepspeed

print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
print(f'  GPU count: {torch.cuda.device_count()}')
print(f'  Transformers: {transformers.__version__}')
print(f'  DeepSpeed: {deepspeed.__version__}')

# 验证模型可加载
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained('/data/msz/models/Qwen3-VL-8B-Instruct')
print(f'  Model: {cfg.model_type} ({cfg.num_hidden_layers} layers)')
print('  Environment OK!')
"

# ─── Step 5: 验证数据 ───
echo ""
echo "[Step 5] Verifying data..."
python3 -c "
import json, os

# Check prompt pool
pp = '${PROJECT_ROOT}/data/prompt_pool.jsonl'
if os.path.exists(pp):
    with open(pp) as f:
        lines = sum(1 for _ in f)
    print(f'  Prompt pool: {lines} prompts')

    # Sample check
    with open(pp) as f:
        sample = json.loads(f.readline())
    print(f'  Sample keys: {list(sample.keys())}')
    print(f'  Sample source: {sample[\"metadata\"][\"source\"]}')
    img = sample['images'][0] if sample['images'] else 'none'
    print(f'  Sample image: {img}')
    print(f'  Image exists: {os.path.exists(img)}')
else:
    print('  WARNING: prompt pool not built yet')

# Check eval subset
ep = '${PROJECT_ROOT}/data/eval_robopoint_500.jsonl'
if os.path.exists(ep):
    with open(ep) as f:
        lines = sum(1 for _ in f)
    print(f'  Eval subset: {lines} samples')
"

# ─── Summary ───
echo ""
echo "============================================"
echo " Deployment Complete!"
echo "============================================"
echo ""
echo "Project: ${PROJECT_ROOT}"
echo ""
echo "Next steps:"
echo "  1. Build prompt pool (if not done):"
echo "     python3 ${PROJECT_ROOT}/scripts/build_prompt_pool.py"
echo ""
echo "  2. Train pointing expert teacher (SFT, using existing scripts):"
echo "     bash /data/msz/point/onekey_expert_sft.sh"
echo ""
echo "  3. Launch teacher server:"
echo "     bash ${PROJECT_ROOT}/training/launch_teacher.sh"
echo ""
echo "  4. Run OPD training:"
echo "     bash ${PROJECT_ROOT}/training/run_opd.sh"
echo ""
echo "  5. Evaluate:"
echo "     bash ${PROJECT_ROOT}/evaluation/run_all_eval.sh <checkpoint>"
echo ""
