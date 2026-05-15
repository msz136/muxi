#!/bin/bash
# OPD Pointing Expert 训练启动脚本
# 基于 slime 框架

set -e

# === 路径配置 ===
SLIME_ROOT="/path/to/slime"
CONFIG="./configs/opd_pointing.yaml"
REWARD_FN="./training/opd_reward.py"
OUTPUT_DIR="/data/checkpoints/opd_pointing_expert"

# === 教师服务 ===
TEACHER_URL="http://localhost:30000/v1/completions"

# === 训练参数 ===
TOTAL_STEPS=2000
EVAL_STEPS=100
SAVE_STEPS=500

echo "=== OPD Pointing Expert Training ==="
echo "Config: ${CONFIG}"
echo "Teacher: ${TEACHER_URL}"
echo "Output: ${OUTPUT_DIR}"

# 检查教师服务是否就绪
echo "Checking teacher server..."
for i in $(seq 1 10); do
    if curl -s "${TEACHER_URL}" > /dev/null 2>&1; then
        echo "Teacher server ready."
        break
    fi
    if [ $i -eq 10 ]; then
        echo "ERROR: Teacher server not responding at ${TEACHER_URL}"
        echo "Please run: bash training/launch_teacher.sh"
        exit 1
    fi
    sleep 5
done

# 启动 OPD 训练
python -m slime.train \
    --model-path "Qwen/Qwen3-VL-8B-Instruct" \
    --rollout-function-path "${SLIME_ROOT}/slime/rollout/on_policy_distillation.py" \
    --custom-rm-path "${REWARD_FN}" \
    --prompt-data "./data/prompt_pool.jsonl" \
    --output-dir "${OUTPUT_DIR}" \
    --teacher-server-url "${TEACHER_URL}" \
    --kl-penalty-coeff 0.05 \
    --advantage-estimator grpo \
    --group-size 4 \
    --temperature 0.7 \
    --max-new-tokens 256 \
    --learning-rate 1e-6 \
    --batch-size 32 \
    --gradient-accumulation-steps 4 \
    --total-steps ${TOTAL_STEPS} \
    --warmup-ratio 0.05 \
    --max-grad-norm 1.0 \
    --eval-steps ${EVAL_STEPS} \
    --save-steps ${SAVE_STEPS} \
    --eval-prompt-data "./evaluation/robopoint_test_500.jsonl" \
    --deepspeed-stage 2 \
    --gradient-checkpointing \
    --bf16 \
    --wandb-project "opd_pointing_expert" \
    --seed 42

echo "=== Training Complete ==="
echo "Checkpoints saved to: ${OUTPUT_DIR}"
