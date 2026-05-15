#!/bin/bash
# 启动教师模型 sglang 服务（Pointing Expert）

MODEL_PATH="/data/checkpoints/8b_grounding_multicontract"
PORT=30000
TP=2  # tensor parallel

echo "=== Launching Pointing Expert Teacher Server ==="
echo "Model: ${MODEL_PATH}"
echo "Port: ${PORT}, TP: ${TP}"

python -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --port ${PORT} \
    --tp ${TP} \
    --dtype bfloat16 \
    --max-total-tokens 8192 \
    --context-length 4096 \
    --enable-torch-compile \
    --log-level warning

echo "Teacher server launched at http://localhost:${PORT}"
