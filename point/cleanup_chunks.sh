#!/usr/bin/env bash
# Cleanup old chunk checkpoints: keep the 5 most recent full checkpoints,
# remove optimizer state from older chunks (keep HF model files = 17G).
# Run this periodically to bound disk usage.

set -euo pipefail

BASE_DIR="${1:-/data/msz/point/outputs/chunked_sft_v1_20260509_105943}"
KEEP="${2:-5}"

if [ ! -d "$BASE_DIR" ]; then
    echo "ERROR: directory not found: $BASE_DIR"
    exit 1
fi

# Find the largest chunk number (numeric sort)
MAX_CHUNK=$(ls -d "$BASE_DIR"/chunk_* 2>/dev/null | sed 's/.*chunk_//' | sort -n | tail -1)

if [ -z "$MAX_CHUNK" ]; then
    echo "no chunk dirs found"
    exit 0
fi

THRESHOLD=$((MAX_CHUNK - KEEP))

echo "[cleanup] max_chunk=$MAX_CHUNK, keep=$KEEP, threshold=$THRESHOLD"
echo "[cleanup] removing checkpoints from chunks with number <= $THRESHOLD"

for d in "$BASE_DIR"/chunk_*; do
    n=$(basename "$d" | sed 's/chunk_//')
    if [ "$n" -le "$THRESHOLD" ] 2>/dev/null; then
        if [ -d "$d/checkpoint-79" ]; then
            echo "[cleanup] removing optimizer state from chunk_$n"
            rm -rf "$d/checkpoint-79"
            echo "[cleanup] chunk_$n: $(du -sh "$d" | cut -f1)"
        fi
    else
        echo "[cleanup] keeping chunk_$n ($(du -sh "$d" | cut -f1))"
    fi
done

echo "[cleanup] done"
