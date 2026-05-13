#!/usr/bin/env python3
import os
import json
import shutil
from collections import OrderedDict

from safetensors.torch import load_file, save_file

SRC = "/data/msz/models/qwen3-vl-32b-text"
DST = "/data/msz/models/qwen3-vl-32b-text-converted"

os.makedirs(DST, exist_ok=True)

# 复制非权重文件
for fn in [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "merges.txt",
    "vocab.json",
    "chat_template.json",
]:
    src = os.path.join(SRC, fn)
    dst = os.path.join(DST, fn)
    if os.path.exists(src):
        shutil.copy2(src, dst)

index_path = os.path.join(SRC, "model.safetensors.index.json")
with open(index_path, "r", encoding="utf-8") as f:
    index_data = json.load(f)

weight_map = index_data["weight_map"]
shards = sorted(set(weight_map.values()))

new_weight_map = OrderedDict()
total_size = 0

def map_key(k: str):
    if k.startswith("model.visual."):
        return None
    if k.startswith("model.language_model.lm_head."):
        return k.replace("model.language_model.lm_head.", "lm_head.", 1)
    if k.startswith("model.language_model."):
        return k.replace("model.language_model.", "model.", 1)
    if k.startswith("language_model.lm_head."):
        return k.replace("language_model.lm_head.", "lm_head.", 1)
    if k.startswith("language_model."):
        return k.replace("language_model.", "model.", 1)
    if k.startswith("lm_head."):
        return k
    return None

for shard in shards:
    src_shard = os.path.join(SRC, shard)
    dst_shard = os.path.join(DST, shard)

    tensors = load_file(src_shard)
    new_tensors = OrderedDict()

    for k, v in tensors.items():
        nk = map_key(k)
        if nk is None:
            continue
        new_tensors[nk] = v
        new_weight_map[nk] = shard
        total_size += v.numel() * v.element_size()

    if new_tensors:
        save_file(new_tensors, dst_shard)

new_index = {
    "metadata": {"total_size": total_size},
    "weight_map": new_weight_map,
}

with open(os.path.join(DST, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
    json.dump(new_index, f, ensure_ascii=False, indent=2)

print("done")
print("src =", SRC)
print("dst =", DST)
print("num_weights =", len(new_weight_map))
