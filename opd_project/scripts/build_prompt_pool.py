#!/usr/bin/env python3
"""
构建 OPD Prompt Pool - 集群版本 v2
从已有数据集中提取 pointing prompts，构建统一格式的 prompt pool。
"""

import json
import random
import argparse
import re
import os
from pathlib import Path

POINTING_SYSTEM_PROMPT = (
    "You are a helpful vision-language assistant. "
    "When the user asks for a location, answer with coordinates "
    "in the range 0 to 1000. Your answer should be formatted as "
    '"<point>[[x1, y1], [x2, y2],...]</point>".'
)

DATA_ROOT = "/data/msz/dataset"


def parse_tuple_points(text: str) -> list:
    """解析 RoboPoint 的 tuple 格式: [(0.461, 0.527), (0.498, 0.521)]"""
    matches = re.findall(r'\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)', text)
    points = []
    for x_str, y_str in matches:
        x = int(float(x_str) * 1000)
        y = int(float(y_str) * 1000)
        points.append([x, y])
    return points


def extract_robopoint(json_path: str, image_root: str, max_n: int) -> list:
    """从 RoboPoint 1.4M json 中提取 pointing prompts。"""
    print(f"  Loading {json_path}...")
    with open(json_path, "r") as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} items")

    # 采样候选
    if len(data) > max_n * 2:
        data = random.sample(data, max_n * 2)

    prompts = []
    for item in data:
        if len(prompts) >= max_n:
            break

        image = item.get("image", "")
        if not image:
            continue

        img_path = f"{image_root}/{image}"

        # 提取对话
        convs = item.get("conversations", [])
        user_query = None
        gpt_answer = None
        for conv in convs:
            if conv.get("from") == "human":
                user_query = conv.get("value", "")
            elif conv.get("from") == "gpt":
                gpt_answer = conv.get("value", "")

        if not user_query or not gpt_answer:
            continue

        # 解析 GT points
        gt_points = parse_tuple_points(gpt_answer)
        if not gt_points:
            continue

        prompts.append({
            "prompt_id": item.get("id", f"robopoint_{len(prompts)}"),
            "images": [img_path],
            "messages": [
                {"role": "system", "content": POINTING_SYSTEM_PROMPT},
                {"role": "user", "content": user_query},
            ],
            "gt_points": gt_points,
            "metadata": {"source": "robopoint", "task_type": "pointing"},
        })

    return prompts


def extract_sharerobot(json_path: str, image_root: str, max_n: int) -> list:
    """从 ShareRobot affordance 中提取 prompts。"""
    with open(json_path, "r") as f:
        data = json.load(f)

    if len(data) > max_n:
        data = random.sample(data, max_n)

    prompts = []
    for item in data:
        img_path = item.get("image_path", "")
        if not img_path:
            continue

        full_img = f"{image_root}/{img_path}"
        instruction = item.get("instruction", "")
        affordance = item.get("affordance", {})

        if not affordance or not instruction:
            continue

        meta = item.get("meta_data", {})
        w = meta.get("original_width", 128)
        h = meta.get("original_height", 128)
        x = int(float(affordance.get("x", 0)) / w * 1000)
        y = int(float(affordance.get("y", 0)) / h * 1000)

        query = (
            f"<image>\nFor the task '{instruction}', "
            "point to the location where the robot should interact."
        )

        prompts.append({
            "prompt_id": f"sharerobot_{item.get('id', len(prompts))}",
            "images": [full_img],
            "messages": [
                {"role": "system", "content": POINTING_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            "gt_points": [[x, y]],
            "metadata": {"source": "sharerobot_affordance", "task_type": "affordance_pointing"},
        })

    return prompts


def extract_replay(jsonl_path: str, image_root: str, max_n: int) -> list:
    """从 keepalive VQA 中提取 replay prompts (防遗忘)。"""
    items = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    items.append(json.loads(line))
                except Exception:
                    continue

    if len(items) > max_n:
        items = random.sample(items, max_n)

    prompts = []
    for item in items:
        convs = item.get("conversations", [])
        images = item.get("image", item.get("images", []))
        if isinstance(images, str):
            images = [images]

        user_query = None
        for conv in convs:
            if conv.get("from") in ("human", "user"):
                user_query = conv.get("value", conv.get("content", ""))
                break

        if not user_query:
            continue

        img_list = []
        for img in images:
            if img.startswith("/"):
                img_list.append(img)
            else:
                img_list.append(f"{image_root}/{img}")

        prompts.append({
            "prompt_id": f"replay_{len(prompts)}",
            "images": img_list,
            "messages": [{"role": "user", "content": user_query}],
            "gt_points": [],
            "metadata": {"source": "general_vqa_replay", "task_type": "general_qa"},
        })

    return prompts


def build_prompt_pool(output_path: str, target_size: int = 50000, seed: int = 42):
    random.seed(seed)
    all_prompts = []

    print("=" * 60)
    print(" Building OPD Prompt Pool v2")
    print(f" Target: {target_size} prompts")
    print(f" Output: {output_path}")
    print("=" * 60)

    # 1. RoboPoint (主力, 60%)
    rp_path = f"{DATA_ROOT}/RoboPoint/robopoint_1432k.json"
    if os.path.exists(rp_path):
        print(f"\n[robopoint] target={int(target_size*0.6)}")
        prompts = extract_robopoint(
            rp_path, f"{DATA_ROOT}/RoboPoint/images", int(target_size * 0.6)
        )
        all_prompts.extend(prompts)
        print(f"  Got: {len(prompts)}")
    else:
        print("\n[robopoint] SKIP: not found")

    # 2. ShareRobot Affordance (10%)
    sr_path = f"{DATA_ROOT}/ShareRobot/affordance/affordance.json"
    if os.path.exists(sr_path):
        print(f"\n[sharerobot] target={int(target_size*0.1)}")
        prompts = extract_sharerobot(
            sr_path, f"{DATA_ROOT}/ShareRobot/affordance/images", int(target_size * 0.1)
        )
        all_prompts.extend(prompts)
        print(f"  Got: {len(prompts)}")
    else:
        print("\n[sharerobot] SKIP: not found")

    # 3. PixMo-Points (15%) - if downloaded
    pixmo_dir = f"{DATA_ROOT}/PixMo-Points"
    if os.path.isdir(pixmo_dir) and len(os.listdir(pixmo_dir)) > 0:
        print(f"\n[pixmo_points] found, implement after checking format")
    else:
        print(f"\n[pixmo_points] SKIP: not downloaded yet")

    # 4. Grasp-Anything (10%) - if downloaded
    grasp_dir = f"{DATA_ROOT}/Grasp-Anything"
    if os.path.isdir(grasp_dir) and len(os.listdir(grasp_dir)) > 0:
        print(f"\n[grasp_anything] found, implement after checking format")
    else:
        print(f"\n[grasp_anything] SKIP: not downloaded yet")

    # 5. General VQA Replay (防遗忘, 10%)
    replay_path = "/data/msz/point/data_expert/keepalive_vqa.jsonl"
    if os.path.exists(replay_path):
        print(f"\n[replay] target={int(target_size*0.1)}")
        prompts = extract_replay(replay_path, DATA_ROOT, int(target_size * 0.1))
        all_prompts.extend(prompts)
        print(f"  Got: {len(prompts)}")
    else:
        print("\n[replay] SKIP: not found")

    # Shuffle and trim
    random.shuffle(all_prompts)
    if len(all_prompts) > target_size:
        all_prompts = all_prompts[:target_size]

    # Save
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for p in all_prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # Stats
    print(f"\n{'=' * 60}")
    print(f" Built: {len(all_prompts)} prompts")
    print(f" Saved: {output_path}")
    print(f"{'=' * 60}")

    source_counts = {}
    for p in all_prompts:
        src = p["metadata"]["source"]
        source_counts[src] = source_counts.get(src, 0) + 1
    print("\nDistribution:")
    for src, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"  {src}: {count} ({count/len(all_prompts)*100:.1f}%)")

    # Save eval subset (robopoint samples with GT points)
    eval_items = [
        p for p in all_prompts
        if p["gt_points"] and p["metadata"]["source"] == "robopoint"
    ]
    if len(eval_items) > 500:
        eval_items = random.sample(eval_items, 500)
    eval_path = str(output).replace("prompt_pool", "eval_robopoint_500")
    with open(eval_path, "w", encoding="utf-8") as f:
        for p in eval_items:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\nEval subset: {len(eval_items)} -> {eval_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/data/msz/opd_project/data/prompt_pool.jsonl")
    parser.add_argument("--target-size", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_prompt_pool(args.output, args.target_size, args.seed)
