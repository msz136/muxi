"""
Prompt Pool 构建脚本
从多个 pointing 数据集中构建统一格式的 OPD prompt pool。
"""

import json
import random
import argparse
from pathlib import Path
from typing import Generator


# 统一的 system prompt
POINTING_SYSTEM_PROMPT = (
    'Your task is to locate several points in the given image according to '
    'the task descriptions. Your answer should be formatted as '
    '"<point>[[x1, y1], [x2, y2],...]</point>". '
    'The point coordinates are normalized to integers between 0 and 1000. '
    'Return the answer in the point format directly.'
)

# 数据集配置
DATASET_CONFIGS = {
    "robopoint": {
        "weight": 0.30,
        "path": "/data/share_data/datasets/embodied_jsons/point_xy_format.jsonl",
        "image_root": "/data/share_data/datasets/RoboPoint/images/",
    },
    "refspatial_3d": {
        "weight": 0.15,
        "path": "/data/share_data/datasets/embodied_jsons/Refspatial_3D_choice_qa_processed.jsonl",
        "image_root": "/data/share_data/datasets/RefSpatial/3D/image",
    },
    "refspatial_sim": {
        "weight": 0.10,
        "path": "/data/share_data/datasets/embodied_jsons/refspatial_sim_pointing.jsonl",
        "image_root": "/data/share_data/datasets/RefSpatial/sim/image",
    },
    "pixmo_points": {
        "weight": 0.15,
        "path": "s3://houzhi/qwen_conversion_json/Pixmo/pixmo_points_full_s3_qwen1000.json",
        "image_root": "s3://houzhi/Pixmo",
    },
    "pacolvis": {
        "weight": 0.10,
        "path": "/data/share_data/datasets/embodied_jsons/paco_lvis_v1_pointing_train_with_system_v2.jsonl",
        "image_root": "/data/luozehang/Qwen3-VL/qwen-vl-finetune/datasets/paco-lvis/images",
    },
    "grasp_anything": {
        "weight": 0.10,
        "path": "s3://houzhi/qwen_conversion_json/Grasp-Anything/grasp_anything_full_s3_qwen1000.json",
        "image_root": "s3://houzhi/Grasp-Anything",
    },
    "sharerobot_affordance": {
        "weight": 0.05,
        "path": "/data/share_data/datasets/embodied_jsons/sharerobot_affordance.jsonl",
        "image_root": "/data/share_data/datasets/ShareRobot/images",
    },
    "general_vlm_replay": {
        "weight": 0.05,
        "path": "/data/share_data/datasets/cambrian_general_qa_subset.jsonl",
        "image_root": "/data/share_data/datasets/cambrian/images",
    },
}


def load_jsonl(path: str) -> Generator[dict, None, None]:
    """加载 JSONL 文件，逐行 yield。"""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def convert_to_prompt(item: dict, dataset_name: str, image_root: str) -> dict | None:
    """
    将原始数据转换为统一的 OPD prompt 格式。

    输出格式:
    {
        "prompt_id": str,
        "images": [str],
        "messages": [{"role": str, "content": str}, ...],
        "metadata": {"source": str, "task_type": str}
    }
    """
    try:
        # 提取图片路径
        images = []
        if "image" in item:
            img = item["image"]
            if isinstance(img, list):
                images = [f"{image_root}/{p}" for p in img]
            else:
                images = [f"{image_root}/{img}"]
        elif "images" in item:
            images = [f"{image_root}/{p}" for p in item["images"]]

        if not images:
            return None

        # 提取用户问题
        conversations = item.get("conversations", item.get("messages", []))
        user_query = None
        for conv in conversations:
            role = conv.get("from", conv.get("role", ""))
            if role in ("human", "user"):
                user_query = conv.get("value", conv.get("content", ""))
                break

        if not user_query:
            return None

        # 构建统一 prompt
        # 对于 replay 数据，不加 pointing system prompt
        if dataset_name == "general_vlm_replay":
            messages = [
                {"role": "user", "content": f"<image>\n{user_query}"}
            ]
        else:
            messages = [
                {"role": "system", "content": POINTING_SYSTEM_PROMPT},
                {"role": "user", "content": f"<image>\n{user_query}"},
            ]

        prompt_id = item.get("id", f"{dataset_name}_{random.randint(0, 999999):06d}")

        return {
            "prompt_id": str(prompt_id),
            "images": images,
            "messages": messages,
            "metadata": {
                "source": dataset_name,
                "task_type": "pointing" if dataset_name != "general_vlm_replay" else "general_qa",
            },
        }
    except (KeyError, TypeError, IndexError):
        return None


def build_prompt_pool(
    output_path: str,
    target_size: int = 50000,
    seed: int = 42,
):
    """构建 prompt pool，按权重从各数据集采样。"""
    random.seed(seed)

    print("=== Building OPD Prompt Pool ===")
    print(f"Target size: {target_size}")

    all_prompts = []

    for name, config in DATASET_CONFIGS.items():
        target_count = int(target_size * config["weight"])
        path = config["path"]
        image_root = config["image_root"]

        print(f"\n[{name}] Loading from {path}")
        print(f"  Target: {target_count} samples (weight={config['weight']})")

        # 跳过 S3 路径（需要实际环境）
        if path.startswith("s3://"):
            print(f"  SKIP: S3 path, requires runtime environment")
            continue

        try:
            dataset_prompts = []
            for item in load_jsonl(path):
                prompt = convert_to_prompt(item, name, image_root)
                if prompt:
                    dataset_prompts.append(prompt)

            # 采样到目标数量
            if len(dataset_prompts) > target_count:
                sampled = random.sample(dataset_prompts, target_count)
            else:
                sampled = dataset_prompts
                print(f"  WARNING: Only {len(sampled)} samples available (< {target_count})")

            all_prompts.extend(sampled)
            print(f"  Added: {len(sampled)} prompts")

        except FileNotFoundError:
            print(f"  SKIP: File not found")
        except Exception as e:
            print(f"  ERROR: {e}")

    # 打乱并保存
    random.shuffle(all_prompts)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as f:
        for prompt in all_prompts:
            f.write(json.dumps(prompt, ensure_ascii=False) + "\n")

    print(f"\n=== Done ===")
    print(f"Total prompts: {len(all_prompts)}")
    print(f"Saved to: {output_path}")

    # 统计
    source_counts = {}
    for p in all_prompts:
        src = p["metadata"]["source"]
        source_counts[src] = source_counts.get(src, 0) + 1

    print("\nDistribution:")
    for src, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"  {src}: {count} ({count/len(all_prompts)*100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="./data/prompt_pool.jsonl")
    parser.add_argument("--target-size", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_prompt_pool(args.output, args.target_size, args.seed)
