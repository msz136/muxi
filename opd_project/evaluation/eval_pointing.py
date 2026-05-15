"""
Pointing 精度评估脚本
评估模型在 pointing 任务上的坐标预测准确性。
"""

import argparse
import json
import re
import math
from pathlib import Path

import numpy as np
from tqdm import tqdm


def parse_points(response: str) -> list[tuple[int, int]]:
    """从模型输出中解析 <point>[[x,y],...]</point> 格式的坐标。"""
    match = re.search(r'<point>\s*(\[.*?\])\s*</point>', response, re.DOTALL)
    if not match:
        return []

    try:
        coords = json.loads(match.group(1))
        if isinstance(coords[0], list):
            return [(int(c[0]), int(c[1])) for c in coords]
        else:
            # 单点情况 [x, y]
            return [(int(coords[0]), int(coords[1]))]
    except (json.JSONDecodeError, IndexError, TypeError):
        return []


def point_distance(pred: tuple[int, int], gt: tuple[int, int]) -> float:
    """计算两点间欧氏距离（坐标归一化到 0-1000）。"""
    return math.sqrt((pred[0] - gt[0]) ** 2 + (pred[1] - gt[1]) ** 2)


def compute_metrics(
    predictions: list[str],
    ground_truths: list[list[tuple[int, int]]],
    thresholds: list[int] = [50, 100, 150],
) -> dict[str, float]:
    """
    计算 pointing 评估指标。

    Args:
        predictions: 模型输出列表
        ground_truths: GT 坐标列表（每个样本可能有多个点）
        thresholds: Acc@threshold 的阈值列表

    Returns:
        指标字典
    """
    all_distances = []
    format_correct = 0
    total = len(predictions)

    for pred_str, gt_points in zip(predictions, ground_truths):
        pred_points = parse_points(pred_str)

        if pred_points:
            format_correct += 1

        if not pred_points or not gt_points:
            # 格式错误或无 GT，记为最大距离
            all_distances.append(1414.0)  # sqrt(1000^2 + 1000^2)
            continue

        # 匹配预测点和 GT 点（贪心最近匹配）
        min_distances = []
        for gt_pt in gt_points:
            if pred_points:
                dists = [point_distance(p, gt_pt) for p in pred_points]
                min_distances.append(min(dists))
            else:
                min_distances.append(1414.0)

        all_distances.append(np.mean(min_distances))

    # 计算指标
    distances = np.array(all_distances)
    metrics = {
        "mean_distance": float(np.mean(distances)),
        "median_distance": float(np.median(distances)),
        "format_accuracy": format_correct / total if total > 0 else 0.0,
    }

    for t in thresholds:
        acc = float(np.mean(distances <= t))
        metrics[f"acc@{t}"] = acc

    return metrics


def check_format(response: str) -> bool:
    """检查输出是否符合 <point>...</point> 格式。"""
    pattern = r'<point>\s*\[\s*\[.*?\]\s*\]\s*</point>'
    return bool(re.search(pattern, response, re.DOTALL))


def main():
    parser = argparse.ArgumentParser(description="Evaluate pointing accuracy")
    parser.add_argument("--predictions", type=str, required=True,
                        help="Path to predictions JSONL (each line: {response, gt_points})")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save metrics JSON")
    parser.add_argument("--thresholds", type=int, nargs="+", default=[50, 100, 150],
                        help="Distance thresholds for Acc@T")
    args = parser.parse_args()

    # 加载预测结果
    predictions = []
    ground_truths = []

    with open(args.predictions, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            predictions.append(item["response"])
            ground_truths.append(
                [tuple(pt) for pt in item["gt_points"]]
            )

    print(f"Loaded {len(predictions)} samples")

    # 计算指标
    metrics = compute_metrics(predictions, ground_truths, args.thresholds)

    # 输出结果
    print("\n=== Pointing Evaluation Results ===")
    print(f"  Format Accuracy:  {metrics['format_accuracy']:.4f}")
    print(f"  Mean Distance:    {metrics['mean_distance']:.2f}")
    print(f"  Median Distance:  {metrics['median_distance']:.2f}")
    for t in args.thresholds:
        print(f"  Acc@{t}:          {metrics[f'acc@{t}']:.4f}")

    # 保存结果
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to: {args.output}")


if __name__ == "__main__":
    main()
