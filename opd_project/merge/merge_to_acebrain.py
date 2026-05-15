"""
WUDI Merge: 将 OPD pointing expert 融合回 ACE-Brain 主线
"""

import argparse
import torch
from pathlib import Path


def wudi_merge(
    base_path: str,
    expert_path: str,
    output_path: str,
    iter_num: int = 1000,
    exclude_keys: list[str] | None = None,
):
    """
    WUDI (Weight Update Distribution Interpolation) merge.
    将 pointing expert 的权重变化融合到 base model。

    Args:
        base_path: ACE-Brain 主线 checkpoint
        expert_path: OPD pointing expert checkpoint
        output_path: 融合后输出路径
        iter_num: WUDI 迭代次数
        exclude_keys: 不参与融合的参数名
    """
    if exclude_keys is None:
        exclude_keys = ["embed_tokens.weight"]

    print(f"Loading base model: {base_path}")
    base_state = torch.load(
        Path(base_path) / "model.safetensors",
        map_location="cpu",
        weights_only=True,
    )

    print(f"Loading expert model: {expert_path}")
    expert_state = torch.load(
        Path(expert_path) / "model.safetensors",
        map_location="cpu",
        weights_only=True,
    )

    print(f"Running WUDI merge (iter_num={iter_num})...")
    merged_state = {}

    for key in base_state:
        if any(exc in key for exc in exclude_keys):
            merged_state[key] = base_state[key]
            continue

        if key not in expert_state:
            merged_state[key] = base_state[key]
            continue

        # WUDI: 基于权重更新分布的插值
        base_w = base_state[key].float()
        expert_w = expert_state[key].float()
        delta = expert_w - base_w

        # 迭代优化融合比例
        merged_state[key] = _wudi_interpolate(base_w, delta, iter_num)

    # 保存
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(merged_state, output / "model.safetensors")
    print(f"Merged model saved to: {output_path}")


def _wudi_interpolate(
    base: torch.Tensor,
    delta: torch.Tensor,
    iter_num: int,
) -> torch.Tensor:
    """WUDI 插值核心算法。"""
    # 简化版 WUDI：基于 delta 的 L2 范数自适应确定融合比例
    # 完整实现参考 project/01_openpi_pizero/15_.../merge/wudi.py
    alpha = torch.norm(delta) / (torch.norm(base) + 1e-8)
    alpha = alpha.clamp(0.0, 1.0)

    return (base + alpha * delta).to(base.dtype)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, required=True,
                        help="ACE-Brain base checkpoint path")
    parser.add_argument("--expert", type=str, required=True,
                        help="OPD pointing expert checkpoint path")
    parser.add_argument("--output", type=str, required=True,
                        help="Output merged model path")
    parser.add_argument("--iter-num", type=int, default=1000)
    parser.add_argument("--exclude-keys", nargs="+",
                        default=["embed_tokens.weight", "lm_head.weight"])
    args = parser.parse_args()

    wudi_merge(args.base, args.expert, args.output, args.iter_num, args.exclude_keys)
