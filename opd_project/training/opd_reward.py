"""
OPD Pointing Expert - Reward Function
接入 slime 框架的 reward function 接口，实现教师 log-prob 提取。

使用方式：通过 --custom-rm-path 指定本文件路径
"""

import asyncio
import json
import logging
from typing import Any

import aiohttp
import numpy as np

logger = logging.getLogger(__name__)


# === slime reward function 接口 ===

async def reward_function(
    samples: list[dict[str, Any]],
    server_url: str = "http://localhost:30000/v1/completions",
    **kwargs,
) -> list[dict[str, Any]]:
    """
    向教师模型请求 token-level log-probs。
    纯蒸馏模式：返回 reward=0.0，学习信号完全来自 KL penalty。

    Args:
        samples: 学生生成的 response 列表，每个包含 input_ids, response_ids 等
        server_url: 教师 sglang 服务地址

    Returns:
        带有 teacher_log_probs 的 sample 列表
    """
    async with aiohttp.ClientSession() as session:
        tasks = [
            _get_teacher_logprobs(session, sample, server_url)
            for sample in samples
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    processed = []
    for sample, result in zip(samples, results):
        if isinstance(result, Exception):
            logger.warning(f"Teacher inference failed: {result}, using zero logprobs")
            response_length = len(sample.get("response_ids", []))
            sample["teacher_log_probs"] = [0.0] * response_length
        else:
            sample["teacher_log_probs"] = result
        sample["reward"] = 0.0  # 纯蒸馏，无外部奖励
        processed.append(sample)

    return processed


async def _get_teacher_logprobs(
    session: aiohttp.ClientSession,
    sample: dict[str, Any],
    server_url: str,
) -> list[float]:
    """向教师模型请求单个样本的 log-probs。"""
    # 构造完整序列 (prompt + student response) 送入教师
    input_ids = sample["input_ids"] + sample["response_ids"]

    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,  # 不生成，只计算 log-prob
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }

    async with session.post(
        server_url,
        json=payload,
        timeout=aiohttp.ClientTimeout(total=60),
    ) as resp:
        resp.raise_for_status()
        data = await resp.json()

    # 提取教师 log-probs，截取 response 部分
    token_logprobs = data["meta_info"]["input_token_logprobs"][1:]
    response_length = len(sample["response_ids"])
    teacher_lp = token_logprobs[-response_length:]

    return teacher_lp


def post_process_rewards(
    samples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    后处理：将 teacher_log_probs 附加到 sample 上供 KL 计算使用。
    slime 框架会在 advantage 计算时读取 sample.teacher_log_probs。
    """
    for sample in samples:
        # 确保 teacher_log_probs 是 numpy array
        if "teacher_log_probs" in sample:
            sample["teacher_log_probs"] = np.array(
                sample["teacher_log_probs"], dtype=np.float32
            )
    return samples
