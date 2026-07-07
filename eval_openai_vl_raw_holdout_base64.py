#!/usr/bin/env python3
"""Evaluate OpenAI-compatible VLM APIs on raw_holdout_eval_v1 with base64 images."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures as futures
import json
import math
import mimetypes
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

COORD_RE = re.compile(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]")
BOX_FLAT_RE = re.compile(
    r"<box>\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]\s*</box>",
    re.I | re.S,
)
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def now() -> str:
    return time.strftime("%F %T")


def log(obj: dict[str, Any]) -> None:
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def answer_of(row: dict[str, Any]) -> str:
    if row.get("gold"):
        return str(row.get("gold"))
    for turn in row.get("conversations") or []:
        if str(turn.get("from", "")).lower() in {"gpt", "assistant"}:
            return str(turn.get("value", ""))
    return ""


def expected_format(answer: str) -> str:
    has_point = "<point>" in answer
    has_box = "<box>" in answer
    if has_point and not has_box:
        return "point"
    if has_box and not has_point:
        return "box"
    if has_point and has_box:
        return "mixed_grounding"
    return "text"


def first_user_text(row: dict[str, Any]) -> str:
    for turn in row.get("conversations") or []:
        if str(turn.get("from", "")).lower() in {"human", "user"}:
            return str(turn.get("value", ""))
    return ""


def system_text(row: dict[str, Any]) -> str:
    for turn in row.get("conversations") or []:
        if str(turn.get("from", "")).lower() == "system":
            return str(turn.get("value", ""))
    return "You are a helpful vision-language assistant."


def expected_format_of_row(row: dict[str, Any]) -> str:
    meta = ((row.get("metadata") or {}).get("raw_holdout_eval") or {})
    gold = answer_of(row)
    return str(meta.get("expected_format") or expected_format(gold))


def parse_points(text: str) -> list[tuple[float, float]]:
    return [(float(x), float(y)) for x, y in COORD_RE.findall(text)]


def parse_box(text: str) -> tuple[float, float, float, float] | None:
    flat = BOX_FLAT_RE.search(text)
    if flat:
        x1, y1, x2, y2 = [float(v) for v in flat.groups()]
        return x1, y1, x2, y2
    box_match = re.search(r"<box>(.*?)</box>", text, re.I | re.S)
    if box_match:
        nums = [float(v) for v in NUMBER_RE.findall(box_match.group(1))]
        if len(nums) >= 4:
            return nums[0], nums[1], nums[2], nums[3]
    coords = parse_points(text)
    if len(coords) >= 2:
        (x1, y1), (x2, y2) = coords[:2]
        return x1, y1, x2, y2
    return None


def valid_box(box: tuple[float, float, float, float] | None) -> bool:
    if box is None:
        return False
    x1, y1, x2, y2 = box
    return 0 <= x1 <= 1000 and 0 <= y1 <= 1000 and 0 <= x2 <= 1000 and 0 <= y2 <= 1000 and x2 > x1 and y2 > y1


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-9)


def center_dist(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    acx, acy = (a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0
    bcx, bcy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
    return math.hypot(acx - bcx, acy - bcy)


def norm_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[\"'`]+|[\"'`.]+$", "", text)
    return text


def text_score(pred: str, gold: str) -> dict[str, float | bool | str | None]:
    p = norm_text(pred)
    g = norm_text(gold)
    exact = p == g
    p_delettered = re.sub(r"^[a-e]\s*[\.\):：、-]\s*", "", p).strip()
    loose = exact or p_delettered == g or (len(g) >= 2 and re.search(rf"(^|\b){re.escape(g)}($|\b)", p) is not None)
    bool_acc = None
    if g in {"true", "false", "yes", "no"}:
        bool_acc = exact or p_delettered == g
    mc_acc = None
    if len(g) == 1 and g in "abcde":
        mc_acc = p[:1] == g or p.startswith({"a": "a.", "b": "b.", "c": "c.", "d": "d.", "e": "e."}[g])
    return {"text_exact": exact, "text_loose": loose, "bool_acc": bool_acc, "mc_acc": mc_acc, "pred_norm": p, "gold_norm": g}


def score_row(pred: str, gold: str, expected: str) -> dict[str, Any]:
    out: dict[str, Any] = {"expected_format": expected}
    if expected == "box":
        gold_box = parse_box(gold)
        pred_box = parse_box(pred)
        out["format_pass"] = "<box>" in pred and valid_box(pred_box)
        out["coord_valid"] = valid_box(pred_box)
        if valid_box(gold_box) and valid_box(pred_box):
            assert gold_box is not None and pred_box is not None
            iou = box_iou(pred_box, gold_box)
            out.update({
                "iou": iou,
                "acc_iou_0_3": iou >= 0.3,
                "acc_iou_0_5": iou >= 0.5,
                "acc_iou_0_75": iou >= 0.75,
                "center_dist": center_dist(pred_box, gold_box),
            })
        else:
            out.update({"iou": None, "acc_iou_0_3": False, "acc_iou_0_5": False, "acc_iou_0_75": False, "center_dist": None})
        return out
    if expected == "point":
        gold_pts = parse_points(gold)
        pred_pts = parse_points(pred)
        coord_valid = bool(pred_pts) and all(0 <= x <= 1000 and 0 <= y <= 1000 for x, y in pred_pts)
        out["format_pass"] = "<point>" in pred and coord_valid
        out["coord_valid"] = coord_valid
        out["pred_point_count"] = len(pred_pts)
        out["gold_point_count"] = len(gold_pts)
        if pred_pts and gold_pts:
            pred_to_gold = [min(math.hypot(px - gx, py - gy) for gx, gy in gold_pts) for px, py in pred_pts]
            gold_to_pred = [min(math.hypot(px - gx, py - gy) for px, py in pred_pts) for gx, gy in gold_pts]
            min_dist = min(pred_to_gold)
            out.update({
                "min_point_dist": min_dist,
                "mean_pred_to_gold_dist": sum(pred_to_gold) / len(pred_to_gold),
                "mean_gold_to_pred_dist": sum(gold_to_pred) / len(gold_to_pred),
                "hit_at_50": min_dist <= 50,
                "hit_at_100": min_dist <= 100,
                "point_count_abs_diff": abs(len(pred_pts) - len(gold_pts)),
            })
        else:
            out.update({
                "min_point_dist": None,
                "mean_pred_to_gold_dist": None,
                "mean_gold_to_pred_dist": None,
                "hit_at_50": False,
                "hit_at_100": False,
                "point_count_abs_diff": abs(len(pred_pts) - len(gold_pts)),
            })
        return out
    text = text_score(pred, gold)
    out["format_pass"] = True
    out.update(text)
    return out


class Metrics:
    def __init__(self) -> None:
        self.n = 0
        self.c = Counter()
        self.s = Counter()

    def add(self, score: dict[str, Any]) -> None:
        self.n += 1
        for key, value in score.items():
            if isinstance(value, bool):
                self.c[key] += int(value)
            elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                self.s[key] += float(value)
                self.c[f"{key}__count"] += 1

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {"n": self.n}
        for key, val in sorted(self.c.items()):
            if key.endswith("__count"):
                continue
            out[key] = val / max(self.n, 1)
        for key, total in sorted(self.s.items()):
            denom = self.c.get(f"{key}__count", self.n)
            out[f"{key}_mean"] = total / max(denom, 1)
        return out


def load_rows(path: Path, limit: int | None, expected_format_filter: str | None = None) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if expected_format_filter and expected_format_of_row(row) != expected_format_filter:
                continue
            rows.append(row)
            if limit and len(rows) >= limit:
                break
    return rows


def image_data_url(image_path: str) -> str:
    path = Path(image_path)
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


POINT_PROTOCOL = (
    "Return only the final answer in this exact format: "
    "<point>[[x1,y1],[x2,y2],...]</point>. "
    "Return 1 to 5 points. Use integer coordinates from 0 to 1000. "
    "Ignore any earlier formatting example such as [(x, y)]. "
    "Do not use parentheses. Do not put coordinates in XML attributes. "
    "The opening tag must be exactly <point> and the closing tag must be exactly </point>. "
    "Do not include any prose."
)


def user_content_from_row(row: dict[str, Any], enforce_point_protocol: bool = False) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for image_path in row.get("image") or []:
        content.append({"type": "image_url", "image_url": {"url": image_data_url(str(image_path))}})
    text = first_user_text(row).replace("<image>", "").replace("<video>", "").strip()
    if enforce_point_protocol and expected_format_of_row(row) == "point":
        text = f"{text}\n\n{POINT_PROTOCOL}".strip()
    if text:
        content.append({"type": "text", "text": text})
    return content


def messages_from_row(row: dict[str, Any], enforce_point_protocol: bool = False) -> list[dict[str, Any]]:
    sys_text = system_text(row)
    if enforce_point_protocol and expected_format_of_row(row) == "point":
        sys_text = f"{sys_text} {POINT_PROTOCOL}"
    return [
        {"role": "system", "content": sys_text},
        {"role": "user", "content": user_content_from_row(row, enforce_point_protocol=enforce_point_protocol)},
    ]


def completion_content(choice_message: Any) -> str:
    if isinstance(choice_message, str):
        return choice_message
    if not isinstance(choice_message, dict):
        return ""
    content = choice_message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return ""


def post_chat_completion(
    *,
    endpoint: str,
    api_key: str,
    model_id: str,
    row: dict[str, Any],
    max_tokens: int,
    temperature: float,
    timeout: float,
    retries: int,
    retry_sleep: float,
    enable_thinking: bool,
    enforce_point_protocol: bool,
) -> tuple[str, str | None]:
    payload = {
        "model": model_id,
        "messages": messages_from_row(row, enforce_point_protocol=enforce_point_protocol),
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if not enable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_error: str | None = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            choices = data.get("choices") or []
            if not choices:
                return "", "empty choices"
            return completion_content(choices[0].get("message")), None
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(retry_sleep * (attempt + 1))
    return "", last_error


def score_prediction(model_name: str, row: dict[str, Any], pred: str, error: str | None) -> dict[str, Any]:
    meta = ((row.get("metadata") or {}).get("raw_holdout_eval") or {})
    gold = answer_of(row)
    expected = expected_format_of_row(row)
    score = score_row(pred, gold, expected)
    pool = str(meta.get("source_pool") or row.get("dataset"))
    group = str(meta.get("group") or "unknown")
    return {
        "model": model_name,
        "eval_index": meta.get("eval_index"),
        "source_pool": pool,
        "group": group,
        "expected_format": expected,
        "gold": gold,
        "prediction": pred,
        "score": score,
        "api_error": error,
        "metadata": meta,
    }


def worker(args: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    row, cfg = args
    pred, error = post_chat_completion(
        endpoint=cfg["endpoint"],
        api_key=cfg["api_key"],
        model_id=cfg["model_id"],
        row=row,
        max_tokens=cfg["max_tokens"],
        temperature=cfg["temperature"],
        timeout=cfg["timeout"],
        retries=cfg["retries"],
        retry_sleep=cfg["retry_sleep"],
        enable_thinking=cfg["enable_thinking"],
        enforce_point_protocol=cfg["enforce_point_protocol"],
    )
    return score_prediction(cfg["model_name"], row, pred, error)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--api-base", required=True, help="Base URL such as http://127.0.0.1:13001/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--eval-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--flush-every", type=int, default=50)
    parser.add_argument("--enable-thinking", action="store_true", help="Keep Qwen3 thinking mode enabled. Default disables it for eval answers.")
    parser.add_argument("--expected-format-filter", default=None, choices=["box", "point", "text", "mixed_grounding"])
    parser.add_argument("--enforce-point-protocol", action="store_true", help="For point rows, append the strict <point>[[x,y],...]</point> output protocol to the prompt.")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"missing API key env var: {args.api_key_env}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    metrics_path = out_dir / "metrics.json"
    rows = load_rows(Path(args.eval_path), args.limit, expected_format_filter=args.expected_format_filter)

    endpoint = args.api_base.rstrip("/") + "/chat/completions"
    cfg = {
        "endpoint": endpoint,
        "api_key": api_key,
        "model_name": args.model_name,
        "model_id": args.model_id or args.model_name,
        "max_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "timeout": args.timeout,
        "retries": args.retries,
        "retry_sleep": args.retry_sleep,
        "enable_thinking": args.enable_thinking,
        "enforce_point_protocol": args.enforce_point_protocol,
    }

    overall = Metrics()
    by_pool: dict[str, Metrics] = defaultdict(Metrics)
    by_format: dict[str, Metrics] = defaultdict(Metrics)
    by_group: dict[str, Metrics] = defaultdict(Metrics)
    errors = 0
    start = time.time()
    processed = 0

    log({
        "stage": "start",
        "model": args.model_name,
        "model_id": cfg["model_id"],
        "endpoint": endpoint,
        "rows": len(rows),
        "workers": args.workers,
        "enable_thinking": args.enable_thinking,
        "expected_format_filter": args.expected_format_filter,
        "enforce_point_protocol": args.enforce_point_protocol,
        "time": now(),
    })

    with pred_path.open("w", encoding="utf-8") as wf:
        if args.workers <= 1:
            iterable = map(worker, ((row, cfg) for row in rows))
            for record in iterable:
                score = record["score"]
                overall.add(score)
                by_pool[record["source_pool"]].add(score)
                by_format[record["expected_format"]].add(score)
                by_group[f"{record['source_pool']}/{record['group']}"].add(score)
                errors += int(record.get("api_error") is not None)
                wf.write(json.dumps(record, ensure_ascii=False) + "\n")
                processed += 1
                if processed % args.flush_every == 0 or processed == len(rows):
                    wf.flush()
                    elapsed = time.time() - start
                    log({
                        "stage": "progress",
                        "model": args.model_name,
                        "processed": processed,
                        "total": len(rows),
                        "api_errors": errors,
                        "samples_per_sec": round(processed / max(elapsed, 1e-6), 4),
                        "time": now(),
                    })
        else:
            with futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
                pending = [ex.submit(worker, (row, cfg)) for row in rows]
                for fut in futures.as_completed(pending):
                    record = fut.result()
                    score = record["score"]
                    overall.add(score)
                    by_pool[record["source_pool"]].add(score)
                    by_format[record["expected_format"]].add(score)
                    by_group[f"{record['source_pool']}/{record['group']}"].add(score)
                    errors += int(record.get("api_error") is not None)
                    wf.write(json.dumps(record, ensure_ascii=False) + "\n")
                    processed += 1
                    if processed % args.flush_every == 0 or processed == len(rows):
                        wf.flush()
                        elapsed = time.time() - start
                        log({
                            "stage": "progress",
                            "model": args.model_name,
                            "processed": processed,
                            "total": len(rows),
                            "api_errors": errors,
                            "samples_per_sec": round(processed / max(elapsed, 1e-6), 4),
                            "time": now(),
                        })

    summary = {
        "model": args.model_name,
        "model_path": f"{args.api_base}#{cfg['model_id']}",
        "eval_path": args.eval_path,
        "rows": len(rows),
        "batch_size": None,
        "workers": args.workers,
        "max_new_tokens": args.max_new_tokens,
        "api_errors": errors,
        "enable_thinking": args.enable_thinking,
        "expected_format_filter": args.expected_format_filter,
        "enforce_point_protocol": args.enforce_point_protocol,
        "seconds": round(time.time() - start, 3),
        "overall": overall.summary(),
        "by_pool": {k: v.summary() for k, v in sorted(by_pool.items())},
        "by_format": {k: v.summary() for k, v in sorted(by_format.items())},
        "by_group": {k: v.summary() for k, v in sorted(by_group.items())},
        "prediction_file": str(pred_path),
    }
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log({"stage": "done", "model": args.model_name, "metrics": str(metrics_path), "seconds": summary["seconds"], "api_errors": errors, "time": now()})


if __name__ == "__main__":
    main()
