#!/usr/bin/env python3
"""Collect per-model raw holdout metrics into compact comparison tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


KEYS = [
    "format_pass",
    "coord_valid",
    "iou_mean",
    "acc_iou_0_3",
    "acc_iou_0_5",
    "acc_iou_0_75",
    "center_dist_mean",
    "hit_at_50",
    "hit_at_100",
    "min_point_dist_mean",
    "mean_pred_to_gold_dist_mean",
    "text_exact",
    "text_loose",
    "bool_acc",
    "mc_acc",
]


def pick(metrics: dict[str, Any]) -> dict[str, Any]:
    return {k: metrics[k] for k in KEYS if k in metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    models = []
    by_pool: dict[str, dict[str, Any]] = {}
    by_format: dict[str, dict[str, Any]] = {}
    for metrics_path in sorted(run_dir.glob("*/metrics.json")):
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        name = data["model"]
        models.append({
            "model": name,
            "rows": data.get("rows"),
            "seconds": data.get("seconds"),
            "samples_per_sec": data.get("rows", 0) / max(float(data.get("seconds", 1)), 1e-9),
            **pick(data.get("overall") or {}),
        })
        for pool, metrics in (data.get("by_pool") or {}).items():
            by_pool.setdefault(pool, {})[name] = {"n": metrics.get("n"), **pick(metrics)}
        for fmt, metrics in (data.get("by_format") or {}).items():
            by_format.setdefault(fmt, {})[name] = {"n": metrics.get("n"), **pick(metrics)}
    summary = {
        "run_dir": str(run_dir),
        "models": models,
        "by_pool": by_pool,
        "by_format": by_format,
    }
    out_path = run_dir / "comparison_summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
