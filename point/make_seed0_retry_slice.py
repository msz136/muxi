#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-step", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    args = parser.parse_args()

    start_pos = args.start_step * args.world_size * args.per_device_batch_size * args.grad_accum
    rows: list[str] = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(line if line.endswith("\n") else f"{line}\n")
                if len(rows) >= args.limit:
                    break

    if start_pos >= len(rows):
        raise ValueError(f"start_pos={start_pos} >= rows={len(rows)}")

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    order = torch.randperm(len(rows), generator=generator).tolist()
    selected = order[start_pos:]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for idx in selected:
            f.write(rows[idx])

    print(
        f"wrote {len(selected)} rows to {output}; "
        f"source_rows={len(rows)} start_step={args.start_step} start_pos={start_pos}"
    )


if __name__ == "__main__":
    main()
