#!/usr/bin/env python3
"""Deterministically shuffle OPD student JSONL files with bounded memory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path


def now() -> str:
    return time.strftime("%F %T")


def log(obj: dict) -> None:
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def shuffle_file(path: Path, *, buckets: int, seed: int) -> dict:
    start = time.time()
    tmp_dir = path.parent / f".{path.name}.shuffle_buckets_{os.getpid()}"
    tmp_out = path.parent / f".{path.name}.shuffled_{os.getpid()}.tmp"
    if tmp_dir.exists() or tmp_out.exists():
        raise RuntimeError(f"temporary path already exists for {path}")

    tmp_dir.mkdir(parents=True)
    handles = []
    rows_in = 0
    try:
        for bucket_id in range(buckets):
            handles.append((tmp_dir / f"bucket_{bucket_id:04d}.jsonl").open("wb"))

        log({"stage": "bucket_start", "path": str(path), "buckets": buckets, "time": now()})
        seed_prefix = f"opd_student_v1_shuffle_seed={seed}\0".encode()
        with path.open("rb") as rf:
            for rows_in, line in enumerate(rf, start=1):
                digest = hashlib.blake2b(seed_prefix + line, digest_size=8).digest()
                bucket_id = int.from_bytes(digest, "big") % buckets
                handles[bucket_id].write(line)
                if rows_in % 100_000 == 0:
                    log({"stage": "bucket_progress", "path": str(path), "rows": rows_in, "time": now()})
    finally:
        for handle in handles:
            handle.close()

    bucket_ids = list(range(buckets))
    random.Random(seed + 17).shuffle(bucket_ids)
    rows_out = 0
    log({"stage": "merge_start", "path": str(path), "time": now()})
    with tmp_out.open("wb") as wf:
        for rank, bucket_id in enumerate(bucket_ids, start=1):
            bucket_path = tmp_dir / f"bucket_{bucket_id:04d}.jsonl"
            with bucket_path.open("rb") as bf:
                lines = bf.readlines()
            random.Random(seed + 1_000_003 * bucket_id).shuffle(lines)
            rows_out += len(lines)
            wf.writelines(lines)
            if rank % 32 == 0 or rank == len(bucket_ids):
                log({"stage": "merge_progress", "path": str(path), "buckets_done": rank, "rows_out": rows_out, "time": now()})

    if rows_in != rows_out:
        raise RuntimeError(f"row-count mismatch for {path}: input={rows_in}, output={rows_out}")

    os.replace(tmp_out, path)
    shutil.rmtree(tmp_dir)
    return {
        "path": str(path),
        "rows": rows_out,
        "buckets": buckets,
        "seed": seed,
        "seconds": round(time.time() - start, 3),
    }


def update_summary(out_root: Path, results: list[dict], seed: int) -> None:
    for rel in ["summary.json", "manifests/opd_student_v1_build_summary.json"]:
        path = out_root / rel
        data = json.loads(path.read_text(encoding="utf-8"))
        data["shuffle"] = {
            "method": "hash bucket external shuffle, bucket-internal random shuffle, random bucket merge",
            "seed": seed,
            "finished_at": now(),
            "files": results,
        }
        tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default="/data/msz/point/opd_student_v1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-buckets", type=int, default=256)
    parser.add_argument("--eval-buckets", type=int, default=64)
    args = parser.parse_args()

    out_root = Path(args.out_root)
    results = [
        shuffle_file(out_root / "train_prompts.jsonl", buckets=args.train_buckets, seed=args.seed + 10_001),
        shuffle_file(out_root / "eval_prompts.jsonl", buckets=args.eval_buckets, seed=args.seed + 20_001),
    ]
    update_summary(out_root, results, args.seed)
    log({"stage": "done", "out_root": str(out_root), "results": results, "time": now()})


if __name__ == "__main__":
    main()
