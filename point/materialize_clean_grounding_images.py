#!/usr/bin/env python3
"""Extract archive-backed images referenced by clean grounding pools."""

from __future__ import annotations

import argparse
import collections
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any


def iter_archive_refs(clean_dir: Path) -> dict[tuple[str, str], str]:
    refs: dict[tuple[str, str], str] = {}
    for path in sorted(clean_dir.glob("*_clean_v1.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                meta = row.get("metadata") or {}
                archive = meta.get("image_archive")
                member = meta.get("image_member")
                images = row.get("image") or []
                if not archive or not member or not images:
                    continue
                refs[(str(archive), str(member))] = str(images[0])
    return refs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-dir", default="/data/msz/point/data_grounding_clean_v1")
    parser.add_argument("--summary", default=None)
    parser.add_argument("--validate-existing", action="store_true")
    args = parser.parse_args()

    clean_dir = Path(args.clean_dir)
    summary_path = Path(args.summary) if args.summary else clean_dir / "materialized_images_summary.json"
    refs = iter_archive_refs(clean_dir)
    by_archive: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
    for (archive, member), target in refs.items():
        by_archive[archive].append((member, target))

    stats: dict[str, Any] = {
        "clean_dir": str(clean_dir),
        "unique_archive_members": len(refs),
        "archives": {},
        "created": 0,
        "existing": 0,
        "missing_member": 0,
        "errors": 0,
    }
    for archive, items in sorted(by_archive.items()):
        archive_stats = collections.Counter()
        archive_path = Path(archive)
        if not archive_path.exists():
            archive_stats["missing_archive"] += len(items)
            stats["errors"] += len(items)
            stats["archives"][archive] = dict(archive_stats)
            continue
        with zipfile.ZipFile(archive_path) as z:
            names = set(z.namelist())
            for member, target in items:
                target_path = Path(target)
                if target_path.exists() and target_path.stat().st_size > 0:
                    archive_stats["existing"] += 1
                    stats["existing"] += 1
                    continue
                if member not in names:
                    archive_stats["missing_member"] += 1
                    stats["missing_member"] += 1
                    continue
                target_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with z.open(member) as src, target_path.open("wb") as dst:
                        shutil.copyfileobj(src, dst, length=1024 * 1024)
                    archive_stats["created"] += 1
                    stats["created"] += 1
                except Exception:
                    archive_stats["errors"] += 1
                    stats["errors"] += 1
        stats["archives"][archive] = dict(archive_stats)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
