#!/usr/bin/env python3
"""Prepare AceBrain-style Qwen-VL grounding data from existing raw datasets.

This script does data conversion only. It never starts training and never
downloads dependencies or data.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

SYSTEM_PROMPT = (
    "You are a helpful vision-language assistant. When the user asks for a "
    "location, answer with coordinates in the range 0 to 1000."
)

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def log(msg: str) -> None:
    print(msg, flush=True)


def read_json_or_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    yield obj
        return

    if path.suffix != ".json":
        return

    # Big RoboPoint-style JSON arrays should be streamed instead of json.load.
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        head = f.read(4096)
        f.seek(0)
        first = head.lstrip()[:1]
        if first == "[":
            decoder = json.JSONDecoder()
            buf = ""
            eof = False
            started = False
            while True:
                if not eof:
                    chunk = f.read(1024 * 1024)
                    if chunk:
                        buf += chunk
                    else:
                        eof = True
                pos = 0
                made_progress = False
                while True:
                    while pos < len(buf) and buf[pos].isspace():
                        pos += 1
                    if not started:
                        if pos < len(buf) and buf[pos] == "[":
                            pos += 1
                            started = True
                        else:
                            break
                    while pos < len(buf) and buf[pos] in " \r\n\t,":
                        pos += 1
                    if pos < len(buf) and buf[pos] == "]":
                        return
                    try:
                        obj, end = decoder.raw_decode(buf, pos)
                    except json.JSONDecodeError:
                        break
                    if isinstance(obj, dict):
                        yield obj
                    pos = end
                    made_progress = True
                buf = buf[pos:]
                if eof:
                    return
                if not made_progress and len(buf) > 16 * 1024 * 1024:
                    raise RuntimeError(f"Cannot stream JSON object from {path}")
        else:
            obj = json.load(f)
            if isinstance(obj, list):
                for item in obj:
                    if isinstance(item, dict):
                        yield item
            elif isinstance(obj, dict):
                # Known container keys (list-valued)
                for key in ("data", "annotations", "items", "samples", "train", "generated", "real"):
                    val = obj.get(key)
                    if isinstance(val, list):
                        for item in val:
                            if isinstance(item, dict):
                                yield item
                        return
                # UUID-keyed dicts (EmbSpatial): each value is a list of task dicts
                uuid_pattern = re.compile(
                    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I
                )
                uuid_like = [k for k in obj if uuid_pattern.match(str(k))]
                if uuid_like and len(uuid_like) >= max(1, len(obj) * 0.5):
                    for k in obj:
                        val = obj[k]
                        if isinstance(val, list):
                            for item in val:
                                if isinstance(item, dict):
                                    yield item
                    return
                yield obj


def read_parquet(path: Path) -> Iterable[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception:
        try:
            import pandas as pd  # type: ignore
        except Exception:
            log(f"[skip] parquet reader unavailable: {path}")
            return
        df = pd.read_parquet(path)
        for row in df.to_dict("records"):
            yield dict(row)
        return

    table = pq.ParquetFile(path)
    for batch in table.iter_batches(batch_size=4096):
        cols = batch.to_pydict()
        keys = list(cols)
        for i in range(batch.num_rows):
            yield {k: cols[k][i] for k in keys}


def candidate_files(base: Path, names: list[str], exts: tuple[str, ...] = (".json", ".jsonl")) -> list[Path]:
    out: list[Path] = []
    for name in names:
        p = base / name
        if p.exists():
            out.append(p)
    for ext in exts:
        out += sorted(base.glob(f"*{ext}"))
        out += sorted((base / "data").glob(f"*{ext}")) if (base / "data").exists() else []
        out += sorted((base / "json_qwen").glob(f"*{ext}")) if (base / "json_qwen").exists() else []
    seen: set[Path] = set()
    uniq: list[Path] = []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def walk_strings(obj: Any) -> Iterable[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from walk_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk_strings(v)


def strip_format_hint(text: str) -> str:
    text = re.sub(r"Your answer should[^.\n]*(?:\.|\n|$)", "", text, flags=re.I)
    text = re.sub(r"The coordinates should[^.\n]*(?:\.|\n|$)", "", text, flags=re.I)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def media_path(value: Any, data_path: str | Path, roots: list[Path]) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip()
    if value.startswith("http://") or value.startswith("https://"):
        return value
    p = Path(value)
    if p.is_absolute() and p.exists():
        return str(p)
    tries = [Path(data_path) / value, Path(data_path) / p.name]
    for root in roots:
        tries += [
            root / value,
            root / p.name,
            root / "images" / value,
            root / "images" / p.name,
            root / "image" / value,
            root / "videos" / value,
            root / "videos" / p.name,
        ]
    for t in tries:
        if t.exists():
            return str(t)
    return value


def get_image_size(path: str | None) -> tuple[int, int] | None:
    if not path or path.startswith("http"):
        return None
    try:
        from PIL import Image  # type: ignore
    except Exception:
        return None
    try:
        with Image.open(path) as im:
            return im.size
    except Exception:
        return None


def to_float_pair_list(value: Any) -> list[list[float]]:
    if value is None:
        return []
    if isinstance(value, str):
        s = value.strip()
        for wrapper in ("<point>", "</point>"):
            s = s.replace(wrapper, "")
        # Fast path: regex number extraction (much faster than ast.literal_eval)
        nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", s)]
        if len(nums) >= 2:
            return [[nums[i], nums[i + 1]] for i in range(0, len(nums) - 1, 2)]
        # Fallback: try literal eval for complex nested structures
        try:
            value = ast.literal_eval(s)
        except Exception:
            return []
    if isinstance(value, dict):
        for key in ("points", "point", "coords", "coordinate", "trajectory"):
            if key in value:
                return to_float_pair_list(value[key])
        if "x" in value and "y" in value:
            return [[float(value["x"]), float(value["y"])]]
        return []
    if isinstance(value, (list, tuple)):
        if len(value) >= 2 and all(isinstance(x, (int, float, str)) for x in value[:2]):
            return [[float(value[0]), float(value[1])]]
        pts: list[list[float]] = []
        for item in value:
            pts.extend(to_float_pair_list(item))
        return pts
    return []


def scale_points(points: list[list[float]], width: int | None = None, height: int | None = None) -> list[list[int]]:
    if not points:
        return []
    max_abs = max(abs(v) for p in points for v in p[:2])
    out = []
    for x, y in points:
        if width and height and max_abs > 100:
            sx, sy = x / width * 1000, y / height * 1000
        elif max_abs <= 1.05:
            sx, sy = x * 1000, y * 1000
        elif max_abs <= 100:
            sx, sy = x * 10, y * 10
        else:
            sx, sy = x, y
        out.append([int(round(max(0, min(1000, sx)))), int(round(max(0, min(1000, sy))))])
    return out


def point_answer(points: list[list[int]]) -> str | None:
    return f"<point>{json.dumps(points, separators=(',', ':'))}</point>" if points else None


def make_sample(
    dataset: str,
    data_path: str,
    image: list[str] | None,
    video: list[str] | None,
    user: str,
    answer: str,
    system: str = SYSTEM_PROMPT,
) -> dict[str, Any] | None:
    if not user or not answer:
        return None
    image = image or []
    video = video or []
    user = strip_format_hint(str(user))
    if image and user.count("<image>") != len(image):
        user = (" ".join(["<image>"] * len(image)) + "\n" + user.replace("<image>", "")).strip()
    if video and user.count("<video>") != len(video):
        user = (" ".join(["<video>"] * len(video)) + "\n" + user.replace("<video>", "")).strip()
    return {
        "dataset": dataset,
        "data_path": data_path,
        "image": image,
        "video": video,
        "conversations": [
            {"from": "system", "value": system},
            {"from": "human", "value": user},
            {"from": "gpt", "value": answer},
        ],
    }


class Writer:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        self.files = {
            "grounding": (out_dir / "grounding_point.jsonl").open("w", encoding="utf-8"),
            "keepalive": (out_dir / "keepalive_vqa.jsonl").open("w", encoding="utf-8"),
        }
        self.counts: Counter[str] = Counter()
        self.bad = (out_dir / "bad_records.jsonl").open("w", encoding="utf-8")

    def write(self, kind: str, sample: dict[str, Any] | None) -> None:
        if not sample:
            return
        self.files[kind].write(json.dumps(sample, ensure_ascii=False) + "\n")
        self.counts[f"{kind}:{sample['dataset']}"] += 1
        n = sum(self.counts.values())
        if n and n % 10000 == 0:
            log(f"[progress] written={n} counts={dict(self.counts)}")

    def bad_record(self, dataset: str, reason: str, rec: Any) -> None:
        self.bad.write(json.dumps({"dataset": dataset, "reason": reason, "record": rec}, ensure_ascii=False) + "\n")

    def close(self) -> None:
        for f in self.files.values():
            f.close()
        self.bad.close()


def convert_robopoint(root: Path, writer: Writer, limit: int) -> None:
    base = root / "RoboPoint"
    if not base.exists():
        return
    data_path = str(base / "images") if (base / "images").exists() else str(base)
    files = candidate_files(base, ["robopoint_annotation.json", "robopoint_1432k.json", "metadata.json"])
    n = 0
    for path in files:
        log(f"[RoboPoint] reading {path}")
        for rec in read_json_or_jsonl(path):
            conv = rec.get("conversations")
            img = rec.get("image") or rec.get("img") or rec.get("image_path")
            if not isinstance(conv, list) or not img:
                continue
            user, ans = None, None
            for m in conv:
                role = str(m.get("from", m.get("role", ""))).lower()
                val = m.get("value", m.get("content", ""))
                if role in {"human", "user"} and user is None:
                    user = str(val)
                elif role in {"gpt", "assistant"}:
                    ans = val
            pts = scale_points(to_float_pair_list(ans))
            out = point_answer(pts)
            if not out:
                continue
            image = [media_path(img, data_path, [base]) or str(img)]
            writer.write("grounding", make_sample("robopoint", data_path, image, [], user or "Point to the target object.", out))
            n += 1
            if limit and n >= limit:
                return


def convert_pixmo(root: Path, writer: Writer, limit: int, allow_url: bool) -> None:
    base = root / "pixmo-points"
    if not base.exists():
        return
    files = candidate_files(base, ["pixmo_points_annotation_struct2d_format.jsonl"], (".json", ".jsonl", ".parquet"))
    n = 0
    for path in files:
        log(f"[PixMo] reading {path}")
        records = read_parquet(path) if path.suffix == ".parquet" else read_json_or_jsonl(path)
        for rec in records:
            label = rec.get("label") or rec.get("referring_expression") or rec.get("expression") or rec.get("text")
            pts = rec.get("points") or rec.get("point") or rec.get("coords")
            img = rec.get("image") or rec.get("image_path") or rec.get("image_url") or rec.get("url")
            img_path = media_path(img, base, [base]) if img else None
            if img_path and img_path.startswith("http") and not allow_url:
                continue
            size = get_image_size(img_path)
            scaled = scale_points(to_float_pair_list(pts), *(size or (None, None)))
            out = point_answer(scaled)
            if not (label and img_path and out):
                continue
            user = f"Point to {label}."
            writer.write("grounding", make_sample("pixmopoints", str(base), [img_path], [], user, out))
            n += 1
            if limit and n >= limit:
                return


def convert_sharerobot_json(root: Path, writer: Writer, limit: int) -> None:
    base = root / "ShareRobot"
    embodied = root / "embodied_jsons"
    # Prefer AceBrain-style already converted files if they exist.
    for name, data_path in [
        ("sharerobot_converted_trajectory.jsonl", base / "trajectory" / "images"),
        ("sharerobot_converted_affordance.jsonl", base / "affordance" / "images"),
    ]:
        path = embodied / name
        if path.exists():
            log(f"[ShareRobot] reading converted {path}")
            n = 0
            for rec in read_json_or_jsonl(path):
                sample = normalize_existing_qwen(rec, "sharerobot", str(data_path), [base, embodied])
                writer.write("grounding", sample)
                n += 1
                if limit and n >= limit:
                    break

    if not base.exists():
        return
    files = candidate_files(base, ["trajectory/trajectory.json", "affordance/affordance.json"])
    n = 0
    for path in files:
        kind = "trajectory" if "trajectory" in str(path).lower() else "affordance"
        data_path = base / kind / "images"
        log(f"[ShareRobot:{kind}] reading {path}")
        for rec in read_json_or_jsonl(path):
            img = rec.get("image") or rec.get("image_path") or rec.get("frame") or rec.get("id")
            if isinstance(img, int):
                img = f"{img}.jpg"
            img_path = media_path(img, data_path, [base, path.parent]) if img else None
            width = rec.get("width") or rec.get("image_width") or rec.get("original_width")
            height = rec.get("height") or rec.get("image_height") or rec.get("original_height")
            if not width or not height:
                size = get_image_size(img_path)
                if size:
                    width, height = size
            pts = rec.get("points") or rec.get("trajectory") or rec.get("point") or rec.get("affordance")
            if isinstance(pts, dict) and {"x", "y", "w", "h"} <= set(pts):
                pts = [[float(pts["x"]) + float(pts["w"]) / 2, float(pts["y"]) + float(pts["h"]) / 2]]
            scaled = scale_points(to_float_pair_list(pts), int(width) if width else None, int(height) if height else None)
            out = point_answer(scaled)
            prompt = rec.get("instruction") or rec.get("question") or (
                "Predict the robot end-effector trajectory." if kind == "trajectory" else "Point to the actionable affordance location."
            )
            if img_path and out:
                writer.write("grounding", make_sample(f"sharerobot_{kind}", str(data_path), [img_path], [], str(prompt), out))
                n += 1
                if limit and n >= limit:
                    return


def convert_embspatial(root: Path, writer: Writer, limit: int, allow_url: bool) -> None:
    base = root / "EmbSpatial"
    if not base.exists():
        return
    files = candidate_files(base, ["all.json", "all_v2.json", "transformed_all.json", "transformed_all_v2.json"])
    n = 0
    for path in files:
        log(f"[EmbSpatial] reading {path}")
        for rec in read_json_or_jsonl(path):
            query = rec.get("query") or rec.get("task_query") or rec.get("question")
            label = rec.get("label") or rec.get("task_label")
            img_url = rec.get("image_url") or rec.get("image") or rec.get("img")
            if not (query and isinstance(label, (bool, int))):
                continue
            # Build a conversational answer
            constraint = rec.get("constraint_name", "") or rec.get("constraint", "")
            if isinstance(constraint, list):
                constraint = constraint[0].get("constraint_name", "") if constraint else ""
            if isinstance(label, bool):
                answer = "true" if label else "false"
            else:
                answer = str(int(label))
            # Make the question natural
            if constraint:
                user = (
                    f"<image>\n"
                    f"Determine whether the following spatial relationship holds in the image: {query}\n"
                    f"Answer with 'true' or 'false'."
                )
            else:
                user = (
                    f"<image>\n"
                    f"{query}\n"
                    f"Answer with 'true' or 'false'."
                )
            img_path = media_path(img_url, base, [base]) if img_url else None
            if img_path and img_path.startswith("http") and not allow_url:
                continue
            images = [img_path] if img_path else []
            writer.write(
                "keepalive",
                make_sample("embspatial", str(base), images, [], user, answer),
            )
            n += 1
            if limit and n >= limit:
                return
    log(f"[EmbSpatial] wrote {n} samples")


def convert_robo2vlm1(root: Path, writer: Writer, limit: int) -> None:
    base = root / "Robo2VLM-1"
    if not base.exists():
        return
    # Extract embedded images to local dir
    img_dir = base / "_extracted_images"
    img_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(base.glob("data/train-*.parquet"))
    n = 0
    for path in files:
        log(f"[Robo2VLM-1] reading {path}")
        for rec in read_parquet(path):
            q = rec.get("question")
            choices = rec.get("choices")
            correct_idx = rec.get("correct_answer")
            img_data = rec.get("image")
            if not (q and isinstance(correct_idx, int)):
                continue
            # Parse choices
            if isinstance(choices, str):
                try:
                    choices = ast.literal_eval(choices)
                except Exception:
                    choices = [x.strip().strip("'\"") for x in choices.strip("[]").split(",")]
            if not isinstance(choices, list) or correct_idx < 0 or correct_idx >= len(choices):
                continue
            answer = str(choices[correct_idx])
            # Extract embedded image
            img_path = None
            if isinstance(img_data, dict):
                raw = img_data.get("bytes")
                img_path_str = img_data.get("path") or rec.get("id", "image")
                if isinstance(raw, bytes):
                    safe_id = re.sub(r"[^a-zA-Z0-9_.-]", "_", str(img_path_str))[:120]
                    fname = f"{n:08d}_{safe_id}.png"
                    dst = img_dir / fname
                    if not dst.exists():
                        dst.write_bytes(raw)
                    img_path = str(dst)
            if not img_path:
                continue
            # Form user prompt with choices
            choice_lines = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices))
            user = (
                f"<image>\n"
                f"{q}\n\n"
                f"Options:\n{choice_lines}\n\n"
                f"Select the correct option."
            )
            # Format answer as the choice text, not just the letter
            writer.write(
                "keepalive",
                make_sample("robo2vlm-1", str(img_dir), [img_path], [], user, answer),
            )
            n += 1
            if limit and n >= limit:
                return
    log(f"[Robo2VLM-1] wrote {n} samples")


def normalize_existing_qwen(rec: dict[str, Any], dataset: str, data_path: str, roots: list[Path]) -> dict[str, Any] | None:
    conv = rec.get("conversations") or rec.get("messages")
    if not isinstance(conv, list):
        return None
    system = SYSTEM_PROMPT
    user = None
    ans = None
    for m in conv:
        role = str(m.get("from", m.get("role", ""))).lower()
        val = m.get("value", m.get("content", ""))
        if role == "system":
            system = str(val)
        elif role in {"human", "user"} and user is None:
            user = str(val)
        elif role in {"gpt", "assistant"}:
            ans = str(val)
    images = rec.get("image") or rec.get("images") or []
    videos = rec.get("video") or rec.get("videos") or []
    if isinstance(images, str):
        images = [images]
    if isinstance(videos, str):
        videos = [videos]
    images = [media_path(x, data_path, roots) or str(x) for x in images]
    videos = [media_path(x, data_path, roots) or str(x) for x in videos]
    return make_sample(dataset, data_path, images, videos, user or "", ans or "", system)


def convert_existing_or_keepalive(root: Path, writer: Writer, dataset: str, rel: str, preferred: list[str], limit: int) -> None:
    base = root / rel
    if not base.exists():
        return
    files = candidate_files(base, preferred, (".json", ".jsonl", ".parquet"))
    n = 0
    for path in files:
        log(f"[{dataset}] reading {path}")
        records = read_parquet(path) if path.suffix == ".parquet" else read_json_or_jsonl(path)
        for rec in records:
            sample = normalize_existing_qwen(rec, dataset, str(base), [base])
            if sample:
                answer = sample["conversations"][-1]["value"]
                kind = "grounding" if "<point>" in answer or "<box>" in answer else "keepalive"
                writer.write(kind, sample)
                n += 1
                if limit and n >= limit:
                    return
                continue
            question = rec.get("question") or rec.get("instruction") or rec.get("prompt") or rec.get("query") or rec.get("text") or rec.get("task_query")
            answer = rec.get("answer") or rec.get("output") or rec.get("response") or rec.get("caption") or rec.get("chosen")
            # Bool labels → text
            if not answer and "label" in rec:
                label = rec["label"]
                if isinstance(label, bool):
                    answer = "true" if label else "false"
                elif isinstance(label, (int, float)):
                    answer = str(int(label)) if isinstance(rec.get("choices"), (list, str)) else str(label)
                else:
                    answer = str(label)
            # integer correct_answer index with choices list
            if not answer and "correct_answer" in rec and "choices" in rec:
                idx = rec["correct_answer"]
                choices = rec["choices"]
                if isinstance(choices, str):
                    try:
                        choices = ast.literal_eval(choices)
                    except Exception:
                        choices = [x.strip() for x in choices.strip("[]").split(",")]
                if isinstance(choices, list) and isinstance(idx, int) and 0 <= idx < len(choices):
                    answer = str(choices[idx])
            # formatted answer from correct_answer if it's a simple value
            if not answer and "correct_answer" in rec:
                ans = rec["correct_answer"]
                if isinstance(ans, (int, float)):
                    answer = str(int(ans))
                else:
                    answer = str(ans)
            if not (question and answer):
                continue
            imgs: list[str] = []
            vids: list[str] = []
            for s in walk_strings(rec):
                low = s.lower()
                if Path(low).suffix in IMG_EXT:
                    mp = media_path(s, base, [base])
                    if mp and mp not in imgs:
                        imgs.append(mp)
                elif Path(low).suffix in VID_EXT:
                    mp = media_path(s, base, [base])
                    if mp and mp not in vids:
                        vids.append(mp)
            writer.write("keepalive", make_sample(dataset, str(base), imgs, vids, str(question), str(answer)))
            n += 1
            if limit and n >= limit:
                return


def merge_outputs(out_dir: Path, seed: int) -> Counter[str]:
    rows: list[str] = []
    counts: Counter[str] = Counter()
    for name in ("grounding_point.jsonl", "keepalive_vqa.jsonl"):
        path = out_dir / name
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rows.append(line)
                try:
                    obj = json.loads(line)
                    counts[obj.get("dataset", "unknown")] += 1
                except Exception:
                    counts["unknown"] += 1
    random.Random(seed).shuffle(rows)
    with (out_dir / "expert_grounding_mix.jsonl").open("w", encoding="utf-8") as w:
        w.writelines(rows)
    return counts


def write_report(out_dir: Path, counts: Counter[str], args: argparse.Namespace) -> None:
    manifest = {
        "data_root": str(args.data_root),
        "out_dir": str(out_dir),
        "seed": args.seed,
        "allow_url_images": args.allow_url_images,
        "counts": dict(counts),
        "outputs": {
            "grounding": str(out_dir / "grounding_point.jsonl"),
            "keepalive": str(out_dir / "keepalive_vqa.jsonl"),
            "mix": str(out_dir / "expert_grounding_mix.jsonl"),
            "bad": str(out_dir / "bad_records.jsonl"),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Expert grounding data report", "", "| dataset | samples |", "|---|---:|"]
    for k, v in counts.most_common():
        lines.append(f"| {k} | {v} |")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def convert_old_robopoint_format(input_path: Path, writer: Writer) -> int:
    """Convert old RoboPoint data ([(x,y)] format in 0-1) to new <point> format (0-1000)."""
    n = 0
    with input_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            conv = rec.get("conversations")
            if not isinstance(conv, list) or len(conv) < 3:
                continue
            # Get the answer from gpt/assistant
            ans = None
            for m in conv:
                role = str(m.get("from", "")).lower()
                if role in ("gpt", "assistant"):
                    ans = m.get("value", "")
            if not ans:
                continue
            # Convert old tuple format to <point> format
            pts = to_float_pair_list(ans)
            scaled = scale_points(pts)
            new_ans = point_answer(scaled)
            if not new_ans:
                continue
            # Build new conversation with consistent system prompt
            system = SYSTEM_PROMPT
            user = None
            for m in conv:
                role = str(m.get("from", "")).lower()
                if role in ("system",):
                    system = str(m.get("value", system))
                elif role in ("human", "user") and user is None:
                    user = str(m.get("value", ""))
            user = user or ""
            # Try to resolve images from walk_strings
            imgs = rec.get("image") or []
            if isinstance(imgs, str):
                imgs = [imgs]
            sample = {
                "dataset": "robopoint",
                "data_path": str(input_path.parent),
                "image": imgs,
                "video": rec.get("video") or [],
                "conversations": [
                    {"from": "system", "value": system},
                    {"from": "human", "value": user},
                    {"from": "gpt", "value": new_ans},
                ],
            }
            writer.write("grounding", sample)
            n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/data/msz/dataset"))
    parser.add_argument("--out-dir", type=Path, default=Path("/data/msz/point/data_expert"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-per-dataset", type=int, default=0)
    parser.add_argument("--allow-url-images", action="store_true")
    parser.add_argument("--skip-datasets", type=str, default="", help="Comma-separated dataset names to skip")
    parser.add_argument("--import-existing", type=Path, default=None, help="Path to existing jsonl file to import and reformat")
    args = parser.parse_args()

    skip = set(s.strip().lower() for s in args.skip_datasets.split(",") if s.strip())

    writer = Writer(args.out_dir)
    try:
        # Import existing jsonl (e.g., old RoboPoint data) and reformat
        if args.import_existing and args.import_existing.exists():
            log(f"[import] converting existing data from {args.import_existing}")
            n = convert_old_robopoint_format(args.import_existing, writer)
            log(f"[import] imported {n} samples")

        if "robopoint" not in skip:
            convert_robopoint(args.data_root, writer, args.limit_per_dataset)
        if "pixmopoints" not in skip and "pixmo" not in skip:
            convert_pixmo(args.data_root, writer, args.limit_per_dataset, args.allow_url_images)
        if "sharerobot" not in skip:
            convert_sharerobot_json(args.data_root, writer, args.limit_per_dataset)

        if "embspatial" not in skip:
            convert_embspatial(args.data_root, writer, args.limit_per_dataset, args.allow_url_images)
        if "robo2vlm-1" not in skip and "robo2vlm1" not in skip:
            convert_robo2vlm1(args.data_root, writer, args.limit_per_dataset)

        if "struct2d" not in skip and "struct2d-set" not in skip:
            if (args.data_root / "Struct2D-Set").exists() and any((args.data_root / "Struct2D-Set").iterdir()):
                convert_existing_or_keepalive(
                    args.data_root, writer, "struct2d-set", "Struct2D-Set",
                    ["struct2d_annotation.jsonl"], args.limit_per_dataset,
                )
        if "robovqa" not in skip:
            if (args.data_root / "robovqa").exists() and any(p.suffix in (".json", ".jsonl") for p in (args.data_root / "robovqa").rglob("*")):
                convert_existing_or_keepalive(
                    args.data_root, writer, "robovqa", "robovqa",
                    ["json_qwen/train_qwen.json", "train_qwen.json", "train.json"], args.limit_per_dataset,
                )
        if "phys100k" not in skip:
            if (args.data_root / "Phys100k").exists() and any((args.data_root / "Phys100k").iterdir()):
                convert_existing_or_keepalive(
                    args.data_root, writer, "phys100k", "Phys100k",
                    ["phys100k_ft_qwen.json", "train.json"], args.limit_per_dataset,
                )
    finally:
        writer.close()

    counts = merge_outputs(args.out_dir, args.seed)
    write_report(args.out_dir, counts, args)
    log(f"[done] total={sum(counts.values())} counts={dict(counts)}")
    log(f"[done] mix={args.out_dir / 'expert_grounding_mix.jsonl'}")


if __name__ == "__main__":
    main()

