#!/usr/bin/env python3
"""Build cleaned grounding data pools from public and in-house sources.

The output is intentionally source-sharded. Training mixes should sample from
these clean pools with explicit ratios instead of letting Visual Genome dominate.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
import re
import textwrap
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable
import xml.etree.ElementTree as ET

import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont


SYSTEM_BOX = (
    "You are a grounding assistant. Given an image and target information, "
    "return only the target bounding box using normalized 0-1000 coordinates."
)
SYSTEM_KEEPALIVE = "You are a helpful vision-language assistant."

SPATIAL_RELATION_KEYWORDS = (
    "on",
    "in",
    "inside",
    "under",
    "above",
    "over",
    "behind",
    "front",
    "left",
    "right",
    "near",
    "next to",
    "beside",
    "between",
    "around",
    "below",
    "beneath",
)


@dataclass(frozen=True)
class ImageRef:
    path: str
    archive: str | None
    member: str | None
    width: int
    height: int


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w", encoding="utf-8")
        self.rows = 0

    def write(self, row: dict[str, Any]) -> None:
        self.file.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.rows += 1

    def close(self) -> None:
        self.file.close()


def clean_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def clamp_box_xyxy(box: Iterable[float], width: float, height: float) -> tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = map(float, box)
    if not (math.isfinite(x1) and math.isfinite(y1) and math.isfinite(x2) and math.isfinite(y2)):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    if width <= 0 or height <= 0:
        return None
    if x1 >= width or y1 >= height or x2 <= 0 or y2 <= 0:
        return None
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def box_area_frac(box: tuple[float, float, float, float], width: float, height: float) -> float:
    return ((box[2] - box[0]) * (box[3] - box[1])) / max(width * height, 1.0)


def normalize_box_1000(box: tuple[float, float, float, float], width: float, height: float) -> list[list[int]]:
    x1, y1, x2, y2 = box
    vals = [
        round(x1 / width * 1000),
        round(y1 / height * 1000),
        round(x2 / width * 1000),
        round(y2 / height * 1000),
    ]
    vals = [max(0, min(1000, int(v))) for v in vals]
    if vals[2] <= vals[0]:
        vals[2] = min(1000, vals[0] + 1)
    if vals[3] <= vals[1]:
        vals[3] = min(1000, vals[1] + 1)
    return [[vals[0], vals[1]], [vals[2], vals[3]]]


def validate_box(
    raw_box: Iterable[float],
    width: float,
    height: float,
    counters: collections.Counter[str],
    min_area: float,
    max_area: float,
) -> tuple[tuple[float, float, float, float], list[list[int]], float] | None:
    clipped = clamp_box_xyxy(raw_box, width, height)
    if clipped is None:
        counters["drop_bad_box"] += 1
        return None
    area = box_area_frac(clipped, width, height)
    if area < min_area:
        counters["drop_tiny_box"] += 1
        return None
    if area > max_area:
        counters["drop_huge_box"] += 1
        return None
    return clipped, normalize_box_1000(clipped, width, height), area


def make_prompt(mode: str, payload: dict[str, Any]) -> str:
    if mode == "obj_relation":
        body = json.dumps(
            {
                "object_name": payload["object_name"],
                "relation": payload["relation"],
                "anchor_object": payload["anchor_object"],
            },
            ensure_ascii=False,
        )
        return (
            "<image>\nTarget object information:\n"
            f"{body}\n"
            "Return only the bounding box as <box>[[x1,y1],[x2,y2]]</box>."
        )
    if mode == "object_name":
        body = json.dumps({"object_name": payload["object_name"]}, ensure_ascii=False)
        return (
            "<image>\nTarget object information:\n"
            f"{body}\n"
            "Return only the bounding box as <box>[[x1,y1],[x2,y2]]</box>."
        )
    if mode == "region_description":
        body = json.dumps({"description": payload["description"]}, ensure_ascii=False)
        return (
            "<image>\nTarget region information:\n"
            f"{body}\n"
            "Return only the bounding box as <box>[[x1,y1],[x2,y2]]</box>."
        )
    raise ValueError(f"unknown prompt mode: {mode}")


def make_box_row(
    dataset: str,
    image_ref: ImageRef,
    prompt_mode: str,
    target: dict[str, Any],
    box_1000: list[list[int]],
    source_meta: dict[str, Any],
) -> dict[str, Any]:
    answer = f"<box>{json.dumps(box_1000, separators=(',', ':'))}</box>"
    return {
        "dataset": dataset,
        "image": [image_ref.path],
        "video": [],
        "target": {
            **target,
            "box": box_1000,
            "image_width": image_ref.width,
            "image_height": image_ref.height,
        },
        "conversations": [
            {"from": "system", "value": SYSTEM_BOX},
            {"from": "human", "value": make_prompt(prompt_mode, target)},
            {"from": "gpt", "value": answer},
        ],
        "metadata": {
            "task_type": "box_grounding",
            "prompt_mode": prompt_mode,
            "image_archive": image_ref.archive,
            "image_member": image_ref.member,
            **source_meta,
        },
    }


def expected_coco_path(root: Path, file_name: str) -> Path:
    split = "val2014" if "val2014" in file_name else "train2014"
    return root / "RefCOCO" / "COCO2014" / split / file_name


def coco_archive_member(file_name: str) -> tuple[str, str]:
    split = "val2014" if "val2014" in file_name else "train2014"
    return f"/data/msz/dataset/RefCOCO/COCO2014/{split}.zip", f"{split}/{file_name}"


def load_vg_image_maps(root: Path) -> tuple[dict[int, tuple[int, int]], dict[int, tuple[str, str, str]]]:
    with zipfile.ZipFile(root / "VisualGenome" / "image_data.json.zip") as z:
        with z.open(z.namelist()[0]) as f:
            image_data = json.load(f)
    dims: dict[int, tuple[int, int]] = {}
    for row in image_data:
        dims[int(row["image_id"])] = (int(row["width"]), int(row["height"]))

    member_map: dict[int, tuple[str, str, str]] = {}
    for zip_name in ("images.zip", "images2.zip"):
        archive = root / "VisualGenome" / zip_name
        with zipfile.ZipFile(archive) as z:
            for member in z.namelist():
                if not member.lower().endswith((".jpg", ".jpeg")):
                    continue
                try:
                    image_id = int(Path(member).stem)
                except ValueError:
                    continue
                expected_path = root / "VisualGenome" / member
                member_map[image_id] = (str(expected_path), str(archive), member)
    return dims, member_map


def build_refcoco(args: argparse.Namespace, writers: dict[str, JsonlWriter], summary: dict[str, Any]) -> None:
    root = Path(args.dataset_root)
    stats = collections.Counter()
    for ds_name in ("refcoco", "refcocoplus", "refcocog"):
        data_dir = root / "RefCOCO" / ds_name / "data"
        for parquet_path in sorted(data_dir.glob("*.parquet")):
            table = pq.read_table(
                parquet_path,
                columns=["bbox", "raw_image_info", "captions", "split", "ann_id", "ref_id", "image_id"],
            )
            for row in table.to_pylist():
                stats["input_refs"] += 1
                info = json.loads(row["raw_image_info"])
                width, height = float(info["width"]), float(info["height"])
                checked = validate_box(row["bbox"], width, height, stats, args.min_area, args.max_area)
                if checked is None:
                    continue
                _, box_1000, area = checked
                captions = [clean_text(x) for x in row.get("captions") or [] if clean_text(x)]
                if not captions:
                    stats["drop_empty_text"] += 1
                    continue
                file_name = info["file_name"]
                archive, member = coco_archive_member(file_name)
                image_ref = ImageRef(
                    path=str(expected_coco_path(root, file_name)),
                    archive=archive,
                    member=member,
                    width=int(width),
                    height=int(height),
                )
                for caption in captions:
                    stats["output_rows"] += 1
                    out = make_box_row(
                        dataset=f"{ds_name}_clean_v1",
                        image_ref=image_ref,
                        prompt_mode="region_description",
                        target={"description": caption},
                        box_1000=box_1000,
                        source_meta={
                            "source": ds_name,
                            "split": row.get("split"),
                            "ann_id": row.get("ann_id"),
                            "ref_id": row.get("ref_id"),
                            "image_id": row.get("image_id"),
                            "area_frac": round(area, 6),
                        },
                    )
                    writers["refcoco"].write(out)
    summary["refcoco"] = dict(stats)


def parse_flickr_sentence_entities(text: str) -> list[tuple[str, str]]:
    entities: list[tuple[str, str]] = []
    for match in re.finditer(r"\[/EN#([^/]+)/(?:[^\s\]]+)\s+([^\]]+)\]", text):
        phrase_id = match.group(1)
        phrase = clean_text(match.group(2))
        if phrase:
            entities.append((phrase_id, phrase))
    return entities


def build_flickr(args: argparse.Namespace, writers: dict[str, JsonlWriter], summary: dict[str, Any]) -> None:
    root = Path(args.dataset_root)
    ann_zip = root / "Flickr30kEntities" / "flickr30k_entities" / "annotations.zip"
    stats = collections.Counter()
    with zipfile.ZipFile(ann_zip) as z:
        xml_members = [n for n in z.namelist() if n.startswith("Annotations/") and n.endswith(".xml")]
        txt_members = {Path(n).stem: n for n in z.namelist() if n.startswith("Sentences/") and n.endswith(".txt")}
        for xml_member in xml_members:
            stats["input_images"] += 1
            image_id = Path(xml_member).stem
            txt_member = txt_members.get(image_id)
            if txt_member is None:
                stats["drop_missing_sentence_file"] += 1
                continue
            root_xml = ET.fromstring(z.read(xml_member))
            size = root_xml.find("size")
            if size is None:
                stats["drop_missing_size"] += 1
                continue
            width = int(float(size.findtext("width", "0")))
            height = int(float(size.findtext("height", "0")))
            boxes_by_id: dict[str, list[tuple[float, float, float, float]]] = collections.defaultdict(list)
            for obj in root_xml.findall("object"):
                name = clean_text(obj.findtext("name", ""))
                box = obj.find("bndbox")
                if not name or box is None:
                    continue
                raw_box = [
                    float(box.findtext("xmin", "nan")),
                    float(box.findtext("ymin", "nan")),
                    float(box.findtext("xmax", "nan")),
                    float(box.findtext("ymax", "nan")),
                ]
                clipped = clamp_box_xyxy(raw_box, width, height)
                if clipped is None:
                    stats["drop_bad_xml_box"] += 1
                    continue
                boxes_by_id[name].append(clipped)
            if not boxes_by_id:
                stats["drop_xml_without_box"] += 1
                continue
            sentence_text = z.read(txt_member).decode("utf-8", errors="ignore")
            image_ref = ImageRef(
                path=str(root / "Flickr30k" / "flickr30k-images" / f"{image_id}.jpg"),
                archive=str(root / "Flickr30k" / "flickr30k-images.zip"),
                member=f"flickr30k-images/{image_id}.jpg",
                width=width,
                height=height,
            )
            for line_no, line in enumerate(sentence_text.splitlines()):
                for phrase_id, phrase in parse_flickr_sentence_entities(line):
                    boxes = boxes_by_id.get(phrase_id)
                    if not boxes:
                        stats["drop_entity_without_box"] += 1
                        continue
                    # Entity chains may have multiple boxes; train on the union box so
                    # the target remains a single rectangle.
                    union = (
                        min(b[0] for b in boxes),
                        min(b[1] for b in boxes),
                        max(b[2] for b in boxes),
                        max(b[3] for b in boxes),
                    )
                    checked = validate_box(union, width, height, stats, args.min_area, args.max_area)
                    if checked is None:
                        continue
                    _, box_1000, area = checked
                    stats["output_rows"] += 1
                    out = make_box_row(
                        dataset="flickr30k_entities_clean_v1",
                        image_ref=image_ref,
                        prompt_mode="region_description",
                        target={"description": phrase},
                        box_1000=box_1000,
                        source_meta={
                            "source": "flickr30k_entities",
                            "image_id": image_id,
                            "phrase_id": phrase_id,
                            "sentence_line": line_no,
                            "box_count": len(boxes),
                            "area_frac": round(area, 6),
                        },
                    )
                    writers["flickr30k_entities"].write(out)
    summary["flickr30k_entities"] = dict(stats)


def load_zip_json(path: Path) -> Any:
    with zipfile.ZipFile(path) as z:
        with z.open(z.namelist()[0]) as f:
            return json.load(f)


def vg_object_name(obj: dict[str, Any]) -> str:
    names = obj.get("names") or []
    if names:
        return clean_text(names[0])
    return clean_text(obj.get("name"))


def vg_xywh_to_xyxy(obj: dict[str, Any], width_key: str = "w", height_key: str = "h") -> list[float]:
    return [
        float(obj.get("x") or 0),
        float(obj.get("y") or 0),
        float(obj.get("x") or 0) + float(obj.get(width_key) or 0),
        float(obj.get("y") or 0) + float(obj.get(height_key) or 0),
    ]


def image_ref_for_vg(root: Path, image_id: int, dims: dict[int, tuple[int, int]], members: dict[int, tuple[str, str, str]]) -> ImageRef | None:
    if image_id not in dims or image_id not in members:
        return None
    width, height = dims[image_id]
    path, archive, member = members[image_id]
    return ImageRef(path=path, archive=archive, member=member, width=width, height=height)


def build_vg_objects(
    args: argparse.Namespace,
    writers: dict[str, JsonlWriter],
    summary: dict[str, Any],
    dims: dict[int, tuple[int, int]],
    members: dict[int, tuple[str, str, str]],
) -> None:
    root = Path(args.dataset_root)
    stats = collections.Counter()
    data = load_zip_json(root / "VisualGenome" / "objects.json.zip")
    for image in data:
        image_id = int(image["image_id"])
        image_ref = image_ref_for_vg(root, image_id, dims, members)
        if image_ref is None:
            stats["drop_missing_image_ref"] += len(image.get("objects") or [])
            continue
        for obj in image.get("objects") or []:
            stats["input_objects"] += 1
            name = vg_object_name(obj)
            if not name:
                stats["drop_empty_text"] += 1
                continue
            checked = validate_box(
                vg_xywh_to_xyxy(obj),
                image_ref.width,
                image_ref.height,
                stats,
                args.min_area,
                args.max_area,
            )
            if checked is None:
                continue
            _, box_1000, area = checked
            stats["output_rows"] += 1
            out = make_box_row(
                dataset="visual_genome_object_clean_v1",
                image_ref=image_ref,
                prompt_mode="object_name",
                target={"object_name": name},
                box_1000=box_1000,
                source_meta={
                    "source": "visual_genome_objects",
                    "image_id": image_id,
                    "object_id": obj.get("object_id"),
                    "area_frac": round(area, 6),
                },
            )
            writers["visual_genome_object"].write(out)
    summary["visual_genome_object"] = dict(stats)


def build_vg_regions(
    args: argparse.Namespace,
    writers: dict[str, JsonlWriter],
    summary: dict[str, Any],
    dims: dict[int, tuple[int, int]],
    members: dict[int, tuple[str, str, str]],
) -> None:
    root = Path(args.dataset_root)
    stats = collections.Counter()
    data = load_zip_json(root / "VisualGenome" / "region_descriptions.json.zip")
    for image in data:
        regions = image.get("regions") or []
        image_id = int((regions[0].get("image_id") if regions else image.get("image_id")) or image.get("id") or 0)
        image_ref = image_ref_for_vg(root, image_id, dims, members)
        if image_ref is None:
            stats["drop_missing_image_ref"] += len(regions)
            continue
        for region in regions:
            stats["input_regions"] += 1
            phrase = clean_text(region.get("phrase"))
            if not phrase:
                stats["drop_empty_text"] += 1
                continue
            checked = validate_box(
                vg_xywh_to_xyxy(region, width_key="width", height_key="height"),
                image_ref.width,
                image_ref.height,
                stats,
                args.min_area,
                args.max_area,
            )
            if checked is None:
                continue
            _, box_1000, area = checked
            stats["output_rows"] += 1
            out = make_box_row(
                dataset="visual_genome_region_clean_v1",
                image_ref=image_ref,
                prompt_mode="region_description",
                target={"description": phrase},
                box_1000=box_1000,
                source_meta={
                    "source": "visual_genome_regions",
                    "image_id": image_id,
                    "region_id": region.get("region_id"),
                    "area_frac": round(area, 6),
                },
            )
            writers["visual_genome_region"].write(out)
    summary["visual_genome_region"] = dict(stats)


def is_spatial_relation(predicate: str) -> bool:
    pred = predicate.lower()
    return any(keyword in pred for keyword in SPATIAL_RELATION_KEYWORDS)


def build_vg_relationships(
    args: argparse.Namespace,
    writers: dict[str, JsonlWriter],
    summary: dict[str, Any],
    dims: dict[int, tuple[int, int]],
    members: dict[int, tuple[str, str, str]],
) -> None:
    root = Path(args.dataset_root)
    stats = collections.Counter()
    data = load_zip_json(root / "VisualGenome" / "relationships.json.zip")
    for image in data:
        image_id = int(image["image_id"])
        image_ref = image_ref_for_vg(root, image_id, dims, members)
        rels = image.get("relationships") or []
        if image_ref is None:
            stats["drop_missing_image_ref"] += len(rels)
            continue
        for rel in rels:
            stats["input_relationships"] += 1
            predicate = clean_text(rel.get("predicate")).lower()
            if not predicate:
                stats["drop_empty_relation"] += 1
                continue
            if not is_spatial_relation(predicate):
                stats["drop_non_spatial_relation"] += 1
                continue
            subject = rel.get("subject") or {}
            obj = rel.get("object") or {}
            subject_name = vg_object_name(subject)
            object_name = vg_object_name(obj)
            if not subject_name or not object_name:
                stats["drop_empty_text"] += 1
                continue
            checked = validate_box(
                vg_xywh_to_xyxy(subject),
                image_ref.width,
                image_ref.height,
                stats,
                args.min_area,
                args.max_area,
            )
            if checked is None:
                continue
            _, box_1000, area = checked
            stats["output_rows"] += 1
            out = make_box_row(
                dataset="visual_genome_relationship_clean_v1",
                image_ref=image_ref,
                prompt_mode="obj_relation",
                target={
                    "object_name": subject_name,
                    "relation": predicate,
                    "anchor_object": object_name,
                },
                box_1000=box_1000,
                source_meta={
                    "source": "visual_genome_relationships",
                    "image_id": image_id,
                    "relationship_id": rel.get("relationship_id"),
                    "area_frac": round(area, 6),
                },
            )
            writers["visual_genome_relationship"].write(out)
    summary["visual_genome_relationship"] = dict(stats)


def has_valid_turns(row: dict[str, Any]) -> bool:
    roles = [str(t.get("from", "")).lower() for t in row.get("conversations") or []]
    return any(r in {"human", "user"} for r in roles) and any(r in {"gpt", "assistant"} for r in roles)


def build_jsonl_passthrough(args: argparse.Namespace, writers: dict[str, JsonlWriter], summary: dict[str, Any]) -> None:
    sources = {
        "semantic_nav_box": Path(args.semantic_nav_box),
        "keepalive_vqa": Path(args.keepalive_vqa),
        "grounding_point": Path(args.grounding_point),
    }
    box_re = re.compile(r"<box>\s*\[\[(\d+),(\d+)\],\[(\d+),(\d+)\]\]\s*</box>")
    for name, path in sources.items():
        stats = collections.Counter()
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                stats["input_rows"] += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    stats["drop_bad_json"] += 1
                    continue
                if not has_valid_turns(row):
                    stats["drop_bad_turns"] += 1
                    continue
                if name == "semantic_nav_box":
                    answer = " ".join(
                        str(t.get("value", ""))
                        for t in row.get("conversations") or []
                        if str(t.get("from", "")).lower() in {"gpt", "assistant"}
                    ).replace(" ", "")
                    match = box_re.search(answer)
                    if not match:
                        stats["drop_bad_box_tag"] += 1
                        continue
                    x1, y1, x2, y2 = map(int, match.groups())
                    if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
                        stats["drop_bad_box_tag"] += 1
                        continue
                row = dict(row)
                row.setdefault("metadata", {})
                row["metadata"] = dict(row["metadata"])
                row["metadata"]["clean_source"] = name
                row["metadata"]["clean_version"] = "grounding_clean_v1"
                writers[name].write(row)
                stats["output_rows"] += 1
        summary[name] = dict(stats)


def read_image_for_preview(row: dict[str, Any]) -> Image.Image | None:
    meta = row.get("metadata") or {}
    archive = meta.get("image_archive")
    member = meta.get("image_member")
    if archive and member and Path(archive).exists():
        try:
            with zipfile.ZipFile(archive) as z:
                with z.open(member) as f:
                    return Image.open(BytesIO(f.read())).convert("RGB")
        except Exception:
            return None
    for item in row.get("image") or []:
        path = Path(str(item))
        if path.exists():
            try:
                return Image.open(path).convert("RGB")
            except Exception:
                return None
    return None


def extract_box(row: dict[str, Any]) -> list[list[int]] | None:
    target = row.get("target") or {}
    box = target.get("box")
    if isinstance(box, list) and len(box) == 2:
        return box
    answer = " ".join(
        str(t.get("value", ""))
        for t in row.get("conversations") or []
        if str(t.get("from", "")).lower() in {"gpt", "assistant"}
    )
    match = re.search(r"<box>\s*\[\[(\d+),(\d+)\],\[(\d+),(\d+)\]\]\s*</box>", answer.replace(" ", ""))
    if not match:
        return None
    x1, y1, x2, y2 = map(int, match.groups())
    return [[x1, y1], [x2, y2]]


def preview_label(row: dict[str, Any]) -> str:
    target = row.get("target") or {}
    if target.get("description"):
        return clean_text(target["description"])[:80]
    if target.get("object_name") and target.get("relation"):
        return f"{target['object_name']} {target['relation']} {target.get('anchor_object', '')}"[:80]
    if target.get("object_name"):
        return clean_text(target["object_name"])[:80]
    return clean_text((row.get("dataset") or ""))[:80]


def collect_preview_rows(out_dir: Path, rng: random.Random, per_source: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(out_dir.glob("*_clean_v1.jsonl")):
        if path.name in {"keepalive_vqa_clean_v1.jsonl", "grounding_point_clean_v1.jsonl"}:
            continue
        sample: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f, start=1):
                if len(sample) < per_source:
                    sample.append(json.loads(line))
                else:
                    j = rng.randrange(idx)
                    if j < per_source:
                        sample[j] = json.loads(line)
        rows.extend(sample)
    rng.shuffle(rows)
    return rows


def make_preview(out_dir: Path, preview_path: Path, rng: random.Random, per_source: int = 4) -> dict[str, Any]:
    rows = collect_preview_rows(out_dir, rng, per_source)
    tiles: list[Image.Image] = []
    skipped = 0
    font = ImageFont.load_default()
    for row in rows:
        box = extract_box(row)
        img = read_image_for_preview(row)
        if box is None or img is None:
            skipped += 1
            continue
        img.thumbnail((260, 200))
        canvas = Image.new("RGB", (280, 250), "white")
        xoff = (280 - img.width) // 2
        canvas.paste(img, (xoff, 8))
        draw = ImageDraw.Draw(canvas)
        sx = img.width / 1000.0
        sy = img.height / 1000.0
        x1, y1 = box[0]
        x2, y2 = box[1]
        rect = [xoff + x1 * sx, 8 + y1 * sy, xoff + x2 * sx, 8 + y2 * sy]
        draw.rectangle(rect, outline=(0, 114, 255), width=3)
        title = str(row.get("dataset", ""))[:38]
        label = textwrap.wrap(preview_label(row), width=42)[:2]
        draw.text((8, 212), title, fill=(0, 0, 0), font=font)
        draw.text((8, 226), "\n".join(label), fill=(40, 40, 40), font=font)
        tiles.append(canvas)
    if not tiles:
        raise RuntimeError("no previewable rows")
    cols = 4
    rows_n = math.ceil(len(tiles) / cols)
    sheet = Image.new("RGB", (cols * 280, rows_n * 250), "white")
    for idx, tile in enumerate(tiles):
        sheet.paste(tile, ((idx % cols) * 280, (idx // cols) * 250))
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(preview_path)
    return {"preview_rows": len(tiles), "preview_skipped": skipped, "preview_path": str(preview_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="/data/msz/dataset")
    parser.add_argument("--out-dir", default="/data/msz/point/data_grounding_clean_v1")
    parser.add_argument("--semantic-nav-box", default="/data/msz/opd_project/data/semantic_nav_box_v1/semantic_nav_box_grounding_full_object_ref_v1_high_quality.jsonl")
    parser.add_argument("--keepalive-vqa", default="/data/msz/point/data_expert/keepalive_vqa.jsonl")
    parser.add_argument("--grounding-point", default="/data/msz/point/data_expert/grounding_point.jsonl")
    parser.add_argument("--min-area", type=float, default=0.0005)
    parser.add_argument("--max-area", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--skip-vg-object", action="store_true")
    parser.add_argument("--skip-vg-region", action="store_true")
    parser.add_argument("--skip-vg-relationship", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writers = {
        "refcoco": JsonlWriter(out_dir / "refcoco_clean_v1.jsonl"),
        "flickr30k_entities": JsonlWriter(out_dir / "flickr30k_entities_clean_v1.jsonl"),
        "visual_genome_object": JsonlWriter(out_dir / "visual_genome_object_clean_v1.jsonl"),
        "visual_genome_region": JsonlWriter(out_dir / "visual_genome_region_clean_v1.jsonl"),
        "visual_genome_relationship": JsonlWriter(out_dir / "visual_genome_relationship_clean_v1.jsonl"),
        "semantic_nav_box": JsonlWriter(out_dir / "semantic_nav_box_clean_v1.jsonl"),
        "keepalive_vqa": JsonlWriter(out_dir / "keepalive_vqa_clean_v1.jsonl"),
        "grounding_point": JsonlWriter(out_dir / "grounding_point_clean_v1.jsonl"),
    }
    summary: dict[str, Any] = {
        "clean_version": "grounding_clean_v1",
        "rules": {
            "skipped_sources": ["PhraseCut", "Talk2Car image version", "old region synthetic datasets", "RoboRefIt abnormal-box rows/source"],
            "min_area_frac": args.min_area,
            "max_area_frac": args.max_area,
            "bbox": "clip partially out-of-image boxes, drop bad/tiny/huge boxes, normalize to 0-1000",
            "visual_genome_relationships": "keep spatial predicates only and train subject-as-target relation-to-object",
        },
    }
    try:
        build_refcoco(args, writers, summary)
        build_flickr(args, writers, summary)
        dims, members = load_vg_image_maps(Path(args.dataset_root))
        if not args.skip_vg_object:
            build_vg_objects(args, writers, summary, dims, members)
        if not args.skip_vg_region:
            build_vg_regions(args, writers, summary, dims, members)
        if not args.skip_vg_relationship:
            build_vg_relationships(args, writers, summary, dims, members)
        build_jsonl_passthrough(args, writers, summary)
    finally:
        for writer in writers.values():
            writer.close()

    summary["files"] = {
        name: {"path": str(writer.path), "rows": writer.rows, "bytes": writer.path.stat().st_size}
        for name, writer in writers.items()
    }
    preview_info = make_preview(out_dir, out_dir / "clean_grounding_preview.png", random.Random(args.seed))
    summary["preview"] = preview_info
    summary_path = out_dir / "clean_grounding_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({"summary": str(summary_path), **summary["files"], "preview": preview_info}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
