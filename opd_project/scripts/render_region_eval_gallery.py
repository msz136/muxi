#!/usr/bin/env python3
"""Render the full RegionRef eval set as an auditable image/HTML gallery."""

from __future__ import annotations

import html
import json
import math
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


EVAL_BASE = Path("/data/msz/opd_project/data/semantic_nav_region_box_v1/eval/semantic_nav_region_solution_a_bidir_eval_v1_base_manifest.jsonl")
OUT_DIR = Path("/data/msz/opd_project/data/semantic_nav_region_box_v1/eval_gallery_full")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def box_text(box: list[list[int]]) -> str:
    return f"<box>[[{box[0][0]},{box[0][1]}],[{box[1][0]},{box[1][1]}]]</box>"


def draw_box(draw: ImageDraw.ImageDraw, box: list[list[int]], image_size: tuple[int, int], color: tuple[int, int, int], width: int) -> None:
    w, h = image_size
    x1 = int(box[0][0] / 1000 * w)
    y1 = int(box[0][1] / 1000 * h)
    x2 = int(box[1][0] / 1000 * w)
    y2 = int(box[1][1] / 1000 * h)
    draw.rectangle([x1, y1, x2, y2], outline=(*color, 255), width=width)


def load_fonts() -> tuple[ImageFont.ImageFont | None, ImageFont.ImageFont | None, ImageFont.ImageFont | None]:
    try:
        return (
            ImageFont.truetype("DejaVuSans.ttf", 18),
            ImageFont.truetype("DejaVuSans.ttf", 13),
            ImageFont.truetype("DejaVuSans.ttf", 11),
        )
    except Exception:
        return None, None, None


def render_card(record: dict, idx: int, out_path: Path) -> dict:
    title_font, body_font, small_font = load_fonts()
    image = Image.open(record["image"]).convert("RGB")
    max_w, max_h = 720, 520
    w, h = image.size
    scale = min(max_w / w, max_h / h, 1.0)
    view = image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
    overlay = view.convert("RGBA")
    draw = ImageDraw.Draw(overlay, "RGBA")

    seed_box = record["box"]
    pred_box = record.get("description_to_box_validation", {}).get("pred_box")
    draw_box(draw, seed_box, view.size, (148, 163, 184), 3)
    if pred_box:
        draw_box(draw, pred_box, view.size, (0, 112, 255), 5)

    desc = record["unique_description"]
    expected_seed = box_text(seed_box)
    expected_pred = box_text(pred_box) if pred_box else "N/A"
    metrics = record.get("description_to_box_validation", {})

    text_lines = [
        f"#{idx:02d}  relation={record.get('relation')}  category={record.get('region_category')}  points={record.get('point_count')}",
        f"description: {desc}",
        f"anchor: {record.get('anchor_phrase')} | relation_to_anchor: {record.get('relation_to_anchor')}",
        f"seed/eval label (gray): {expected_seed}",
        f"description->box label (blue): {expected_pred}",
        "metrics: "
        + f"target_cov={metrics.get('target_coverage')} point_recall={metrics.get('point_recall')} "
        + f"area_ratio={metrics.get('area_ratio_pred_over_target')} center_dist={metrics.get('center_distance')}",
        f"prompt_id: {record.get('prompt_id')}",
    ]
    wrapped: list[str] = []
    for line in text_lines:
        wrapped.extend(textwrap.wrap(line, width=110) or [""])

    line_h = 18
    pad = 14
    legend_h = 34
    text_h = pad * 2 + len(wrapped) * line_h
    canvas = Image.new("RGB", (max(view.size[0], 760), view.size[1] + legend_h + text_h), (248, 250, 252))
    canvas.paste(overlay.convert("RGB"), (0, 0))
    cdraw = ImageDraw.Draw(canvas, "RGBA")
    y = view.size[1] + 8
    cdraw.rectangle([14, y + 5, 34, y + 21], outline=(148, 163, 184, 255), width=3)
    cdraw.text((42, y + 4), "gray = original point-derived seed/eval box", fill=(51, 65, 85), font=small_font)
    cdraw.rectangle([330, y + 5, 350, y + 21], outline=(0, 112, 255, 255), width=4)
    cdraw.text((358, y + 4), "blue = description->box prediction", fill=(51, 65, 85), font=small_font)
    y += legend_h
    cdraw.rectangle([0, y, canvas.size[0], canvas.size[1]], fill=(255, 255, 255, 255))
    y += pad
    for i, line in enumerate(wrapped):
        font = title_font if i == 0 else body_font
        fill = (15, 23, 42) if i == 0 else (30, 41, 59)
        cdraw.text((pad, y), line, fill=fill, font=font)
        y += line_h if i else 24

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return {
        "idx": idx,
        "prompt_id": record.get("prompt_id"),
        "image": record.get("image"),
        "description": desc,
        "relation": record.get("relation"),
        "region_category": record.get("region_category"),
        "seed_eval_box": seed_box,
        "description_to_box": pred_box,
        "expected_seed_output": expected_seed,
        "expected_predbox_output": expected_pred,
        "card": str(out_path),
    }


def render_contact_sheets(cards: list[Path], out_dir: Path) -> list[Path]:
    sheets = []
    cols = 2
    rows_per_sheet = 4
    thumb_w = 520
    thumb_h = 430
    per_sheet = cols * rows_per_sheet
    for sheet_idx in range(math.ceil(len(cards) / per_sheet)):
        subset = cards[sheet_idx * per_sheet : (sheet_idx + 1) * per_sheet]
        sheet = Image.new("RGB", (cols * thumb_w, rows_per_sheet * thumb_h + 44), (241, 245, 249))
        draw = ImageDraw.Draw(sheet)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 18)
        except Exception:
            font = None
        draw.text((12, 12), f"RegionRef Eval Gallery Sheet {sheet_idx + 1}", fill=(15, 23, 42), font=font)
        for i, card_path in enumerate(subset):
            img = Image.open(card_path).convert("RGB")
            scale = min((thumb_w - 14) / img.size[0], (thumb_h - 14) / img.size[1])
            thumb = img.resize((max(1, int(img.size[0] * scale)), max(1, int(img.size[1] * scale))), Image.Resampling.LANCZOS)
            x = (i % cols) * thumb_w + 7
            y = (i // cols) * thumb_h + 44
            sheet.paste(thumb, (x, y))
        path = out_dir / f"semantic_nav_region_eval_gallery_sheet_{sheet_idx + 1:02d}.png"
        sheet.save(path)
        sheets.append(path)
    return sheets


def write_html(index: list[dict], sheets: list[Path], out_dir: Path) -> None:
    cards_html = []
    for item in index:
        rel_card = Path(item["card"]).name
        cards_html.append(
            f"""
            <article class="card">
              <img src="cards/{html.escape(rel_card)}" alt="case {item['idx']}">
              <pre>{html.escape(json.dumps({k: v for k, v in item.items() if k != 'card'}, ensure_ascii=False, indent=2))}</pre>
            </article>
            """
        )
    sheet_links = "\n".join(f'<li><a href="{html.escape(p.name)}">{html.escape(p.name)}</a></li>' for p in sheets)
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>RegionRef Eval Full Gallery</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #f8fafc; color: #0f172a; }}
    h1 {{ font-size: 24px; }}
    .note {{ color: #475569; line-height: 1.45; }}
    .card {{ background: white; border: 1px solid #cbd5e1; border-radius: 6px; padding: 12px; margin: 18px 0; }}
    img {{ max-width: 100%; height: auto; display: block; }}
    pre {{ white-space: pre-wrap; background: #f1f5f9; padding: 12px; border-radius: 4px; font-size: 12px; }}
  </style>
</head>
<body>
  <h1>RegionRef Eval Full Gallery</h1>
  <p class="note">Gray box is the original point-derived seed/eval label. Blue box is the description-&gt;box prediction, which is the candidate label under the new predbox-label design.</p>
  <h2>Contact Sheets</h2>
  <ul>{sheet_links}</ul>
  <h2>Cases</h2>
  {''.join(cards_html)}
</body>
</html>
"""
    (out_dir / "index.html").write_text(html_text, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cards_dir = OUT_DIR / "cards"
    cards_dir.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(EVAL_BASE)
    index = []
    card_paths = []
    for idx, record in enumerate(records, 1):
        card_path = cards_dir / f"case_{idx:02d}.png"
        index.append(render_card(record, idx, card_path))
        card_paths.append(card_path)
    sheets = render_contact_sheets(card_paths, OUT_DIR)
    write_json = {
        "source": str(EVAL_BASE),
        "base_cases": len(records),
        "gallery_dir": str(OUT_DIR),
        "index_html": str(OUT_DIR / "index.html"),
        "contact_sheets": [str(p) for p in sheets],
        "cards_dir": str(cards_dir),
        "legend": {
            "gray": "original point-derived seed/eval box",
            "blue": "description->box prediction/candidate predbox label",
        },
        "cases": index,
    }
    (OUT_DIR / "semantic_nav_region_eval_gallery_index.json").write_text(
        json.dumps(write_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_html(index, sheets, OUT_DIR)
    print(json.dumps({k: write_json[k] for k in ["source", "base_cases", "gallery_dir", "index_html", "contact_sheets"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
