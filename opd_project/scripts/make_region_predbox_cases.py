#!/usr/bin/env python3
"""Create a 10-case preview for RegionRef predbox-label data."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    root = Path("/data/msz/opd_project/data/semantic_nav_region_box_v1/predbox_label_v1")
    base = [json.loads(line) for line in (root / "semantic_nav_region_predbox_label_v1_base_accepted.jsonl").open(encoding="utf-8")]
    buckets: dict[str, list[dict]] = {}
    for row in base:
        buckets.setdefault(str(row.get("source_solution_reason")), []).append(row)
    for rows in buckets.values():
        rows.sort(key=lambda r: (r.get("relation", ""), r.get("prompt_id", "")))

    selected = []
    used = set()
    for reason, count in [("desc_to_box_geometry", 4), ("no_pred_box", 3), ("accepted", 3)]:
        rel_seen = set()
        for row in buckets.get(reason, []):
            if row["prompt_id"] in used or row.get("relation") in rel_seen:
                continue
            selected.append(row)
            used.add(row["prompt_id"])
            rel_seen.add(row.get("relation"))
            if sum(1 for x in selected if str(x.get("source_solution_reason")) == reason) >= count:
                break
    for row in base:
        if len(selected) >= 10:
            break
        if row["prompt_id"] not in used:
            selected.append(row)
            used.add(row["prompt_id"])
    selected = selected[:10]

    prompts = {}
    for line in (root / "semantic_nav_region_predbox_label_v1_high_quality.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        md = row.get("metadata", {})
        if md.get("prompt_mode") == "semantic_json" and md.get("original_prompt_id") in used:
            prompts[md["original_prompt_id"]] = row

    cases = []
    for idx, row in enumerate(selected, 1):
        prompt_row = prompts.get(row["prompt_id"])
        box = row["final_box"]
        cases.append(
            {
                "case": idx,
                "prompt_id": row["prompt_id"],
                "image": row["image"],
                "source_solution_reason": row.get("source_solution_reason"),
                "relation": row.get("relation"),
                "region_category": row.get("region_category"),
                "description": row.get("unique_description"),
                "seed_box_old_point_label": row.get("seed_box"),
                "final_box_new_label": box,
                "expected_output": f"<box>[[{box[0][0]},{box[0][1]}],[{box[1][0]},{box[1][1]}]]</box>",
                "model_input": prompt_row["conversations"][1]["value"] if prompt_row else None,
                "seed_to_final_metrics": row.get("seed_to_final_metrics"),
            }
        )
    (root / "semantic_nav_region_predbox_label_v1_cases10.json").write_text(
        json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    cols = 2
    cell_w = 520
    cell_h = 360
    header_h = 48
    sheet = Image.new("RGB", (cols * cell_w, ((len(selected) + cols - 1) // cols) * cell_h + header_h), (248, 250, 252))
    draw = ImageDraw.Draw(sheet, "RGBA")
    try:
        title_font = ImageFont.truetype("DejaVuSans.ttf", 18)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except Exception:
        title_font = None
        small_font = None
    draw.text(
        (12, 12),
        "10 RegionRef predbox-label cases: gray=old point seed, blue=new description->box label",
        fill=(15, 23, 42),
        font=title_font,
    )

    for idx, row in enumerate(selected):
        x0 = (idx % cols) * cell_w + 8
        y0 = (idx // cols) * cell_h + header_h
        image = Image.open(row["image"]).convert("RGB")
        w, h = image.size
        scale = min((cell_w - 16) / w, 250 / h)
        thumb = image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
        sheet.paste(thumb, (x0, y0))
        box_draw = ImageDraw.Draw(sheet, "RGBA")

        def draw_box(box: list[list[int]], color: tuple[int, int, int], width: int) -> None:
            bx1 = x0 + int(box[0][0] / 1000 * thumb.size[0])
            by1 = y0 + int(box[0][1] / 1000 * thumb.size[1])
            bx2 = x0 + int(box[1][0] / 1000 * thumb.size[0])
            by2 = y0 + int(box[1][1] / 1000 * thumb.size[1])
            box_draw.rectangle([bx1, by1, bx2, by2], outline=(*color, 255), width=width)

        draw_box(row["seed_box"], (148, 163, 184), 2)
        draw_box(row["final_box"], (0, 112, 255), 4)
        text_y = y0 + thumb.size[1] + 6
        lines = [
            f"#{idx + 1} {row.get('source_solution_reason')} rel={row.get('relation')} cat={row.get('region_category')}"
        ]
        lines.extend(textwrap.wrap(row.get("unique_description", ""), width=66)[:3])
        metrics = row["seed_to_final_metrics"]
        lines.append(
            f"old={row['seed_box']} new={row['final_box']} pts={metrics['point_recall']} area={metrics['area_ratio_final_over_seed']}"
        )
        for line in lines:
            box_draw.text((x0, text_y), line, fill=(15, 23, 42), font=small_font)
            text_y += 16

    sheet.save(root / "semantic_nav_region_predbox_label_v1_cases10_preview.png")
    print(
        json.dumps(
            {
                "cases_json": str(root / "semantic_nav_region_predbox_label_v1_cases10.json"),
                "preview": str(root / "semantic_nav_region_predbox_label_v1_cases10_preview.png"),
                "cases": len(cases),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
