"""Taxonomy-driven side-by-side Qwen grounding for stable answer zones.

This script fixes two problems in the earlier grounding pass:
1. It no longer assumes every target field lives on page 2.
2. It saves stable answer zones, not overly tight text boxes, so later
   extraction has enough context for short and long handwritten values.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import fitz
import numpy as np

from src.canonical_extractor import CanonicalExtractor
from src.form300_templates import FORM300_PAGE_TEMPLATES, PAGE_TYPE_TO_PAGE_NUM, bbox_to_pixels
from src.surgical_field_taxonomy import FocusFieldSpec, get_focus_field, iter_focus_fields


HOT_GROUND_URL = "http://127.0.0.1:8765/ground_field"
HOT_EXTRACT_URL = "http://127.0.0.1:8765/extract_crop"
PAGE_NUM_TO_TYPE = {v: k for k, v in PAGE_TYPE_TO_PAGE_NUM.items()}


def post_json(url: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode("utf-8"))


def render_page(pdf_path: Path, page_num: int, dpi: int = 200) -> np.ndarray:
    doc = fitz.open(str(pdf_path))
    page = doc.load_page(page_num - 1)
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    doc.close()
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
    if pix.n == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def expand_box(
    box: Tuple[int, int, int, int],
    width: int,
    height: int,
    x_pad: int,
    y_pad_top: int,
    y_pad_bottom: int,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        max(0, x1 - x_pad),
        max(0, y1 - y_pad_top),
        min(width, x2 + x_pad),
        min(height, y2 + y_pad_bottom),
    )


def clip_box(
    box: Tuple[int, int, int, int],
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return x1, y1, x2, y2


def candidate_from_local(
    local_bbox: List[int],
    search_box: Tuple[int, int, int, int],
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    sx1, sy1, _, _ = search_box
    lx1, ly1, lx2, ly2 = [int(v) for v in local_bbox]
    return clip_box((sx1 + lx1, sy1 + ly1, sx1 + lx2, sy1 + ly2), width, height)


def build_stable_answer_zone(
    candidate_boxes: List[Tuple[int, int, int, int]],
    legacy_box: Tuple[int, int, int, int],
    search_box: Tuple[int, int, int, int],
    width: int,
    height: int,
    min_size: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    lx1, ly1, lx2, ly2 = legacy_box
    legacy_w = max(1, lx2 - lx1)
    legacy_h = max(1, ly2 - ly1)
    legacy_cx = (lx1 + lx2) / 2.0
    legacy_cy = (ly1 + ly2) / 2.0
    sx1, sy1, sx2, sy2 = search_box
    min_w, min_h = min_size

    if candidate_boxes:
        arr = np.asarray(candidate_boxes, dtype=np.float32)
        widths = arr[:, 2] - arr[:, 0]
        heights = arr[:, 3] - arr[:, 1]
        centers_x = (arr[:, 0] + arr[:, 2]) / 2.0
        centers_y = (arr[:, 1] + arr[:, 3]) / 2.0
        union_x1 = float(np.min(arr[:, 0]))
        union_y1 = float(np.min(arr[:, 1]))
        union_x2 = float(np.max(arr[:, 2]))
        union_y2 = float(np.max(arr[:, 3]))
        union_w = max(1.0, union_x2 - union_x1)
        union_h = max(1.0, union_y2 - union_y1)

        cx = 0.70 * float(np.median(centers_x)) + 0.30 * legacy_cx
        cy = 0.75 * float(np.median(centers_y)) + 0.25 * legacy_cy
        target_w = int(
            round(
                max(
                    float(min_w),
                    float(union_w + 48.0),
                    float(np.percentile(widths, 75) + 36.0),
                    legacy_w * 0.92,
                )
            )
        )
        target_h = int(
            round(
                max(
                    float(min_h),
                    float(union_h + 28.0),
                    float(np.percentile(heights, 75) + 24.0),
                    legacy_h * 1.18,
                )
            )
        )
    else:
        cx = legacy_cx
        cy = legacy_cy
        target_w = max(min_w, int(round(legacy_w * 1.05)))
        target_h = max(min_h, int(round(legacy_h * 1.20)))

    row_top = max(sy1, ly1 - int(round(legacy_h * 0.45)))
    row_bottom = min(sy2, ly2 + int(round(legacy_h * 0.55)))
    band_h = max(1, row_bottom - row_top)
    target_h = min(target_h, band_h)
    target_w = min(target_w, max(1, sx2 - sx1))

    nx1 = int(round(cx - target_w / 2.0))
    ny1 = int(round(cy - target_h / 2.0))
    nx1 = max(sx1, min(sx2 - target_w, nx1))
    ny1 = max(row_top, min(row_bottom - target_h, ny1))
    nx2 = nx1 + target_w
    ny2 = ny1 + target_h
    return clip_box((nx1, ny1, nx2, ny2), width, height)


def draw_overlay(image: np.ndarray, items: List[Tuple[str, Tuple[int, int, int, int]]]) -> np.ndarray:
    overlay = image.copy()
    for label, (x1, y1, x2, y2) in items:
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 180, 255), 2)
        cv2.putText(
            overlay,
            label,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 140, 255),
            1,
            cv2.LINE_AA,
        )
    return overlay


def default_prompt_variants(spec: FocusFieldSpec) -> List[str]:
    field_label = spec.public_id.replace("_", " ")
    return [
        f"Find the handwritten value area for {field_label} only.",
        f"Locate only the answer region for {field_label}. Exclude nearby labels and neighboring fields.",
    ]


def resolve_specs(fields_arg: str, bundle: str, include_unsupported: bool) -> List[FocusFieldSpec]:
    requested: List[FocusFieldSpec] = []
    seen = set()

    if bundle == "surgical25":
        for spec in iter_focus_fields(supported_only=not include_unsupported):
            if spec.public_id not in seen:
                requested.append(spec)
                seen.add(spec.public_id)

    if fields_arg.strip():
        for token in [part.strip() for part in fields_arg.split(",") if part.strip()]:
            spec = get_focus_field(token)
            if spec is None:
                raise ValueError(f"Unknown focus field: {token}")
            if not include_unsupported and not spec.supported:
                continue
            if spec.public_id not in seen:
                requested.append(spec)
                seen.add(spec.public_id)

    if not requested:
        for token in ("first_name", "last_name", "date_of_birth", "marital_status", "address_line1"):
            spec = get_focus_field(token)
            if spec is not None and spec.public_id not in seen:
                requested.append(spec)
                seen.add(spec.public_id)

    return requested


def extract_value_if_needed(
    crop_bgr: np.ndarray,
    field_name: str,
    tmpdir_path: Path,
) -> dict:
    crop_path = tmpdir_path / f"{field_name}_final.png"
    cv2.imwrite(str(crop_path), crop_bgr)
    return post_json(HOT_EXTRACT_URL, {"image_path": str(crop_path), "field_name": field_name})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path")
    parser.add_argument("--page", type=int, default=0, help="Optional: limit grounding to a single page number")
    parser.add_argument("--fields", default="")
    parser.add_argument("--bundle", choices=["", "surgical25"], default="")
    parser.add_argument("--include-unsupported", action="store_true")
    parser.add_argument("--output", default="")
    parser.add_argument("--verify", action="store_true", help="Also ask Qwen to read the final crop")
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    specs = resolve_specs(args.fields, args.bundle, args.include_unsupported)
    if args.page:
        specs = [spec for spec in specs if spec.actual_page_num == args.page]
    if not specs:
        raise SystemExit("No focus fields selected after filtering")

    if args.output:
        output_path = Path(args.output)
    elif args.bundle == "surgical25":
        suffix = "all" if not args.page else f"page{args.page}"
        output_path = pdf_path.with_name(f"{pdf_path.stem}_surgical25_qwen_zones_{suffix}.json")
    else:
        output_path = pdf_path.with_name(f"{pdf_path.stem}_qwen_side_by_side.json")

    extractor = CanonicalExtractor(audit_dir=str(Path("data") / "qwen_side_by_side_audit" / pdf_path.stem))
    page_specs: Dict[int, List[FocusFieldSpec]] = defaultdict(list)
    output: Dict[str, dict] = {}

    for spec in specs:
        if not spec.supported or spec.template_field is None or spec.actual_page_num is None:
            output[spec.public_id] = {
                "public_id": spec.public_id,
                "template_field": spec.template_field,
                "actual_page_num": spec.actual_page_num,
                "route_group": spec.route_group,
                "supported": False,
                "status": "unsupported",
                "notes": spec.notes,
            }
            continue
        page_specs[spec.actual_page_num].append(spec)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="qwen_side_by_side_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        for page_num in sorted(page_specs):
            page_type = PAGE_NUM_TO_TYPE.get(page_num)
            if page_type not in FORM300_PAGE_TEMPLATES:
                for spec in page_specs[page_num]:
                    output[spec.public_id] = {
                        "public_id": spec.public_id,
                        "template_field": spec.template_field,
                        "actual_page_num": page_num,
                        "route_group": spec.route_group,
                        "supported": False,
                        "status": "missing_template",
                        "notes": f"No page template found for page {page_num}",
                    }
                continue

            print(f"Page {page_num}: aligning and grounding {len(page_specs[page_num])} field(s)")
            image = render_page(pdf_path, page_num)
            registration = extractor.align_page(image, page_num)
            aligned = registration.aligned_bgr
            template = FORM300_PAGE_TEMPLATES[page_type]
            height, width = aligned.shape[:2]
            overlay_items: List[Tuple[str, Tuple[int, int, int, int]]] = []

            for spec in page_specs[page_num]:
                field = next((item for item in template.fields if item.name == spec.template_field), None)
                if field is None:
                    output[spec.public_id] = {
                        "public_id": spec.public_id,
                        "template_field": spec.template_field,
                        "actual_page_num": page_num,
                        "route_group": spec.route_group,
                        "supported": False,
                        "status": "missing_field_template",
                        "notes": f"Template field '{spec.template_field}' not found on page {page_num}",
                    }
                    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
                    print(f"{spec.public_id}: template field missing on page {page_num}")
                    continue

                legacy_box = bbox_to_pixels(field.bbox_norm, width, height, pad_norm=0.0)
                search_box = expand_box(legacy_box, width, height, *spec.search_pad)
                sx1, sy1, sx2, sy2 = search_box
                search_crop = aligned[sy1:sy2, sx1:sx2]
                search_path = tmpdir_path / f"page{page_num}_{spec.public_id}_search.png"
                cv2.imwrite(str(search_path), search_crop)

                prompt_variants = list(spec.prompt_variants) or default_prompt_variants(spec)
                candidate_boxes: List[Tuple[int, int, int, int]] = []
                ground_details = []

                for idx, prompt_text in enumerate(prompt_variants, start=1):
                    try:
                        grounded_res = post_json(
                            HOT_GROUND_URL,
                            {
                                "image_path": str(search_path),
                                "field_name": spec.public_id,
                                "field_description": prompt_text,
                                "renderer": field.renderer,
                                "page_num": page_num,
                            },
                        )
                        local_bbox = grounded_res.get("bbox") or [0, 0, search_crop.shape[1], search_crop.shape[0]]
                        candidate_box = candidate_from_local(local_bbox, search_box, width, height)
                        candidate_boxes.append(candidate_box)
                        ground_details.append(
                            {
                                "variant": idx,
                                "prompt": prompt_text,
                                "bbox": list(candidate_box),
                                "ground_confidence": grounded_res.get("confidence", 0.0),
                            }
                        )
                        print(f"{spec.public_id} variant {idx}: bbox={candidate_box}")
                    except Exception as exc:
                        ground_details.append(
                            {
                                "variant": idx,
                                "prompt": prompt_text,
                                "error": str(exc),
                            }
                        )
                        print(f"{spec.public_id} variant {idx}: failed -> {exc}")

                final_box = build_stable_answer_zone(
                    candidate_boxes=candidate_boxes,
                    legacy_box=legacy_box,
                    search_box=search_box,
                    width=width,
                    height=height,
                    min_size=spec.min_size,
                )
                overlay_items.append((spec.public_id, final_box))

                result = {
                    "public_id": spec.public_id,
                    "template_field": spec.template_field,
                    "actual_page_num": page_num,
                    "route_group": spec.route_group,
                    "supported": True,
                    "search_box": list(search_box),
                    "legacy_box": list(legacy_box),
                    "bbox": list(final_box),
                    "grounding_variants": ground_details,
                    "status": "ok" if candidate_boxes else "fallback_legacy_zone",
                    "notes": spec.notes,
                }

                if args.verify:
                    x1, y1, x2, y2 = final_box
                    final_crop = aligned[y1:y2, x1:x2]
                    verified = extract_value_if_needed(final_crop, spec.public_id, tmpdir_path)
                    result["value"] = verified.get("value")
                    result["value_confidence"] = verified.get("confidence", 0.0)
                    print(
                        f"{spec.public_id}: {result.get('value')} "
                        f"(bbox={result['bbox']}, conf={result.get('value_confidence', 0.0)})"
                    )
                else:
                    print(f"{spec.public_id}: final stable zone={result['bbox']}")

                output[spec.public_id] = result
                output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

            overlay = draw_overlay(aligned, overlay_items)
            audit_dir = Path("data") / "qwen_side_by_side_audit" / pdf_path.stem
            audit_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(audit_dir / f"page_{page_num:02d}_stable_zone_overlay.png"), overlay)

    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
