"""Audit registered template boxes against a local PDF or page image.

This is a geometry QA tool:
1. Render pages from the input PDF.
2. Register each target page to the local blank sample reference.
3. Draw template boxes on the aligned page and on the source page.
4. Save a JSON metrics report for each audited page.
"""

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np

from src.config import TARGET_DPI
from src.fixed_form_registration import FixedFormRegistrar
from src.form300_templates import FORM300_PAGE_TEMPLATES, PAGE_TYPE_TO_PAGE_NUM, bbox_to_pixels


PAGE_NUM_TO_TYPE = {v: k for k, v in PAGE_TYPE_TO_PAGE_NUM.items()}


def parse_pages(spec: str) -> List[int]:
    if not spec:
        return []
    if "," in spec:
        return [int(part.strip()) for part in spec.split(",") if part.strip()]
    if "-" in spec:
        start, end = spec.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(spec)]


def render_pdf_pages(pdf_path: Path, page_nums: Iterable[int]) -> List[tuple[int, np.ndarray]]:
    import fitz

    doc = fitz.open(str(pdf_path))
    rendered = []
    for page_num in page_nums:
        if page_num < 1 or page_num > doc.page_count:
            continue
        page = doc.load_page(page_num - 1)
        pix = page.get_pixmap(dpi=TARGET_DPI, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        rendered.append((page_num, img))
    doc.close()
    return rendered


def draw_source_overlay(
    source_bgr: np.ndarray,
    template,
    registrar: FixedFormRegistrar,
    registration,
) -> np.ndarray:
    overlay = source_bgr.copy()
    ref_w, ref_h = registration.reference_size
    for field in template.fields:
        ref_bbox = bbox_to_pixels(field.bbox_norm, ref_w, ref_h, pad_norm=0.0)
        src_bbox = registrar.aligned_bbox_to_source_bbox(ref_bbox, registration, source_bgr.shape)
        x1, y1, x2, y2 = src_bbox
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            overlay,
            field.name,
            (x1, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    return overlay


def audit_pdf(pdf_path: Path, pages: List[int], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    registrar = FixedFormRegistrar()

    metrics = {}
    for page_num, source_bgr in render_pdf_pages(pdf_path, pages):
        page_type = PAGE_NUM_TO_TYPE.get(page_num)
        if page_type not in FORM300_PAGE_TEMPLATES:
            continue

        template = FORM300_PAGE_TEMPLATES[page_type]
        registration = registrar.register(source_bgr, page_num)

        aligned_path = output_dir / f"{pdf_path.stem}_page{page_num:02d}_aligned.png"
        aligned_overlay_path = output_dir / f"{pdf_path.stem}_page{page_num:02d}_aligned_overlay.png"
        source_overlay_path = output_dir / f"{pdf_path.stem}_page{page_num:02d}_source_overlay.png"

        aligned_overlay = registrar.draw_template_overlay(registration.aligned_bgr, template)
        source_overlay = draw_source_overlay(source_bgr, template, registrar, registration)

        cv2.imwrite(str(aligned_path), registration.aligned_bgr)
        cv2.imwrite(str(aligned_overlay_path), aligned_overlay)
        cv2.imwrite(str(source_overlay_path), source_overlay)

        metrics[page_num] = {
            "page_type": page_type,
            "registration_success": registration.success,
            "registration_method": registration.method,
            "match_count": registration.match_count,
            "inlier_count": registration.inlier_count,
            "inlier_ratio": registration.inlier_ratio,
            "mean_reprojection_error": registration.mean_reprojection_error,
            "ecc_correlation": registration.ecc_correlation,
            "structure_overlap": registration.structure_overlap,
            "failure_reason": registration.failure_reason,
            "aligned_image": str(aligned_path),
            "aligned_overlay": str(aligned_overlay_path),
            "source_overlay": str(source_overlay_path),
        }

    report_path = output_dir / f"{pdf_path.stem}_registration_audit.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"Audit report written to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Audit fixed-template registration overlays")
    parser.add_argument("pdf_path", help="Path to input PDF")
    parser.add_argument(
        "--pages",
        default="2,3,5,6,7,10,13,14,15,16,18,28",
        help="Pages to audit: '2,3,5' or '2-7'",
    )
    parser.add_argument(
        "--output-dir",
        default="data/registration_audit",
        help="Directory for aligned pages, overlays, and JSON report",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    output_dir = Path(args.output_dir)
    pages = parse_pages(args.pages)
    audit_pdf(pdf_path, pages, output_dir)


if __name__ == "__main__":
    main()
