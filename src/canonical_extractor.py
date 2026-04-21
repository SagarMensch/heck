"""Reference-aligned canonical extractor for LIC Form 300.

This module keeps the old name so legacy scripts still import it, but the
implementation is now production-oriented:
1. Register a page to a local blank reference page using FixedFormRegistrar.
2. Crop fields from the aligned page using the shared normalized templates.
3. Optionally save aligned overlays for audit.

The old contour-only warp and hand-written canonical coordinates were too
fragile for real scanned forms.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from src.fixed_form_registration import FixedFormRegistrar, RegistrationResult
from src.form300_templates import (
    FORM300_PAGE_TEMPLATES,
    PAGE_TYPE_TO_PAGE_NUM,
    bbox_to_pixels,
)
from src.qwen_bbox_grounder import QwenReferenceBBoxStore


CANONICAL_TEMPLATE_SIZE = (1000, 1400)
CANONICAL_TEMPLATE_PAGE_NUM = 2


def _to_legacy_canonical(bbox_norm: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
    width, height = CANONICAL_TEMPLATE_SIZE
    x1, y1, x2, y2 = bbox_norm
    return (
        int(round(x1 * width)),
        int(round(y1 * height)),
        int(round(x2 * width)),
        int(round(y2 * height)),
    )


CANONICAL_FIELDS = {
    field.name: _to_legacy_canonical(field.bbox_norm)
    for field in FORM300_PAGE_TEMPLATES["proposer_details"].fields
}


PAGE_NUM_TO_TYPE = {v: k for k, v in PAGE_TYPE_TO_PAGE_NUM.items()}


class CanonicalExtractor:
    def __init__(self, audit_dir: Optional[str] = None):
        self.template_w, self.template_h = CANONICAL_TEMPLATE_SIZE
        self.registrar = FixedFormRegistrar()
        self.qwen_bbox_store = QwenReferenceBBoxStore()
        self.audit_dir = Path(audit_dir) if audit_dir else None
        if self.audit_dir is not None:
            self.audit_dir.mkdir(parents=True, exist_ok=True)

    def warp_to_canonical(self, image: np.ndarray, page_num: int = CANONICAL_TEMPLATE_PAGE_NUM) -> np.ndarray:
        """Align a page image to the locally cached reference page geometry."""
        registration = self.registrar.register(image, page_num)
        return registration.aligned_bgr

    def align_page(self, image: np.ndarray, page_num: int) -> RegistrationResult:
        return self.registrar.register(image, page_num)

    def extract_field_crop(self, aligned_image: np.ndarray, field_name: str, page_num: int = CANONICAL_TEMPLATE_PAGE_NUM) -> np.ndarray:
        """Extract a single field crop from an aligned page image."""
        page_type = PAGE_NUM_TO_TYPE.get(page_num)
        if page_type not in FORM300_PAGE_TEMPLATES:
            raise ValueError(f"No template for page {page_num}")

        template = FORM300_PAGE_TEMPLATES[page_type]
        field = next((item for item in template.fields if item.name == field_name), None)
        if field is None:
            raise ValueError(f"Unknown field '{field_name}' for page {page_num}")

        h_img, w_img = aligned_image.shape[:2]
        x1, y1, x2, y2 = self._resolve_field_bbox(field, page_num, w_img, h_img)
        return aligned_image[y1:y2, x1:x2]

    def extract_template_fields(self, image: np.ndarray, page_num: int) -> Dict[str, np.ndarray]:
        """Register the page and return all field crops from the shared template."""
        registration = self.align_page(image, page_num)
        page_type = PAGE_NUM_TO_TYPE.get(page_num)
        if page_type not in FORM300_PAGE_TEMPLATES:
            return {}

        template = FORM300_PAGE_TEMPLATES[page_type]
        aligned = registration.aligned_bgr
        cleaned = self.registrar.subtract_reference_form(aligned, page_num)
        h_img, w_img = aligned.shape[:2]
        crops: Dict[str, np.ndarray] = {}

        for field in template.fields:
            x1, y1, x2, y2 = self._resolve_field_bbox(field, page_num, w_img, h_img)
            source = cleaned if field.renderer == "text" else aligned
            crop = source[y1:y2, x1:x2]
            if crop.size != 0 and field.renderer == "text" and np.mean(crop) > 252:
                crop = aligned[y1:y2, x1:x2]
            if crop.size != 0:
                crops[field.name] = crop

        if self.audit_dir is not None:
            overlay = self.registrar.draw_template_overlay(aligned, template)
            cv2.imwrite(str(self.audit_dir / f"page_{page_num:02d}_aligned.png"), aligned)
            cv2.imwrite(str(self.audit_dir / f"page_{page_num:02d}_cleaned.png"), cleaned)
            cv2.imwrite(str(self.audit_dir / f"page_{page_num:02d}_overlay.png"), overlay)

        return crops

    def process_pdf_page(self, image_path: str, page_num: int = CANONICAL_TEMPLATE_PAGE_NUM):
        """Legacy helper kept for compatibility with older scripts."""
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(image_path)
        aligned = self.warp_to_canonical(img, page_num=page_num)
        crops = self.extract_template_fields(img, page_num=page_num)
        return crops, aligned

    def _resolve_field_bbox(self, field, page_num: int, width: int, height: int) -> Tuple[int, int, int, int]:
        override = self.qwen_bbox_store.get_field_bbox(
            page_num,
            field.name,
            target_size=(width, height),
        )
        if override is not None:
            return override
        return bbox_to_pixels(field.bbox_norm, width, height, pad_norm=0.0)


def run_canonical_extraction(image_path: str, output_dir: str, page_num: int = CANONICAL_TEMPLATE_PAGE_NUM):
    extractor = CanonicalExtractor(audit_dir=output_dir)
    print(f"Processing {image_path}...")
    crops, aligned = extractor.process_pdf_page(image_path, page_num=page_num)

    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, "aligned_full.png"), aligned)

    for name, crop in crops.items():
        path = os.path.join(output_dir, f"{name}.png")
        cv2.imwrite(path, crop)
        print(f"  Saved crop: {name} ({crop.shape})")

    print(f"All crops saved to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run canonical extraction with fixed-form registration")
    parser.add_argument("image_path", help="Input page image path")
    parser.add_argument("--page-num", type=int, default=CANONICAL_TEMPLATE_PAGE_NUM)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    out_dir = args.output_dir or args.image_path.replace(".png", "_crops")
    run_canonical_extraction(args.image_path, out_dir, page_num=args.page_num)
