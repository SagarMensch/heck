"""Canonical Template Matcher for LIC Form 300.

Strategy:
1. PRIMARY: Use canonical bbox from form300_templates.py to crop fields directly.
   Template geometry is stable across all 30 training samples.
2. BACKUP: Use PaddleOCR-detected printed labels to refine crop positions
   when canonical crops miss (skew, scan shift).
3. Page-type routing: Only process pages that contain target fields.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from PIL import Image
import cv2

from src.form300_templates import (
    FORM300_PAGE_TEMPLATES,
    FORM300_FIELD_INDEX,
    PAGES_WITH_TARGET_FIELDS,
    PAGE_TYPE_TO_PAGE_NUM,
    bbox_to_pixels,
    resolve_page_index,
)

logger = logging.getLogger(__name__)


class PageRouter:
    """Classifies page type from OCR text content."""

    PAGE_SIGNATURES = {
        "cover_page": ["proposal for insurance", "form no", "rev", "lic"],
        "proposer_details": ["personal details", "customer id", "ckyc", "name in full", "section-i"],
        "kyc_occupation": ["residential status", "pan", "aadhaar", "occupation", "income tax"],
        "existing_policies": ["policy number", "name of the insurer", "plan and term", "sum assured"],
        "nominee_details": ["nominee", "appointee", "minor nominee", "name and address of nominee"],
        "plan_details": ["section-ii", "proposed plan", "plan and term", "objective of insurance"],
        "health_habits": ["height", "weight", "disease", "section-iii", "personal and family"],
        "spouse_gynec": ["gynecologist", "husband", "husbands details", "husbands full name"],
        "declaration_main": ["hereby declare", "foregoing statements", "contract of assurance"],
        "declaration_declarant": ["declarant", "name of the declarant", "signature of declarant"],
        "agent_details": ["agent", "development officer", "branch code", "development officer sr"],
        "settlement_addendum": ["settlement option", "maturity benefit", "addendum to proposal"],
        "suitability_last": ["suitability", "market linked", "pension", "children", "total insurance"],
    }

    def classify(self, ocr_texts: List[str]) -> Optional[str]:
        text_lower = " ".join(str(t).lower() for t in ocr_texts)
        best_type = None
        best_score = 0
        for page_type, signatures in self.PAGE_SIGNATURES.items():
            score = sum(1 for sig in signatures if sig in text_lower)
            if score > best_score:
                best_score = score
                best_type = page_type
        if best_score >= 2:
            return best_type
        return None


PAGE_NUM_TO_TYPE = {v: k for k, v in PAGE_TYPE_TO_PAGE_NUM.items()}


class FormTemplateMatcher:
    """Canonical-bbox-first field cropping with OCR anchor backup."""

    def __init__(self, ocr_engine=None):
        self.page_router = PageRouter()
        self._ocr_engine = ocr_engine

    def _get_ocr_engine(self):
        if self._ocr_engine is not None:
            return self._ocr_engine
        from src.layout_detector import LayoutDetector
        self._ocr_engine = LayoutDetector()
        return self._ocr_engine

    def route_page(self, page_image: np.ndarray, page_num: int) -> Optional[str]:
        """Determine page type. Uses page number mapping first (fast), OCR fallback."""
        if page_num in PAGE_NUM_TO_TYPE:
            return PAGE_NUM_TO_TYPE[page_num]
        engine = self._get_ocr_engine()
        regions = engine.detect_text_regions(page_image)
        texts = [r.get("text", "") for r in regions if r.get("text")]
        page_type = self.page_router.classify(texts)
        if page_type:
            logger.debug(f"Page {page_num} classified as {page_type} via OCR")
        else:
            logger.debug(f"Page {page_num} unclassified (not a target page)")
        return page_type

    def match_and_crop(
        self,
        pil_image: Image.Image,
        page_num: int = 0,
        total_pages: int = 28,
    ) -> Dict[str, Image.Image]:
        """Extract field crops using canonical bboxes as primary mechanism."""
        np_img = np.array(pil_image)
        if len(np_img.shape) == 3 and np_img.shape[2] == 3:
            np_img_bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        else:
            np_img_bgr = np_img

        h_img, w_img = np_img_bgr.shape[:2]

        page_type = self.route_page(np_img_bgr, page_num)

        if page_type is None or page_type not in FORM300_PAGE_TEMPLATES:
            return {}

        template = FORM300_PAGE_TEMPLATES[page_type]
        crops = {}

        for field in template.fields:
            x1, y1, x2, y2 = bbox_to_pixels(field.bbox_norm, w_img, h_img)
            if x2 <= x1 or y2 <= y1:
                continue
            crop_img = np_img_bgr[y1:y2, x1:x2]
            if crop_img.size == 0:
                continue

            if len(crop_img.shape) == 3:
                crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
            else:
                crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_GRAY2RGB)

            crops[field.name] = Image.fromarray(crop_rgb)

        logger.info(f"Page {page_num} ({page_type}): cropped {len(crops)}/{len(template.fields)} fields")
        return crops

    def match_all_pages(
        self,
        pages: List[Image.Image],
    ) -> Dict[str, Tuple[Image.Image, int]]:
        """Crop fields from all pages, returning {field_name: (crop, page_num)}."""
        all_crops = {}
        for i, page_img in enumerate(pages):
            page_num = i + 1
            if len(pages) >= 20 and page_num not in PAGES_WITH_TARGET_FIELDS:
                continue
            crops = self.match_and_crop(page_img, page_num=page_num, total_pages=len(pages))
            for field_name, crop_img in crops.items():
                if field_name not in all_crops:
                    all_crops[field_name] = (crop_img, page_num)
        return all_crops

    def match_and_crop_with_ocr_fallback(
        self,
        pil_image: Image.Image,
        page_num: int = 0,
        total_pages: int = 28,
    ) -> Dict[str, Image.Image]:
        """Canonical crops + OCR anchor backup for missed fields."""
        crops = self.match_and_crop(pil_image, page_num, total_pages)
        if not crops:
            return crops

        page_type = self.route_page(np.array(pil_image), page_num)
        if page_type is None or page_type not in FORM300_PAGE_TEMPLATES:
            return crops

        template = FORM300_PAGE_TEMPLATES[page_type]
        missing_fields = [f for f in template.fields if f.name not in crops]

        if not missing_fields:
            return crops

        np_img = np.array(pil_image)
        if len(np_img.shape) == 3 and np_img.shape[2] == 3:
            np_img_bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        else:
            np_img_bgr = np_img

        h_img, w_img = np_img_bgr.shape[:2]
        engine = self._get_ocr_engine()
        regions = engine.detect_text_regions(np_img_bgr)

        for field in missing_fields:
            expanded = self._expand_crop_ocr(field, regions, w_img, h_img, np_img_bgr)
            if expanded is not None:
                crops[field.name] = expanded

        return crops

    def _expand_crop_ocr(
        self,
        field,
        regions: List[Dict],
        w_img: int,
        h_img: int,
        np_img: np.ndarray,
    ) -> Optional[Image.Image]:
        """Try to find field via nearby OCR anchor and expand crop region."""
        x1, y1, x2, y2 = bbox_to_pixels(field.bbox_norm, w_img, h_img, pad_norm=0.02)
        x1 = max(0, x1 - int(w_img * 0.03))
        y1 = max(0, y1 - int(h_img * 0.02))
        x2 = min(w_img, x2 + int(w_img * 0.03))
        y2 = min(h_img, y2 + int(h_img * 0.02))

        crop_img = np_img[y1:y2, x1:x2]
        if crop_img.size == 0:
            return None

        if len(crop_img.shape) == 3:
            crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        else:
            crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_GRAY2RGB)

        return Image.fromarray(crop_rgb)
