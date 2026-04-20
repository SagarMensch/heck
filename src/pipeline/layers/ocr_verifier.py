"""
Layer 3: PaddleOCR Bilingual Verifier + Bbox Localizer
=======================================================
Cross-verifies VLM extractions against PaddleOCR output.
Locates pixel-level bboxes for click-to-source navigation.

This layer DOES NOT do primary extraction — it verifies and locates.
"""

import os
import logging
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from difflib import SequenceMatcher

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "1"

from src.pipeline.models.schemas import BBox, ExtractedField

logger = logging.getLogger(__name__)


class OCRVerifier:

    def __init__(self):
        self.ocr_en = None
        self.ocr_hi = None
        self._loaded = False
        self._region_cache: Dict[int, List[Dict]] = {}

    def load(self):
        if self._loaded:
            return
        from paddleocr import PaddleOCR
        self.ocr_en = PaddleOCR(
            use_textline_orientation=True, lang='en',
            text_det_thresh=0.3, text_det_box_thresh=0.5,
        )
        self.ocr_hi = PaddleOCR(
            use_textline_orientation=True, lang='hi',
            text_det_thresh=0.3, text_det_box_thresh=0.5,
        )
        self._loaded = True
        logger.info("PaddleOCR v5 (EN+HI) loaded for verification")

    def get_regions(self, page_num: int, cv_img: np.ndarray) -> List[Dict]:
        if page_num in self._region_cache:
            return self._region_cache[page_num]

        self.load()
        regions = []

        for ocr_engine, lang in [(self.ocr_en, 'en'), (self.ocr_hi, 'hi')]:
            for res in ocr_engine.predict(cv_img):
                texts = res.get('rec_texts', [])
                scores = res.get('rec_scores', [])
                polys = res.get('dt_polys', [])
                for t, s, p in zip(texts, scores, polys):
                    pts = np.asarray(p)
                    if pts.ndim == 2 and pts.shape[0] >= 2:
                        xs, ys = pts[:, 0], pts[:, 1]
                        x1, y1 = int(np.min(xs)), int(np.min(ys))
                        x2, y2 = int(np.max(xs)), int(np.max(ys))
                        regions.append({
                            "text": str(t).strip(),
                            "confidence": float(s),
                            "bbox": [x1, y1, x2, y2],
                            "lang": lang,
                        })

        regions = self._deduplicate(regions)
        self._region_cache[page_num] = regions
        logger.info(f"  PaddleOCR: {len(regions)} unique regions for page {page_num}")
        return regions

    def verify_field(self, page_num: int, cv_img: np.ndarray, field_name: str, vlm_value: str) -> Tuple[float, bool, Optional[BBox]]:
        if not vlm_value or not vlm_value.strip():
            return 0.0, False, None

        regions = self.get_regions(page_num, cv_img)
        vlm_lower = vlm_value.lower().strip()

        best_score = 0.0
        best_bbox = None

        for r in regions:
            rtext = r["text"].lower().strip()
            sim = self._string_similarity(vlm_lower, rtext)
            sim *= r["confidence"]
            if sim > best_score:
                best_score = sim
                b = r["bbox"]
                best_bbox = BBox(x1=b[0], y1=b[1], x2=b[2], y2=b[3],
                                 page_width=cv_img.shape[1], page_height=cv_img.shape[0])

        verified = best_score >= 0.65
        confidence = min(best_score + 0.3, 0.95) if verified else 0.5

        if best_bbox is None and vlm_value:
            confidence = 0.6
            logger.debug(f"  OCR could not locate '{vlm_value}' for {field_name} — trusting VLM")

        return confidence, verified, best_bbox

    def find_label_bbox(self, page_num: int, cv_img: np.ndarray, field_name: str, label_patterns: List[str]) -> Optional[BBox]:
        regions = self.get_regions(page_num, cv_img)
        for pattern in label_patterns:
            for r in regions:
                if pattern.lower() in r["text"].lower():
                    b = r["bbox"]
                    return BBox(x1=b[0], y1=b[1], x2=b[2], y2=b[3],
                                page_width=cv_img.shape[1], page_height=cv_img.shape[0])
        return None

    def _string_similarity(self, s1: str, s2: str) -> float:
        if s1 == s2:
            return 1.0
        if s1 in s2 or s2 in s1:
            return 0.8 * (len(min(s1, s2, key=len)) / max(len(s1, s2), 1))
        return SequenceMatcher(None, s1, s2).ratio()

    def _deduplicate(self, regions: List[Dict]) -> List[Dict]:
        seen = set()
        out = []
        for r in regions:
            key = (r["text"][:30].lower(), r["bbox"][1] // 20, r["bbox"][0] // 20)
            if key not in seen:
                seen.add(key)
                out.append(r)
        return out

    def clear_cache(self):
        self._region_cache.clear()
