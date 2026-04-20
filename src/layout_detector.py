"""Layout Detector using PaddleOCR
=================================
Dual-language (Hindi+English) OCR with page-type classification support.
"""

import logging
import numpy as np
from typing import Dict, List, Optional
from PIL import Image

logger = logging.getLogger(__name__)


class LayoutDetector:
    """Dual-language PaddleOCR for layout analysis and text detection."""

    def __init__(self, lang: str = "hi"):
        self._engine_hi = None
        self._engine_en = None
        self._primary_lang = lang

    def _get_engine(self, lang: str = None):
        """Lazy-load PaddleOCR engine for specified language."""
        lang = lang or self._primary_lang

        if lang == "hi":
            if self._engine_hi is None:
                try:
                    from paddleocr import PaddleOCR
                    self._engine_hi = PaddleOCR(lang="hi", use_textline_orientation=True)
                    logger.info("PaddleOCR Hindi engine initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize PaddleOCR Hindi: {e}")
                    self._engine_hi = None
            return self._engine_hi
        else:
            if self._engine_en is None:
                try:
                    from paddleocr import PaddleOCR
                    self._engine_en = PaddleOCR(lang="en", use_textline_orientation=True)
                    logger.info("PaddleOCR English engine initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize PaddleOCR English: {e}")
                    self._engine_en = None
            return self._engine_en

    def detect_text_regions(self, image: np.ndarray, lang: str = None) -> List[Dict]:
        """Detect all text regions in an image using both Hindi and English engines."""
        lang = lang or self._primary_lang
        engine = self._get_engine(lang)
        if engine is None:
            return []

        try:
            result = engine.ocr(image)
            regions = []

            if result and len(result) > 0:
                res0 = result[0]
                texts = res0.get("rec_texts") if hasattr(res0, "get") else None
                scores = res0.get("rec_scores") if hasattr(res0, "get") else None
                polys = res0.get("rec_polys") if hasattr(res0, "get") else None

                if texts is not None and scores is not None and polys is not None:
                    if isinstance(texts, list) and isinstance(scores, list) and isinstance(polys, list):
                        for text, score, poly in zip(texts, scores, polys):
                            if not isinstance(text, str):
                                text = str(text)
                            try:
                                confidence = float(score)
                            except (ValueError, TypeError):
                                confidence = 0.0

                            if isinstance(poly, (list, tuple, np.ndarray)):
                                pts = np.asarray(poly)
                                if pts.ndim == 2 and pts.shape[0] >= 4 and pts.shape[1] == 2:
                                    xs = pts[:, 0]
                                    ys = pts[:, 1]
                                    x1, x2 = int(np.min(xs)), int(np.max(xs))
                                    y1, y2 = int(np.min(ys)), int(np.max(ys))
                                    rect = [x1, y1, x2, y2]
                                    quad = [[int(xs[i]), int(ys[i])] for i in range(4)]
                                    regions.append({
                                        "bbox": rect,
                                        "quad": quad,
                                        "text": text,
                                        "confidence": confidence,
                                        "lang": lang,
                                    })

                if not regions and isinstance(result, list) and result and isinstance(result[0], list):
                    for line in result[0]:
                        if isinstance(line, (list, tuple)) and len(line) >= 2:
                            bbox = line[0]
                            second = line[1]
                            if isinstance(second, (list, tuple)) and len(second) == 2:
                                text, confidence = second
                            elif isinstance(second, str):
                                text = second
                                confidence = 1.0
                            else:
                                continue
                            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                                xs = [p[0] for p in bbox]
                                ys = [p[1] for p in bbox]
                            else:
                                continue
                            rect = [min(xs), min(ys), max(xs), max(ys)]
                            regions.append({
                                "bbox": rect,
                                "quad": bbox,
                                "text": text,
                                "confidence": float(confidence),
                                "lang": lang,
                            })
            return regions
        except Exception as e:
            logger.error(f"Text region detection failed: {e}")
            return []

    def detect_text_regions_bilingual(self, image: np.ndarray) -> List[Dict]:
        """Detect text using both Hindi and English engines, merge results."""
        hi_regions = self.detect_text_regions(image, lang="hi")
        en_regions = self.detect_text_regions(image, lang="en")

        seen_texts = set()
        merged = []

        for r in hi_regions + en_regions:
            key = (r["text"].strip().lower(), tuple(r["bbox"]))
            if key not in seen_texts:
                seen_texts.add(key)
                merged.append(r)

        return merged

    def detect_fields(self, image: np.ndarray,
                      field_coords: Optional[Dict[str, List[int]]] = None) -> Dict[str, np.ndarray]:
        """Detect and crop field regions from form image."""
        crops = {}

        if field_coords:
            for field_name, bbox in field_coords.items():
                x1, y1, x2, y2 = [int(c) for c in bbox]
                crop = image[y1:y2, x1:x2]
                if crop.size > 0:
                    crops[field_name] = crop
        else:
            regions = self.detect_text_regions(image)
            for i, region in enumerate(regions):
                x1, y1, x2, y2 = [int(c) for c in region["bbox"]]
                crop = image[y1:y2, x1:x2]
                if crop.size > 0:
                    field_name = "detected_field_{:03d}".format(i)
                    crops[field_name] = crop

        return crops

    def get_paddle_text(self, image: np.ndarray, lang: str = None) -> str:
        """Quick OCR text extraction using PaddleOCR."""
        regions = self.detect_text_regions(image, lang=lang)
        texts = [r["text"] for r in regions]
        return " ".join(texts)
