"""
Layout Detector using PaddleOCR
===============================
Detects text regions, table structures, and field bounding boxes.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from PIL import Image
from pathlib import Path
import cv2

logger = logging.getLogger(__name__)


class LayoutDetector:
    """Uses PaddleOCR PP-Structure for layout analysis and field detection."""

    def __init__(self):
        self._engine = None

    def _get_engine(self):
        """Lazy-load PaddleOCR engine."""
        if self._engine is None:
            try:
                from paddleocr import PaddleOCR
                self._engine = PaddleOCR(
                    lang='en',
                    use_angle_cls=True
                )
                logger.info("PaddleOCR engine initialized")
            except Exception as e:
                logger.error(f"Failed to initialize PaddleOCR: {e}")
                self._engine = None
        return self._engine

    def detect_text_regions(self, image: np.ndarray) -> List[Dict]:
        """Detect all text regions in an image."""
        engine = self._get_engine()
        if engine is None:
            return []

        try:
            result = engine.ocr(image)
            # result is a list with one OCRResult object (dict-like)
            regions = []

            if result and len(result) > 0:
                res0 = result[0]
                # The OCRResult object has dict-like .get method for accessing fields
                texts = res0.get('rec_texts')
                scores = res0.get('rec_scores')
                polys = res0.get('rec_polys')
                if texts is not None and scores is not None and polys is not None:
                    # Ensure they are lists
                    if isinstance(texts, list) and isinstance(scores, list) and isinstance(polys, list):
                        for text, score, poly in zip(texts, scores, polys):
                            # Ensure we have string and float
                            if not isinstance(text, str):
                                text = str(text)
                            try:
                                confidence = float(score)
                            except (ValueError, TypeError):
                                confidence = 0.0
                            # Convert polygon to bounding box
                            if isinstance(poly, (list, tuple, np.ndarray)):
                                # poly might be numpy array; convert to list of points
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
                                    })
            # Fallback: if the above didn't produce regions, try the old format parsing
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
                            if len(bbox) == 4:
                                xs = [bbox[0], bbox[2], bbox[0], bbox[2]]
                                ys = [bbox[1], bbox[1], bbox[3], bbox[3]]
                            else:
                                continue
                        rect = [min(xs), min(ys), max(xs), max(ys)]
                        regions.append({
                            "bbox": rect,
                            "quad": bbox,
                            "text": text,
                            "confidence": float(confidence),
                        })
            return regions
        except Exception as e:
            logger.error(f"Text region detection failed: {e}")
            return []

    def detect_fields(self, image: np.ndarray,
                      field_coords: Optional[Dict[str, List[int]]] = None) -> Dict[str, np.ndarray]:
        """
        Detect and crop field regions from form image.

        Args:
            image: Full form page image
            field_coords: Optional pre-defined field coordinates {name: [x1,y1,x2,y2]}

        Returns:
            Dict of {field_name: cropped_image}
        """
        crops = {}

        if field_coords:
            # Use pre-defined coordinates
            for field_name, bbox in field_coords.items():
                x1, y1, x2, y2 = [int(c) for c in bbox]
                crop = image[y1:y2, x1:x2]
                if crop.size > 0:
                    crops[field_name] = crop
        else:
            # Auto-detect using PaddleOCR
            regions = self.detect_text_regions(image)
            for i, region in enumerate(regions):
                x1, y1, x2, y2 = [int(c) for c in region["bbox"]]
                crop = image[y1:y2, x1:x2]
                if crop.size > 0:
                    field_name = f"detected_field_{i:03d}"
                    crops[field_name] = crop

        return crops

    def detect_photo_signature(self, image: np.ndarray) -> Dict[str, Optional[np.ndarray]]:
        """Detect and extract photo and signature regions from form."""
        h, w = image.shape[:2]

        # Heuristic: photo usually in top-right area, signature near bottom
        # This is form-specific and should be calibrated on actual Form 300
        results = {
            "photo": None,
            "signature": None,
        }

        # Photo detection: look for rectangular regions with face-like content
        # in top-right quadrant
        photo_region = image[0:h//3, 2*w//3:w]
        if photo_region.size > 0:
            results["photo"] = photo_region

        # Signature detection: look in bottom quarter
        sig_region = image[3*h//4:h, w//4:3*w//4]
        if sig_region.size > 0:
            results["signature"] = sig_region

        return results

    def get_paddle_text(self, image: np.ndarray) -> str:
        """Quick OCR text extraction using PaddleOCR (for printed text)."""
        regions = self.detect_text_regions(image)
        texts = [r["text"] for r in regions]
        return " ".join(texts)