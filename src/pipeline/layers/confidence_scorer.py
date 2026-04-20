"""
Layer 5: Confidence Scorer + Click-to-Source Localizer
=======================================================
Final scoring combining VLM confidence, OCR verification,
validation status, and KB corrections.

Produces:
  - Per-field confidence score (0.0-1.0)
  - Review category (auto_accepted / needs_review / low_confidence / rejected)
  - Click-to-source bbox (pixel coordinates on original image)
  - Overlay image with colored bounding boxes + labels
  - JSON output with normalized bbox coordinates for UI integration
"""

import os
import logging
import numpy as np
import cv2
import json
from typing import List, Optional, Dict, Any
from PIL import Image

from src.pipeline.models.schemas import (
    ExtractedField, BBox, ValidationStatus, ReviewCategory, PageImage
)

logger = logging.getLogger(__name__)


class ConfidenceScorer:

    FIELD_COLORS = {
        "Proposer_Full_Name": (0, 255, 0),
        "Proposer_First_Name": (0, 255, 0),
        "Proposer_Last_Name": (0, 255, 0),
        "Proposer_Father_Husband_Name": (255, 165, 0),
        "Proposer_Mother_Name": (255, 165, 0),
        "Proposer_Date_of_Birth": (0, 0, 255),
        "Proposer_Age": (0, 0, 255),
        "Proposer_Address_Line1": (255, 0, 255),
        "Proposer_City": (255, 0, 255),
        "Proposer_State": (255, 0, 255),
        "Proposer_Pincode": (255, 0, 255),
        "Proposer_Mobile_Number": (255, 255, 0),
        "Proposer_PAN": (255, 0, 0),
        "Proposer_Aadhaar": (255, 0, 0),
        "Proposer_Email": (0, 255, 255),
    }
    DEFAULT_COLOR = (0, 200, 0)

    def score_fields(self, fields: List[ExtractedField]) -> List[ExtractedField]:
        for f in fields:
            if not f.value:
                f.confidence = 0.0
                f.review_category = ReviewCategory.MISSING.value
                f.needs_human_review = False
                continue

            score = f.confidence  # Start with VLM/OCR confidence

            # Boost for KB-corrected values (we verified and fixed them)
            if f.kb_corrected:
                score = min(score + 0.05, 0.98)

            # Penalize for validation failures
            if f.validation_status == ValidationStatus.INVALID.value:
                score *= 0.5
            elif f.validation_status == ValidationStatus.HALLUCINATION.value:
                score = 0.0

            # Penalize for cross-field issues
            if f.cross_field_issues:
                score *= 0.85

            # Boost for OCR verification
            if f.ocr_verified:
                score = min(score + 0.10, 0.99)

            f.confidence = round(max(0.0, min(1.0, score)), 4)

            # Assign review category
            if f.confidence >= 0.85 and f.validation_status == ValidationStatus.VALID.value:
                f.review_category = ReviewCategory.AUTO_ACCEPTED.value
                f.needs_human_review = False
            elif f.confidence >= 0.70:
                f.review_category = ReviewCategory.NEEDS_REVIEW.value
                f.needs_human_review = True
            elif f.confidence >= 0.50:
                f.review_category = ReviewCategory.LOW_CONFIDENCE.value
                f.needs_human_review = True
            else:
                f.review_category = ReviewCategory.REJECTED.value
                f.needs_human_review = True

        return fields

    def draw_overlay(self, page: PageImage, fields: List[ExtractedField], output_path: str):
        vis = page.cv_image.copy()
        for f in fields:
            if not f.value or not f.value_bbox:
                continue

            color = self.FIELD_COLORS.get(f.field_name, self.DEFAULT_COLOR)

            # Highlight accepted vs needs-review differently
            if f.review_category == ReviewCategory.AUTO_ACCEPTED.value:
                thickness = 2
            elif f.review_category == ReviewCategory.NEEDS_REVIEW.value:
                thickness = 3
                color = (0, 165, 255)  # Orange for review
            else:
                thickness = 2
                color = (0, 0, 255)  # Red for low conf/rejected

            bbox = f.value_bbox
            x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

            label = f"{f.field_name}: {f.value[:25]}"
            font_scale = 0.35
            thickness_text = 1
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness_text)
            cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(vis, label, (x1 + 2, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness_text)

            # Draw confidence badge
            conf_label = f"{f.confidence:.0%}"
            cv2.putText(vis, conf_label, (x2 - 40, y2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

        cv2.imwrite(output_path, vis)
        logger.info(f"Overlay saved: {output_path}")

    def build_click_to_source_map(self, fields: List[ExtractedField], page_width: int, page_height: int) -> Dict[str, Any]:
        source_map = {}
        for f in fields:
            entry: Dict[str, Any] = {
                "field_name": f.field_name,
                "value": f.value,
                "confidence": f.confidence,
                "review_category": f.review_category,
                "needs_human_review": f.needs_human_review,
                "validation_status": f.validation_status,
                "kb_corrected": f.kb_corrected,
                "click_to_source": None,
            }

            if f.value_bbox:
                entry["click_to_source"] = {
                    "bbox_pixels": f.value_bbox.to_list(),
                    "bbox_normalized": f.value_bbox.normalized(),
                    "page_num": f.page_num,
                }

            if f.label_bbox:
                entry["label_location"] = {
                    "bbox_pixels": f.label_bbox.to_list(),
                    "bbox_normalized": f.label_bbox.normalized(),
                }

            if f.kb_corrected:
                entry["kb_original_value"] = f.kb_original_value
                entry["kb_correction_reason"] = f.kb_correction_reason

            if f.cross_field_issues:
                entry["cross_field_issues"] = f.cross_field_issues

            source_map[f.field_name] = entry

        return source_map
