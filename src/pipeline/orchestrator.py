"""
Pipeline Orchestrator
=====================
Wires all 5 layers together, processes PDF pages, produces FormResult.

Layer 1: Preprocessor → PageImage
Layer 2: VLM Extractor → {field: value} dict + raw_outputs
Layer 3: OCR Verifier → confidence boost + bbox localization per field
Layer 4: Validation+KB → regex, fuzzy, Verhoeff, cross-field checks
Layer 5: Confidence Scorer → final score, review category, overlay, click-to-source JSON
"""

import os
import sys
import io
import json
import logging
import time
from typing import Dict, List, Optional, Any

import fitz
from PIL import Image
import numpy as np

from src.pipeline.models.schemas import (
    ExtractedField, BBox, PageImage, PageResult, FormResult,
    ExtractionSource, ValidationStatus, ReviewCategory,
)
from src.pipeline.layers.preprocessor import Preprocessor
from src.pipeline.layers.vlm_extractor import VLMExtractor
from src.pipeline.layers.ocr_verifier import OCRVerifier
from src.pipeline.layers.validation_kb import ValidationKB
from src.pipeline.layers.confidence_scorer import ConfidenceScorer

logger = logging.getLogger(__name__)

LABEL_PATTERNS = {
    "Proposer_Full_Name": ["Full Name", "Name of Proposer", "Proposer Name"],
    "Proposer_First_Name": ["First Name", "Name"],
    "Proposer_Father_Husband_Name": ["Father", "Husband", "Father/Husband"],
    "Proposer_Mother_Name": ["Mother"],
    "Proposer_Date_of_Birth": ["Date of Birth", "DOB", "Birth Date"],
    "Proposer_Age": ["Age"],
    "Proposer_Gender": ["Gender", "Sex"],
    "Proposer_Marital_Status": ["Marital Status", "Marital"],
    "Proposer_City": ["City", "Town"],
    "Proposer_State": ["State", "Province"],
    "Proposer_Pincode": ["PIN", "Pincode", "PIN Code"],
    "Proposer_Mobile_Number": ["Mobile", "Phone", "Telephone"],
    "Proposer_PAN": ["PAN"],
    "Proposer_Aadhaar": ["Aadhaar", "Aadhar", "UID"],
    "Proposer_Email": ["Email", "E-mail"],
    "Proposer_Address_Line1": ["House", "Flat", "Address"],
    "Proposer_Birth_Place": ["Birth Place", "Place of Birth"],
    "Proposer_Nationality": ["Nationality"],
    "Proposer_Citizenship": ["Citizenship"],
    "Sum_Assured": ["Sum Assured", "Sum Assd"],
    "Premium_Amount": ["Premium", "Prem."],
    "Plan_Name": ["Plan", "Plan Name"],
    "Nominee_Name": ["Nominee"],
    "Bank_Name": ["Bank", "Bank Name"],
    "Bank_IFSC": ["IFSC"],
}


class PipelineOrchestrator:

    def __init__(
        self,
        vlm_model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        dpi: int = 150,
        tile_cols: int = 2,
        tile_rows: int = 3,
        tile_overlap: int = 50,
        output_dir: str = "data/output",
    ):
        self.dpi = dpi
        self.tile_cols = tile_cols
        self.tile_rows = tile_rows
        self.tile_overlap = tile_overlap
        self.output_dir = output_dir

        self.preprocessor = Preprocessor()
        self.vlm = VLMExtractor(model_name=vlm_model)
        self.ocr = OCRVerifier()
        self.validation = ValidationKB()
        self.scorer = ConfidenceScorer()

        os.makedirs(output_dir, exist_ok=True)

    def process_pdf(
        self,
        pdf_path: str,
        pages: Optional[List[int]] = None,
    ) -> FormResult:
        t0_total = time.time()
        logger.info(f"=" * 60)
        logger.info(f"PIPELINE START: {pdf_path}")
        logger.info(f"=" * 60)

        doc = fitz.open(pdf_path)
        total_pages = doc.page_count

        if pages is None:
            pages = list(range(1, total_pages + 1))

        form_result = FormResult(
            pdf_path=pdf_path,
            total_pages=total_pages,
            pages_processed=pages,
        )

        for page_num in pages:
            page_idx = page_num - 1
            if page_idx < 0 or page_idx >= total_pages:
                logger.warning(f"Page {page_num} out of range, skipping")
                continue

            logger.info(f"\n{'='*50}")
            logger.info(f"PROCESSING PAGE {page_num}/{total_pages}")
            logger.info(f"{'='*50}")

            page_result = self._process_single_page(doc, page_idx, page_num)
            form_result.page_results.append(page_result)

        doc.close()

        elapsed_total = (time.time() - t0_total) * 1000
        form_result.processing_time_ms = round(elapsed_total, 1)

        all_fields = form_result.all_fields()
        found_fields = [f for f in all_fields if f.value]
        if found_fields:
            form_result.overall_confidence = round(
                sum(f.confidence for f in found_fields) / len(found_fields), 4
            )
        form_result.kpis = self._compute_kpis(all_fields)

        auto = sum(1 for f in found_fields if f.review_category == ReviewCategory.AUTO_ACCEPTED.value)
        review = sum(1 for f in found_fields if f.review_category == ReviewCategory.NEEDS_REVIEW.value)
        low = sum(1 for f in found_fields if f.review_category == ReviewCategory.LOW_CONFIDENCE.value)
        rej = sum(1 for f in found_fields if f.review_category == ReviewCategory.REJECTED.value)
        missing = sum(1 for f in all_fields if not f.value)

        if rej > 0 or low > auto:
            form_result.form_status = "needs_review"
        elif review > 0:
            form_result.form_status = "mostly_confident"
        else:
            form_result.form_status = "auto_accepted"

        self._save_results(form_result, pdf_path)
        self.ocr.clear_cache()

        logger.info(f"\n{'='*60}")
        logger.info(f"PIPELINE COMPLETE in {elapsed_total/1000:.1f}s")
        logger.info(f"  Fields found: {len(found_fields)}/{len(all_fields)}")
        logger.info(f"  Auto-accepted: {auto} | Needs review: {review} | Low conf: {low} | Rejected: {rej} | Missing: {missing}")
        logger.info(f"  Overall confidence: {form_result.overall_confidence:.1%}")
        logger.info(f"  Form status: {form_result.form_status}")
        logger.info(f"{'='*60}")

        return form_result

    def _process_single_page(self, doc: fitz.Document, page_idx: int, page_num: int) -> PageResult:
        t0 = time.time()

        # ── Layer 1: Preprocess ──
        logger.info(f"[L1] Preprocessing page {page_num}...")
        page = doc[page_idx]
        mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        pil_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        page_image = self.preprocessor.process_page(page_num, pil_img, dpi=self.dpi)
        logger.info(f"[L1] Done — quality={page_image.quality_score:.1f}, steps={page_image.preprocessing_applied}")

        # ── Layer 2: VLM Extract ──
        logger.info(f"[L2] VLM extraction on page {page_num}...")
        tiles = self.preprocessor.create_tiles(
            page_image, cols=self.tile_cols, rows=self.tile_rows, overlap=self.tile_overlap
        )
        logger.info(f"[L2] {len(tiles)} tiles created")

        vlm_fields, raw_outputs = self.vlm.extract_page(tiles)
        found_vlm = sum(1 for v in vlm_fields.values() if v is not None)
        logger.info(f"[L2] Done — {found_vlm} fields extracted by VLM")

        # Convert VLM dict → ExtractedField list
        extracted_fields = []
        for field_name, value in vlm_fields.items():
            ef = ExtractedField(
                field_name=field_name,
                value=value if value else "",
                confidence=0.75 if value else 0.0,
                source=ExtractionSource.VLM_MERGED.value if value else ExtractionSource.NOT_FOUND.value,
                validation_status=ValidationStatus.MISSING.value if not value else ValidationStatus.VALID.value,
                page_num=page_num,
                tile_origins=["vlm_merged"],
            )
            extracted_fields.append(ef)

        # ── Layer 3: OCR Verify + Locate ──
        logger.info(f"[L3] OCR verification on page {page_num}...")
        for ef in extracted_fields:
            if not ef.value:
                continue

            ocr_conf, verified, bbox = self.ocr.verify_field(
                page_num, page_image.cv_image, ef.field_name, ef.value
            )

            ef.ocr_verified = verified
            if bbox:
                ef.value_bbox = bbox

            if verified:
                ef.confidence = ocr_conf
                ef.ocr_alternatives.append({"ocr_confidence": ocr_conf, "verified": True})
            else:
                ef.confidence = max(ef.confidence * 0.9, 0.3)
                ef.ocr_alternatives.append({"ocr_confidence": ocr_conf, "verified": False})

            # Locate label bbox
            patterns = LABEL_PATTERNS.get(ef.field_name, [])
            if patterns:
                label_bbox = self.ocr.find_label_bbox(
                    page_num, page_image.cv_image, ef.field_name, patterns
                )
                if label_bbox:
                    ef.label_bbox = label_bbox

        verified_count = sum(1 for f in extracted_fields if f.ocr_verified)
        located_count = sum(1 for f in extracted_fields if f.value_bbox)
        logger.info(f"[L3] Done — {verified_count} verified, {located_count} bbox-located")

        # ── Layer 4: Validate + KB ──
        logger.info(f"[L4] Validation + KB correction...")
        for ef in extracted_fields:
            self.validation.validate_field(ef)

        extracted_fields = self.validation.cross_field_validate(extracted_fields)

        corrected = sum(1 for f in extracted_fields if f.kb_corrected)
        invalid = sum(1 for f in extracted_fields if f.validation_status == ValidationStatus.INVALID.value)
        hallucinated = sum(1 for f in extracted_fields if f.hallucination_flag)
        logger.info(f"[L4] Done — {corrected} corrected, {invalid} invalid, {hallucinated} hallucinated")

        # ── Layer 5: Score + Localize ──
        logger.info(f"[L5] Confidence scoring + overlay...")
        extracted_fields = self.scorer.score_fields(extracted_fields)

        page_result = PageResult(
            page_num=page_num,
            fields=extracted_fields,
            vlm_raw_outputs=raw_outputs,
            ocr_region_count=sum(1 for f in extracted_fields if f.ocr_verified),
            processing_time_ms=round((time.time() - t0) * 1000, 1),
        )

        # Draw bbox overlay
        pdf_basename = os.path.splitext(os.path.basename(doc.name if hasattr(doc, 'name') else "unknown"))[0]
        overlay_path = os.path.join(self.output_dir, f"{pdf_basename}_page{page_num}_overlay.png")
        self.scorer.draw_overlay(page_image, extracted_fields, overlay_path)

        # Build click-to-source JSON
        source_map = self.scorer.build_click_to_source_map(
            extracted_fields, page_image.width, page_image.height
        )
        source_map_path = os.path.join(self.output_dir, f"{pdf_basename}_page{page_num}_click2source.json")
        with open(source_map_path, "w", encoding="utf-8") as jf:
            json.dump(source_map, jf, indent=2, ensure_ascii=False)

        auto = sum(1 for f in extracted_fields if f.review_category == ReviewCategory.AUTO_ACCEPTED.value)
        rev = sum(1 for f in extracted_fields if f.review_category == ReviewCategory.NEEDS_REVIEW.value)
        logger.info(f"[L5] Done — {auto} auto_accepted, {rev} needs_review")

        # Print field summary
        self._print_field_summary(extracted_fields)

        return page_result

    def _print_field_summary(self, fields: List[ExtractedField]):
        logger.info(f"\n{'─'*60}")
        logger.info(f"{'Field':<35} {'Value':<20} {'Conf':>5} {'Status':<10} {'Review'}")
        logger.info(f"{'─'*60}")
        for f in sorted(fields, key=lambda x: x.field_name):
            val = f.value[:18] if f.value else "—"
            conf = f"{f.confidence:.0%}" if f.value else "—"
            status = f.validation_status[:8] if f.value else "MISSING"
            review = f.review_category[:12] if f.value else "—"
            kb_flag = " ✎" if f.kb_corrected else ""
            ocr_flag = " ✓" if f.ocr_verified else ""
            logger.info(f"{f.field_name:<35} {val:<20} {conf:>5} {status:<10} {review}{kb_flag}{ocr_flag}")
        logger.info(f"{'─'*60}")

    def _compute_kpis(self, fields: List[ExtractedField]) -> Dict[str, Any]:
        found = [f for f in fields if f.value]
        return {
            "total_fields": len(fields),
            "fields_extracted": len(found),
            "fields_missing": len(fields) - len(found),
            "auto_accepted": sum(1 for f in found if f.review_category == ReviewCategory.AUTO_ACCEPTED.value),
            "needs_review": sum(1 for f in found if f.review_category == ReviewCategory.NEEDS_REVIEW.value),
            "low_confidence": sum(1 for f in found if f.review_category == ReviewCategory.LOW_CONFIDENCE.value),
            "rejected": sum(1 for f in found if f.review_category == ReviewCategory.REJECTED.value),
            "kb_corrected": sum(1 for f in found if f.kb_corrected),
            "ocr_verified": sum(1 for f in found if f.ocr_verified),
            "bbox_located": sum(1 for f in found if f.value_bbox is not None),
            "hallucinations_caught": sum(1 for f in found if f.hallucination_flag),
            "avg_confidence": round(sum(f.confidence for f in found) / max(len(found), 1), 4),
        }

    def _save_results(self, form_result: FormResult, pdf_path: str):
        pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]

        # Full JSON
        json_path = os.path.join(self.output_dir, f"{pdf_basename}_full_result.json")
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(form_result.to_dict(), jf, indent=2, ensure_ascii=False)
        logger.info(f"Full result saved: {json_path}")

        # Flat field CSV
        csv_path = os.path.join(self.output_dir, f"{pdf_basename}_fields.csv")
        with open(csv_path, "w", encoding="utf-8") as cf:
            cf.write("field_name,value,confidence,source,validation_status,review_category,"
                     "kb_corrected,ocr_verified,bbox_located,needs_human_review\n")
            for f in form_result.all_fields():
                cf.write(
                    f'"{f.field_name}","{f.value}",{f.confidence},{f.source},'
                    f'{f.validation_status},{f.review_category},'
                    f'{f.kb_corrected},{f.ocr_verified},{f.value_bbox is not None},'
                    f'{f.needs_human_review}\n'
                )
        logger.info(f"CSV saved: {csv_path}")
