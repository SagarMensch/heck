"""Reference-registered final extraction runner.

This script is kept as a simple production-style runner for direct PDF tests:
1. Render target pages from the PDF.
2. Register each page to the local blank LIC Form 300 reference page.
3. Crop fields from the shared normalized templates.
4. OCR each crop with PaddleOCR EN/HI.
5. Save a flat JSON report with registration metadata.
"""

import json
import os
import sys
import io
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

from src.canonical_extractor import CanonicalExtractor
from src.extractor import QwenExtractor
from src.form300_templates import FORM300_PAGE_TEMPLATES, PAGE_TYPE_TO_PAGE_NUM
from src.lic_encyclopedia import LicCleaner
from src.verhoeff_validator import extract_aadhaar_from_text

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "1"
from paddleocr import PaddleOCR

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


PAGE_NUM_TO_TYPE = {v: k for k, v in PAGE_TYPE_TO_PAGE_NUM.items()}
TARGET_PAGES = sorted(PAGE_NUM_TO_TYPE.keys())

cleaner = LicCleaner()
ocr_en = None
ocr_hi = None
qwen_extractor = None


def get_ocr_en():
    global ocr_en
    if ocr_en is None:
        print("Loading PaddleOCR English (GPU)...")
        ocr_en = PaddleOCR(lang="en", use_textline_orientation=True)
    return ocr_en


def get_ocr_hi():
    global ocr_hi
    if ocr_hi is None:
        print("Loading PaddleOCR Hindi (GPU)...")
        ocr_hi = PaddleOCR(lang="hi", use_textline_orientation=True)
    return ocr_hi


def get_qwen_extractor():
    global qwen_extractor
    if qwen_extractor is None:
        print("Loading Qwen-VL for crop extraction...")
        qwen_extractor = QwenExtractor()
    return qwen_extractor


def ocr_crop(crop_img: np.ndarray, field_name: str):
    """Read the crop with Qwen-VL directly instead of PaddleOCR."""
    crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    crop_pil = Image.fromarray(crop_rgb)
    extractor = get_qwen_extractor()
    result = extractor.extract_field_fallback(crop_pil, field_name)
    value = str(result.get("value") or "").strip()
    confidence = float(result.get("confidence") or 0.0)
    return value, "qwen", confidence


def _parse_ocr_result(result):
    texts = []
    scores = []
    if not result:
        return texts, scores

    if isinstance(result, list):
        for entry in result:
            if hasattr(entry, "get"):
                rec_texts = entry.get("rec_texts") or []
                rec_scores = entry.get("rec_scores") or []
                for text, score in zip(rec_texts, rec_scores):
                    if str(text).strip():
                        texts.append(str(text).strip())
                        scores.append(float(score))
            elif isinstance(entry, list):
                for item in entry:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        second = item[1]
                        if isinstance(second, (list, tuple)) and len(second) == 2:
                            text, score = second
                            if str(text).strip():
                                texts.append(str(text).strip())
                                scores.append(float(score))
    return texts, scores


def postprocess_value(field_name: str, value: str, confidence: float):
    val = value.strip()
    conf = confidence

    if "pincode" in field_name or field_name.endswith("_pin"):
        parsed = cleaner.extract_pincode(val)
        if parsed:
            val, conf = parsed, max(conf, 0.99)
    elif "aadhaar" in field_name:
        parsed = extract_aadhaar_from_text(val)
        if parsed:
            val, conf = parsed, max(conf, 0.99)
    elif "gender" in field_name:
        parsed = cleaner.correct_gender(val)
        if parsed:
            val, conf = parsed, max(conf, 0.98)
    elif "city" in field_name or "birth" in field_name:
        parsed = cleaner.correct_city(val)
        if parsed:
            val, conf = parsed, max(conf, 0.95)
    elif "occupation" in field_name:
        parsed = cleaner.correct_occupation(val)
        if parsed:
            val, conf = parsed, max(conf, 0.92)

    return val, conf


def _write_partial_results(output_path: Optional[str], results: Dict[str, Dict]) -> None:
    if not output_path:
        return

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


def extract_from_pdf(
    pdf_path: str,
    pages: Optional[List[int]] = None,
    audit_dir: Optional[str] = None,
    output_path: Optional[str] = None,
):
    import fitz

    extractor = CanonicalExtractor(audit_dir=audit_dir)
    doc = fitz.open(pdf_path)
    page_nums = pages if pages else TARGET_PAGES

    final_results: Dict[str, Dict] = {}
    _write_partial_results(output_path, final_results)

    for pnum in page_nums:
        if pnum < 1 or pnum > doc.page_count:
            continue
        page_type = PAGE_NUM_TO_TYPE.get(pnum)
        if page_type not in FORM300_PAGE_TEMPLATES:
            continue

        print(f"Processing Page {pnum} ({page_type})...")
        page = doc.load_page(pnum - 1)
        pix = page.get_pixmap(dpi=200, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        registration = extractor.align_page(img, pnum)
        field_crops = extractor.extract_template_fields(img, pnum)

        for field_name, crop in field_crops.items():
            text, lang, confidence = ocr_crop(crop, field_name)
            if not text:
                continue

            value, confidence = postprocess_value(field_name, text, confidence)
            final_results[f"Page{pnum}_{field_name}"] = {
                "value": value,
                "confidence": round(float(confidence), 4),
                "lang": lang,
                "page_num": pnum,
                "page_type": page_type,
                "registration_method": registration.method,
                "registration_success": registration.success,
                "registration_overlap": round(float(registration.structure_overlap), 4),
            }

        _write_partial_results(output_path, final_results)
        print(f"Completed Page {pnum}: cumulative fields={len(final_results)}")

    doc.close()
    _write_partial_results(output_path, final_results)
    return final_results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_final_extraction.py <pdf_path> [pages]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    pages = None
    if len(sys.argv) >= 3 and sys.argv[2].strip():
        raw = sys.argv[2].strip()
        if "," in raw:
            pages = [int(part.strip()) for part in raw.split(",") if part.strip()]
        elif "-" in raw:
            start, end = raw.split("-", 1)
            pages = list(range(int(start), int(end) + 1))
        else:
            pages = [int(raw)]

    audit_dir = str(Path("data") / "final_extraction_audit" / Path(pdf_path).stem)
    print(f"Running Final Extraction on {pdf_path}...")
    output_path = str(Path(pdf_path).with_suffix("")) + "_final_result.json"
    results = extract_from_pdf(
        pdf_path,
        pages=pages,
        audit_dir=audit_dir,
        output_path=output_path,
    )

    print("\n" + "=" * 80)
    print("FINAL EXTRACTION RESULTS")
    print("=" * 80)
    for key, value in results.items():
        print(
            f"{key}: {value['value']} "
            f"(Conf: {value['confidence']:.2f}, Lang: {value['lang']}, Reg: {value['registration_method']})"
        )

    print(f"\nSaved to: {output_path}")
    print(f"Audit overlays saved under: {audit_dir}")
