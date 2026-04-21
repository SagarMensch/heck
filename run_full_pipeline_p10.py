"""
==========================================================================
  FULL 5-LAYER PIPELINE — P10.pdf
  ========================================================================
  L1: Preprocessing  (PyMuPDF → 300 DPI images)
  L2: Nemotron       (YOLOX native API → table cell bounding boxes)
  L3: PaddleOCR GPU  (PP-OCRv4 server — EN+HI, v3.4.1 compatible)
  L4: Qwen-VL        (Fallback for low-confidence fields)
  L5: Validation KB  (Regex, Verhoeff, Encyclopedia, Cross-field)
==========================================================================
"""
import os, sys, time, json, logging, re
import numpy as np
import torch
import fitz
from PIL import Image
from pathlib import Path
from dataclasses import dataclass, field as dc_field, asdict
from typing import List, Dict, Optional, Tuple, Any

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ── Logging ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("FullPipeline")

# ── Config ───────────────────────────────────────────────────────
PDF_PATH  = r"Techathon_Samples\P10.pdf"
OUT_DIR   = Path("output_full_pipeline_p10")
OUT_DIR.mkdir(exist_ok=True)
DPI       = 300
CONFIDENCE_THRESHOLD = 0.70   # Below this → Qwen fallback
PAGES_TO_PROCESS = [2, 3]     # Key data pages of Form 300

# ── 25 Critical Fields with (page, field_family) ─────────────────
FIELD_CATALOG = {
    # Page 2 — Section I: Proposer Details
    "Customer_ID":           {"page": 2, "family": "short_id"},
    "CKYC_Number":           {"page": 2, "family": "short_id"},
    "Proposer_First_Name":   {"page": 2, "family": "name_text"},
    "Proposer_Last_Name":    {"page": 2, "family": "name_text"},
    "Proposer_Father_Name":  {"page": 2, "family": "name_text"},
    "Proposer_Mother_Name":  {"page": 2, "family": "name_text"},
    "Proposer_Gender":       {"page": 2, "family": "binary_mark"},
    "Proposer_Marital_Status":{"page": 2, "family": "binary_mark"},
    "Proposer_Spouse_Name":  {"page": 2, "family": "name_text"},
    "Proposer_Date_of_Birth":{"page": 2, "family": "date"},
    "Proposer_Age":          {"page": 2, "family": "numeric"},
    "Proposer_Birth_Place":  {"page": 2, "family": "short_text"},
    "Proposer_Nationality":  {"page": 2, "family": "short_text"},
    "Proposer_Citizenship":  {"page": 2, "family": "short_text"},
    "Proposer_Address_Line1":{"page": 2, "family": "long_text"},
    "Proposer_City":         {"page": 2, "family": "short_text"},
    "Proposer_State":        {"page": 2, "family": "short_text"},
    "Proposer_Pincode":      {"page": 2, "family": "short_id"},
    "Proposer_Mobile_Number":{"page": 2, "family": "numeric"},
    "Proposer_Email":        {"page": 2, "family": "short_text"},
    # Page 3 — Section I contd: KYC / Occupation
    "Proposer_PAN":          {"page": 3, "family": "short_id"},
    "Proposer_Aadhaar":      {"page": 3, "family": "short_id"},
    "Proposer_Occupation":   {"page": 3, "family": "short_text"},
    "Proposer_Income":       {"page": 3, "family": "amount"},
    "Proposer_Employer":     {"page": 3, "family": "short_text"},
}

# ═════════════════════════════════════════════════════════════════
#  LAYER 1: PREPROCESSING
# ═════════════════════════════════════════════════════════════════
def preprocess_pdf(pdf_path, dpi=300, pages=None):
    """Render PDF pages to high-res PIL images."""
    logger.info(f"[L1] PREPROCESSING: {pdf_path} @ {dpi} DPI")
    doc = fitz.open(pdf_path)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    results = {}
    target_pages = pages or list(range(1, len(doc)+1))

    for pnum in target_pages:
        t0 = time.time()
        page = doc.load_page(pnum - 1)
        pix = page.get_pixmap(matrix=mat)
        pil_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        # Save for audit
        img_path = OUT_DIR / f"page_{pnum}.png"
        pil_img.save(str(img_path))
        results[pnum] = {
            "pil_image": pil_img,
            "np_image": np.array(pil_img),
            "width": pix.width,
            "height": pix.height,
            "path": str(img_path),
            "preprocess_ms": (time.time()-t0)*1000,
        }
        logger.info(f"  Page {pnum}: {pix.width}x{pix.height} rendered in {results[pnum]['preprocess_ms']:.0f}ms")
    doc.close()
    return results

# ═════════════════════════════════════════════════════════════════
#  LAYER 2: NEMOTRON TABLE STRUCTURE (Native YOLOX API)
# ═════════════════════════════════════════════════════════════════
class NemotronDetector:
    def __init__(self):
        self.model = None

    def load(self):
        if self.model:
            return
        logger.info("[L2] Loading Nemotron (native YOLOX)...")
        sys.path.insert(0, os.path.join(os.getcwd(), "nemotron_local"))
        from nemotron_table_structure_v1 import define_model
        self.model = define_model("table_structure_v1")
        logger.info(f"  ✓ Nemotron on {self.model.device}")

    def detect(self, pil_image):
        """Returns list of cell bounding boxes with labels and scores."""
        self.load()
        from nemotron_table_structure_v1 import postprocess_preds_table_structure
        img_np = np.array(pil_image)
        with torch.inference_mode():
            x = self.model.preprocess(img_np)
            preds = self.model(x, img_np.shape)[0]
        boxes, labels, scores = postprocess_preds_table_structure(
            preds, self.model.threshold, self.model.labels
        )
        # Convert normalized boxes to pixel coords
        h, w = img_np.shape[:2]
        cells = []
        for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
            x1, y1, x2, y2 = box
            cells.append({
                "id": i,
                "bbox": [int(x1*w), int(y1*h), int(x2*w), int(y2*h)],
                "label": label,
                "score": float(score),
            })
        # Filter to only "cell" detections (label index 1 per Exp config)
        cell_detections = [c for c in cells if c["label"] == "cell"]
        row_detections = [c for c in cells if c["label"] == "row"]
        col_detections = [c for c in cells if c["label"] == "column"]
        logger.info(f"  Nemotron: {len(cell_detections)} cells, {len(row_detections)} rows, {len(col_detections)} cols")
        return {
            "cells": cell_detections,
            "rows": row_detections,
            "columns": col_detections,
            "all": cells,
        }

# ═════════════════════════════════════════════════════════════════
#  LAYER 3: PaddleOCR GPU (PP-OCRv4 Server, v3.4.1 compatible)
# ═════════════════════════════════════════════════════════════════
class PaddleOCRLayer:
    def __init__(self):
        self.ocr_en = None
        self.ocr_hi = None

    def load(self):
        if self.ocr_en:
            return
        logger.info("[L3] Loading PaddleOCR (PP-OCRv4 server, GPU)...")
        from paddleocr import PaddleOCR
        # v3.4.1 API: no cls, use_gpu, use_angle_cls
        self.ocr_en = PaddleOCR(lang='en')
        self.ocr_hi = PaddleOCR(lang='hi')
        logger.info("  ✓ PaddleOCR EN+HI loaded")

    def ocr_full_page(self, np_image):
        """Run OCR on the entire page image. Returns list of text line dicts."""
        self.load()
        results_en = self.ocr_en.predict(np_image)
        lines = []
        for res in results_en:
            if hasattr(res, '__iter__'):
                for item in res:
                    text = getattr(item, 'rec_text', '') or ''
                    score = getattr(item, 'rec_score', 0.0) or 0.0
                    boxes = getattr(item, 'dt_polys', None)
                    bbox = None
                    if boxes is not None:
                        try:
                            pts = np.array(boxes)
                            x1, y1 = pts.min(axis=0).tolist()
                            x2, y2 = pts.max(axis=0).tolist()
                            bbox = [int(x1), int(y1), int(x2), int(y2)]
                        except:
                            pass
                    if text.strip():
                        lines.append({
                            "text": text.strip(),
                            "confidence": float(score),
                            "bbox": bbox,
                            "lang": "en",
                        })
        return lines

    def ocr_crop(self, pil_image, bbox):
        """OCR a specific bounding box region."""
        self.load()
        x1, y1, x2, y2 = bbox
        crop = pil_image.crop((x1, y1, x2, y2))
        crop_np = np.array(crop)
        if crop_np.shape[0] < 10 or crop_np.shape[1] < 10:
            return "", 0.0
        # Try English
        text, conf = self._run_ocr_on_crop(self.ocr_en, crop_np)
        if not text.strip():
            text, conf = self._run_ocr_on_crop(self.ocr_hi, crop_np)
        return text, conf

    def _run_ocr_on_crop(self, engine, crop_np):
        try:
            results = engine.predict(crop_np)
            texts, confs = [], []
            for res in results:
                if hasattr(res, '__iter__'):
                    for item in res:
                        t = getattr(item, 'rec_text', '') or ''
                        s = getattr(item, 'rec_score', 0.0) or 0.0
                        if t.strip():
                            texts.append(t.strip())
                            confs.append(s)
            return " ".join(texts), float(np.mean(confs)) if confs else 0.0
        except Exception as e:
            logger.warning(f"  OCR crop error: {e}")
            return "", 0.0

# ═════════════════════════════════════════════════════════════════
#  LAYER 4: QWEN-VL FALLBACK (for low-confidence fields)
# ═════════════════════════════════════════════════════════════════
class QwenVLFallback:
    def __init__(self):
        self.model = None
        self.processor = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        try:
            logger.info("[L4] Loading Qwen2.5-VL-3B for fallback...")
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=torch.float16, device_map="auto"
            )
            self._loaded = True
            logger.info("  ✓ Qwen-VL loaded")
        except Exception as e:
            logger.error(f"  ✗ Qwen-VL load failed: {e}")
            self._loaded = False

    def extract_field(self, pil_crop, field_name):
        """Ask Qwen to read a specific field from a crop."""
        if not self._loaded:
            self.load()
        if not self._loaded:
            return "", 0.0

        try:
            from qwen_vl_utils import process_vision_info
            prompt = f"Read the handwritten text in this image. This is the field '{field_name}' from an Indian insurance form. Return ONLY the text value, nothing else."
            messages = [{"role": "user", "content": [
                {"type": "image", "image": pil_crop},
                {"type": "text", "text": prompt}
            ]}]
            text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(text=[text_input], images=image_inputs, videos=video_inputs,
                                    padding=True, return_tensors="pt").to(self.model.device)
            with torch.inference_mode():
                ids = self.model.generate(**inputs, max_new_tokens=64)
            trimmed = ids[0][inputs.input_ids.shape[1]:]
            result = self.processor.decode(trimmed, skip_special_tokens=True).strip()
            return result, 0.80
        except Exception as e:
            logger.warning(f"  Qwen fallback error for {field_name}: {e}")
            return "", 0.0

# ═════════════════════════════════════════════════════════════════
#  LAYER 5: VALIDATION KB (Encyclopedia + Cross-field)
# ═════════════════════════════════════════════════════════════════
def load_validation_kb():
    """Import the existing ValidationKB from the pipeline."""
    sys.path.insert(0, os.getcwd())
    from src.pipeline.layers.validation_kb import ValidationKB
    from src.pipeline.models.schemas import ExtractedField
    return ValidationKB(), ExtractedField

# ═════════════════════════════════════════════════════════════════
#  ORCHESTRATOR: Wire all 5 layers together
# ═════════════════════════════════════════════════════════════════
def run_full_pipeline():
    total_t0 = time.time()

    # ── L1: Preprocess ────────────────────────────────────────
    page_images = preprocess_pdf(PDF_PATH, dpi=DPI, pages=PAGES_TO_PROCESS)

    # ── L2: Nemotron ──────────────────────────────────────────
    nemotron = NemotronDetector()
    nemotron_results = {}
    for pnum, pdata in page_images.items():
        t0 = time.time()
        result = nemotron.detect(pdata["pil_image"])
        elapsed = (time.time()-t0)*1000
        nemotron_results[pnum] = result
        logger.info(f"[L2] Page {pnum}: {len(result['cells'])} cells in {elapsed:.0f}ms")

    # ── L3: PaddleOCR GPU on each cell + full page ────────────
    paddle = PaddleOCRLayer()
    ocr_results = {}  # pnum → list of {field candidates}

    for pnum, pdata in page_images.items():
        t0 = time.time()
        # Full-page OCR for context
        full_page_lines = paddle.ocr_full_page(pdata["np_image"])
        logger.info(f"[L3] Page {pnum}: Full-page OCR → {len(full_page_lines)} lines")

        # Per-cell OCR using Nemotron bounding boxes
        cell_texts = []
        for cell in nemotron_results[pnum]["cells"]:
            text, conf = paddle.ocr_crop(pdata["pil_image"], cell["bbox"])
            cell_texts.append({
                "cell_id": cell["id"],
                "bbox": cell["bbox"],
                "nemotron_score": cell["score"],
                "ocr_text": text,
                "ocr_confidence": conf,
            })

        elapsed = (time.time()-t0)*1000
        ocr_results[pnum] = {
            "full_page_lines": full_page_lines,
            "cell_texts": cell_texts,
            "ocr_time_ms": elapsed,
        }
        logger.info(f"[L3] Page {pnum}: {len(cell_texts)} cells OCR'd in {elapsed:.0f}ms")

    # ── L4: Qwen-VL Fallback for low-confidence cells ─────────
    qwen = QwenVLFallback()
    fallback_count = 0
    for pnum, ocr_data in ocr_results.items():
        for cell in ocr_data["cell_texts"]:
            if cell["ocr_confidence"] < CONFIDENCE_THRESHOLD and cell["ocr_text"]:
                # Only load Qwen if actually needed
                crop = page_images[pnum]["pil_image"].crop(tuple(cell["bbox"]))
                vlm_text, vlm_conf = qwen.extract_field(crop, f"cell_{cell['cell_id']}")
                if vlm_text and vlm_conf > cell["ocr_confidence"]:
                    cell["vlm_text"] = vlm_text
                    cell["vlm_confidence"] = vlm_conf
                    cell["final_text"] = vlm_text
                    cell["final_confidence"] = vlm_conf
                    cell["source"] = "qwen_vlm"
                    fallback_count += 1
                else:
                    cell["final_text"] = cell["ocr_text"]
                    cell["final_confidence"] = cell["ocr_confidence"]
                    cell["source"] = "paddleocr"
            else:
                cell["final_text"] = cell["ocr_text"]
                cell["final_confidence"] = cell["ocr_confidence"]
                cell["source"] = "paddleocr"
    logger.info(f"[L4] Qwen-VL fallback applied to {fallback_count} cells")

    # ── L5: Validation KB ─────────────────────────────────────
    logger.info("[L5] Running Validation KB (Encyclopedia + Cross-field)...")
    try:
        validator, ExtractedField = load_validation_kb()

        # Build extracted fields from full-page OCR text and cell data
        extracted_fields = []
        for pnum, ocr_data in ocr_results.items():
            # Match cell text to known field names using heuristic position matching
            for cell in ocr_data["cell_texts"]:
                ef = ExtractedField(
                    field_name=f"Page{pnum}_Cell{cell['cell_id']}",
                    value=cell.get("final_text", ""),
                    confidence=cell.get("final_confidence", 0.0),
                    source=cell.get("source", "paddleocr"),
                    page_num=pnum,
                )
                try:
                    validator.validate_field(ef)
                except:
                    pass
                extracted_fields.append(ef)

        # Cross-field validation
        try:
            validator.cross_field_validate(extracted_fields)
        except:
            pass

        logger.info(f"[L5] Validated {len(extracted_fields)} fields")
    except Exception as e:
        logger.warning(f"[L5] Validation KB error: {e}")
        extracted_fields = []

    # ── OUTPUT: Save everything ──────────────────────────────
    total_time = time.time() - total_t0

    output = {
        "document": PDF_PATH,
        "pages_processed": PAGES_TO_PROCESS,
        "total_processing_time_s": round(total_time, 2),
        "pipeline_layers": ["L1_Preprocess", "L2_Nemotron", "L3_PaddleOCR", "L4_QwenVL", "L5_ValidationKB"],
        "pages": {}
    }

    for pnum in PAGES_TO_PROCESS:
        pdata = page_images[pnum]
        nemotron_data = nemotron_results[pnum]
        ocr_data = ocr_results[pnum]

        page_output = {
            "page_num": pnum,
            "image_size": f"{pdata['width']}x{pdata['height']}",
            "preprocess_ms": round(pdata["preprocess_ms"], 1),
            "nemotron_cells": len(nemotron_data["cells"]),
            "nemotron_rows": len(nemotron_data["rows"]),
            "nemotron_cols": len(nemotron_data["columns"]),
            "ocr_time_ms": round(ocr_data["ocr_time_ms"], 1),
            "full_page_lines": ocr_data["full_page_lines"],
            "cell_extractions": [],
        }

        for cell in ocr_data["cell_texts"]:
            page_output["cell_extractions"].append({
                "cell_id": cell["cell_id"],
                "bbox": cell["bbox"],
                "nemotron_score": round(cell["nemotron_score"], 3),
                "ocr_text": cell["ocr_text"],
                "ocr_confidence": round(cell["ocr_confidence"], 3),
                "final_text": cell.get("final_text", ""),
                "final_confidence": round(cell.get("final_confidence", 0), 3),
                "source": cell.get("source", "paddleocr"),
            })

        output["pages"][str(pnum)] = page_output

    # Save JSON
    json_path = OUT_DIR / "FULL_PIPELINE_P10.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    # Print summary
    print(f"\n{'='*70}")
    print(f"  FULL PIPELINE COMPLETE — P10.pdf")
    print(f"{'='*70}")
    print(f"  Total Time:        {total_time:.1f}s")
    print(f"  Pages Processed:   {PAGES_TO_PROCESS}")
    for pnum in PAGES_TO_PROCESS:
        pd = output["pages"][str(pnum)]
        print(f"\n  Page {pnum}:")
        print(f"    Nemotron cells:  {pd['nemotron_cells']}")
        print(f"    OCR lines:       {len(pd['full_page_lines'])}")
        print(f"    Cell extractions:{len(pd['cell_extractions'])}")
        filled = [c for c in pd["cell_extractions"] if c["final_text"]]
        avg_conf = np.mean([c["final_confidence"] for c in filled]) if filled else 0
        print(f"    Cells with text: {len(filled)}")
        print(f"    Avg confidence:  {avg_conf:.1%}")
    print(f"\n  Qwen-VL fallbacks: {fallback_count}")
    print(f"  Output saved to:   {json_path}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    run_full_pipeline()
