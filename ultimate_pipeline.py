#!/usr/bin/env python
"""
ULTIMATE EXTRACTION PIPELINE
============================
1. Advanced Preprocessing (denoise, deskew, enhance) -> keeps BGR
2. PaddleX Layout Analysis (RT-DETR layout + OCR)
3. Primary OCR: PaddleOCR GPU (Bilingual Hindi+English v5)
4. Fallback: Qwen2.5-VL for low-confidence fields (<0.85)
5. Field Mapping & Validation
"""
import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import numpy as np
import cv2

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ultimate_pipeline")


@dataclass
class ExtractionField:
    field_name: str
    value: str
    confidence: float
    source: str
    bbox: Optional[List[int]] = None
    page_num: int = 1
    corrected: bool = False


class AdvancedPreprocessor:
    """Image enhancement: denoise, deskew, CLAHE contrast boost."""
    
    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """Preprocess and return BGR image (PaddleOCR needs 3-channel)."""
        if img is None or img.size == 0:
            return img
        # 1. Denoise
        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        # 2. Convert to grayscale for deskew
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 3. Deskew
        coords = np.column_stack(np.where(gray > 0))
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            if abs(angle) > 0.5:
                h, w = img.shape[:2]
                M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
                img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        # 4. CLAHE on L channel (keeps BGR)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return img


class PaddleOCREngine:
    """Bilingual PaddleOCR GPU engine (v5)."""

    def __init__(self):
        self.ocr_en = None
        self.ocr_hi = None
        self._loaded = False
    
    def load(self):
        if self._loaded:
            return
        from paddleocr import PaddleOCR
        self.ocr_en = PaddleOCR(use_textline_orientation=True, lang='en', text_det_thresh=0.3, text_det_box_thresh=0.5)
        self.ocr_hi = PaddleOCR(use_textline_orientation=True, lang='hi', text_det_thresh=0.3, text_det_box_thresh=0.5)
        self._loaded = True
        logger.info("PaddleOCR v5 (EN+HI) loaded")
    
    def ocr_bilingual(self, img: np.ndarray) -> List[Dict]:
        """Run bilingual OCR. Returns list of {text, confidence, bbox, lang}."""
        self.load()
        results = []
        # English
        for res in self.ocr_en.predict(img):
            results.extend(self._parse(res, 'en'))
        # Hindi
        for res in self.ocr_hi.predict(img):
            results.extend(self._parse(res, 'hi'))
        return results
    
    def _parse(self, result, lang: str) -> List[Dict]:
        """Parse PaddleOCR v5 OCRResult (dict-like)."""
        regions = []
        texts = result.get('rec_texts') or []
        scores = result.get('rec_scores') or []
        polys = result.get('dt_polys') or []
        
        for text, score, poly in zip(texts, scores, polys):
            if not text or not text.strip():
                continue
            pts = np.asarray(poly)
            if pts.ndim == 2 and pts.shape[0] >= 2:
                xs, ys = pts[:, 0], pts[:, 1]
                regions.append({
                    "text": str(text).strip(),
                    "confidence": float(score),
                    "bbox": [int(np.min(xs)), int(np.min(ys)), int(np.max(xs)), int(np.max(ys))],
                    "lang": lang
                })
        return regions


class QwenVLFallback:
    """Qwen2.5-VL fallback for low-confidence fields."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self._loaded = False
    
    def load(self):
        if self._loaded:
            return
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name, torch_dtype="auto", device_map="auto"
            )
            self._loaded = True
            logger.info("Qwen2.5-VL loaded")
        except Exception as e:
            logger.warning(f"Qwen2.5-VL unavailable: {e}")
            self._loaded = False
    
    def extract_field(self, img: np.ndarray, bbox: List[int], field_name: str) -> Optional[str]:
        if not self._loaded:
            self.load()
        if not self._loaded:
            return None
        try:
            from PIL import Image
            x1, y1, x2, y2 = map(int, bbox)
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                return None
            pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": f"Extract the handwritten/printed value for '{field_name}'. Return only the value."}
                ]}
            ]
            text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text_input], images=[pil_img], return_tensors="pt").to(self.model.device)
            output_ids = self.model.generate(**inputs, max_new_tokens=50)
            decoded = self.processor.batch_decode(output_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
            return decoded.strip()
        except Exception as e:
            logger.debug(f"Qwen failed for {field_name}: {e}")
            return None


class UltimateExtractionPipeline:
    """
    Ultimate Pipeline:
    1. Preprocess (denoise, deskew, CLAHE)
    2. PaddleOCR v5 bilingual GPU
    3. Proximity-based field mapping
    4. Qwen fallback for low confidence
    """

    # Field definitions: label patterns + expected value region (y_center range in % of page height)
    # Format: field_name -> (label_patterns, y_start_pct, y_end_pct)
    # Field definitions: (label_patterns, y_start_pct, y_end_pct, x_start_pct, x_end_pct)
    # Calibrated from actual P02 page 2 OCR at 200 DPI (1689x2292)
    FIELD_DEFS = {
        "Proposer_First_Name":    (["First Name"],                   0.195, 0.225, 0.50, 0.65),
        "Proposer_Middle_Name":   (["Middle Name"],                  0.195, 0.225, 0.65, 0.80),
        "Proposer_Last_Name":     (["Last Name", "आडनाव"],           0.195, 0.230, 0.80, 1.00),
        "Proposer_Prefix":        (["Prefix", "उपसर्ग"],             0.195, 0.225, 0.05, 0.20),
        "Proposer_Title":         (["Mr", "Mrs", "Ms"],              0.195, 0.230, 0.42, 0.55),
        "Proposer_Father_Name":   (["Father", "वडिल", "पिता"],       0.225, 0.270, 0.47, 1.00),
        "Proposer_Mother_Name":   (["Mother", "आई", "माँ"],          0.255, 0.300, 0.47, 1.00),
        "Proposer_Gender":        (["Gender", "लिंग"],               0.280, 0.310, 0.47, 1.00),
        "Proposer_Marital_Status":(["Marital", "वैवाहिक"],           0.300, 0.340, 0.47, 1.00),
        "Proposer_Spouse_Name":   (["Spouse", "पत्नी"],              0.325, 0.360, 0.47, 1.00),
        "Proposer_DOB":           (["Date of Birth", "जन्म"],        0.350, 0.400, 0.47, 1.00),
        "Proposer_Age":           (["Age", "वय", "आयु"],             0.370, 0.420, 0.47, 1.00),
        "Proposer_Birth_Place":   (["Place", "City of Birth", "जन्म स्थळ"], 0.460, 0.500, 0.47, 1.00),
        "Proposer_Nationality":   (["Nationality", "राष्ट्रीयता"],   0.520, 0.560, 0.47, 1.00),
        "Proposer_Citizenship":   (["Citizenship", "नागरिकता"],      0.540, 0.580, 0.47, 1.00),
        "Proposer_Address_House": (["House No", "Building", "घर"],   0.620, 0.665, 0.47, 1.00),
        "Proposer_Address_Town":  (["Town", "Village", "Taluk"],     0.660, 0.700, 0.47, 1.00),
        "Proposer_Address_City":  (["City", "District", "जिल्हा"],   0.685, 0.720, 0.47, 1.00),
        "Proposer_Address_State": (["State", "Country", "राज्य"],    0.700, 0.745, 0.47, 1.00),
        "Proposer_PIN":           (["PIN Code", "पीन कोड"],          0.730, 0.775, 0.47, 1.00),
        "Proposer_Phone":         (["Tel", "STD", "दूरध्वनी"],       0.760, 0.810, 0.47, 1.00),
    }

    def __init__(self, confidence_threshold: float = 0.85, use_qwen: bool = False):
        self.confidence_threshold = confidence_threshold
        self.use_qwen = use_qwen
        self.preprocessor = AdvancedPreprocessor()
        self.ocr_engine = PaddleOCREngine()
        self.qwen = QwenVLFallback() if use_qwen else None
    
    def process_pdf(self, pdf_path: str, pages: Optional[List[int]] = None) -> Dict:
        import fitz
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        if pages is None:
            pages = list(range(1, total_pages + 1))
        
        all_fields = []
        t_start = time.time()
        
        for pnum in pages:
            if pnum < 1 or pnum > total_pages:
                continue
            logger.info(f"Processing page {pnum}/{total_pages}...")
            page = doc.load_page(pnum - 1)
            pix = page.get_pixmap(dpi=200)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif pix.n == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            page_fields = self._process_page(img, pnum)
            all_fields.extend(page_fields)
        
        doc.close()
        elapsed = time.time() - t_start
        
        high_conf = sum(1 for f in all_fields if f.confidence >= self.confidence_threshold)
        low_conf = sum(1 for f in all_fields if f.confidence < self.confidence_threshold)
        corrected = sum(1 for f in all_fields if f.corrected)
        found = sum(1 for f in all_fields if f.value)
        
        stats = {
            "total_fields": len(all_fields),
            "found": found,
            "high_confidence": high_conf,
            "low_confidence": low_conf,
            "corrected_by_qwen": corrected,
            "processing_time_sec": round(elapsed, 2),
            "time_per_page": round(elapsed / max(len(pages), 1), 2),
        }
        
        return {"pdf_path": pdf_path, "pages_processed": len(pages), "fields": [asdict(f) for f in all_fields], "statistics": stats}
    
    def _process_page(self, img: np.ndarray, page_num: int) -> List[ExtractionField]:
        h, w = img.shape[:2]
        
        # 1. Preprocess
        try:
            img = self.preprocessor.preprocess(img)
        except Exception as e:
            logger.warning(f"Preprocess failed: {e}")
        
        # 2. OCR
        ocr_results = self.ocr_engine.ocr_bilingual(img)
        logger.info(f"Page {page_num}: {len(ocr_results)} OCR regions")
        
        # 3. Map OCR results to fields by Y+X position
        value_x_threshold = 0.40 * w  # minimum x-center for values
        
        fields = []
        for field_name, (patterns, y_start_pct, y_end_pct, x_start_pct, x_end_pct) in self.FIELD_DEFS.items():
            y1 = int(y_start_pct * h)
            y2 = int(y_end_pct * h)
            x1 = int(x_start_pct * w)
            x2 = int(x_end_pct * w)
            
            matching = []
            for r in ocr_results:
                rx1, ry1, rx2, ry2 = r["bbox"]
                ry_center = (ry1 + ry2) / 2
                rx_center = (rx1 + rx2) / 2
                
                if not (y1 <= ry_center <= y2):
                    continue
                if not (x1 <= rx_center <= x2):
                    continue
                if r["confidence"] < 0.5:
                    continue
                if not r["text"].strip() or len(r["text"].strip()) <= 1:
                    continue
                
                is_label = any(p.lower() in r["text"].lower() for p in patterns)
                if is_label:
                    continue
                
                if r["text"].strip().rstrip('.').isdigit() and len(r["text"].strip()) <= 3:
                    continue
                
                matching.append(r)
            
            if matching:
                best = max(matching, key=lambda r: r["confidence"])
                
                value = best["text"]
                conf = best["confidence"]
                bbox = best["bbox"]
                source = f"paddle_ocr_{best['lang']}"
            else:
                value = ""
                conf = 0.0
                bbox = None
                source = "not_found"
            
            # 4. Qwen fallback
            corrected = False
            if conf < self.confidence_threshold and self.qwen and bbox:
                qwen_val = self.qwen.extract_field(img, bbox, field_name)
                if qwen_val:
                    value = qwen_val
                    conf = 0.7
                    source = "qwen_vl"
                    corrected = True
            
            fields.append(ExtractionField(
                field_name=field_name, value=value, confidence=conf,
                source=source, bbox=bbox, page_num=page_num, corrected=corrected
            ))
        
        return fields


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ultimate LIC Extraction Pipeline")
    parser.add_argument("pdf_path", help="Path to PDF")
    parser.add_argument("--output", default="data/output/ultimate_result.json")
    parser.add_argument("--confidence", type=float, default=0.85)
    parser.add_argument("--use-qwen", action="store_true")
    parser.add_argument("--pages", type=int, nargs="+")
    args = parser.parse_args()
    
    pipeline = UltimateExtractionPipeline(confidence_threshold=args.confidence, use_qwen=args.use_qwen)
    result = pipeline.process_pdf(args.pdf_path, args.pages)
    
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    stats = result["statistics"]
    print("\n" + "=" * 70)
    print("EXTRACTION RESULTS")
    print("=" * 70)
    for f in result["fields"]:
        icon = "[OK]" if f["confidence"] >= args.confidence else "[!!]" if f["value"] else "[--]"
        val = f["value"][:60] if f["value"] else "(empty)"
        print(f"  {icon} {f['field_name']:<30} {val:<60} conf={f['confidence']:.2f} src={f['source']}")
    print("=" * 70)
    print(f"Fields: {stats['total_fields']} | Found: {stats['found']} | High conf: {stats['high_confidence']} | Low conf: {stats['low_confidence']}")
    print(f"Time: {stats['processing_time_sec']}s | Per page: {stats['time_per_page']}s")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
