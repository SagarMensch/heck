#!/usr/bin/env python
"""
ULTIMATE EXTRACTION PIPELINE v2
================================
1. PP-Structure Layout Analysis (template-agnostic)
2. PaddleOCR v5 Bilingual GPU (Hindi + English)
3. AI-Driven Spatial Reasoning (proximity-based KV association)
4. Qwen2.5-VL Fallback for low-confidence fields
5. Semantic Field Localization (bbox linking to source)
"""
import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import json
import time
import logging
import re
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pipeline_v2")


# ──────────────────────────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────────────────────────

@dataclass
class OCRRegion:
    text: str
    confidence: float
    bbox: List[int]  # [x1, y1, x2, y2]
    lang: str
    is_label: bool = False
    matched_field: Optional[str] = None

    @property
    def cx(self): return (self.bbox[0] + self.bbox[2]) / 2
    @property
    def cy(self): return (self.bbox[1] + self.bbox[3]) / 2
    @property
    def width(self): return self.bbox[2] - self.bbox[0]
    @property
    def height(self): return self.bbox[3] - self.bbox[1]
    @property
    def right(self): return self.bbox[2]
    @property
    def bottom(self): return self.bbox[3]


@dataclass
class ExtractedField:
    field_name: str
    value: str
    confidence: float
    source: str  # 'ppstructure', 'paddle_ocr_en', 'paddle_ocr_hi', 'qwen_vl', 'not_found'
    source_bbox: Optional[List[int]] = None
    label_bbox: Optional[List[int]] = None
    page_num: int = 1
    direction: str = ""  # 'right', 'below', 'below_right', 'inside'
    corrected: bool = False


# ──────────────────────────────────────────────────────────────────
# Layer 1: Advanced Preprocessor
# ──────────────────────────────────────────────────────────────────

class AdvancedPreprocessor:
    def preprocess(self, img: np.ndarray) -> np.ndarray:
        if img is None or img.size == 0:
            return img
        try:
            img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        except Exception:
            pass
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        coords = np.column_stack(np.where(gray > 0))
        if len(coords) > 100:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            if abs(angle) > 0.5:
                h, w = img.shape[:2]
                M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
                img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return img


# ──────────────────────────────────────────────────────────────────
# Layer 2: PP-Structure Layout Analysis
# ──────────────────────────────────────────────────────────────────

class PPStructureAnalyzer:
    """PaddleOCR PP-Structure for template-agnostic layout parsing."""

    def __init__(self):
        self.engine_hi = None
        self.engine_en = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        from paddleocr import PPStructure
        self.engine_hi = PPStructure(show_log=False, lang="hi")
        self.engine_en = PPStructure(show_log=False, lang="en")
        self._loaded = True
        logger.info("PP-Structure (Hindi + English) loaded")

    def analyze(self, img: np.ndarray) -> List[Dict]:
        """Return structured layout regions with text."""
        if not self._loaded:
            self.load()
        if not self._loaded:
            return []

        results = []
        try:
            for engine, lang in [(self.engine_en, 'en'), (self.engine_hi, 'hi')]:
                raw = engine(img)
                for region in raw:
                    rtype = region.get("type", "text")
                    bbox = region.get("bbox", [0, 0, 0, 0])
                    score = region.get("score", 0.0)
                    res_list = region.get("res", [])
                    texts = []
                    if isinstance(res_list, list):
                        for item in res_list:
                            if isinstance(item, dict) and "text" in item:
                                texts.append(item["text"])
                            elif isinstance(item, str):
                                texts.append(item)
                    combined = " ".join(texts).strip()
                    if combined:
                        results.append({
                            "type": rtype,
                            "bbox": bbox,
                            "confidence": score,
                            "text": combined,
                            "lang": lang,
                        })
        except Exception as e:
            logger.warning(f"PP-Structure failed: {e}")

        return results


# ──────────────────────────────────────────────────────────────────
# Layer 3: PaddleOCR v5 Bilingual GPU
# ──────────────────────────────────────────────────────────────────

class PaddleOCREngine:
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
        logger.info("PaddleOCR v5 (EN+HI) loaded on GPU")

    def ocr_bilingual(self, img: np.ndarray) -> List[OCRRegion]:
        self.load()
        regions = []
        for res in self.ocr_en.predict(img):
            regions.extend(self._parse(res, 'en'))
        for res in self.ocr_hi.predict(img):
            regions.extend(self._parse(res, 'hi'))
        return regions

    def _parse(self, result, lang: str) -> List[OCRRegion]:
        out = []
        texts = result.get('rec_texts') or []
        scores = result.get('rec_scores') or []
        polys = result.get('dt_polys') or []
        for text, score, poly in zip(texts, scores, polys):
            if not text or not text.strip():
                continue
            pts = np.asarray(poly)
            if pts.ndim == 2 and pts.shape[0] >= 2:
                xs, ys = pts[:, 0], pts[:, 1]
                out.append(OCRRegion(
                    text=str(text).strip(),
                    confidence=float(score),
                    bbox=[int(np.min(xs)), int(np.min(ys)), int(np.max(xs)), int(np.max(ys))],
                    lang=lang,
                ))
        return out


# ──────────────────────────────────────────────────────────────────
# Layer 4: AI-Driven Spatial Reasoning Engine
# ──────────────────────────────────────────────────────────────────

# Canonical field names and their label patterns (English + Hindi + Marathi)
FIELD_LABELS = {
    "Proposer_Name":           ["name", "naam", "नाव", "नाम", "नाम / name"],
    "Proposer_First_Name":     ["first name", "पहिले नाव", "प्रथम नाम"],
    "Proposer_Middle_Name":    ["middle name", "मध्यम नाव", "मध्य नाम"],
    "Proposer_Last_Name":      ["last name", "surname", "आडनाव", "अंतिम नाम"],
    "Proposer_Prefix":         ["prefix", "उपसर्ग"],
    "Proposer_Title":          ["mr", "mrs", "ms", "श्री", "श्रीमती", "सुश्री"],
    "Proposer_Father_Name":    ["father", "पिता", "वडिल"],
    "Proposer_Mother_Name":    ["mother", "माँ", "आई"],
    "Proposer_Gender":         ["gender", "लिंग", "sex"],
    "Proposer_Marital_Status": ["marital", "वैवाहिक", "स्थिती"],
    "Proposer_Spouse_Name":    ["spouse", "पत्नी", "पति"],
    "Proposer_DOB":            ["date of birth", "जन्म दिनांक", "जन्म तारीख", "dob"],
    "Proposer_Age":            ["age", "वय", "आयु"],
    "Proposer_Birth_Place":    ["place of birth", "city of birth", "जन्म स्थळ", "जन्म स्थान"],
    "Proposer_Nationality":    ["nationality", "राष्ट्रीयता"],
    "Proposer_Citizenship":    ["citizenship", "नागरिकता"],
    "Proposer_Customer_ID":    ["customer id", "संकेतांक", "customer code"],
    "Proposer_CKYC":           ["ckyc", "central kyc"],
    "Proposer_Address":        ["permanent address", "स्थई पत्ता", "स्थायी पता"],
    "Proposer_House_No":       ["house no", "building", "घर क्र", "मकान नं"],
    "Proposer_Street":         ["street", "पथ", "सडक"],
    "Proposer_City":           ["city", "district", "शहर", "जिल्हा", "जिला"],
    "Proposer_State":          ["state", "country", "राज्य"],
    "Proposer_PIN":            ["pin code", "pin", "पीन कोड", "पिन कोड"],
    "Proposer_Phone":          ["tel", "telephone", "दूरध्वनी", "दूरभाष", "std code"],
    "Proposer_Email":          ["email", "ईमेल"],
    "Proposer_Aadhaar":        ["aadhaar", "aadhar", "आधार"],
    "Proposer_PAN":            ["pan", "पैन"],
    "Proposer_Nominee_Name":   ["nominee", "नामित"],
    "Proposer_Sum_Assured":    ["sum assured", "विमा रक्कम", "बीमा राशि"],
    "Proposer_Plan_Name":      ["plan", "योजना", "plan name"],
    "Proposer_Policy_Term":    ["policy term", "पॉलिसी मुदत"],
    "Proposer_Premium_Mode":   ["premium mode", "हप्ता पद्धत"],
    "Proposer_Annual_Premium": ["annual premium", "वार्षिक हप्ता"],
}

# Also collect ALL label patterns for label detection
ALL_LABEL_PATTERNS = set()
for patterns in FIELD_LABELS.values():
    ALL_LABEL_PATTERNS.update([p.lower() for p in patterns])

# Common non-value patterns to skip
SKIP_PATTERNS = re.compile(
    r'^[\d]{1,2}\.$|'      # Just field numbers like "3.", "14."
    r'^[\*]+$|'             # Just asterisks
    r'^[\-/\\/\s]+$|'       # Just slashes/dashes
    r'^section|^\d+\)$'     # Section headers
)


class SpatialKVEngine:
    """
    AI-Driven Spatial Reasoning Engine.
    Finds labels → pairs with nearest value using multi-directional proximity scoring.
    Template-agnostic. No hardcoded coordinates.
    """

    def __init__(self, page_height: int, page_width: int):
        self.h = page_height
        self.w = page_width
        self.max_search_x = 0.5 * page_width
        self.max_search_y = 0.15 * page_height

    def identify_labels(self, regions: List[OCRRegion]) -> List[OCRRegion]:
        """Mark regions that are labels based on text patterns."""
        for r in regions:
            text_lower = r.text.lower().strip()
            if any(p in text_lower for p in ALL_LABEL_PATTERNS):
                r.is_label = True
        return regions

    def match_field_to_label(self, label_region: OCRRegion, field_name: str) -> Optional[str]:
        """Map a label OCR region to a canonical field name."""
        text_lower = label_region.text.lower().strip()
        patterns = FIELD_LABELS.get(field_name, [])
        for p in patterns:
            if p in text_lower:
                return field_name
        return None

    def find_best_value(self, label: OCRRegion, candidates: List[OCRRegion]) -> Tuple[Optional[OCRRegion], str]:
        """
        Multi-directional proximity search.
        Searches RIGHT, BELOW, BELOW-RIGHT, and INSIDE.
        Returns (best_candidate, direction).
        """
        scored = []

        for cand in candidates:
            if cand.is_label:
                continue
            if cand.matched_field is not None:
                continue
            text = cand.text.strip()
            if not text or len(text) <= 1:
                continue
            if SKIP_PATTERNS.match(text):
                continue
            if cand.confidence < 0.3:
                continue

            # Calculate spatial distances
            h_dist, v_dist, direction = self._spatial_distance(label, cand)

            # Skip if too far
            if h_dist > self.max_search_x and v_dist > self.max_search_y:
                continue

            # Compute composite score (higher = better match)
            euclidean = (h_dist ** 2 + v_dist ** 2) ** 0.5 + 1
            direction_bonus = 0.0
            if direction == 'right':
                direction_bonus = 0.30
            elif direction == 'below_right':
                direction_bonus = 0.20
            elif direction == 'below':
                direction_bonus = 0.10
            elif direction == 'inside':
                direction_bonus = 0.35

            # Confidence bonus
            conf_bonus = cand.confidence * 0.25

            # Text quality: prefer longer, more meaningful text
            text_bonus = min(len(text) / 20.0, 0.15)

            # X-alignment bonus for below/below-right (same column)
            x_alignment = 1.0 - min(abs(label.cx - cand.cx) / self.w, 1.0)
            align_bonus = x_alignment * 0.10 if direction in ('below', 'below_right') else 0.0

            score = (1.0 / euclidean) + direction_bonus + conf_bonus + text_bonus + align_bonus
            scored.append((score, cand, direction))

        if not scored:
            return None, ""

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1], scored[0][2]

    def _spatial_distance(self, label: OCRRegion, candidate: OCRRegion) -> Tuple[float, float, str]:
        """
        Compute horizontal and vertical distance from label to candidate.
        Returns (h_dist, v_dist, direction).
        """
        # Horizontal: from label's right edge to candidate's left edge (for right-side)
        h_right = candidate.bbox[0] - label.right
        # Horizontal: from label's left edge to candidate's left edge (for below)
        h_same_col = candidate.cx - label.cx
        # Vertical: from label's bottom to candidate's top
        v_below = candidate.bbox[1] - label.bottom
        # Vertical: overlap (same row)
        v_overlap = max(0, min(label.bbox[3], candidate.bbox[3]) - max(label.bbox[1], candidate.bbox[1]))

        # Check INSIDE first
        if (label.bbox[0] <= candidate.bbox[0] and label.bbox[1] <= candidate.bbox[1] and
            label.bbox[2] >= candidate.bbox[2] and label.bbox[3] >= candidate.bbox[3]):
            return 0.0, 0.0, 'inside'

        # RIGHT: candidate is to the right, same-ish row
        if h_right > -20 and v_overlap > label.height * 0.3:
            return max(0, h_right), max(0, -v_overlap), 'right'

        # BELOW-RIGHT: candidate is below and to the right
        if v_below > 0 and h_right > -50:
            return max(0, h_right), v_below, 'below_right'

        # BELOW: candidate is below, same column
        if v_below > 0:
            return abs(h_same_col), v_below, 'below'

        # Default: compute raw euclidean direction
        dx = candidate.cx - label.cx
        dy = candidate.cy - label.cy
        if dx > 0 and abs(dy) < label.height:
            return abs(dx), abs(dy), 'right'
        elif dy > 0:
            return abs(dx), dy, 'below_right'
        else:
            return abs(dx), abs(dy), 'right'

    def extract_all_fields(self, regions: List[OCRRegion]) -> List[ExtractedField]:
        """Main entry: identify labels, find values, return extracted fields."""
        regions = self.identify_labels(regions)
        non_label = [r for r in regions if not r.is_label and r.confidence >= 0.3]

        fields = []
        matched_regions = set()

        for field_name, patterns in FIELD_LABELS.items():
            # Find all label regions matching this field
            label_matches = []
            for r in regions:
                if r.is_label:
                    text_lower = r.text.lower().strip()
                    if any(p in text_lower for p in patterns):
                        label_matches.append(r)

            if not label_matches:
                fields.append(ExtractedField(
                    field_name=field_name, value="", confidence=0.0,
                    source="not_found", page_num=1
                ))
                continue

            # Use the best (most specific) label match
            best_label = max(label_matches, key=lambda r: r.confidence * len(r.text))

            # Find best value candidate
            available = [r for r in non_label if id(r) not in matched_regions]
            best_value, direction = self.find_best_value(best_label, available)

            if best_value:
                best_value.matched_field = field_name
                matched_regions.add(id(best_value))
                fields.append(ExtractedField(
                    field_name=field_name,
                    value=best_value.text,
                    confidence=best_value.confidence,
                    source=f"paddle_ocr_{best_value.lang}",
                    source_bbox=best_value.bbox,
                    label_bbox=best_label.bbox,
                    direction=direction,
                ))
            else:
                fields.append(ExtractedField(
                    field_name=field_name, value="", confidence=0.0,
                    source="not_found", label_bbox=best_label.bbox,
                ))

        return fields


# ──────────────────────────────────────────────────────────────────
# Layer 5: Qwen2.5-VL Fallback
# ──────────────────────────────────────────────────────────────────

class QwenVLFallback:
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
            logger.info("Qwen2.5-VL-7B loaded for fallback")
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
                    {"type": "text", "text": f"Extract the handwritten/printed value for '{field_name}'. Return only the value, nothing else."}
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

    def extract_full_page(self, img: np.ndarray, fields: List[str]) -> Optional[Dict[str, str]]:
        """VLM full-page extraction as last resort."""
        if not self._loaded:
            self.load()
        if not self._loaded:
            return None
        try:
            from PIL import Image
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            field_list = ", ".join(fields)
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": f"Extract all fields [{field_list}] from this LIC form image as JSON. Return only JSON."}
                ]}
            ]
            text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text_input], images=[pil_img], return_tensors="pt").to(self.model.device)
            output_ids = self.model.generate(**inputs, max_new_tokens=500)
            decoded = self.processor.batch_decode(output_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
            try:
                return json.loads(decoded)
            except json.JSONDecodeError:
                return {"raw_vlm_output": decoded}
        except Exception as e:
            logger.debug(f"Qwen full-page failed: {e}")
            return None


# ──────────────────────────────────────────────────────────────────
# Main Pipeline Orchestrator
# ──────────────────────────────────────────────────────────────────

class UltimatePipeline:
    """
    Orchestrates: Preprocess → PP-Structure → PaddleOCR → Spatial KV → Qwen Fallback
    """

    def __init__(self, confidence_threshold: float = 0.85, use_qwen: bool = False, use_ppstructure: bool = True):
        self.confidence_threshold = confidence_threshold
        self.use_qwen = use_qwen
        self.use_ppstructure = use_ppstructure
        self.preprocessor = AdvancedPreprocessor()
        self.ppstructure = PPStructureAnalyzer() if use_ppstructure else None
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
        low_conf = sum(1 for f in all_fields if f.confidence < self.confidence_threshold and f.value)
        not_found = sum(1 for f in all_fields if not f.value)
        corrected = sum(1 for f in all_fields if f.corrected)
        found = sum(1 for f in all_fields if f.value)

        stats = {
            "total_fields": len(all_fields),
            "found": found,
            "high_confidence": high_conf,
            "low_confidence": low_conf,
            "not_found": not_found,
            "corrected_by_qwen": corrected,
            "accuracy_estimate": round(found / max(len(all_fields), 1) * 100, 1),
            "processing_time_sec": round(elapsed, 2),
            "time_per_page": round(elapsed / max(len(pages), 1), 2),
        }

        return {
            "pdf_path": pdf_path,
            "pages_processed": len(pages),
            "fields": [asdict(f) for f in all_fields],
            "statistics": stats,
        }

    def _process_page(self, img: np.ndarray, page_num: int) -> List[ExtractedField]:
        h, w = img.shape[:2]

        # Step 1: Preprocess
        try:
            img = self.preprocessor.preprocess(img)
        except Exception as e:
            logger.warning(f"Preprocess failed: {e}")

        # Step 2: PP-Structure (optional, for layout context)
        layout_regions = []
        if self.ppstructure:
            layout_regions = self.ppstructure.analyze(img)
            logger.info(f"  PP-Structure: {len(layout_regions)} layout regions")

        # Step 3: PaddleOCR Bilingual
        ocr_regions = self.ocr_engine.ocr_bilingual(img)
        logger.info(f"  PaddleOCR: {len(ocr_regions)} text regions")

        # Deduplicate overlapping regions (same text, similar position)
        ocr_regions = self._deduplicate(ocr_regions)
        logger.info(f"  After dedup: {len(ocr_regions)} regions")

        # Step 4: Spatial KV Reasoning
        kv_engine = SpatialKVEngine(h, w)
        fields = kv_engine.extract_all_fields(ocr_regions)

        # Step 5: Qwen fallback for low-confidence fields
        if self.qwen:
            for i, f in enumerate(fields):
                if f.confidence < self.confidence_threshold and f.source_bbox:
                    qwen_val = self.qwen.extract_field(img, f.source_bbox, f.field_name)
                    if qwen_val:
                        f.value = qwen_val
                        f.confidence = 0.7
                        f.source = "qwen_vl"
                        f.corrected = True

        # Set page number
        for f in fields:
            f.page_num = page_num

        return fields

    def _deduplicate(self, regions: List[OCRRegion]) -> List[OCRRegion]:
        """Remove duplicate detections (same text at similar position from different languages)."""
        seen = []
        out = []
        for r in regions:
            key = (r.text.strip().lower()[:30], r.cx // 20, r.cy // 20)
            if key not in seen:
                seen.append(key)
                out.append(r)
        return out


# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ultimate LIC Extraction Pipeline v2")
    parser.add_argument("pdf_path", help="Path to PDF")
    parser.add_argument("--output", default="data/output/pipeline_v2_result.json")
    parser.add_argument("--confidence", type=float, default=0.85)
    parser.add_argument("--use-qwen", action="store_true", help="Enable Qwen2.5-VL fallback")
    parser.add_argument("--no-ppstructure", action="store_true", help="Disable PP-Structure")
    parser.add_argument("--pages", type=int, nargs="+")
    args = parser.parse_args()

    pipeline = UltimatePipeline(
        confidence_threshold=args.confidence,
        use_qwen=args.use_qwen,
        use_ppstructure=not args.no_ppstructure,
    )
    result = pipeline.process_pdf(args.pdf_path, args.pages)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    stats = result["statistics"]
    print("\n" + "=" * 80)
    print("EXTRACTION RESULTS — AI-Driven Spatial Reasoning")
    print("=" * 80)
    for f in result["fields"]:
        if f["value"]:
            icon = "[OK]" if f["confidence"] >= args.confidence else "[!!]"
        else:
            icon = "[--]"
        val = f["value"][:50] if f["value"] else "(not found)"
        dir_str = f" ({f['direction']})" if f.get("direction") else ""
        print(f"  {icon} {f['field_name']:<30} {val:<52} conf={f['confidence']:.2f}{dir_str}")
    print("=" * 80)
    print(f"  Total: {stats['total_fields']} | Found: {stats['found']} | High: {stats['high_confidence']} | "
          f"Low: {stats['low_confidence']} | Missing: {stats['not_found']}")
    print(f"  Accuracy: {stats['accuracy_estimate']}% | Time: {stats['processing_time_sec']}s | "
          f"Per page: {stats['time_per_page']}s")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
