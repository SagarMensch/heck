"""PaddleOCR-First Field Extractor
====================================
PRIMARY: PaddleOCR (Hindi+English) on full page → proximity-match to canonical bboxes
FALLBACK: Qwen-VL with CoT + self-consistency for low-confidence fields
CONSENSUS: Dual-model agreement scoring for confidence calibration

Replaces TrOCR as primary extractor. TrOCR was trained on Western handwriting,
produces gibberish on Hindi handwritten text. PaddleOCR devanagari model
handles Hindi handwriting natively.

Speed: ~0.5-1s/page PaddleOCR, ~5s/page Qwen fallback (only for missed fields)
Target: 50 docs × 30 pages in <1hr
"""

import re
import json
import time
import logging
from collections import Counter
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
from PIL import Image

from src.config import (
    QWEN_MODEL_ID, DEVICE, TROCR_FALLBACK_THRESHOLD,
    FIELD_FAMILY_EXTRACTOR, REVERSE_FIELD_NAME_MAP,
    FORM_300_FIELDS, FIELD_NAMES,
)
from src.form300_templates import (
    FORM300_PAGE_TEMPLATES, FORM300_FIELD_INDEX,
    PAGES_WITH_TARGET_FIELDS, PAGE_TYPE_TO_PAGE_NUM,
    bbox_to_pixels, FieldTemplate,
)

logger = logging.getLogger(__name__)

PAGE_NUM_TO_TYPE = {v: k for k, v in PAGE_TYPE_TO_PAGE_NUM.items()}


class PaddleOCRFieldExtractor:
    """Primary extractor: PaddleOCR on full page, proximity-match to canonical bboxes."""

    def __init__(self):
        self._engine_hi = None
        self._engine_en = None

    def _get_hindi(self):
        if self._engine_hi is None:
            import os
            os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "1"
            from paddleocr import PaddleOCR
            self._engine_hi = PaddleOCR(lang="hi", use_textline_orientation=True)
            logger.info("PaddleOCR Hindi engine loaded (GPU)")
        return self._engine_hi

    def _get_english(self):
        if self._engine_en is None:
            import os
            os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "1"
            from paddleocr import PaddleOCR
            self._engine_en = PaddleOCR(lang="en", use_textline_orientation=True)
            logger.info("PaddleOCR English engine loaded (GPU)")
        return self._engine_en

    def _ocr_page(self, np_image: np.ndarray, lang: str = "hi") -> List[Dict]:
        engine = self._get_hindi() if lang == "hi" else self._get_english()
        try:
            result = engine.ocr(np_image)
            return self._parse_paddle_result(result, lang)
        except Exception as e:
            logger.error(f"PaddleOCR {lang} failed: {e}")
            return []

    def _ocr_page_bilingual(self, np_image: np.ndarray) -> List[Dict]:
        hi_regions = self._ocr_page(np_image, "hi")
        en_regions = self._ocr_page(np_image, "en")
        seen = set()
        merged = []
        for r in hi_regions + en_regions:
            key = (r["text"].strip().lower()[:30],
                   tuple(r["bbox"]))
            if key not in seen:
                seen.add(key)
                merged.append(r)
        return merged

    def _parse_paddle_result(self, result, lang: str) -> List[Dict]:
        regions = []
        if not result or not result[0]:
            return regions

        res0 = result[0]

        texts = res0.get("rec_texts") if hasattr(res0, "get") else None
        scores = res0.get("rec_scores") if hasattr(res0, "get") else None
        polys = res0.get("rec_polys") if hasattr(res0, "get") else None

        if texts is not None and scores is not None and polys is not None:
            if isinstance(texts, list):
                for text, score, poly in zip(texts, scores, polys):
                    if not isinstance(text, str):
                        text = str(text)
                    try:
                        confidence = float(score)
                    except (ValueError, TypeError):
                        confidence = 0.0
                    if isinstance(poly, (list, tuple, np.ndarray)):
                        pts = np.asarray(poly)
                        if pts.ndim == 2 and pts.shape[0] >= 2 and pts.shape[1] == 2:
                            xs, ys = pts[:, 0], pts[:, 1]
                            regions.append({
                                "bbox": [int(np.min(xs)), int(np.min(ys)),
                                         int(np.max(xs)), int(np.max(ys))],
                                "text": text.strip(),
                                "confidence": confidence,
                                "lang": lang,
                            })
                return regions

        if isinstance(result, list) and result and isinstance(result[0], list):
            for line in result[0]:
                if isinstance(line, (list, tuple)) and len(line) >= 2:
                    bbox_data = line[0]
                    second = line[1]
                    if isinstance(second, (list, tuple)) and len(second) == 2:
                        text, confidence = second
                    elif isinstance(second, str):
                        text, confidence = second, 1.0
                    else:
                        continue
                    if isinstance(bbox_data, (list, tuple)) and len(bbox_data) == 4:
                        xs = [p[0] for p in bbox_data]
                        ys = [p[1] for p in bbox_data]
                        regions.append({
                            "bbox": [min(xs), min(ys), max(xs), max(ys)],
                            "text": str(text).strip(),
                            "confidence": float(confidence),
                            "lang": lang,
                        })

        return regions

    def extract_page(self, page_image: np.ndarray, page_num: int) -> Dict[str, Dict]:
        if page_num not in PAGE_NUM_TO_TYPE:
            return {}

        page_type = PAGE_NUM_TO_TYPE[page_num]
        if page_type not in FORM300_PAGE_TEMPLATES:
            return {}

        template = FORM300_PAGE_TEMPLATES[page_type]
        h_img, w_img = page_image.shape[:2]

        ocr_regions = self._ocr_page_bilingual(page_image)
        logger.info(f"Page {page_num} ({page_type}): {len(ocr_regions)} OCR regions, {len(template.fields)} template fields")

        fields = {}
        for field in template.fields:
            fx1, fy1, fx2, fy2 = bbox_to_pixels(field.bbox_norm, w_img, h_img)
            f_cx = (fx1 + fx2) / 2
            f_cy = (fy1 + fy2) / 2
            f_w = fx2 - fx1
            f_h = fy2 - fy1

            best_match = None
            best_dist = float("inf")
            best_iou = 0.0

            for region in ocr_regions:
                rx1, ry1, rx2, ry2 = region["bbox"]
                r_cx = (rx1 + rx2) / 2
                r_cy = (ry1 + ry2) / 2

                dist = np.sqrt(((r_cx - f_cx) / max(w_img, 1)) ** 2 +
                               ((r_cy - f_cy) / max(h_img, 1)) ** 2)

                ix1 = max(fx1, rx1)
                iy1 = max(fy1, ry1)
                ix2 = min(fx2, rx2)
                iy2 = min(fy2, ry2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                r_area = max(1, (rx2 - rx1) * (ry2 - ry1))
                f_area = max(1, f_w * f_h)
                iou = inter / max(1, min(f_area, r_area))

                overlap_ratio = inter / r_area if r_area > 0 else 0

                if overlap_ratio > 0.4:
                    score = dist - iou * 0.5
                    if score < best_dist:
                        best_dist = score
                        best_match = region
                        best_iou = overlap_ratio

                elif dist < 0.05:
                    score = dist
                    if score < best_dist:
                        best_dist = score
                        best_match = region
                        best_iou = overlap_ratio

            if best_match and (best_iou > 0.3 or best_dist < 0.05):
                value = best_match["text"]
                confidence = best_match["confidence"]

                value = self._postprocess_field(value, field)
                confidence = self._calibrate_confidence(value, field, confidence)

                fields[field.name] = {
                    "value": value,
                    "confidence": confidence,
                    "source": f"PaddleOCR-{best_match['lang']}",
                    "field_family": field.family,
                    "page_num": page_num,
                    "ocr_bbox": best_match["bbox"],
                    "template_bbox_norm": list(field.bbox_norm),
                }
            else:
                fields[field.name] = {
                    "value": "",
                    "confidence": 0.0,
                    "source": "none",
                    "field_family": field.family,
                    "page_num": page_num,
                }

        return fields

    def _postprocess_field(self, value: str, field: FieldTemplate) -> str:
        val = value.strip()

        if field.family == "numeric":
            digits = re.sub(r'[^\d]', '', val)
            if digits:
                return digits
        elif field.family == "amount":
            digits = re.sub(r'[^\d]', '', val)
            if digits:
                return digits
        elif field.family == "date":
            val = val.replace(" ", "")
            date_pat = re.search(r'(\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4})', val)
            if date_pat:
                return date_pat.group(1).replace(".", "/").replace("-", "/")
        elif field.family == "short_id":
            if field.name == "pan_number":
                val = re.sub(r'[^A-Za-z0-9]', '', val).upper()
            elif field.name in ("aadhaar_last_or_id_number",):
                val = re.sub(r'[^0-9]', '', val)
            elif field.name == "mobile_number":
                val = re.sub(r'[^0-9]', '', val)
                if len(val) == 12 and val.startswith("91"):
                    val = val[2:]
                elif len(val) == 11 and val.startswith("0"):
                    val = val[1:]
            elif field.name == "email":
                val = val.strip()
            elif "customer_id" in field.name or "ckyc" in field.name:
                val = re.sub(r'[^A-Za-z0-9]', '', val)
            elif field.name == "agent_code":
                val = re.sub(r'[^A-Za-z0-9]', '', val)
            elif field.name == "branch_code":
                val = re.sub(r'[^A-Za-z0-9]', '', val)
        elif field.family == "binary_mark":
            val_lower = val.lower().strip()
            if any(k in val_lower for k in ["yes", "ha", "हाँ", "✓", "tick", "checked"]):
                return "YES"
            elif any(k in val_lower for k in ["no", "nahi", "नहीं", "unchecked"]):
                return "NO"
        elif field.family == "signature_presence" or field.family == "photo_presence":
            val_lower = val.lower().strip()
            if val_lower:
                return "PRESENT"

        return val

    def _calibrate_confidence(self, value: str, field: FieldTemplate, ocr_conf: float) -> float:
        conf = ocr_conf

        if not value or len(value) < 1:
            return 0.0

        if field.family == "numeric" and not re.search(r'\d', value):
            return max(0.1, conf * 0.3)

        if field.family == "amount" and not re.search(r'\d', value):
            return max(0.1, conf * 0.3)

        if field.name == "pan_number" and not re.match(r'^[A-Z]{5}\d{4}[A-Z]$', value):
            conf *= 0.7

        if "mobile" in field.name and (len(value) != 10 or value[0] not in "6789"):
            conf *= 0.6

        return min(1.0, max(0.0, conf))


FIELD_CONSTRAINTS = {
    "pan_number": {
        "format_hint": "5 uppercase letters + 4 digits + 1 uppercase letter (e.g. ABCDE1234F)",
        "regex": "^[A-Z]{5}\\d{4}[A-Z]$",
        "example": "ABCDE1234F",
    },
    "aadhaar_last_or_id_number": {
        "format_hint": "12 digits, no spaces",
        "regex": "^\\d{12}$",
        "example": "123456789012",
    },
    "mobile_number": {
        "format_hint": "10 digit Indian mobile starting with 6-9",
        "regex": "^[6-9]\\d{9}$",
        "example": "9876543210",
    },
    "address_pincode": {
        "format_hint": "6 digit Indian pincode",
        "regex": "^\\d{6}$",
        "example": "400001",
    },
    "date_of_birth": {
        "format_hint": "Date in DD/MM/YYYY format",
        "regex": "^\\d{2}/\\d{2}/\\d{4}$",
        "example": "15/06/1985",
    },
    "annual_income": {
        "format_hint": "Numeric amount in rupees, no symbols",
        "regex": "^\\d+$",
        "example": "500000",
    },
    "proposed_sum_assured": {
        "format_hint": "Numeric amount in rupees",
        "regex": "^\\d+$",
        "example": "1000000",
    },
    "proposed_premium_amount": {
        "format_hint": "Numeric amount in rupees",
        "regex": "^\\d+$",
        "example": "50000",
    },
}

COT_FIELD_PROMPT_TEMPLATE = (
    "You are reading a handwritten LIC insurance form field.\n\n"
    "Field: {field_name}\n"
    "Field type: {field_family}\n"
    "Format: {format_hint}\n"
    "Example: {example}\n\n"
    "Step 1: Look at the image carefully. Describe what you see in the handwriting area.\n"
    "Step 2: Read the handwritten text literally, character by character.\n"
    "Step 3: Apply the format constraint. The output MUST match: {regex}\n"
    "Step 4: Output the final value.\n\n"
    "Respond with ONLY this JSON:\n"
    '{{"observation": "what you see", "raw_read": "literal reading", "value": "final formatted value", "confidence": 0.95}}'
)

COT_GENERIC_PROMPT_TEMPLATE = (
    "You are reading a handwritten LIC insurance form field.\n\n"
    "Field: {field_name}\n"
    "Field type: {field_family}\n\n"
    "Step 1: Look at the image carefully. Describe what handwritten text you see.\n"
    "Step 2: Read the handwritten text literally.\n"
    "Step 3: Output the value. If Hindi, transliterate to English.\n\n"
    "Respond with ONLY this JSON:\n"
    '{{"observation": "what you see", "raw_read": "literal reading", "value": "extracted value", "confidence": 0.95}}'
)


class QwenCoTExtractor:
    """Qwen-VL with Chain-of-Thought prompting and self-consistency voting."""

    def __init__(self, model_manager=None):
        self.manager = model_manager
        self._qwen_model = None
        self._qwen_processor = None

    def _load_qwen(self):
        if self._qwen_model is not None:
            return self._qwen_model, self._qwen_processor

        import torch
        import gc

        if self.manager is not None:
            model, processor = self.manager.get_qwen()
            self._qwen_model = model
            self._qwen_processor = processor
            return model, processor

        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        logger.info(f"Loading {QWEN_MODEL_ID}...")
        self._qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            QWEN_MODEL_ID,
            dtype=torch.float16,
            device_map="auto",
            attn_implementation="eager",
        )
        self._qwen_processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID)
        return self._qwen_model, self._qwen_processor

    def extract_field_cot(self, page_image: Image.Image, field_name: str,
                          field_family: str, num_votes: int = 3) -> Dict:
        model, processor = self._load_qwen()

        constraint = FIELD_CONSTRAINTS.get(field_name)
        if constraint:
            prompt = COT_FIELD_PROMPT_TEMPLATE.format(
                field_name=field_name,
                field_family=field_family,
                format_hint=constraint["format_hint"],
                example=constraint["example"],
                regex=constraint["regex"],
            )
        else:
            prompt = COT_GENERIC_PROMPT_TEMPLATE.format(
                field_name=field_name,
                field_family=field_family,
            )

        messages = [
            {"role": "system", "content": "You are an expert Indian handwritten form reader. Always think step by step. Output only valid JSON."},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": page_image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)

        import torch
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(model.device)

        votes = []
        for vote_idx in range(num_votes):
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.3 if num_votes > 1 else 0.0,
                    do_sample=num_votes > 1,
                    num_beams=1,
                )

            generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
            output_text = processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()

            parsed = self._parse_cot_json(output_text)
            if parsed and parsed.get("value"):
                votes.append(parsed)

        if not votes:
            return {"value": "", "confidence": 0.0, "source": "Qwen-CoT", "votes": 0}

        return self._majority_vote(votes, field_name, field_family)

    def _parse_cot_json(self, text: str) -> Optional[Dict]:
        try:
            json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                return {
                    "observation": parsed.get("observation", ""),
                    "raw_read": parsed.get("raw_read", ""),
                    "value": str(parsed.get("value", "")).strip(),
                    "confidence": float(parsed.get("confidence", 0.5)),
                }
        except (json.JSONDecodeError, ValueError):
            pass

        value_match = re.search(r'"value"\s*:\s*"([^"]*)"', text)
        if value_match:
            return {
                "observation": "",
                "raw_read": "",
                "value": value_match.group(1).strip(),
                "confidence": 0.4,
            }

        return None

    def _majority_vote(self, votes: List[Dict], field_name: str,
                       field_family: str) -> Dict:
        values = [v["value"] for v in votes if v["value"]]
        if not values:
            return {"value": "", "confidence": 0.0, "source": "Qwen-CoT", "votes": len(votes)}

        counter = Counter(values)
        most_common_val, count = counter.most_common(1)[0]
        agreement_ratio = count / len(values)

        avg_conf = np.mean([v["confidence"] for v in votes if v["value"] == most_common_val])

        final_conf = min(1.0, avg_conf * (0.7 + 0.3 * agreement_ratio))

        if agreement_ratio < 0.5:
            final_conf *= 0.5

        return {
            "value": most_common_val,
            "confidence": final_conf,
            "source": f"Qwen-CoT-vote{len(votes)}",
            "agreement": agreement_ratio,
            "votes": len(votes),
            "all_values": values,
        }


class DualModelConsensus:
    """Score agreement between PaddleOCR and Qwen for confidence calibration."""

    def compute_consensus(self, paddle_result: Dict, qwen_result: Dict,
                          field_name: str) -> Dict:
        p_val = str(paddle_result.get("value", "")).strip().lower()
        q_val = str(qwen_result.get("value", "")).strip().lower()
        p_conf = paddle_result.get("confidence", 0.0)
        q_conf = qwen_result.get("confidence", 0.0)

        if not p_val and not q_val:
            return {
                "value": "",
                "confidence": 0.0,
                "consensus": "both_empty",
                "source": "consensus",
            }

        if not p_val:
            return {
                "value": qwen_result.get("value", ""),
                "confidence": q_conf * 0.85,
                "consensus": "qwen_only",
                "source": "Qwen-Only",
            }

        if not q_val:
            return {
                "value": paddle_result.get("value", ""),
                "confidence": p_conf,
                "consensus": "paddle_only",
                "source": "PaddleOCR-Only",
            }

        if p_val == q_val:
            return {
                "value": paddle_result.get("value", ""),
                "confidence": min(1.0, max(p_conf, q_conf) * 1.1),
                "consensus": "full_agreement",
                "source": "Consensus-Both",
            }

        if p_val.replace(" ", "") == q_val.replace(" ", ""):
            return {
                "value": paddle_result.get("value", ""),
                "confidence": max(p_conf, q_conf) * 1.05,
                "consensus": "agree_no_space",
                "source": "Consensus-Both",
            }

        p_digits = re.sub(r'\D', '', p_val)
        q_digits = re.sub(r'\D', '', q_val)
        if p_digits and q_digits and p_digits == q_digits:
            return {
                "value": p_digits if field_name in (
                    "mobile_number", "aadhaar_last_or_id_number",
                    "address_pincode", "annual_income",
                    "proposed_sum_assured", "proposed_premium_amount",
                ) else paddle_result.get("value", ""),
                "confidence": max(p_conf, q_conf) * 0.95,
                "consensus": "numeric_agree",
                "source": "Consensus-Numeric",
            }

        lev_sim = self._levenshtein_similarity(p_val, q_val)
        if lev_sim > 0.7:
            best_val = p_val if p_conf >= q_conf else q_val
            return {
                "value": best_val,
                "confidence": max(p_conf, q_conf) * (0.5 + 0.5 * lev_sim),
                "consensus": f"partial_agree_{lev_sim:.2f}",
                "source": "Consensus-Partial",
            }

        return {
            "value": p_val if p_conf >= q_conf else q_val,
            "confidence": max(p_conf, q_conf) * 0.6,
            "consensus": f"disagree_lev{lev_sim:.2f}",
            "source": "Consensus-Disagree",
            "paddle_value": paddle_result.get("value", ""),
            "qwen_value": qwen_result.get("value", ""),
        }

    @staticmethod
    def _levenshtein_similarity(s1: str, s2: str) -> float:
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0
        m, n = len(s1), len(s2)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, n + 1):
                temp = dp[j]
                if s1[i - 1] == s2[j - 1]:
                    dp[j] = prev
                else:
                    dp[j] = 1 + min(prev, dp[j], dp[j - 1])
                prev = temp
        return 1.0 - dp[n] / max(m, n)


class HybridExtractor:
    """Full hybrid pipeline: PaddleOCR primary → Qwen CoT fallback → Consensus.

    Phase 1: PaddleOCR (Hindi+English) on each target page
    Phase 2: Qwen CoT + self-consistency on low-confidence/empty fields
    Phase 3: Dual-model consensus for confidence calibration
    """

    def __init__(self, qwen_model_manager=None, use_consensus: bool = True,
                 qwen_votes: int = 3, qwen_fallback_threshold: float = 0.70,
                 use_qwen_fallback: bool = True):
        self.paddle = PaddleOCRFieldExtractor()
        self.qwen = QwenCoTExtractor(model_manager=qwen_model_manager) if use_qwen_fallback else None
        self.consensus = DualModelConsensus() if use_consensus else None
        self.use_consensus = use_consensus
        self.use_qwen_fallback = use_qwen_fallback
        self.qwen_votes = qwen_votes
        self.fallback_threshold = qwen_fallback_threshold

    def extract(self, page_images: List[np.ndarray],
                page_num_map: List[int] = None) -> Dict:
        all_fields = {}
        low_conf_fields = []
        page_image_cache = {}

        if page_num_map is None:
            page_num_map = list(range(1, len(page_images) + 1))

        t0 = time.time()

        for i, (np_img, page_num) in enumerate(zip(page_images, page_num_map)):
            if page_num not in PAGES_WITH_TARGET_FIELDS:
                continue

            if len(np_img.shape) == 3 and np_img.shape[2] == 3:
                pass
            elif len(np_img.shape) == 2:
                np_img = np.stack([np_img] * 3, axis=-1)

            page_fields = self.paddle.extract_page(np_img, page_num)

            for field_name, field_data in page_fields.items():
                field_data["selected_model"] = "PaddleOCR"
                all_fields[field_name] = field_data

                conf = field_data.get("confidence", 0.0)
                val = field_data.get("value", "")
                family = field_data.get("field_family", "text")

                if conf < self.fallback_threshold or not val.strip() or len(val.strip()) < 2:
                    if family not in ("binary_mark", "signature_presence", "photo_presence"):
                        low_conf_fields.append((field_name, page_num, family))

                page_image_cache[page_num] = np_img

        paddle_time = time.time() - t0
        logger.info(f"Phase 1 (PaddleOCR): {len(all_fields)} fields in {paddle_time:.1f}s, "
                     f"{len(low_conf_fields)} need Qwen fallback")

        if low_conf_fields and self.use_qwen_fallback and self.qwen is not None:
            t1 = time.time()
            logger.info(f"Phase 2 (Qwen CoT): Processing {len(low_conf_fields)} low-confidence fields...")

            fallback_pages_needed = set(f[1] for f in low_conf_fields)
            pil_cache = {}
            for pg in fallback_pages_needed:
                if pg in page_image_cache:
                    pil_cache[pg] = Image.fromarray(page_image_cache[pg])

            for field_name, page_num, family in low_conf_fields:
                pil_img = pil_cache.get(page_num)
                if pil_img is None:
                    continue

                qwen_result = self.qwen.extract_field_cot(
                    pil_img, field_name, family, num_votes=self.qwen_votes
                )

                paddle_result = all_fields.get(field_name, {})

                if self.use_consensus and paddle_result.get("value"):
                    consensus_result = self.consensus.compute_consensus(
                        paddle_result, qwen_result, field_name
                    )
                    all_fields[field_name] = {
                        **all_fields[field_name],
                        "value": consensus_result["value"],
                        "confidence": consensus_result["confidence"],
                        "consensus": consensus_result.get("consensus", ""),
                        "selected_model": consensus_result.get("source", "Qwen-CoT"),
                        "qwen_value": qwen_result.get("value", ""),
                        "paddle_value": paddle_result.get("value", ""),
                    }
                elif qwen_result.get("value"):
                    all_fields[field_name] = {
                        **all_fields[field_name],
                        "value": qwen_result["value"],
                        "confidence": max(0.5, qwen_result["confidence"]),
                        "selected_model": qwen_result.get("source", "Qwen-CoT"),
                        "qwen_votes": qwen_result.get("votes", 0),
                        "qwen_agreement": qwen_result.get("agreement", 0),
                    }

            qwen_time = time.time() - t1
            logger.info(f"Phase 2 (Qwen CoT): done in {qwen_time:.1f}s")

        mapped_fields = self._map_field_names(all_fields)

        total_expected = len(FIELD_NAMES)
        extracted = sum(1 for f in mapped_fields.values()
                       if f.get("value") and f.get("confidence", 0) > 0.3)
        missing = total_expected - extracted

        confidences = [f.get("confidence", 0) for f in mapped_fields.values() if f.get("value")]
        overall_conf = float(np.mean(confidences)) if confidences else 0.0

        return {
            "fields": mapped_fields,
            "raw_template_fields": all_fields,
            "extraction_summary": {
                "total_fields_expected": total_expected,
                "fields_extracted": extracted,
                "fields_missing": missing,
                "fields_paddle_primary": sum(1 for f in all_fields.values()
                                              if f.get("selected_model") == "PaddleOCR"),
                "fields_qwen_fallback": sum(1 for f in all_fields.values()
                                             if "Qwen" in f.get("selected_model", "")),
                "fields_consensus": sum(1 for f in all_fields.values()
                                         if "Consensus" in f.get("selected_model", "")),
                "overall_confidence": round(overall_conf, 4),
            },
            "models_used": ["PaddleOCR-GPU", "Qwen-VL-CoT"],
            "total_pages": len(page_images),
            "extraction_time_seconds": time.time() - t0,
        }

    def _map_field_names(self, template_fields: Dict) -> Dict:
        mapped = {}
        for template_name, field_data in template_fields.items():
            config_name = REVERSE_FIELD_NAME_MAP.get(template_name)
            if config_name:
                if config_name not in mapped or field_data.get("confidence", 0) > mapped[config_name].get("confidence", 0):
                    mapped[config_name] = {**field_data, "template_name": template_name}
            else:
                mapped[template_name] = field_data
        return mapped

    def cleanup(self):
        pass
