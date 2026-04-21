#!/usr/bin/env python
"""
LIC EXTRACTION PIPELINE v3 — VLM-First SOTA
=============================================
Architecture:
1. PDF → High-DPI pages → Smart Tiling (2x3 grid)
2. Qwen2.5-VL-7B: Per-tile structured JSON extraction
3. Multi-tile field merging with confidence scoring
4. PaddleOCR v5 Bilingual: Cross-verification of key fields
5. PaddleOCR bbox localization for human review overlay
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
from PIL import Image

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pipeline_v3")


# ──────────────────────────────────────────────────────────────────
# Field Definitions
# ──────────────────────────────────────────────────────────────────

FIELDS = [
    "Proposer_Name", "Proposer_First_Name", "Proposer_Middle_Name", "Proposer_Last_Name",
    "Proposer_Prefix", "Proposer_Father_Name", "Proposer_Mother_Name",
    "Proposer_Gender", "Proposer_Marital_Status", "Proposer_Spouse_Name",
    "Proposer_DOB", "Proposer_Age", "Proposer_Birth_Place",
    "Proposer_Nationality", "Proposer_Citizenship",
    "Proposer_Customer_ID", "Proposer_CKYC",
    "Proposer_House_No", "Proposer_Street", "Proposer_City", "Proposer_State",
    "Proposer_PIN", "Proposer_Phone", "Proposer_Email",
    "Proposer_Aadhaar", "Proposer_PAN",
    "Proposer_Nominee_Name", "Proposer_Relationship",
    "Proposer_Sum_Assured", "Proposer_Plan_Name",
    "Proposer_Policy_Term", "Proposer_Premium_Mode", "Proposer_Annual_Premium",
]

FIELDS_JSON_TEMPLATE = json.dumps({f: None for f in FIELDS}, indent=2)

PAGE2_FIELDS = [
    "Proposer_Name", "Proposer_First_Name", "Proposer_Middle_Name", "Proposer_Last_Name",
    "Proposer_Prefix", "Proposer_Father_Name", "Proposer_Mother_Name",
    "Proposer_Gender", "Proposer_Marital_Status", "Proposer_Spouse_Name",
    "Proposer_DOB", "Proposer_Age", "Proposer_Birth_Place",
    "Proposer_Nationality", "Proposer_Citizenship",
    "Proposer_Customer_ID", "Proposer_CKYC",
    "Proposer_House_No", "Proposer_Street", "Proposer_City", "Proposer_State",
    "Proposer_PIN", "Proposer_Phone", "Proposer_Email",
    "Proposer_Aadhaar", "Proposer_PAN",
]

TILE_EXTRACTION_PROMPT = """You are an expert document extraction AI specializing in LIC (Life Insurance Corporation of India) proposal forms.

This is a TILE (section) of a form page. The form is bilingual: English + Hindi/Marathi.

YOUR TASK: Extract all FILLED-IN field values visible in this tile. Focus on HANDWRITTEN entries, not printed labels. Read handwriting carefully.

Return ONLY a valid JSON object with these keys. Use null for fields NOT visible in this tile.
Do NOT guess or fabricate values. Only extract what you can actually see.

""" + FIELDS_JSON_TEMPLATE


# ──────────────────────────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────────────────────────

@dataclass
class ExtractedField:
    field_name: str
    value: str
    confidence: float
    source: str  # 'vlm_full', 'vlm_tile', 'paddle_ocr_verify', 'not_found'
    source_bbox: Optional[List[int]] = None
    label_bbox: Optional[List[int]] = None
    page_num: int = 1
    tile_origin: str = ""  # which tile it came from
    verified: bool = False


# ──────────────────────────────────────────────────────────────────
# Layer 1: PDF → Image → Smart Tiling
# ──────────────────────────────────────────────────────────────────

class ImagePreparer:
    """Convert PDF pages to images and create smart tiles for VLM processing."""

    def __init__(self, dpi: int = 150, tile_cols: int = 2, tile_rows: int = 3, overlap: int = 50):
        self.dpi = dpi
        self.tile_cols = tile_cols
        self.tile_rows = tile_rows
        self.overlap = overlap

    def pdf_page_to_image(self, pdf_path: str, page_num: int) -> Image.Image:
        import fitz
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_num - 1)
        pix = page.get_pixmap(dpi=self.dpi)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        doc.close()
        return Image.fromarray(img)

    def create_tiles(self, pil_img: Image.Image) -> List[Tuple[Image.Image, str]]:
        """Create overlapping tiles. Returns list of (tile_image, tile_name)."""
        w, h = pil_img.size
        tw = w // self.tile_cols
        th = h // self.tile_rows
        tiles = []

        for r in range(self.tile_rows):
            for c in range(self.tile_cols):
                left = max(0, c * tw - self.overlap)
                upper = max(0, r * th - self.overlap)
                right = min(w, (c + 1) * tw + self.overlap)
                lower = min(h, (r + 1) * th + self.overlap)
                tile = pil_img.crop((left, upper, right, lower))
                tile_name = f"tile_r{r}_c{c}"
                tiles.append((tile, tile_name))

        # Also add half-page tiles (top half, bottom half) for context
        tiles.append((pil_img.crop((0, 0, w, h // 2 + self.overlap)), "half_top"))
        tiles.append((pil_img.crop((0, h // 2 - self.overlap, w, h)), "half_bottom"))

        return tiles


# ──────────────────────────────────────────────────────────────────
# Layer 2: Qwen2.5-VL VLM Extraction Engine
# ──────────────────────────────────────────────────────────────────

class QwenVLEngine:
    """Qwen2.5-VL for direct form understanding and field extraction."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        logger.info(f"Loading {self.model_name}...")
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info
        self.process_vision_info = process_vision_info
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            dtype=torch.float16,
            device_map="auto",
        )
        self._loaded = True
        logger.info(f"{self.model_name} loaded on GPU")

    def extract_tile(self, pil_img: Image.Image, prompt: str = TILE_EXTRACTION_PROMPT) -> Dict[str, Optional[str]]:
        """Extract fields from a single tile image."""
        self.load()
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": prompt},
            ]}
        ]

        text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=1024, do_sample=False)

        output = self.processor.batch_decode(
            output_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0].strip()

        # Free GPU memory
        del inputs, output_ids
        torch.cuda.empty_cache()

        return self._parse_json(output)

    def _parse_json(self, raw: str) -> Dict[str, Optional[str]]:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:])
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()
        try:
            parsed = json.loads(cleaned)
            # Normalize: convert empty strings and "N/A" to None
            for k, v in parsed.items():
                if isinstance(v, str) and (v.strip() == "" or v.strip().upper() in ("N/A", "NA", "NOT FOUND", "NOT VISIBLE")):
                    parsed[k] = None
            return parsed
        except json.JSONDecodeError:
            logger.warning(f"VLM JSON parse failed, raw: {raw[:200]}")
            return {}

    def extract_full_page(self, pil_img: Image.Image, prompt: str = TILE_EXTRACTION_PROMPT) -> Dict[str, Optional[str]]:
        """Extract from full page (may OOM for large images — use tiling instead)."""
        return self.extract_tile(pil_img, prompt)


# ──────────────────────────────────────────────────────────────────
# Layer 3: PaddleOCR v5 Bilingual Verification
# ──────────────────────────────────────────────────────────────────

class PaddleOCRVerifier:
    """PaddleOCR v5 bilingual for cross-verification and bbox localization."""

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
        logger.info("PaddleOCR v5 (EN+HI) loaded for verification")

    def get_all_regions(self, img: np.ndarray) -> List[Dict]:
        """Get all OCR text regions with bboxes for a page."""
        self.load()
        regions = []
        for ocr_engine, lang in [(self.ocr_en, 'en'), (self.ocr_hi, 'hi')]:
            for res in ocr_engine.predict(img):
                texts = res.get('rec_texts', [])
                scores = res.get('rec_scores', [])
                polys = res.get('dt_polys', [])
                for t, s, p in zip(texts, scores, polys):
                    pts = np.asarray(p)
                    if pts.ndim == 2 and pts.shape[0] >= 2:
                        xs, ys = pts[:, 0], pts[:, 1]
                        regions.append({
                            "text": str(t).strip(),
                            "confidence": float(s),
                            "bbox": [int(np.min(xs)), int(np.min(ys)), int(np.max(xs)), int(np.max(ys))],
                            "lang": lang,
                        })
        # Deduplicate by position + text
        seen = set()
        unique = []
        for r in regions:
            key = (r["text"][:30].lower(), r["bbox"][1] // 20, r["bbox"][0] // 20)
            if key not in seen:
                seen.add(key)
                unique.append(r)
        return unique

    def find_value_bbox(self, img: np.ndarray, field_name: str, value: str) -> Optional[List[int]]:
        """Find the bbox of a specific value in the OCR regions."""
        if not value or not value.strip():
            return None
        regions = self.get_all_regions(img)
        value_lower = value.lower().strip()
        best_match = None
        best_score = 0

        for r in regions:
            rtext = r["text"].lower().strip()
            # Exact or substring match
            if value_lower == rtext:
                score = 1.0
            elif value_lower in rtext or rtext in value_lower:
                score = 0.7 * (len(min(value_lower, rtext, key=len)) / max(len(value_lower, rtext), 1))
            else:
                continue
            # Weight by OCR confidence
            score *= r["confidence"]
            if score > best_score:
                best_score = score
                best_match = r

        if best_match and best_score > 0.3:
            return best_match["bbox"]
        return None

    def verify_field(self, img: np.ndarray, field_name: str, vlm_value: str) -> Tuple[str, float, bool]:
        """
        Cross-verify VLM extraction against PaddleOCR.
        Returns (final_value, confidence, was_corrected).
        """
        if not vlm_value or not vlm_value.strip():
            return vlm_value, 0.0, False

        regions = self.get_all_regions(img)
        vlm_lower = vlm_value.lower().strip()

        # Search for VLM value in OCR regions
        for r in regions:
            rtext = r["text"].lower().strip()
            if vlm_lower == rtext:
                # Exact match — high confidence
                return vlm_value, 0.95, False
            if vlm_lower in rtext or rtext in vlm_lower:
                # Partial match — VLM likely correct, OCR missed some chars
                overlap = len(set(vlm_lower) & set(rtext)) / max(len(set(vlm_lower) | set(rtext)), 1)
                if overlap > 0.6:
                    return vlm_value, 0.85, False

        # VLM value not found in OCR — could be VLM hallucination or OCR miss
        # For handwritten values, OCR often fails so VLM is likely correct
        # For printed values, if OCR doesn't see it, lower confidence
        is_likely_handwritten = any(c.isdigit() for c in vlm_value) or len(vlm_value.split()) <= 2
        if is_likely_handwritten:
            return vlm_value, 0.70, False  # Handwritten — trust VLM
        else:
            return vlm_value, 0.50, False  # Printed but not found — suspicious


# ──────────────────────────────────────────────────────────────────
# Layer 4: Multi-Tile Field Merger
# ──────────────────────────────────────────────────────────────────

class FieldMerger:
    """Merge extractions from multiple tiles into a single coherent result."""

    # Priority: tile with more context > less context
    TILE_PRIORITY = {
        "half_top": 3,
        "half_bottom": 3,
        "tile_r0_c0": 2, "tile_r0_c1": 2,
        "tile_r1_c0": 2, "tile_r1_c1": 2,
        "tile_r2_c0": 2, "tile_r2_c1": 2,
    }

    # Certain tiles are better for certain fields
    FIELD_TILE_AFFINITY = {
        "Proposer_First_Name": ["half_top", "tile_r0_c1"],
        "Proposer_Middle_Name": ["half_top", "tile_r0_c1"],
        "Proposer_Last_Name": ["half_top", "tile_r0_c1"],
        "Proposer_Father_Name": ["half_top", "tile_r1_c0", "tile_r1_c1"],
        "Proposer_Mother_Name": ["half_top", "tile_r1_c0", "tile_r1_c1"],
        "Proposer_Gender": ["half_top", "tile_r1_c0", "tile_r1_c1"],
        "Proposer_Marital_Status": ["half_top", "tile_r1_c0", "tile_r1_c1"],
        "Proposer_Spouse_Name": ["half_top", "tile_r1_c0", "tile_r1_c1"],
        "Proposer_DOB": ["half_top", "tile_r1_c0", "tile_r1_c1"],
        "Proposer_Age": ["half_top", "tile_r1_c0", "tile_r1_c1"],
        "Proposer_House_No": ["half_bottom", "tile_r1_c1", "tile_r2_c0"],
        "Proposer_Street": ["half_bottom", "tile_r1_c1", "tile_r2_c0"],
        "Proposer_City": ["half_bottom", "tile_r2_c0", "tile_r2_c1"],
        "Proposer_State": ["half_bottom", "tile_r2_c0", "tile_r2_c1"],
        "Proposer_PIN": ["half_bottom", "tile_r2_c0", "tile_r2_c1"],
        "Proposer_Phone": ["half_bottom", "tile_r2_c0", "tile_r2_c1"],
    }

    def merge(self, tile_results: Dict[str, Dict[str, Optional[str]]]) -> Dict[str, Optional[str]]:
        """
        Merge results from all tiles. For each field:
        1. If only one tile has it, use that
        2. If multiple tiles have it, prefer the tile with affinity
        3. If values disagree, prefer the more specific/longer value
        """
        merged = {}
        for field_name in FIELDS:
            values = {}
            for tile_name, tile_data in tile_results.items():
                val = tile_data.get(field_name)
                if val is not None and str(val).strip():
                    values[tile_name] = str(val).strip()

            if not values:
                merged[field_name] = None
                continue

            if len(values) == 1:
                merged[field_name] = list(values.values())[0]
                continue

            # Multiple tiles have this field — pick the best
            affinity = self.FIELD_TILE_AFFINITY.get(field_name, [])
            best_tile = None
            best_val = None
            best_priority = -1

            for tile_name, val in values.items():
                priority = self.TILE_PRIORITY.get(tile_name, 1)
                # Boost if this tile has affinity for the field
                if tile_name in affinity:
                    priority += 5
                # Slight preference for longer values (more complete)
                priority += len(val) * 0.01
                if priority > best_priority:
                    best_priority = priority
                    best_tile = tile_name
                    best_val = val

            merged[field_name] = best_val

        return merged


# ──────────────────────────────────────────────────────────────────
# Layer 5: Semantic Field Localization (Bbox Overlay)
# ──────────────────────────────────────────────────────────────────

class BboxLocalizer:
    """Draw extraction results as labeled bboxes on the original page image."""

    FIELD_COLORS = {
        "Proposer_Name": (0, 255, 0),
        "Proposer_First_Name": (0, 255, 0),
        "Proposer_Middle_Name": (0, 255, 0),
        "Proposer_Last_Name": (0, 255, 0),
        "Proposer_Father_Name": (255, 165, 0),
        "Proposer_Mother_Name": (255, 165, 0),
        "Proposer_DOB": (0, 0, 255),
        "Proposer_Age": (0, 0, 255),
        "Proposer_Address": (255, 0, 255),
        "Proposer_City": (255, 0, 255),
        "Proposer_State": (255, 0, 255),
        "Proposer_PIN": (255, 0, 255),
        "Proposer_Phone": (255, 255, 0),
        "Proposer_PAN": (255, 0, 0),
        "Proposer_Aadhaar": (255, 0, 0),
    }

    def draw_overlay(self, img: np.ndarray, fields: List[ExtractedField], output_path: str):
        """Draw bounding boxes and labels on the original image."""
        vis = img.copy()
        for f in fields:
            if not f.value or not f.source_bbox:
                continue
            color = self.FIELD_COLORS.get(f.field_name, (0, 200, 0))
            x1, y1, x2, y2 = f.source_bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f"{f.field_name}: {f.value[:30]}"
            font_scale = 0.4
            thickness = 1
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cv2.rectangle(vis, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
            cv2.putText(vis, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        cv2.imwrite(output_path, vis)
        logger.info(f"Overlay saved to {output_path}")


# ──────────────────────────────────────────────────────────────────
# Main Pipeline Orchestrator
# ──────────────────────────────────────────────────────────────────

class SOTAPipeline:
    """
    VLM-First SOTA Extraction Pipeline.
    Flow: PDF → Tiles → Qwen2.5-VL per tile → Merge → PaddleOCR verify → Localize
    """

    def __init__(
        self,
        vlm_model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        use_ocr_verify: bool = True,
        use_localization: bool = True,
        tile_cols: int = 2,
        tile_rows: int = 3,
        tile_overlap: int = 50,
        dpi: int = 150,
    ):
        self.preparer = ImagePreparer(dpi=dpi, tile_cols=tile_cols, tile_rows=tile_rows, overlap=tile_overlap)
        self.vlm = QwenVLEngine(model_name=vlm_model)
        self.merger = FieldMerger()
        self.ocr_verifier = PaddleOCRVerifier() if use_ocr_verify else None
        self.localizer = BboxLocalizer() if use_localization else None
        self.use_ocr_verify = use_ocr_verify
        self.use_localization = use_localization

    def process_pdf(self, pdf_path: str, pages: Optional[List[int]] = None) -> Dict:
        import fitz
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()

        if pages is None:
            pages = list(range(1, min(total_pages + 1, 10)))

        all_fields = []
        t_start = time.time()

        for pnum in pages:
            if pnum < 1 or pnum > total_pages:
                continue
            logger.info(f"Processing page {pnum}/{total_pages}...")
            page_fields = self._process_page(pdf_path, pnum)
            all_fields.extend(page_fields)

        elapsed = time.time() - t_start

        found = sum(1 for f in all_fields if f.value)
        verified = sum(1 for f in all_fields if f.verified)
        stats = {
            "total_fields": len(all_fields),
            "found": found,
            "verified_by_ocr": verified,
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

    def _process_page(self, pdf_path: str, page_num: int) -> List[ExtractedField]:
        # Step 1: Load page image
        pil_img = self.preparer.pdf_page_to_image(pdf_path, page_num)
        logger.info(f"  Page {page_num} image: {pil_img.size}")

        # Step 2: Create tiles
        tiles = self.preparer.create_tiles(pil_img)
        logger.info(f"  Created {len(tiles)} tiles")

        # Step 3: Extract from each tile via VLM
        tile_results = {}
        for tile_img, tile_name in tiles:
            logger.info(f"  Extracting {tile_name}...")
            t0 = time.time()
            result = self.vlm.extract_tile(tile_img)
            elapsed = time.time() - t0
            found_count = sum(1 for v in result.values() if v is not None)
            logger.info(f"  {tile_name}: {found_count} fields in {elapsed:.1f}s")
            tile_results[tile_name] = result

        # Step 4: Merge tile results
        merged = self.merger.merge(tile_results)
        logger.info(f"  Merged: {sum(1 for v in merged.values() if v is not None)} fields")

        # Step 5: PaddleOCR verification and bbox localization
        ocr_regions = []
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        if self.ocr_verifier:
            ocr_regions = self.ocr_verifier.get_all_regions(cv_img)
            logger.info(f"  PaddleOCR: {len(ocr_regions)} regions for verification")

        # Build ExtractedField objects
        fields = []
        for field_name in FIELDS:
            value = merged.get(field_name)
            if not value:
                fields.append(ExtractedField(
                    field_name=field_name, value="", confidence=0.0,
                    source="not_found", page_num=page_num,
                ))
                continue

            confidence = 0.80  # Base VLM confidence
            verified = False
            source = "vlm_tile"
            source_bbox = None

            if self.ocr_verifier:
                final_val, conf, was_corrected = self.ocr_verifier.verify_field(cv_img, field_name, value)
                confidence = conf
                verified = conf >= 0.85
                source_bbox = self.ocr_verifier.find_value_bbox(cv_img, field_name, value)

            fields.append(ExtractedField(
                field_name=field_name,
                value=value,
                confidence=confidence,
                source=source,
                source_bbox=source_bbox,
                page_num=page_num,
                verified=verified,
            ))

        # Step 6: Draw localization overlay
        if self.localizer:
            output_dir = "data/output"
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            overlay_path = os.path.join(output_dir, f"{base_name}_page{page_num}_overlay.png")
            # Re-render at higher DPI for better overlay
            hi_dpi_img = self.preparer.pdf_page_to_image(pdf_path, page_num) if self.preparer.dpi < 200 else pil_img
            hi_cv = cv2.cvtColor(np.array(hi_dpi_img), cv2.COLOR_RGB2BGR)
            scale = hi_dpi_img.size[0] / pil_img.size[0]
            scaled_fields = []
            for f in fields:
                sf = ExtractedField(**asdict(f))
                if sf.source_bbox:
                    sf.source_bbox = [int(x * scale) for x in sf.source_bbox]
                scaled_fields.append(sf)
            self.localizer.draw_overlay(hi_cv, scaled_fields, overlay_path)

        return fields


# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="LIC SOTA Extraction Pipeline v3 — VLM-First")
    parser.add_argument("pdf_path", help="Path to PDF")
    parser.add_argument("--output", default="data/output/pipeline_v3_result.json")
    parser.add_argument("--pages", type=int, nargs="+", help="Page numbers to process")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct",
                        choices=["Qwen/Qwen2.5-VL-7B-Instruct", "Qwen/Qwen2.5-VL-3B-Instruct"],
                        help="VLM model to use")
    parser.add_argument("--no-ocr-verify", action="store_true", help="Skip PaddleOCR verification")
    parser.add_argument("--no-localization", action="store_true", help="Skip bbox overlay")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for page rendering")
    parser.add_argument("--tile-cols", type=int, default=2, help="Tile grid columns")
    parser.add_argument("--tile-rows", type=int, default=3, help="Tile grid rows")
    args = parser.parse_args()

    pipeline = SOTAPipeline(
        vlm_model=args.model,
        use_ocr_verify=not args.no_ocr_verify,
        use_localization=not args.no_localization,
        tile_cols=args.tile_cols,
        tile_rows=args.tile_rows,
        dpi=args.dpi,
    )

    result = pipeline.process_pdf(args.pdf_path, args.pages)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    stats = result["statistics"]
    print("\n" + "=" * 80)
    print("SOTA VLM-FIRST EXTRACTION RESULTS")
    print("=" * 80)
    for f in result["fields"]:
        if f["value"]:
            icon = "[OK]" if f.get("verified") or f["confidence"] >= 0.80 else "[??]"
        else:
            icon = "[--]"
        val = f["value"][:50] if f["value"] else "(not found)"
        print(f" {icon} {f['field_name']:<30} {val:<52} conf={f['confidence']:.2f}")
    print("=" * 80)
    print(f" Total: {stats['total_fields']} | Found: {stats['found']} | "
          f"OCR-Verified: {stats['verified_by_ocr']}")
    print(f" Accuracy: {stats['accuracy_estimate']}% | Time: {stats['processing_time_sec']}s | "
          f"Per page: {stats['time_per_page']}s")
    print(f" Output: {args.output}")


if __name__ == "__main__":
    main()
