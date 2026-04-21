"""
Nemotron-Based Extraction Pipeline v2
========================================
Layer 1: Preprocessing
Layer 2: Nemotron Table Structure Detection
Layer 3: PaddleOCR GPU (EN+HI) - Primary
Layer 4: Qwen-VL - Fallback only
Layer 5: Encyclopedia Validation + Post-processing
"""

import os
import sys
import time
import logging
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, field
from PIL import Image
import fitz
import numpy as np
import json
import torch

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """Bounding box with normalized coordinates support."""
    x1: int
    y1: int
    x2: int
    y2: int
    page_width: int = 0
    page_height: int = 0
    
    @property
    def width(self) -> int:
        return abs(self.x2 - self.x1)
    
    @property
    def height(self) -> int:
        return abs(self.y2 - self.y1)
    
    def to_list(self) -> List[int]:
        return [self.x1, self.y1, self.x2, self.y2]
    
    def normalized(self) -> Tuple[float, float, float, float]:
        if self.page_width == 0 or self.page_height == 0:
            return (0.0, 0.0, 0.0, 0.0)
        return (
            self.x1 / self.page_width,
            self.y1 / self.page_height,
            self.x2 / self.page_width,
            self.y2 / self.page_height
        )
    
    def crop(self, image: Image.Image) -> Image.Image:
        return image.crop((self.x1, self.y1, self.x2, self.y2))


@dataclass
class Field:
    """Extracted field with full provenance."""
    name: str
    value: str = ""
    confidence: float = 0.0
    source: str = "unknown"  # "nemotron", "paddleocr", "qwen"
    bbox: Optional[BoundingBox] = None
    ocr_confidence: float = 0.0
    vlm_confidence: float = 0.0
    validation_status: str = "pending"
    corrected_value: str = ""
    validation_issues: List[str] = field(default_factory=list)
    needs_human_review: bool = False


@dataclass
class PageResult:
    """Page extraction result."""
    page_num: int
    fields: List[Field] = field(default_factory=list)
    processing_time_ms: float = 0.0
    nemotron_time_ms: float = 0.0
    ocr_time_ms: float = 0.0
    vlm_time_ms: float = 0.0
    validation_time_ms: float = 0.0


@dataclass
class DocumentResult:
    """Complete document result."""
    document_id: str
    file_path: str
    pages: List[PageResult] = field(default_factory=list)
    total_processing_time_ms: float = 0.0
    overall_confidence: float = 0.0
    form_status: str = "pending"
    
    def all_fields(self) -> List[Field]:
        return [f for page in self.pages for f in page.fields]
    
    def to_dict(self) -> Dict:
        return {
            "document_id": self.document_id,
            "file_path": self.file_path,
            "total_pages": len(self.pages),
            "total_time_ms": self.total_processing_time_ms,
            "overall_confidence": self.overall_confidence,
            "form_status": self.form_status,
            "pages": [
                {
                    "page_num": p.page_num,
                    "processing_time_ms": p.processing_time_ms,
                    "fields": [
                        {
                            "name": f.name,
                            "value": f.value,
                            "confidence": f.confidence,
                            "source": f.source,
                            "validation_status": f.validation_status,
                            "needs_human_review": f.needs_human_review
                        }
                        for f in p.fields
                    ]
                }
                for p in self.pages
            ]
        }


class NemotronTableDetector:
    """
    Layer 2: Nemotron Table Structure Detection
    Uses nvidia/nemotron-table-structure-v1
    """
    
    def __init__(self, model_name: str = "./nemotron_local"):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self._loaded = False
        self._cache = {}
        
    def load(self):
        """Load Nemotron model from HuggingFace."""
        if self._loaded:
            return
            
        try:
            from transformers import AutoModelForObjectDetection, AutoProcessor
            
            logger.info(f"Loading Nemotron: {self.model_name}")
            
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            self.model = AutoModelForObjectDetection.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            self._loaded = True
            logger.info("✓ Nemotron loaded on GPU")
            
        except Exception as e:
            logger.error(f"✗ Failed to load Nemotron: {e}")
            logger.warning("Will use fallback detection")
            self._loaded = False
    
    def detect(self, image: Image.Image, page_num: int = 0) -> Dict[str, Any]:
        """
        Detect table cells using Nemotron.
        
        Returns:
            Dict with 'cells' list containing bbox coordinates
        """
        cache_key = f"page_{page_num}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        if not self._loaded:
            self.load()
        
        if not self._loaded:
            return self._fallback_detection(image)
        
        try:
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            target_sizes = torch.tensor([image.size[::-1]])
            results = self.processor.post_process_object_detection(
                outputs,
                threshold=0.3,
                target_sizes=target_sizes
            )[0]
            
            cells = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                x1, y1, x2, y2 = box.tolist()
                cells.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "score": float(score),
                    "label": int(label),
                    "row": -1,
                    "col": -1
                })
            
            # Organize into rows
            cells = self._organize_cells(cells, image.size)
            
            result = {"cells": cells, "method": "nemotron", "count": len(cells)}
            self._cache[cache_key] = result
            return result
            
        except Exception as e:
            logger.error(f"Nemotron detection failed: {e}")
            return self._fallback_detection(image)
    
    def _fallback_detection(self, image: Image.Image) -> Dict:
        """Fallback: rule-based table detection."""
        import cv2
        
        img = np.array(image)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Detect lines
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        
        cells = []
        if lines is not None:
            h_lines, v_lines = [], []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(x2-x1) > abs(y2-y1):
                    h_lines.append((y1+y2)//2)
                else:
                    v_lines.append((x1+x2)//2)
            
            h_lines = sorted(list(set([int(y) for y in h_lines])))
            v_lines = sorted(list(set([int(x) for x in v_lines])))
            
            for i in range(len(h_lines)-1):
                for j in range(len(v_lines)-1):
                    cells.append({
                        "bbox": [v_lines[j], h_lines[i], v_lines[j+1], h_lines[i+1]],
                        "row": i,
                        "col": j,
                        "score": 0.5
                    })
        
        return {"cells": cells, "method": "fallback", "count": len(cells)}
    
    def _organize_cells(self, cells: List[Dict], img_size: tuple) -> List[Dict]:
        """Organize cells by row."""
        if not cells:
            return cells
        
        cells_sorted = sorted(cells, key=lambda c: c["bbox"][1])
        rows = []
        current_row = []
        prev_y = cells_sorted[0]["bbox"][1]
        
        for cell in cells_sorted:
            y = cell["bbox"][1]
            if abs(y - prev_y) > 30:
                if current_row:
                    rows.append(sorted(current_row, key=lambda c: c["bbox"][0]))
                current_row = [cell]
                prev_y = y
            else:
                current_row.append(cell)
        
        if current_row:
            rows.append(sorted(current_row, key=lambda c: c["bbox"][0]))
        
        organized = []
        for row_idx, row in enumerate(rows):
            for col_idx, cell in enumerate(row):
                cell["row"] = row_idx
                cell["col"] = col_idx
                organized.append(cell)
        
        return organized


class PaddleOCRGPU:
    """
    Layer 3: PaddleOCR GPU-based (EN + HI)
    """
    
    def __init__(self):
        self.ocr_en = None
        self.ocr_hi = None
        self._loaded = False
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "1"
        
    def load(self):
        """Load PaddleOCR models."""
        if self._loaded:
            return
            
        try:
            from paddleocr import PaddleOCR
            
            logger.info("Loading PaddleOCR GPU...")
            
            self.ocr_en = PaddleOCR(
                lang='en',
                det_db_thresh=0.3,
                rec_batch_num=10
            )
            
            self.ocr_hi = PaddleOCR(
                lang='hi',
                det_db_thresh=0.3
            )
            
            self._loaded = True
            logger.info("✓ PaddleOCR GPU loaded (EN+HI)")
            
        except Exception as e:
            logger.error(f"✗ PaddleOCR load failed: {e}")
            raise
    
    def recognize(self, image: Image.Image, bbox: Optional[BoundingBox] = None) -> Tuple[str, float, str]:
        """
        OCR on image or region.
        
        Returns: (text, confidence, language)
        """
        if not self._loaded:
            self.load()
        
        if bbox:
            image = bbox.crop(image)
        
        img_array = np.array(image)
        
        try:
            # Try English first
            result = self.ocr_en.ocr(img_array)
            lang = "en"
            
            if not result or not result[0]:
                # Try Hindi
                result = self.ocr_hi.ocr(img_array)
                lang = "hi"
            
            if result and result[0]:
                texts = []
                confs = []
                for line in result[0]:
                    if line:
                        texts.append(line[1][0])
                        confs.append(line[1][1])
                
                text = " ".join(texts).strip()
                conf = np.mean(confs) if confs else 0.0
                return text, float(conf), lang
            
            return "", 0.0, "unknown"
            
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return "", 0.0, "error"


class ExtractionPipeline:
    """
    Complete 5-Layer Pipeline
    """
    
    def __init__(self):
        self.nemotron = NemotronTableDetector()
        self.paddleocr = PaddleOCRGPU()
        self.vlm = None  # Lazy load
        self.validator = None  # Lazy load
        
        # Field mapping for Form 300
        self.FIELD_MAP = {
            "Proposer_First_Name": (2, 1),
            "Proposer_Middle_Name": (2, 2),
            "Proposer_Last_Name": (2, 3),
            "Proposer_Gender": (5, 1),
            "Proposer_Marital_Status": (6, 1),
            "Proposer_Date_of_Birth": (8, 1),
            "Proposer_Age": (9, 1),
            "Proposer_Birth_Place": (10, 1),
            "Proposer_Nationality": (12, 1),
            "Proposer_Citizenship": (13, 1),
            "Proposer_Address_Line1": (14, 1),
            "Proposer_City": (14, 1),
            "Proposer_State": (14, 1),
            "Proposer_Pincode": (14, 1),
            "Proposer_Mobile_Number": (14, 1),
        }
        
    def process(self, pdf_path: str, pages: Optional[List[int]] = None) -> DocumentResult:
        """Process PDF document."""
        t0_total = time.time()
        
        doc = fitz.open(pdf_path)
        if pages is None:
            pages = list(range(1, doc.page_count + 1))
        
        page_results = []
        
        for page_num in pages:
            page_result = self._process_page(doc, page_num)
            page_results.append(page_result)
        
        doc.close()
        
        total_time = (time.time() - t0_total) * 1000
        
        # Calculate overall metrics
        all_fields = [f for p in page_results for f in p.fields]
        extracted = [f for f in all_fields if f.value]
        avg_conf = np.mean([f.confidence for f in extracted]) if extracted else 0.0
        
        result = DocumentResult(
            document_id=Path(pdf_path).stem,
            file_path=pdf_path,
            pages=page_results,
            total_processing_time_ms=total_time,
            overall_confidence=avg_conf,
            form_status="completed"
        )
        
        return result
    
    def _process_page(self, doc: fitz.Document, page_num: int) -> PageResult:
        """Process single page through all layers."""
        t0 = time.time()
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing Page {page_num}")
        logger.info(f"{'='*60}")
        
        # --- Layer 1: Preprocess ---
        t1 = time.time()
        page = doc[page_num - 1]
        mat = fitz.Matrix(150/72, 150/72)  # 150 DPI
        pix = page.get_pixmap(matrix=mat)
        pil_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        logger.info(f"[L1] Preprocess: {(time.time()-t1)*1000:.0f}ms")
        
        # --- Layer 2: Nemotron Table Detection ---
        t2 = time.time()
        table_result = self.nemotron.detect(pil_img, page_num)
        nemotron_time = (time.time() - t2) * 1000
        logger.info(f"[L2] Nemotron: {table_result['count']} cells, {nemotron_time:.0f}ms")
        
        # --- Layer 3: PaddleOCR Primary ---
        t3 = time.time()
        fields = []
        vlm_fallbacks = []
        
        for field_name, (row, col) in self.FIELD_MAP.items():
            # Find matching cell
            cell = self._find_cell(table_result["cells"], row, col)
            
            if cell:
                bbox = BoundingBox(*cell["bbox"])
                text, conf, lang = self.paddleocr.recognize(pil_img, bbox)
                
                field = Field(
                    name=field_name,
                    value=text,
                    confidence=conf,
                    source="paddleocr",
                    bbox=bbox,
                    ocr_confidence=conf
                )
                
                # Queue for VLM fallback if low confidence
                if conf < 0.7:
                    vlm_fallbacks.append(field)
            else:
                field = Field(name=field_name, value="", confidence=0.0, source="missed")
                vlm_fallbacks.append(field)
            
            fields.append(field)
        
        ocr_time = (time.time() - t3) * 1000
        logger.info(f"[L3] PaddleOCR: {len(fields)} fields, {ocr_time:.0f}ms")
        
        # --- Layer 4: Qwen-VL Fallback ---
        t4 = time.time()
        if vlm_fallbacks:
            self._vlm_fallback(pil_img, vlm_fallbacks[:5])  # Max 5 VLM calls
        
        vlm_time = (time.time() - t4) * 1000
        logger.info(f"[L4] Qwen-VL Fallback: {len(vlm_fallbacks)} fields, {vlm_time:.0f}ms")
        
        # --- Layer 5: Encyclopedia Validation ---
        t5 = time.time()
        self._validate_fields(fields)
        
        validation_time = (time.time() - t5) * 1000
        logger.info(f"[L5] Validation: {validation_time:.0f}ms")
        
        total_time = (time.time() - t0) * 1000
        
        return PageResult(
            page_num=page_num,
            fields=fields,
            processing_time_ms=total_time,
            nemotron_time_ms=nemotron_time,
            ocr_time_ms=ocr_time,
            vlm_time_ms=vlm_time,
            validation_time_ms=validation_time
        )
    
    def _find_cell(self, cells: List[Dict], row: int, col: int) -> Optional[Dict]:
        """Find cell by row/col."""
        for cell in cells:
            if cell.get("row") == row and cell.get("col") == col:
                return cell
        return None
    
    def _vlm_fallback(self, image: Image.Image, fields: List[Field]):
        """Use Qwen-VL for low confidence fields."""
        if self.vlm is None:
            try:
                from src.pipeline.layers.vlm_extractor import VLMExtractor
                self.vlm = VLMExtractor()
            except Exception as e:
                logger.error(f"VLM load failed: {e}")
                return
        
        for field in fields:
            try:
                if field.bbox:
                    crop = field.bbox.crop(image)
                    result = self.vlm.extract_tile(crop)
                    
                    if field.name in result and result[field.name]:
                        field.value = result[field.name]
                        field.confidence = 0.85
                        field.source = "qwen"
                        field.vlm_confidence = 0.85
            except Exception as e:
                logger.warning(f"VLM fallback failed for {field.name}: {e}")
    
    def _validate_fields(self, fields: List[Field]):
        """Validate using existing ValidationKB."""
        if self.validator is None:
            try:
                from src.pipeline.layers.validation_kb import ValidationKB
                from src.pipeline.models.schemas import ExtractedField
                self.validator = ValidationKB()
                self.ExtractedField = ExtractedField
            except Exception as e:
                logger.warning(f"Validator load failed: {e}")
                return
        
        for field in fields:
            try:
                ef = self.ExtractedField(
                    field_name=field.name,
                    value=field.value,
                    confidence=field.confidence,
                    source=field.source
                )
                
                self.validator.validate_field(ef)
                
                field.validation_status = ef.validation_status
                if ef.kb_corrected:
                    field.corrected_value = ef.value
                field.validation_issues = ef.cross_field_issues
                field.needs_human_review = ef.needs_human_review
                
            except Exception as e:
                logger.warning(f"Validation failed for {field.name}: {e}")


# CLI Entry point
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Nemotron-Based Extraction Pipeline v2")
    parser.add_argument("pdf_path", help="Path to PDF")
    parser.add_argument("--pages", default=None, help="Pages: '2', '1-5', '2,3,4'")
    parser.add_argument("--output", default="data/output_v2")
    
    args = parser.parse_args()
    
    # Parse pages
    pages = None
    if args.pages:
        if '-' in args.pages:
            start, end = map(int, args.pages.split('-'))
            pages = list(range(start, end + 1))
        elif ',' in args.pages:
            pages = [int(p.strip()) for p in args.pages.split(',')]
        else:
            pages = [int(args.pages)]
    
    print(f"\n{'='*70}")
    print(f"NEMOTRON-BASED EXTRACTION PIPELINE v2")
    print(f"{'='*70}")
    print(f"PDF: {args.pdf_path}")
    print(f"Pages: {pages if pages else 'ALL'}")
    print(f"{'='*70}\n")
    
    pipeline = ExtractionPipeline()
    result = pipeline.process(args.pdf_path, pages)
    
    # Save results
    os.makedirs(args.output, exist_ok=True)
    output_path = os.path.join(args.output, f"{result.document_id}_v2.json")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"COMPLETE")
    print(f"{'='*70}")
    print(f"Total Time: {result.total_processing_time_ms/1000:.1f}s")
    print(f"Overall Confidence: {result.overall_confidence:.1%}")
    print(f"Fields: {len([f for f in result.all_fields() if f.value])}/{len(result.all_fields())}")
    print(f"Output: {output_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
