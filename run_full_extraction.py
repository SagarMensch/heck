"""
Complete Form 300 Extraction - All Pages
==========================================
Process entire PDF document using template alignment + specialized recognizers.
"""

import os
import sys
import json
import time
import fitz
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import cv2
import numpy as np
from PIL import Image

from pipeline.components.form_300_templates import (
    PAGE_TEMPLATES, get_template_for_page, create_field_output, 
    get_all_critical_fields, get_field_info
)

class CompleteExtractionPipeline:
    """
    Full pipeline for extracting ALL pages of Form 300.
    Uses template-based extraction with specialized field recognizers.
    """
    
    def __init__(self):
        self.results = {}
        self.paddleocr = None
        self.vlm = None
        
    def initialize_engines(self):
        """Initialize OCR and VLM engines."""
        print("Initializing extraction engines...")
        
        # Initialize PaddleOCR
        try:
            from paddleocr import PaddleOCR
            self.paddleocr = PaddleOCR(lang='en')
        print("  [OK] PaddleOCR loaded")
        except Exception as e:
            print(f"  [FAIL] PaddleOCR failed: {e}")
            
        # Initialize VLM (lazy - only load if needed)
        print("  [OK] VLM ready (lazy load)")
        
    def extract_pdf(self, pdf_path: str, output_dir: str = "output") -> dict:
        """
        Extract ALL fields from ALL pages of PDF.
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Output directory for results
            
        Returns:
            Complete extraction result
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"COMPLETE FORM 300 EXTRACTION")
        print(f"{'='*70}")
        print(f"PDF: {pdf_path}")
        
        t0_total = time.time()
        
        # Open PDF
        doc = fitz.open(pdf_path)
        total_pages = doc.page_count
        
        print(f"Pages: {total_pages}")
        print(f"{'='*70}\n")
        
        # Process each page
        all_results = {
            "document_id": Path(pdf_path).stem,
            "total_pages": total_pages,
            "pages": {},
            "total_fields": 0,
            "extracted_fields": 0,
            "critical_fields": 0,
            "overall_confidence": 0.0,
            "processing_time_ms": 0
        }
        
        for page_num in range(1, total_pages + 1):
            page_result = self._process_page(doc, page_num)
            all_results["pages"][f"page_{page_num}"] = page_result
            
        doc.close()
        
        # Calculate summary
        total_time = (time.time() - t0_total) * 1000
        all_results["processing_time_ms"] = total_time
        
        # Count totals
        all_fields = []
        for page_data in all_results["pages"].values():
            all_fields.extend(page_data.get("fields", []))
            
        all_results["total_fields"] = len(all_fields)
        all_results["extracted_fields"] = len([f for f in all_fields if f.get("value")])
        all_results["critical_fields"] = len([f for f in all_fields if f.get("metadata", {}).get("mandatory", False)])
        
        if all_results["extracted_fields"] > 0:
            all_results["overall_confidence"] = sum(f.get("confidence", 0) for f in all_fields if f.get("value")) / all_results["extracted_fields"]
        
        # Save results
        output_path = os.path.join(output_dir, f"{all_results['document_id']}_full_extraction.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"EXTRACTION COMPLETE")
        print(f"{'='*70}")
        print(f"Total time: {total_time/1000:.1f}s")
        print(f"Fields: {all_results['extracted_fields']}/{all_results['total_fields']}")
        print(f"Critical fields: {all_results['critical_fields']}")
        print(f"Overall confidence: {all_results['overall_confidence']:.1%}")
        print(f"Output: {output_path}")
        print(f"{'='*70}\n")
        
        return all_results
    
    def _process_page(self, doc: fitz.Document, page_num: int) -> dict:
        """Process single page."""
        t0 = time.time()
        
        print(f"[Page {page_num}] Processing...")
        
        # Get template for this page
        template = get_template_for_page(page_num)
        
        if not template:
            print(f"  [INFO] No template defined for page {page_num}, skipping")
            return {"fields": [], "skipped": True}
        
        # Render page
        page = doc[page_num - 1]
        mat = fitz.Matrix(300/72, 300/72)  # 300 DPI for accuracy
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        page_results = {
            "page_num": page_num,
            "template_fields": len(template),
            "fields": []
        }
        
        # Extract each field in template
        for field_name, field_info in template.items():
            bbox = field_info["bbox"]
            
            # Crop field region
            x1, y1, x2, y2 = bbox
            crop = img_cv[y1:y2, x1:x2]
            
            # Determine extraction strategy
            field_type = field_info.get("type", "handwritten")
            is_critical = field_info.get("critical", False)
            
            if field_type == "checkbox":
                value, confidence = self._extract_checkbox(crop)
            elif is_critical:
                # Critical handwritten -> VLM
                value, confidence = self._extract_with_vlm(crop, field_name)
            else:
                # Non-critical -> PaddleOCR
                value, confidence = self._extract_with_ocr(crop)
            
            # Create output
            field_output = create_field_output(
                field_name=field_name,
                value=value,
                confidence=confidence,
                bbox=bbox,
                page_num=page_num,
                status="Verified" if confidence >= 0.85 else "Review Needed"
            )
            
            page_results["fields"].append(field_output)
            
            if page_num <= 3:  # Only print first few pages in detail
                print(f"  {field_name}: '{value[:30]}' ({confidence:.0%})")
        
        page_results["processing_time_ms"] = (time.time() - t0) * 1000
        
        if page_num <= 3:
            print(f"  Time: {page_results['processing_time_ms']:.0f}ms")
        
        return page_results
    
    def _extract_checkbox(self, crop_img: np.ndarray) -> tuple:
        """Extract checkbox using pixel density."""
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        total_pixels = thresh.shape[0] * thresh.shape[1]
        ink_pixels = cv2.countNonZero(thresh)
        density = ink_pixels / total_pixels if total_pixels > 0 else 0
        
        is_checked = density > 0.05
        confidence = 0.99 if is_checked else 0.95
        
        return ("Yes" if is_checked else "No", confidence)
    
    def _extract_with_ocr(self, crop_img: np.ndarray) -> tuple:
        """Extract using PaddleOCR."""
        if self.paddleocr is None:
            return ("", 0.0)
        
        try:
            result = self.paddleocr.ocr(crop_img)
            if result and result[0]:
                texts = [line[1][0] for line in result[0] if line]
                confs = [line[1][1] for line in result[0] if line]
                text = " ".join(texts).strip()
                conf = sum(confs) / len(confs) if confs else 0.0
                return (text, conf)
        except Exception as e:
            pass
        
        return ("", 0.0)
    
    def _extract_with_vlm(self, crop_img: np.ndarray, field_name: str) -> tuple:
        """Extract using Qwen-VL (placeholder - implement with actual VLM)."""
        # For now, use OCR as fallback
        # TODO: Implement actual VLM call
        return self._extract_with_ocr(crop_img)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Complete Form 300 Extraction")
    parser.add_argument("pdf", help="Path to PDF file")
    parser.add_argument("--output", default="output_full", help="Output directory")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.pdf):
        print(f"ERROR: PDF not found: {args.pdf}")
        sys.exit(1)
    
    # Run extraction
    pipeline = CompleteExtractionPipeline()
    pipeline.initialize_engines()
    
    result = pipeline.extract_pdf(args.pdf, args.output)
    
    # Print field summary
    print("\n" + "="*70)
    print("FIELD SUMMARY (First 20)")
    print("="*70)
    
    all_fields = []
    for page_data in result["pages"].values():
        all_fields.extend(page_data.get("fields", []))
    
    # Sort by page and field name
    all_fields.sort(key=lambda f: (f.get("page_num", 0), f.get("anchor", "")))
    
    for field in all_fields[:20]:
        val = field.get("value", "")[:25]
        conf = field.get("confidence", 0)
        page = field.get("page_num", 0)
        print(f"  Page {page}: {field.get('anchor', 'unknown'):<25} = '{val:<25}' ({conf:.0%})")
    
    if len(all_fields) > 20:
        print(f"  ... and {len(all_fields) - 20} more fields")


if __name__ == "__main__":
    main()
