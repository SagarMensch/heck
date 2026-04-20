"""
ACCURACY-FIRST PIPELINE for LIC Form 300
Goal: 99% Field-Level Accuracy
Strategy:
  1. PaddleX Layout Parsing (Structure + OCR)
  2. Forensic Field Mapping (Encyclopedia + Constraints)
  3. Verhoeff Validation (Aadhaar)
  4. Qwen2.5-VL Fallback (for confidence < 0.90)
"""
import os
import json
import time
import logging
import fitz  # PyMuPDF
from typing import Dict, List, Any, Optional
from pathlib import Path

from .layout_extractor import PaddleXLayoutExtractor
from .forensic_mapper import ForensicFieldMapper, ForensicResult

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

class AccuracyFirstPipeline:
    def __init__(self, use_qwen_fallback: bool = False, confidence_threshold: float = 0.90):
        self.use_qwen_fallback = use_qwen_fallback
        self.confidence_threshold = confidence_threshold
        self.layout_extractor = PaddleXLayoutExtractor()
        self.forensic_mapper = ForensicFieldMapper()
        self.stats = {'total_fields': 0, 'high_conf': 0, 'corrected': 0, 'failed': 0}
    
    def process_pdf(self, pdf_path: str, pages: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Process PDF with accuracy-first approach.
        Returns: Dict with 'fields', 'statistics', 'raw_regions'
        """
        logger.info("="*60)
        logger.info("STARTING ACCURACY-FIRST PIPELINE")
        logger.info("="*60)
        
        start_time = time.time()
        doc = fitz.open(pdf_path)
        page_nums = pages if pages else range(1, len(doc) + 1)
        
        all_results = []
        raw_regions = {}
        
        for pnum in page_nums:
            logger.info(f"Processing Page {pnum}/{len(doc)}...")
            page = doc.load_page(pnum - 1)
            pix = page.get_pixmap(dpi=150)  # High DPI for accuracy
            
            # Temporary image path
            import tempfile
            tmpdir = tempfile.gettempdir()
            img_path = os.path.join(tmpdir, f"lic_page_{pnum}.png")
            pix.save(img_path)
            
            try:
                # Step 1: Extract Layout & OCR
                regions = self.layout_extractor.extract_from_image(img_path, pnum)
                raw_regions[pnum] = regions
                logger.info(f"  Extracted {len(regions)} regions")
                
                # Step 2: Forensic Mapping
                forensic_results = self.forensic_mapper.map_and_validate(regions, pnum)
                all_results.extend(forensic_results)
                
                # Update stats
                for r in forensic_results:
                    self.stats['total_fields'] += 1
                    if r.confidence >= self.confidence_threshold:
                        self.stats['high_conf'] += 1
                    if r.validation_status == 'corrected':
                        self.stats['corrected'] += 1
                    if r.validation_status == 'invalid':
                        self.stats['failed'] += 1
                
                # Step 3: Qwen Fallback (if enabled and needed)
                if self.use_qwen_fallback:
                    for i, r in enumerate(forensic_results):
                        if r.confidence < self.confidence_threshold:
                            logger.warning(f"  Low confidence ({r.confidence:.2f}) on {r.field_name}. Triggering Qwen fallback...")
                            # TODO: Implement Qwen fallback here
                            # For now, just log it
                            r.notes += " | Qwen fallback needed but not implemented"
                
            finally:
                if os.path.exists(img_path):
                    try:
                        os.remove(img_path)
                    except:
                        pass
        
        doc.close()
        elapsed = time.time() - start_time
        
        # Compile final results
        final_output = {
            'extracted_fields': self.forensic_mapper.to_dict_list(all_results),
            'statistics': {
                **self.stats,
                'processing_time_s': elapsed,
                'pages_processed': len(page_nums),
                'speed_s_per_page': elapsed / max(len(page_nums), 1)
            },
            'metadata': {
                'pdf_path': pdf_path,
                'pipeline': 'AccuracyFirst-v1',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        logger.info("="*60)
        logger.info("PIPELINE COMPLETE")
        logger.info(f"Total Fields: {self.stats['total_fields']}")
        logger.info(f"High Confidence (>{self.confidence_threshold}): {self.stats['high_conf']}")
        logger.info(f"Corrected by Encyclopedia: {self.stats['corrected']}")
        logger.info(f"Failed/Invalid: {self.stats['failed']}")
        logger.info(f"Time: {elapsed:.1f}s ({elapsed/max(len(page_nums),1):.1f}s/page)")
        logger.info("="*60)
        
        return final_output
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save results to JSON"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {output_path}")


def run_accuracy_pipeline(pdf_path: str, output_path: str = None, pages: List[int] = None):
    """Convenience function"""
    pipeline = AccuracyFirstPipeline(use_qwen_fallback=False)  # Enable Qwen later if needed
    results = pipeline.process_pdf(pdf_path, pages)
    
    if output_path:
        pipeline.save_results(results, output_path)
    
    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python accuracy_pipeline.py <pdf_path> [output_path]")
        sys.exit(1)
    
    pdf = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    run_accuracy_pipeline(pdf, out)
