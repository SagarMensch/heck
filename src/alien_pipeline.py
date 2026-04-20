"""
ALIEN-TECH PIPELINE
Modular, SOTA extraction pipeline for LIC Form 300
"""
import os
import json
import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

# Import modules
from .advanced_preprocessor import AlienPreprocessor, PreprocessResult
from .layout_extractor import PaddleXLayoutExtractor, TextRegion
from .field_mapper import FieldMapper, MappedField
from .qwen_fallback import QwenFallback, QwenResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Complete extraction result for a document"""
    field_name: str
    value: str
    raw_text: str
    confidence: float
    source_page: int
    source_bbox: List[float]
    validation_status: str
    used_fallback: bool
    fallback_reasoning: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)


class AlienTechPipeline:
    """
    ALIEN-TECH EXTRACTION PIPELINE
    Modular, high-performance extraction for LIC Form 300
    """
    
    def __init__(self, confidence_threshold: float = 0.85):
        self.confidence_threshold = confidence_threshold
        
        # Lazy-loaded components
        self.preprocessor = None
        self.layout_extractor = None
        self.field_mapper = None
        self.qwen_fallback = None
        
        # Statistics
        self.stats = {
            'total_fields': 0,
            'high_confidence': 0,
            'fallback_used': 0,
            'fallback_success': 0,
            'processing_time': 0
        }
    
    def _ensure_loaded(self):
        """Lazy load all components"""
        if self.preprocessor is None:
            logger.info("Loading AlienPreprocessor...")
            self.preprocessor = AlienPreprocessor()
        
        if self.layout_extractor is None:
            logger.info("Loading PaddleXLayoutExtractor...")
            self.layout_extractor = PaddleXLayoutExtractor()
            self.layout_extractor.load()
        
        if self.field_mapper is None:
            logger.info("Loading FieldMapper...")
            self.field_mapper = FieldMapper()
    
    def process_page(self, img_path: str, page_num: int = 1) -> List[MappedField]:
        """
        Process a single page through the pipeline
        
        Args:
            img_path: Path to page image
            page_num: Page number (1-indexed)
        
        Returns:
            List of MappedField objects
        """
        self._ensure_loaded()
        
        # Step 1: Load and preprocess image
        logger.info(f"Page {page_num}: Preprocessing...")
        img = np.array(img_path) if isinstance(img_path, str) else img_path
        if isinstance(img_path, str):
            import cv2
            img = cv2.imread(img_path)
        
        preprocess_result = self.preprocessor.preprocess(img)
        logger.info(f"  Quality score: {preprocess_result.quality_score:.2f}, "
                   f"Enhancements: {', '.join(preprocess_result.enhancements_applied)}")
        
        # Step 2: Extract layout regions
        logger.info(f"Page {page_num}: Extracting layout...")
        regions = self.layout_extractor.extract_from_image(img_path, page_num)
        logger.info(f"  Found {len(regions)} regions")
        
        # Step 3: Map to canonical fields
        logger.info(f"Page {page_num}: Mapping fields...")
        mapped_fields = self.field_mapper.map_regions_to_fields(regions, page_num)
        logger.info(f"  Mapped {len(mapped_fields)} fields")
        
        return mapped_fields
    
    def process_pdf(self, pdf_path: str, pages: Optional[List[int]] = None) -> Dict[str, ExtractionResult]:
        """
        Process entire PDF
        
        Args:
            pdf_path: Path to PDF
            pages: Optional list of page numbers to process
        
        Returns:
            Dict mapping field_name -> ExtractionResult
        """
        import fitz  # PyMuPDF
        import tempfile
        
        self._ensure_loaded()
        start_time = time.time()
        
        doc = fitz.open(pdf_path)
        page_nums = pages if pages else range(1, len(doc) + 1)
        
        all_results = {}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            for pnum in page_nums:
                logger.info(f"Processing page {pnum}/{len(doc)}...")
                
                # Convert page to image
                page = doc.load_page(pnum - 1)
                pix = page.get_pixmap(dpi=150)
                img_path = os.path.join(tmpdir, f"page_{pnum}.png")
                pix.save(img_path)
                
                # Extract layout
                regions = self.layout_extractor.extract_from_image(img_path, pnum)
                
                # Map fields
                mapped = self.field_mapper.map_regions_to_fields(regions, pnum)
                
                # Process each mapped field
                for field in mapped:
                    # Check confidence
                    if field.confidence >= self.confidence_threshold:
                        # High confidence - accept
                        result = ExtractionResult(
                            field_name=field.field_name,
                            value=field.value,
                            raw_text=field.raw_text,
                            confidence=field.confidence,
                            source_page=field.page_num,
                            source_bbox=field.source_bbox,
                            validation_status=field.validation_status,
                            used_fallback=False
                        )
                        self.stats['high_confidence'] += 1
                    else:
                        # Low confidence - would use Qwen fallback here
                        # For now, just mark as low confidence
                        result = ExtractionResult(
                            field_name=field.field_name,
                            value=field.value,
                            raw_text=field.raw_text,
                            confidence=field.confidence,
                            source_page=field.page_num,
                            source_bbox=field.source_bbox,
                            validation_status='needs_review',
                            used_fallback=False,
                            fallback_reasoning='Low confidence, fallback not configured'
                        )
                        self.stats['fallback_used'] += 1
                    
                    all_results[field.field_name] = result
                    self.stats['total_fields'] += 1
        
        doc.close()
        
        self.stats['processing_time'] = time.time() - start_time
        logger.info(f"Pipeline complete in {self.stats['processing_time']:.1f}s")
        logger.info(f"  Total fields: {self.stats['total_fields']}")
        logger.info(f"  High confidence: {self.stats['high_confidence']}")
        logger.info(f"  Fallback used: {self.stats['fallback_used']}")
        
        return all_results
    
    def save_results(self, results: Dict[str, ExtractionResult], output_path: str):
        """Save results to JSON"""
        output = {
            'results': {k: v.to_dict() for k, v in results.items()},
            'statistics': self.stats
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_path}")


def run_pipeline(pdf_path: str, output_path: str = None, pages: List[int] = None) -> Dict[str, ExtractionResult]:
    """
    Convenience function to run the full pipeline
    
    Args:
        pdf_path: Path to PDF
        output_path: Optional path to save results
        pages: Optional list of page numbers
    
    Returns:
        Dict of extraction results
    """
    pipeline = AlienTechPipeline()
    results = pipeline.process_pdf(pdf_path, pages)
    
    if output_path:
        pipeline.save_results(results, output_path)
    
    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python alien_pipeline.py <pdf_path> [output_path]")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    results = run_pipeline(pdf_path, output_path)
    
    # Print summary
    print("\n=== EXTRACTION SUMMARY ===")
    for field_name, result in results.items():
        status = "✓" if result.validation_status == 'valid' else "!"
        print(f"{status} {field_name}: {result.value} (conf: {result.confidence:.2f})")
