"""
Quick Test Script for Pipeline v2
==================================
Test the Nemotron-based pipeline on a single PDF.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from pipeline_v2.core.pipeline import ExtractionPipeline

def test_single_page():
    """Test on single page of P02.pdf."""
    
    pdf_path = "Techathon_Samples/P02.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"ERROR: {pdf_path} not found")
        return
    
    print("=" * 70)
    print("TESTING PIPELINE v2")
    print("=" * 70)
    print(f"PDF: {pdf_path}")
    print("Page: 2 (Personal Details)")
    print("=" * 70)
    
    pipeline = ExtractionPipeline()
    
    print("\nInitializing pipeline...")
    print("  - Nemotron table detector")
    print("  - PaddleOCR GPU (EN+HI)")
    print("  - Qwen-VL fallback")
    print("  - Encyclopedia validation")
    
    print("\nProcessing page 2...")
    result = pipeline.process(pdf_path, pages=[2])
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Total time: {result.total_processing_time_ms/1000:.1f}s")
    print(f"Overall confidence: {result.overall_confidence:.1%}")
    
    for page in result.pages:
        print(f"\nPage {page.page_num}:")
        print(f"  Nemotron time: {page.nemotron_time_ms:.0f}ms")
        print(f"  OCR time: {page.ocr_time_ms:.0f}ms")
        print(f"  VLM time: {page.vlm_time_ms:.0f}ms")
        print(f"  Validation time: {page.validation_time_ms:.0f}ms")
        print(f"  Total: {page.processing_time_ms:.0f}ms")
        
        print("\n  Fields:")
        print(f"  {'Field':<30} {'Value':<25} {'Conf':>8} {'Source':<10}")
        print("  " + "-" * 75)
        
        for field in sorted(page.fields, key=lambda f: f.name):
            val = field.value[:22] if field.value else "-"
            conf = f"{field.confidence:.0%}" if field.value else "-"
            print(f"  {field.name:<30} {val:<25} {conf:>8} {field.source:<10}")

if __name__ == "__main__":
    test_single_page()
