"""
Test Nemotron Pipeline on P10.pdf
==================================
Run the full 5-layer pipeline on a sample document.
"""

import os
import sys
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

print("="*70)
print("NEMOTRON PIPELINE TEST - P10.pdf")
print("="*70)
print()

# Test PDF path
pdf_path = r"C:\Users\aigcp_gpuadmin\Downloads\LICRFP\LICF\Techathon_Samples\P10.pdf"

if not os.path.exists(pdf_path):
    print(f"ERROR: {pdf_path} not found")
    print("Checking available samples...")
    import glob
    samples = glob.glob("Techathon_Samples/*.pdf")
    if samples:
        pdf_path = samples[0]
        print(f"Using: {pdf_path}")
    else:
        print("No PDF samples found")
        sys.exit(1)

print(f"PDF: {pdf_path}")
print()

# Import and run
from pipeline_v2.core.pipeline import ExtractionPipeline

print("Initializing pipeline...")
print("  [1] Preprocessor")
print("  [2] Nemotron Table Detector (LOCAL)")
print("  [3] PaddleOCR GPU (EN+HI)")
print("  [4] Qwen-VL Fallback")
print("  [5] Encyclopedia Validation")
print()

t0 = time.time()
pipeline = ExtractionPipeline()

print(f"Pipeline initialized in {(time.time()-t0):.1f}s")
print()

print("Processing Page 2 (Personal Details)...")
print()

try:
    result = pipeline.process(pdf_path, pages=[2])
    
    print("\n" + "="*70)
    print("EXTRACTION COMPLETE")
    print("="*70)
    print(f"Total time: {result.total_processing_time_ms/1000:.1f}s")
    print(f"Overall confidence: {result.overall_confidence:.1%}")
    print(f"Fields extracted: {len([f for f in result.all_fields() if f.value])}/{len(result.all_fields())}")
    print()
    
    # Detailed results
    for page in result.pages:
        print(f"Page {page.page_num}:")
        print(f"  Nemotron: {page.nemotron_time_ms:.0f}ms")
        print(f"  OCR: {page.ocr_time_ms:.0f}ms")
        print(f"  VLM: {page.vlm_time_ms:.0f}ms")
        print(f"  Validation: {page.validation_time_ms:.0f}ms")
        print()
        
        print("  Extracted Fields:")
        print(f"  {'Field':<30} {'Value':<25} {'Conf':>8} {'Source':<10}")
        print("  " + "-"*75)
        
        for field in sorted(page.fields, key=lambda f: f.name):
            val = field.value[:22] if field.value else "-"
            conf = f"{field.confidence:.0%}" if field.value else "-"
            print(f"  {field.name:<30} {val:<25} {conf:>8} {field.source:<10}")
        
        print()
    
    # Save results
    import json
    output_path = "p10_nemotron_result.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {output_path}")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
