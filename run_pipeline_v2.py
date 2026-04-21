"""
LIC Extraction Pipeline v2 - Fast Optimized Version
=====================================================

Usage:
    python run_pipeline_v2.py Techathon_Samples/P02.pdf
    python run_pipeline_v2.py Techathon_Samples/P02.pdf --pages 2,3,4
    python run_pipeline_v2.py Techathon_Samples/P02.pdf --fast
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from pipeline_v2.core.pipeline import ExtractionPipeline
from pipeline_v2.core.models import ProcessingConfig

def main():
    parser = argparse.ArgumentParser(
        description="LIC Form Extraction Pipeline v2 - Nemotron + OCR + VLM Fallback"
    )
    
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--pages", default=None,
                       help="Pages to process: '2', '1-5', '2,3,4'")
    parser.add_argument("--output", default="data/output_v2",
                       help="Output directory")
    parser.add_argument("--fast", action="store_true",
                       help="Fast mode: skip VLM fallback")
    parser.add_argument("--dpi", type=int, default=150,
                       help="Render DPI")
    parser.add_argument("--no-nemotron", action="store_true",
                       help="Skip Nemotron, use fallback detection")
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    
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
    
    # Create config
    config = ProcessingConfig(
        dpi=args.dpi,
        enhance_contrast=True,
        remove_shadows=True,
        denoise=True,
        use_cached_structure=True,
        ocr_batch_size=16,
        vlm_confidence_threshold=0.7 if not args.fast else 1.0,  # Skip VLM in fast mode
        max_vlm_calls_per_page=5 if not args.fast else 0,
    )
    
    # Check if Nemotron available
    if args.no_nemotron:
        config.layout_model = "fallback"
        config.use_cached_structure = False
    
    print(f"\n{'='*70}")
    print(f"LIC Form Extraction Pipeline v2")
    print(f"{'='*70}")
    print(f"PDF: {args.pdf_path}")
    print(f"Pages: {pages if pages else 'ALL'}")
    print(f"DPI: {args.dpi}")
    print(f"Fast Mode: {args.fast}")
    print(f"Nemotron: {'Disabled' if args.no_nemotron else 'Enabled'}")
    print(f"VLM Fallback: {'Disabled' if args.fast else 'Enabled'}")
    print(f"{'='*70}\n")
    
    # Initialize pipeline
    try:
        pipeline = ExtractionPipeline(config)
    except Exception as e:
        print(f"ERROR: Failed to initialize pipeline: {e}")
        sys.exit(1)
    
    # Process
    try:
        result = pipeline.process(args.pdf_path, pages=pages)
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"EXTRACTION COMPLETE")
        print(f"{'='*70}")
        print(f"Document ID: {result.document_id}")
        print(f"Pages Processed: {result.total_pages}")
        print(f"Total Time: {result.total_processing_time_ms/1000:.1f}s")
        print(f"Overall Confidence: {result.overall_confidence:.1%}")
        print(f"Form Status: {result.form_status}")
        print(f"\nFields:")
        print(f"  Total: {len(result.all_fields)}")
        print(f"  Extracted: {len(result.extracted_fields)}")
        print(f"  Missing: {len(result.missing_fields)}")
        print(f"  Needs Review: {result.needs_review_count}")
        print(f"{'='*70}")
        
        # Save results
        os.makedirs(args.output, exist_ok=True)
        
        basename = Path(args.pdf_path).stem
        output_path = os.path.join(args.output, f"{basename}_v2_result.json")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {output_path}")
        
        # Print extracted fields
        print(f"\n{'-'*70}")
        print(f"{'Field':<35} {'Value':<25} {'Conf':>8}")
        print(f"{'-'*70}")
        
        for field in sorted(result.all_fields, key=lambda f: f.name):
            val = field.value[:23] if field.value else "-"
            conf = f"{field.confidence:.0%}" if field.value else "-"
            flags = ""
            if field.source == "vlm":
                flags += " [VLM]"
            elif field.source == "ocr":
                flags += " [OCR]"
            print(f"{field.name:<35} {val:<25} {conf:>8}{flags}")
        
        print(f"{'-'*70}")
        
    except Exception as e:
        print(f"ERROR: Processing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
