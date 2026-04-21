"""
LIC Extraction Pipeline — CLI Entry Point
==========================================
Run the full 5-layer extraction pipeline on LIC Proposal Form 300 PDFs.

Usage:
    python run_pipeline.py Techathon_Samples/P02.pdf
    python run_pipeline.py Techathon_Samples/P02.pdf --pages 1-5
    python run_pipeline.py Techathon_Samples/P02.pdf --model Qwen/Qwen2.5-VL-3B-Instruct
"""

import os
import sys
import argparse
import logging

def main():
    parser = argparse.ArgumentParser(description="LIC 5-Layer Extraction Pipeline")
    parser.add_argument("pdf_path", help="Path to LIC Proposal Form PDF")
    parser.add_argument("--pages", default=None, help="Page(s) to process: '2', '1-5', '2,3,5' (default: all pages)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help="VLM model name (default: Qwen/Qwen2.5-VL-7B-Instruct)")
    parser.add_argument("--dpi", type=int, default=150, help="Render DPI (default: 150)")
    parser.add_argument("--output", default="data/output", help="Output directory")
    parser.add_argument("--fast", action="store_true",
                        help="Fast mode: only half_top + half_bottom tiles")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if not os.path.exists(args.pdf_path):
        print(f"ERROR: PDF not found: {args.pdf_path}")
        sys.exit(1)

    pages = None
    if args.pages:
        if '-' in args.pages:
            parts = args.pages.split('-')
            start, end = int(parts[0]), int(parts[1])
            pages = list(range(start, end + 1))
        elif ',' in args.pages:
            pages = [int(p.strip()) for p in args.pages.split(',')]
        else:
            pages = [int(args.pages)]

    tile_cols = 1 if args.fast else 2
    tile_rows = 1 if args.fast else 3

    from src.pipeline.orchestrator import PipelineOrchestrator

    pipeline = PipelineOrchestrator(
        vlm_model=args.model,
        dpi=args.dpi,
        tile_cols=tile_cols,
        tile_rows=tile_rows,
        output_dir=args.output,
    )

    print(f"\n{'='*60}")
    print(f"LIC 5-Layer Extraction Pipeline")
    print(f"  PDF:     {args.pdf_path}")
    print(f"  Pages:   {pages if pages else 'ALL'}")
    print(f"  Model:   {args.model}")
    print(f"  DPI:     {args.dpi}")
    print(f"  Tiles:   {'2 (fast)' if args.fast else '8 (6 grid + 2 half)'}")
    print(f"  Output:  {args.output}")
    print(f"{'='*60}\n")

    result = pipeline.process_pdf(args.pdf_path, pages=pages)

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"  Form status:     {result.form_status}")
    print(f"  Confidence:      {result.overall_confidence:.1%}")
    print(f"  Fields found:    {result.kpis.get('fields_extracted', 0)}/{result.kpis.get('total_fields', 0)}")
    print(f"  Auto-accepted:   {result.kpis.get('auto_accepted', 0)}")
    print(f"  Needs review:    {result.kpis.get('needs_review', 0)}")
    print(f"  Low confidence:  {result.kpis.get('low_confidence', 0)}")
    print(f"  Rejected:        {result.kpis.get('rejected', 0)}")
    print(f"  KB corrected:    {result.kpis.get('kb_corrected', 0)}")
    print(f"  OCR verified:    {result.kpis.get('ocr_verified', 0)}")
    print(f"  Bbox located:    {result.kpis.get('bbox_located', 0)}")
    print(f"  Time:            {result.processing_time_ms/1000:.1f}s")
    print(f"{'='*60}")

    print(f"\nExtracted fields:")
    print(f"{'-'*70}")
    print(f"{'Field':<35} {'Value':<22} {'Conf':>5} {'Status'}")
    print(f"{'-'*70}")
    for f in sorted(result.all_fields(), key=lambda x: x.field_name):
        val = f.value[:20] if f.value else "-"
        conf = f"{f.confidence:.0%}" if f.value else "-"
        flags = ""
        if f.kb_corrected:
            flags += " KB"
        if f.ocr_verified:
            flags += " OCR"
        if f.value_bbox:
            flags += " LOC"
        print(f"{f.field_name:<35} {val:<22} {conf:>5} {f.review_category[:12]}{flags}")
    print(f"{'-'*70}")


if __name__ == "__main__":
    main()
