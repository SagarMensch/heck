#!/usr/bin/env python
"""
Batch Process All LIC PDFs
Processes all 30 PDFs in Techathon_Samples folder
"""
import os
import sys
import json
import time
import logging
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

SAMPLES_DIR = r"C:\Users\aigcp_gpuadmin\Downloads\LICRFP\LICF\Techathon_Samples"
OUTPUT_DIR = r"C:\Users\aigcp_gpuadmin\Downloads\LICRFP\LICF\data\batch_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def batch_process():
    from src.accuracy_pipeline import AccuracyFirstPipeline
    
    print("=" * 70)
    print("BATCH PROCESSING - ALL LIC PDFs")
    print("=" * 70)
    
    # Find all PDFs
    pdf_files = list(Path(SAMPLES_DIR).glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDFs to process")
    
    pipeline = AccuracyFirstPipeline(use_qwen_fallback=False, confidence_threshold=0.85)
    
    all_results = []
    total_start = time.time()
    
    for i, pdf_path in enumerate(pdf_files, 1):
        pdf_name = pdf_path.name
        print(f"\n[{i}/{len(pdf_files)}] Processing {pdf_name}...")
        
        try:
            # Process first 5 pages only for speed
            results = pipeline.process_pdf(str(pdf_path), pages=[1, 2, 3, 4, 5])
            
            # Save individual result
            output_file = os.path.join(OUTPUT_DIR, pdf_name.replace('.pdf', '_result.json'))
            pipeline.save_results(results, output_file)
            
            # Aggregate stats
            all_results.append({
                'pdf': pdf_name,
                'total_fields': results['statistics']['total_fields'],
                'high_confidence': results['statistics']['high_conf'],
                'corrected': results['statistics']['corrected'],
                'failed': results['statistics']['failed'],
                'time': results['statistics']['processing_time_s']
            })
            
            print(f"  ✓ {results['statistics']['total_fields']} fields, "
                  f"{results['statistics']['high_conf']} high conf, "
                  f"{results['statistics']['processing_time_s']:.1f}s")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            all_results.append({
                'pdf': pdf_name,
                'error': str(e)
            })
    
    total_time = time.time() - total_start
    
    # Save batch summary
    summary_file = os.path.join(OUTPUT_DIR, "batch_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'total_pdfs': len(pdf_files),
            'total_time_s': total_time,
            'results': all_results
        }, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "=" * 70)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 70)
    print(f"Total PDFs: {len(pdf_files)}")
    print(f"Total Time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Avg per PDF: {total_time/len(pdf_files):.1f}s")
    
    successful = [r for r in all_results if 'error' not in r]
    if successful:
        avg_fields = sum(r['total_fields'] for r in successful) / len(successful)
        avg_time = sum(r['time'] for r in successful) / len(successful)
        print(f"Successful: {len(successful)}")
        print(f"Avg Fields/PDF: {avg_fields:.1f}")
        print(f"Avg Time/PDF: {avg_time:.1f}s")
    
    print(f"\nDetailed results: {summary_file}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    batch_process()
