import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "1"
import json
import time
import logging
import sys
import numpy as np
import fitz
import cv2
from paddlex import create_pipeline

# Configuration
PDF_PATH = r"Techathon_Samples\P10.pdf"
OUT_DIR = r"output_advanced_paddlex"
IMG_DIR = os.path.join(OUT_DIR, "page_images")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AdvancedPaddleX")

def default_serializer(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return str(o)

def pdf_to_images(pdf_path, dpi=300):
    """Convert PDF pages to high-resolution images."""
    logger.info(f"Rendering PDF: {pdf_path} at {dpi} DPI")
    doc = fitz.open(pdf_path)
    img_paths = []
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    for i in range(len(doc)):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=mat)
        img_path = os.path.join(IMG_DIR, f"page_{i+1}.png")
        pix.save(img_path)
        img_paths.append(img_path)
    doc.close()
    return img_paths

def run_advanced_pipeline():
    # Load the most advanced layout parsing pipeline
    # This pipeline includes layout detection, table recognition, and OCR
    logger.info("Initializing PaddleX 'layout_parsing' pipeline...")
    
    # PaddleX usually detects GPU automatically. 
    # To force GPU, you might use device='gpu:0' if the API supports it,
    # but the standard create_pipeline call is usually enough.
    pipeline = create_pipeline(pipeline="layout_parsing")
    
    img_paths = pdf_to_images(PDF_PATH)
    img_paths = img_paths[:3]  # Process only first 3 pages for the demo
    total_start = time.time()
    
    all_results = []

    for i, img_path in enumerate(img_paths):
        page_num = i + 1
        logger.info(f"Processing Page {page_num}/{len(img_paths)}...")
        
        t0 = time.time()
        # Run prediction
        result = pipeline.predict(img_path)
        # result is a generator in PaddleX
        res = list(result)[0]
        t_elapsed = time.time() - t0
        
        logger.info(f"Page {page_num} processed in {t_elapsed:.2f}s")
        
        # Save structured results
        prl = res.get("parsing_res_list", [])
        page_data = {
            "page": page_num,
            "processing_time": t_elapsed,
            "regions": []
        }
        
        for item in prl:
            page_data["regions"].append({
                "type": item.get("block_label"),
                "text": item.get("block_content"),
                "bbox": item.get("block_bbox"),
                "score": item.get("score")
            })
        
        all_results.append(page_data)

    total_elapsed = time.time() - total_start
    logger.info(f"Pipeline complete! Total time: {total_elapsed:.2f}s")

    # Save structured JSON
    json_path = os.path.join(OUT_DIR, "extraction_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=default_serializer)
    
    # Generate Markdown Summary
    generate_summary_md(all_results, total_elapsed)

def generate_summary_md(results, total_time):
    md_path = os.path.join(OUT_DIR, "EXTRACTION_SUMMARY.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Advanced PaddleX (Layout Parsing) Extraction Summary\n\n")
        f.write(f"- **PDF:** {PDF_PATH}\n")
        f.write(f"- **Total Processing Time:** {total_time:.2f}s\n")
        f.write(f"- **Pages Processed:** {len(results)}\n")
        f.write(f"- **Avg Speed:** {total_time/len(results):.2f}s/page\n\n")
        
        for page in results:
            f.write(f"## Page {page['page']}\n")
            f.write(f"Processing Time: {page['processing_time']:.2f}s\n\n")
            
            for region in page['regions']:
                rtype = (region['type'] or "UNKNOWN").upper()
                text = region['text'] or ""
                if text.strip():
                    f.write(f"### {rtype}\n")
                    f.write(f"{text.strip()}\n\n")
    logger.info(f"Summary saved to {md_path}")

if __name__ == "__main__":
    try:
        run_advanced_pipeline()
    except Exception as e:
        logger.exception("Pipeline failed")
        sys.exit(1)
