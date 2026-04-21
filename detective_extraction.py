import os
import json
import time
import logging
import sys
import numpy as np
import fitz
from pathlib import Path

# Force PaddleX to use GPU explicitly and bypass checks
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from paddlex import create_pipeline

PDF_PATH = r"C:\Users\aigcp_gpuadmin\Downloads\LICRFP\LICF\Techathon_Samples\P02.pdf"
OUT_DIR = r"C:\Users\aigcp_gpuadmin\Downloads\LICRFP\LICF\output_detective"
IMG_DIR = os.path.join(OUT_DIR, "page_images")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DetectivePaddle")

def default_serializer(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return str(o)

def render_pdf_to_images(pdf_path, dpi=300):
    logger.info(f"Rendering {pdf_path} to high-res {dpi} DPI images...")
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
        logger.info(f"Rendered Page {i+1} -> {img_path}")
    doc.close()
    return img_paths

def run_detective_pipeline():
    logger.info("Initializing PaddleX layout_parsing pipeline for FULL detective extraction on GPU...")
    # 'layout_parsing' is the most advanced pipeline in PaddleX for document understanding
    pipeline = create_pipeline(pipeline="layout_parsing", device="gpu:0")
    
    img_paths = render_pdf_to_images(PDF_PATH)
    total_start = time.time()
    
    ground_truth_doc = {
        "document": PDF_PATH,
        "total_pages": len(img_paths),
        "pages": []
    }

    # Process all pages
    for i, img_path in enumerate(img_paths):
        page_num = i + 1
        logger.info(f"==> Processing Page {page_num}/{len(img_paths)}...")
        
        t0 = time.time()
        result_gen = pipeline.predict(img_path)
        res = list(result_gen)[0]
        t_elapsed = time.time() - t0
        
        logger.info(f"Page {page_num} completed in {t_elapsed:.2f}s")
        
        parsing_res_list = res.get("parsing_res_list", [])
        
        page_data = {
            "page_num": page_num,
            "processing_time_s": round(t_elapsed, 2),
            "image_path": img_path,
            "blocks": []
        }
        
        for idx, item in enumerate(parsing_res_list):
            block_type = item.get("block_label", "unknown")
            block_content = item.get("block_content", "")
            block_bbox = item.get("block_bbox", [])
            score = item.get("score")
            
            # If it's a table, let's also capture the structure
            block_data = {
                "id": f"p{page_num}_b{idx}",
                "type": block_type,
                "bbox": block_bbox,
                "confidence": score,
                "text": block_content
            }
            
            page_data["blocks"].append(block_data)
            
        ground_truth_doc["pages"].append(page_data)
        
        # Save incrementally so we don't lose data
        incremental_path = os.path.join(OUT_DIR, "detective_ground_truth_in_progress.json")
        with open(incremental_path, 'w', encoding='utf-8') as f:
            json.dump(ground_truth_doc, f, indent=2, ensure_ascii=False, default=default_serializer)

    total_time = time.time() - total_start
    ground_truth_doc["total_processing_time_s"] = round(total_time, 2)
    
    final_path = os.path.join(OUT_DIR, "FINAL_detective_ground_truth.json")
    with open(final_path, 'w', encoding='utf-8') as f:
        json.dump(ground_truth_doc, f, indent=2, ensure_ascii=False, default=default_serializer)
        
    logger.info(f"DETECTIVE EXTRACTION COMPLETE! Total Time: {total_time:.2f}s for {len(img_paths)} pages.")
    logger.info(f"Saved complete ground truth to {final_path}")

if __name__ == "__main__":
    run_detective_pipeline()
