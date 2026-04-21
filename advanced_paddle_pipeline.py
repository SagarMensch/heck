import os
import sys
import time
import json
import logging
import cv2
import numpy as np
import fitz  # PyMuPDF
from paddleocr import PaddleOCR, PPStructure, draw_structure_result, save_structure_res

# Configuration
PDF_PATH = r"Techathon_Samples\P10.pdf"
OUTPUT_DIR = r"output_advanced_paddle"
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AdvancedPaddle")

def pdf_to_images(pdf_path, dpi=300):
    """Convert PDF pages to high-resolution images."""
    logger.info(f"Rendering PDF: {pdf_path} at {dpi} DPI")
    doc = fitz.open(pdf_path)
    images = []
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    for i in range(len(doc)):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=mat)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        images.append(img)
    doc.close()
    return images

def run_advanced_pipeline():
    # Initialize PP-Structure with GPU and PP-OCRv4
    # Note: lang='hi' for Hindi support in PP-Structure
    logger.info("Initializing PP-Structure v2 (PP-OCRv4) on GPU...")
    
    # Advanced engine parameters
    engine = PPStructure(
        use_gpu=True,
        # use_tensorrt=True, # Enable if TensorRT is installed and configured
        # trt_min_subgraph_size=15,
        # trt_precision='fp16',
        layout=True,
        table=True,
        ocr=True,
        show_log=False,
        lang='hi', # Supports both Hindi and English
        structure_version='PP-StructureV2',
        det_model_dir=None, # Use default PP-OCRv4
        rec_model_dir=None, 
        type='structure'
    )

    images = pdf_to_images(PDF_PATH)
    total_start = time.time()
    
    all_results = []

    for i, img in enumerate(images):
        page_num = i + 1
        logger.info(f"Processing Page {page_num}/{len(images)}...")
        
        t0 = time.time()
        # Run the structure engine
        result = engine(img)
        t_elapsed = time.time() - t0
        
        logger.info(f"Page {page_num} processed in {t_elapsed:.2f}s")

        # Save visual result
        save_structure_res(result, OUTPUT_DIR, f"page_{page_num}")
        
        # Process and structure the output
        page_data = {
            "page": page_num,
            "processing_time": t_elapsed,
            "regions": []
        }
        
        for res in result:
            region_type = res.get('type')
            bbox = res.get('bbox')
            
            region_item = {
                "type": region_type,
                "bbox": bbox,
            }
            
            if region_type == 'table':
                # Table result contains html and cells
                region_item["html"] = res.get('res', {}).get('html')
                # We could also extract cell content here if needed
            else:
                # Text region
                text_res = res.get('res', [])
                lines = []
                if isinstance(text_res, list):
                    for line in text_res:
                        if isinstance(line, dict):
                            lines.append({
                                "text": line.get('text'),
                                "confidence": line.get('confidence'),
                                "bbox": line.get('text_region')
                            })
                region_item["lines"] = lines
            
            page_data["regions"].append(region_item)
        
        all_results.append(page_data)

    total_elapsed = time.time() - total_start
    logger.info(f"Pipeline complete! Total time: {total_elapsed:.2f}s")

    # Save structured JSON
    json_path = os.path.join(OUTPUT_DIR, "extraction_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # Generate a summary Markdown
    generate_summary_md(all_results, total_elapsed)

def generate_summary_md(results, total_time):
    md_path = os.path.join(OUTPUT_DIR, "EXTRACTION_SUMMARY.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Advanced PaddleOCR Extraction Summary\n\n")
        f.write(f"- **Total Processing Time:** {total_time:.2f}s\n")
        f.write(f"- **Pages Processed:** {len(results)}\n")
        f.write(f"- **Avg Speed:** {total_time/len(results):.2f}s/page\n\n")
        
        for page in results:
            f.write(f"## Page {page['page']}\n")
            f.write(f"Processing Time: {page['processing_time']:.2f}s\n\n")
            
            for region in page['regions']:
                rtype = region['type'].upper()
                f.write(f"### {rtype} Region\n")
                if region['type'] == 'table':
                    f.write("Table extracted. (HTML data saved in JSON)\n\n")
                else:
                    for line in region.get('lines', []):
                        f.write(f"- {line['text']} (conf: {line['confidence']:.2f})\n")
                    f.write("\n")
    logger.info(f"Summary saved to {md_path}")

if __name__ == "__main__":
    try:
        run_advanced_pipeline()
    except Exception as e:
        logger.exception("Pipeline failed")
        sys.exit(1)
