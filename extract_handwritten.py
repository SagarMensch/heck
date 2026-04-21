#!/usr/bin/env python
"""
EXTRACT HANDWRITTEN VALUES FROM LIC FORM 300
Direct approach: OCR each cell separately
"""
import os
import sys
import cv2
import numpy as np
sys.path.insert(0, 'src')

from paddleocr import PaddleOCR
from paddlex import create_pipeline
import fitz  # PyMuPDF

# Initialize OCR engines
print("Loading OCR models...")
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "1"
ocr_en = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True, show_log=False)
ocr_hi = PaddleOCR(use_angle_cls=True, lang='hi', use_gpu=True, show_log=False)
layout_pipeline = create_pipeline(pipeline="layout_parsing")
print("Models loaded!")

PDF_PATH = r'Techathon_Samples\P02.pdf'
OUTPUT_DIR = r'data/handwritten_test'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_text_from_image(img):
    """Extract text using bilingual OCR"""
    results = []
    
    # English OCR
    en_result = ocr_en.ocr(img, cls=True)
    if en_result and en_result[0]:
        for line in en_result[0]:
            text = line[1][0]
            conf = line[1][1]
            results.append((text, conf, 'en'))
    
    # Hindi OCR
    hi_result = ocr_hi.ocr(img, cls=True)
    if hi_result and hi_result[0]:
        for line in hi_result[0]:
            text = line[1][0]
            conf = line[1][1]
            results.append((text, conf, 'hi'))
    
    return results

def process_pdf_with_cell_ocr(pdf_path, page_num=2):
    """
    Process a single page, extracting text from table cells
    """
    print(f"\nProcessing {pdf_path}, page {page_num}...")
    
    # Convert PDF page to image
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num - 1)
    pix = page.get_pixmap(dpi=200)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
    doc.close()
    
    # Save for reference
    img_path = os.path.join(OUTPUT_DIR, f'page{page_num}.png')
    cv2.imwrite(img_path, img)
    print(f"Saved page image to {img_path}")
    
    # Get layout from PaddleX
    print("Running layout detection...")
    result = layout_pipeline.predict(img_path)
    r = list(result)[0]
    prl = r["parsing_res_list"]
    
    print(f"\nFound {len(prl)} layout regions")
    
    # Extract text from each region
    extracted_data = []
    for i, item in enumerate(prl):
        txt = item.get("block_content", "").strip()
        label = item.get("block_label", "text")
        bbox = item.get("block_bbox", [])
        
        if txt:
            extracted_data.append({
                'label': label,
                'text': txt[:200],  # First 200 chars
                'bbox': bbox
            })
    
    # Print what we found
    print("\n" + "="*80)
    print("EXTRACTED TABLE CONTENT:")
    print("="*80)
    
    for item in extracted_data[:20]:  # First 20 items
        label = item['label']
        text = item['text']
        print(f"[{label}] {text}")
    
    return extracted_data

if __name__ == "__main__":
    data = process_pdf_with_cell_ocr(PDF_PATH, page_num=2)
    print(f"\nTotal items extracted: {len(data)}")
