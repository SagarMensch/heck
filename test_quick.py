import json
import time
import torch
from pathlib import Path
from src.preprocessing import ImagePreprocessor
from src.extractor import QwenExtractor

def run_quick_test():
    print("--- QUICK EXTRACTION TEST (PAGE 3 ONLY) ---")
    
    # 1. Preprocess only the third page
    file_path = "C:/Users/aigcp_gpuadmin/Downloads/LICRFP/LICF/data/lic_samples/P02.pdf"
    preprocessor = ImagePreprocessor()
    
    # Extract just the third page (index 2) using PyMuPDF to avoid loading all 28 pages
    import fitz
    doc = fitz.open(file_path)
    page = doc.load_page(2)
    pix = page.get_pixmap(dpi=300)
    import numpy as np
    import cv2
    img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img_data = cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3:
        img_data = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
    doc.close()
    
    print("Running preprocessing on Page 3...")
    preprocessed = preprocessor.preprocess_single(img_data, page_num=3)
    
    if not preprocessed["is_usable"]:
        print("Page rejected due to low quality.")
        return
        
    pil_img = preprocessor.numpy_to_pil(preprocessed["preprocessed_color"])
    
    # 2. Run TrOCR-First architecture
    print("Loading Models (TrOCR Primary + Qwen Fallback)...")
    from src.extractor import DualModelExtractor
    extractor = DualModelExtractor()
    
    print("Starting generation on GPU...")
    start = time.time()
    result = extractor.extract([pil_img])
    end = time.time()
    
    print(f"\nExtraction completed in {end-start:.2f} seconds!")
    print("\n--- RAW JSON EXTRACTION OUTPUT ---")
    print(json.dumps(result, indent=2))
    
if __name__ == "__main__":
    run_quick_test()
