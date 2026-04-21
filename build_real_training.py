# Build larger training dataset from real PDFs
import os
import json
import random
from pathlib import Path
import fitz
from PIL import Image
import cv2
import numpy as np

# Paths
PDF_DIR = "data/lic_samples"
OUTPUT_DIR = "data/form300_factory"
NUM_PDFS = 30  # Use more PDFs

def extract_real_crops():
    """Extract real crops from PDFs for training."""
    
    pdf_paths = sorted(Path(PDF_DIR).glob("*.pdf"))
    print(f"Found {len(pdf_paths)} PDFs")
    
    samples = []
    
    for pdf_path in pdf_paths[:NUM_PDFS]:
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(min(3, len(doc))):  # First 3 pages
                page = doc.load_page(page_num)
                pix = page.get_pixmap(dpi=200)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                np_img = np.array(img)
                
                # Use PaddleOCR to detect and recognize text
                from paddleocr import PaddleOCR
                ocr = PaddleOCR(use_textline_orientation=False, lang='en')
                
                result = ocr.ocr(np_img)
                
                if result and len(result) > 0:
                    res = result[0]
                    texts = res.get('rec_texts', [])
                    scores = res.get('rec_scores', [])
                    
                    # Only keep high-confidence detections
                    for i, text in enumerate(texts):
                        if i < len(scores) and float(scores[i]) > 0.7:
                            # Clean the text
                            text = text.strip()
                            if len(text) >= 2 and len(text) <= 50:
                                samples.append({
                                    "text": text,
                                    "pdf": pdf_path.stem,
                                    "page": page_num + 1,
                                    "confidence": float(scores[i])
                                })
            
            doc.close()
            print(f"Processed {pdf_path.stem}: {len(samples)} total samples")
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
        
        if len(samples) >= 500:  # Collect at least 500 samples
            break
    
    print(f"\nTotal samples collected: {len(samples)}")
    
    # Save samples
    output_file = Path(OUTPUT_DIR) / "manifests" / "real_crops_ocr.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    
    print(f"Saved to {output_file}")
    return len(samples)

if __name__ == "__main__":
    extract_real_crops()