"""
Generate Pseudo Labels using Qwen-3B
======================================
Reads real training PDFs, uses FormTemplateMatcher to crop fields,
and asks Qwen-3B to extract text. High confidence extractions
are saved as pseudo-labels for TrOCR fine-tuning.
"""

import os
import json
import logging
from pathlib import Path
from PIL import Image
import fitz
from tqdm import tqdm

from src.preprocessing import ImagePreprocessor
from src.template_matcher import FormTemplateMatcher
from src.extractor import QwenExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    input_dir = Path("data/training_samples")
    output_dir = Path("data/pseudo_labeled")
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_dir.exists():
        logger.error(f"Input dir {input_dir} not found.")
        return
        
    pdfs = list(input_dir.glob("*.pdf"))
    logger.info(f"Found {len(pdfs)} PDFs for pseudo-labeling.")
    
    preprocessor = ImagePreprocessor()
    matcher = FormTemplateMatcher()
    qwen = QwenExtractor()
    
    metadata = []
    total_saved = 0
    
    for pdf_path in tqdm(pdfs, desc="Processing PDFs"):
        try:
            # We only process page 3 as it contains handwriting in these samples
            doc = fitz.open(str(pdf_path))
            if len(doc) >= 3:
                page = doc[2]
            else:
                page = doc[0]
                
            # Render to high-res image
            pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
            img_mode = "RGBA" if pix.alpha else "RGB"
            pil_img = Image.frombytes(img_mode, [pix.width, pix.height], pix.samples)
            if img_mode == "RGBA":
                pil_img = pil_img.convert("RGB")
            
            # Match templates
            crops = matcher.match_and_crop(pil_img)
            
            # Process crops with Qwen
            for field_name, crop_img in crops.items():
                prompt = f"What is the handwritten text in this image crop for the field '{field_name}'? Output ONLY a valid JSON dictionary in this exact format: {{\"fields\": {{\"{field_name}\": {{\"value\": \"extracted text\", \"confidence\": 0.95}}}}}}"
                res = qwen.extract_from_image(crop_img, 3, custom_prompt=prompt)
                
                val = ""
                conf = 0.0
                if "fields" in res and field_name in res["fields"]:
                    val = res["fields"][field_name].get("value", "")
                    conf = res["fields"][field_name].get("confidence", 0.0)
                elif "fields" in res and res["fields"]:
                    first_key = list(res["fields"].keys())[0]
                    val = res["fields"][first_key].get("value", "")
                    conf = res["fields"][first_key].get("confidence", 0.0)
                
                # Validation checks for pseudo-labeling
                is_valid = True
                
                # Strict confidence threshold
                if conf < 0.90:
                    is_valid = False
                
                # Length check
                if not val or len(val) < 2 or val == "extracted text":
                    is_valid = False
                    
                # Field specific regex/sanity checks
                val_clean = val.strip()
                if field_name == "Proposer_Mobile_Number" and not (val_clean.isdigit() and len(val_clean) >= 10):
                    is_valid = False
                elif field_name == "Proposer_PAN" and len(val_clean) != 10:
                    is_valid = False
                elif field_name == "Sum_Assured" and not val_clean.replace(",", "").isdigit():
                    is_valid = False
                
                if is_valid:
                    fname = f"{pdf_path.stem}_{field_name}.png"
                    fpath = images_dir / fname
                    crop_img.save(fpath)
                    
                    metadata.append({
                        "field_type": field_name,
                        "text": val_clean,
                        "image_path": str(fpath),
                    })
                    total_saved += 1
                    
            doc.close()
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            
    # Save metadata
    meta_file = output_dir / "metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
        
    logger.info(f"Pseudo-labeling complete! Saved {total_saved} high-quality real crops.")
    qwen.manager.unload_all()

if __name__ == "__main__":
    main()
