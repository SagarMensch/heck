"""
Quick auto-fill gold labels using PaddleOCR - minimal version.
"""

import csv
import sys

# Use PaddleOCR for auto-labeling
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
import os

print("Initializing PaddleOCR...", flush=True)
ocr = PaddleOCR(use_textline_orientation=False, lang='en', rec_batch_num=1)

# Read the gold label sheet
gold_csv_path = "data/form300_factory/manifests/gold_label_sheet.csv"
output_csv_path = "data/form300_factory/manifests/gold_label_sheet_filled.csv"

rows = []
with open(gold_csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"Processing {len(rows)} gold labels...", flush=True)

# Process only first 10 for quick test
processed = 0
for i, row in enumerate(rows[:10]):
    image_path = row['image_path']
    
    # Convert relative path to absolute if needed
    if not os.path.exists(image_path):
        image_path = os.path.abspath(image_path)
    
    try:
        # Run OCR on the image
        result = ocr.ocr(image_path, cls=False)
        
        extracted_text = ""
        if result and len(result) > 0:
            res0 = result[0]
            texts = res0.get('rec_texts')
            if texts and len(texts) > 0:
                extracted_text = str(texts[0])
        
        row['gold_text'] = extracted_text
        row['review_status'] = 'auto-filled'
        processed += 1
        
        print(f"[{i+1}/10] {row['field_name']}: {extracted_text[:30]}", flush=True)
            
    except Exception as e:
        print(f"Error: {e}", flush=True)
        row['gold_text'] = ''
        row['review_status'] = 'error'

print(f"Auto-filled {processed} gold labels", flush=True)

# Write the output CSV
with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved to {output_csv_path}", flush=True)