"""
SOTA COORDINATE SLICER
1. PaddleX finds the Table Bounding Box.
2. We slice that box into 3 vertical strips (Index, Label, Value).
3. We OCR ONLY the 3rd strip (Value).
NO HTML PARSING. NO GARBAGE.
"""
import os
import cv2
import numpy as np
from paddlex import create_pipeline

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "1"

# Initialize OCR once
from paddleocr import PaddleOCR
ocr_en = PaddleOCR(lang='en')
ocr_hi = PaddleOCR(lang='hi')

def extract_values_only(image_path: str):
    """
    1. Get Table BBox from PaddleX.
    2. Slice image into 3 columns.
    3. OCR only the 3rd column (Values).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    
    h, w = img.shape[:2]
    
    # 1. Get Table BBox
    pipeline = create_pipeline(pipeline="layout_parsing")
    result = pipeline.predict(image_path)
    r = list(result)[0]
    prl = r["parsing_res_list"]
    
    table_bbox = None
    for item in prl:
        if item.get('block_label') == 'table':
            bbox = item.get('block_bbox')
            if bbox:
                table_bbox = bbox
                break
    
    if not table_bbox:
        print("No table found.")
        return []
    
    # Crop to table
    x1, y1, x2, y2 = map(int, table_bbox)
    table_crop = img[y1:y2, x1:x2]
    th, tw = table_crop.shape[:2]
    
    # 2. Slice into 3 Columns
    # Approximate widths: Col1(10%), Col2(40%), Col3(50%)
    # We want Col3 (Values)
    col1_w = int(tw * 0.10)
    col2_w = int(tw * 0.40)
    col3_start = col1_w + col2_w
    
    # Slice ONLY Column 3 (Values)
    value_col_crop = table_crop[:, col3_start:]
    
    # 3. OCR the Value Column
    # We scan row-by-row using horizontal projection or fixed row height
    # Simple approach: Slice horizontal strips every ~50px (adjust based on th)
    # Better: Use PaddleOCR detection on the value_col_crop to find lines
    
    result = ocr_en.predict(value_col_crop)
    lines = []
    if result and len(result) > 0:
        for line in result[0]:
            text = line[1][0]
            lines.append(text)
    
    return lines, value_col_crop

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'Techathon_Samples/P02_page2_render.png'
    
    print(f"Slicing values from {path}...")
    lines, crop_img = extract_values_only(path)
    
    # Save the value column crop for verification
    cv2.imwrite('data/value_column_crop.png', crop_img)
    print(f"Saved value column crop to data/value_column_crop.png")
    
    print("\nExtracted Handwritten Values (Column 3):")
    for i, line in enumerate(lines):
        print(f"{i+1}. {line}")
