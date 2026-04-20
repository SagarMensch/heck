"""
DYNAMIC TABLE EXTRACTOR (The Real SOTA)
1. Use PaddleX to find the table bounding box.
2. Use OpenCV to detect grid lines and split into cells.
3. Identify Column 3 (Value Column).
4. Slice and OCR Column 3 ONLY.
"""
import os
import cv2
import numpy as np
from paddlex import create_pipeline

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "1"

class DynamicTableExtractor:
    def __init__(self):
        self.pipeline = create_pipeline(pipeline="layout_parsing")
    
    def detect_table_grid(self, image: np.ndarray):
        """
        Detect table cells using morphological operations.
        Returns: List of (x, y, w, h) for each cell, sorted row-major.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Invert for line detection
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )
        
        # Detect horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Detect vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
        
        # Combine
        table_mask = cv2.add(horizontal, vertical)
        
        # Find contours
        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cells = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 50 and h > 20:  # Filter noise
                cells.append((x, y, w, h))
        
        # Sort: Top-to-bottom, then Left-to-right
        # Group by Y (with tolerance), then sort by X
        cells.sort(key=lambda k: (k[1] // 10 * 10, k[0]))
        
        return cells
    
    def extract_from_image(self, image_path: str):
        """
        Main extraction pipeline.
        """
        img = cv2.imread(image_path)
        
        # 1. Detect Table Region (Use PaddleX to find table bbox)
        result = self.pipeline.predict(image_path)
        r = list(result)[0]
        prl = r["parsing_res_list"]
        
        table_bbox = None
        table_html = ""
        
        for item in prl:
            if item.get('block_label') == 'table':
                table_html = item.get('block_content', '')
                bbox = item.get('block_bbox', [])
                if len(bbox) == 4:
                    table_bbox = bbox
                break
        
        if not table_bbox:
            print("No table found!")
            return []
        
        # Crop to table region
        x1, y1, x2, y2 = map(int, table_bbox)
        table_crop = img[y1:y2, x1:x2]
        
        # 2. Detect Grid in Table Crop
        cells = self.detect_table_grid(table_crop)
        
        # 3. Organize into Rows/Cols
        # Assume 3 columns: [Index] [Label] [Value]
        # Sort cells into rows
        rows = []
        current_row = []
        if not cells:
            return []
            
        last_y = cells[0][1]
        row_threshold = 20  # Pixels
        
        sorted_cells = sorted(cells, key=lambda k: k[1]) # Sort by Y
        
        for i, cell in enumerate(sorted_cells):
            x, y, w, h = cell
            if abs(y - last_y) > row_threshold:
                # New row
                if current_row:
                    rows.append(sorted(current_row, key=lambda k: k[0])) # Sort row by X
                current_row = [cell]
                last_y = y
            else:
                current_row.append(cell)
        if current_row:
            rows.append(sorted(current_row, key=lambda k: k[0]))
        
        # 4. Extract Value Column (Col Index 2)
        extracted = []
        for r_idx, row in enumerate(rows):
            if len(row) >= 3:
                # Value is in 3rd cell (index 2)
                val_cell = row[2]
                vx, vy, vw, vh = val_cell
                
                # Crop value from table_crop
                val_img = table_crop[vy:vy+vh, vx:vx+vw]
                extracted.append(val_img)
            elif len(row) == 2:
                # Fallback: 2nd cell is value
                val_cell = row[1]
                vx, vy, vw, vh = val_cell
                val_img = table_crop[vy:vy+vh, vx:vx+vw]
                extracted.append(val_img)
        
        return extracted, table_crop

def run_dynamic_extraction(image_path):
    extractor = DynamicTableExtractor()
    cells, table_crop = extractor.extract_from_image(image_path)
    
    print(f"Detected {len(cells)} value cells.")
    
    # OCR each value cell
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(lang='en')
    
    results = []
    for i, cell_img in enumerate(cells):
        result = ocr.predict(cell_img)
        text = ""
        if result and len(result) > 0:
            for line in result[0]:
                text += line[1][0] + " "
        results.append(text.strip())
        print(f"Cell {i}: {text.strip()}")
    
    return results

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python dynamic_table_extractor.py <image_path>")
        sys.exit(1)
    
    run_dynamic_extraction(sys.argv[1])
