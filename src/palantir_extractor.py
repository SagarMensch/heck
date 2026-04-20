"""
PALANTIR-STYLE EXTRCTOR
No black boxes. Pure engineering.
1. Detect table grid
2. Crop each cell
3. OCR individually
4. Map label→value
"""
import os
import cv2
import numpy as np
from typing import List, Dict, Tuple
import re

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "1"

class PalantirExtractor:
    """
    Extract handwritten values from LIC Form 300 using cell-level OCR
    """
    
    def __init__(self):
        self.ocr_en = None
        self.ocr_hi = None
        self._loaded = False
    
    def load_models(self):
        if self._loaded:
            return
        
        from paddleocr import PaddleOCR
        
        print("Loading PaddleOCR English (v3.4+)...")
        self.ocr_en = PaddleOCR(lang='en')
        
        print("Loading PaddleOCR Hindi...")
        self.ocr_hi = PaddleOCR(lang='hi')
        
        self._loaded = True
    
    def detect_table_grid(self, img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect table cells using line detection
        Returns: List of (x, y, w, h) for each cell
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
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
        
        # Extract bounding boxes
        cells = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 50 and h > 20:  # Filter noise
                cells.append((x, y, w, h))
        
        # Sort by position (top-to-bottom, left-to-right)
        cells.sort(key=lambda k: (k[1] // 10 * 10, k[0]))
        
        return cells
    
    def ocr_cell(self, cell_img: np.ndarray) -> str:
        """OCR a single cell"""
        self.load_models()
        
        # Try English first
        result = self.ocr_en.predict(cell_img)
        texts = []
        if result and len(result) > 0:
            for line in result[0]:
                texts.append(line[1][0])
        
        if texts:
            return ' '.join(texts)
        
        # Try Hindi
        result = self.ocr_hi.predict(cell_img)
        texts = []
        if result and len(result) > 0:
            for line in result[0]:
                texts.append(line[1][0])
        
        return ' '.join(texts) if texts else ""
    
    def extract_from_image(self, img_path: str) -> List[Dict]:
        """
        Extract all fields from image
        """
        self.load_models()
        
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read {img_path}")
        
        # Detect cells
        cells = self.detect_table_grid(img)
        
        print(f"Detected {len(cells)} cells")
        
        # OCR each cell
        results = []
        for i, (x, y, w, h) in enumerate(cells):
            # Crop cell with padding
            pad = 5
            cell_img = img[y-pad:y+h+pad, x-pad:x+w+pad]
            
            # OCR
            text = self.ocr_cell(cell_img)
            
            if text.strip():
                results.append({
                    'cell_id': i,
                    'bbox': (x, y, w, h),
                    'text': text.strip(),
                    'confidence': 0.85  # Placeholder
                })
        
        # Pair cells: odd=index 0,2,4... (labels), even=index 1,3,5... (values)
        # Assuming 2-column table
        paired = []
        for i in range(0, len(results) - 1, 2):
            label_cell = results[i]
            value_cell = results[i + 1] if i + 1 < len(results) else None
            
            paired.append({
                'label': label_cell['text'],
                'value': value_cell['text'] if value_cell else '',
                'label_bbox': label_cell['bbox'],
                'value_bbox': value_cell['bbox'] if value_cell else None,
                'confidence': min(label_cell['confidence'], value_cell['confidence'] if value_cell else 0)
            })
        
        return paired

def extract_palantir_style(img_path: str) -> List[Dict]:
    """Convenience function"""
    extractor = PalantirExtractor()
    return extractor.extract_from_image(img_path)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python palantir_extractor.py <image_path>")
        sys.exit(1)
    
    img_path = sys.argv[1]
    
    print("="*80)
    print("PALANTIR-STYLE CELL-LEVEL EXTRACTION")
    print("="*80)
    
    results = extract_palantir_style(img_path)
    
    print(f"\nExtracted {len(results)} label-value pairs:\n")
    
    for i, pair in enumerate(results):
        label = pair['label'][:60]
        value = pair['value'][:60]
        conf = pair['confidence']
        
        print(f"{i+1:3}. Label: {label:<60}")
        print(f"     Value: {value}")
        print(f"     Conf: {conf:.2f}\n")
    
    # Save to JSON
    import json
    output_path = img_path.replace('.png', '_cells.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Saved to: {output_path}")
