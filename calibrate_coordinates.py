"""
CALIBRATION SCRIPT
Draws red boxes on the canonical warped image to verify coordinates.
Goal: Ensure boxes align PERFECTLY with handwritten values.
"""
import cv2
import numpy as np
import os
from src.canonical_extractor import CanonicalExtractor, CANONICAL_FIELDS

def calibrate(image_path: str, output_path: str):
    extractor = CanonicalExtractor()
    
    # 1. Load and Warp
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    
    print("Warping to canonical 1000x1400...")
    warped = extractor.warp_to_canonical(img)
    
    # 2. Draw Boxes
    h, w = warped.shape[:2]
    print(f"Warped image size: {w}x{h}")
    
    # Convert to BGR for drawing (if grayscale)
    if len(warped.shape) == 2:
        draw_img = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
    else:
        draw_img = warped.copy()
    
    print(f"\nDrawing {len(CANONICAL_FIELDS)} calibration boxes...")
    for field_name, coords in CANONICAL_FIELDS.items():
        x1, y1, x2, y2 = coords
        
        # Scale to 1000x1400 (since warped is already 1000x1400, scale=1)
        # But just in case warped is different:
        scale_x = w / 1000.0
        scale_y = h / 1400.0
        
        px1 = int(x1 * scale_x)
        py1 = int(y1 * scale_y)
        px2 = int(x2 * scale_x)
        py2 = int(y2 * scale_y)
        
        # Draw Red Rectangle (Thickness = 2)
        cv2.rectangle(draw_img, (px1, py1), (px2, py2), (0, 0, 255), 2)
        
        # Put Label
        cv2.putText(draw_img, field_name, (px1, py1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    # 3. Save
    cv2.imwrite(output_path, draw_img)
    print(f"\nCalibration image saved to: {output_path}")
    print("CHECK THIS IMAGE:")
    print("- Do red boxes cover ONLY the handwritten value?")
    print("- Do they EXCLUDE the printed labels?")
    print("- If NO, adjust CANONICAL_FIELDS coordinates and re-run.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python calibrate_coordinates.py <image_path>")
        sys.exit(1)
    
    img_path = sys.argv[1]
    out_path = img_path.replace('.png', '_calibrated.png')
    
    calibrate(img_path, out_path)
