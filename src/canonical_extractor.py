"""
CANONICAL TEMPLATE EXTRACTOR (The "Palantir" Approach)
1. Detect 4 corners of the form (Homography).
2. Warp perspective to align perfectly with standard template.
3. Slice pre-defined canonical coordinates for each field.
4. OCR the perfect slices.
NO HTML PARSING. NO GARBAGE. PURE PIXELS.
"""
import cv2
import numpy as np
import os
from typing import List, Dict, Tuple

# Canonical Field Definitions (Normalized 1000x1400 scale)
# Coordinates derived from analyzing the standard LIC Form 300 (Rev 2023)
# Format: (x_start, y_start, x_end, y_end) relative to 1000x1400
CANONICAL_FIELDS = {
    'Proposer_Name': (560, 130, 950, 165),      # Approx coords for Name
    'Proposer_Father': (560, 170, 950, 205),    # Father's Name
    'Proposer_Mother': (560, 210, 950, 245),    # Mother's Name
    'Proposer_Gender': (560, 250, 950, 285),    # Gender
    'Proposer_Marital': (560, 290, 950, 325),   # Marital Status
    'Proposer_DOB': (560, 330, 950, 365),       # DOB
    'Proposer_Age': (560, 370, 950, 405),       # Age
    'Proposer_BirthPlace': (560, 450, 950, 485),# Birth Place
    'Proposer_Nationality': (560, 530, 950, 565),# Nationality
    'Proposer_Citizenship': (560, 570, 950, 605),# Citizenship
    'Proposer_Address': (560, 650, 950, 750),   # Address (Large block)
    'Proposer_PIN': (560, 790, 950, 825),       # PIN
    'Proposer_Phone': (560, 830, 950, 865),     # Phone
}

class CanonicalExtractor:
    def __init__(self):
        self.template_h = 1400
        self.template_w = 1000
    
    def find_form_corners(self, image: np.ndarray) -> np.ndarray:
        """
        Detect the 4 corners of the LIC form in the image.
        Uses contour approximation on the largest rectangular object.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Threshold to find paper boundaries
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Approximate to polygon
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # If we found 4 points, we are good. If not, fallback to bounding box.
        if len(approx) == 4:
            return approx.reshape(4, 2)
        else:
            # Fallback: Bounding rect of largest contour
            x, y, w, h = cv2.boundingRect(largest_contour)
            return np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]])
    
    def warp_to_canonical(self, image: np.ndarray) -> np.ndarray:
        """
        Warp the image to the canonical 1000x1400 template size.
        """
        corners = self.find_form_corners(image)
        
        # Order corners: Top-Left, Top-Right, Bottom-Right, Bottom-Left
        # (Simple sorting for demo; robust version needs angle sorting)
        # Assuming standard scan orientation
        sorted_corners = corners[np.argsort(corners[:, 1])] # Sort by Y
        top_two = sorted_corners[:2]
        bottom_two = sorted_corners[2:]
        
        # Sort top two by X
        tl, tr = top_two[np.argsort(top_two[:, 0])]
        # Sort bottom two by X
        bl, br = bottom_two[np.argsort(bottom_two[:, 0])]
        
        src_pts = np.array([tl, tr, br, bl], dtype=np.float32)
        dst_pts = np.array([
            [0, 0], 
            [self.template_w, 0], 
            [self.template_w, self.template_h], 
            [0, self.template_h]
        ], dtype=np.float32)
        
        # Compute Homography
        M, _ = cv2.findHomography(src_pts, dst_pts)
        
        # Warp
        warped = cv2.warpPerspective(image, M, (self.template_w, self.template_h))
        return warped
    
    def extract_field_crop(self, warped_image: np.ndarray, field_name: str) -> np.ndarray:
        """
        Extract the exact crop for a specific field.
        """
        if field_name not in CANONICAL_FIELDS:
            raise ValueError(f"Unknown field: {field_name}")
        
        x1, y1, x2, y2 = CANONICAL_FIELDS[field_name]
        
        # Scale to actual image size
        h, w = warped_image.shape[:2]
        scale_x = w / 1000.0
        scale_y = h / 1400.0
        
        px1 = int(x1 * scale_x)
        py1 = int(y1 * scale_y)
        px2 = int(x2 * scale_x)
        py2 = int(y2 * scale_y)
        
        crop = warped_image[py1:py2, px1:px2]
        return crop
    
    def process_pdf_page(self, image_path: str) -> Dict[str, np.ndarray]:
        """
        Full pipeline: Load -> Warp -> Extract All Crops
        Returns dict of {field_name: crop_image}
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(image_path)
        
        print(f"  [1/3] Warping to canonical template...")
        warped = self.warp_to_canonical(img)
        
        crops = {}
        for field in CANONICAL_FIELDS.keys():
            crops[field] = self.extract_field_crop(warped, field)
        
        return crops, warped

def run_canonical_extraction(image_path: str, output_dir: str):
    extractor = CanonicalExtractor()
    print(f"Processing {image_path}...")
    
    crops, warped = extractor.process_pdf_page(image_path)
    
    # Save warped full image
    cv2.imwrite(os.path.join(output_dir, 'warped_full.png'), warped)
    
    # Save individual crops
    for name, crop in crops.items():
        path = os.path.join(output_dir, f'{name}.png')
        cv2.imwrite(path, crop)
        print(f"  Saved crop: {name} ({crop.shape})")
    
    print(f"All crops saved to {output_dir}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python canonical_extractor.py <image_path>")
        sys.exit(1)
    
    img_path = sys.argv[1]
    out_dir = img_path.replace('.png', '_crops')
    os.makedirs(out_dir, exist_ok=True)
    
    run_canonical_extraction(img_path, out_dir)
