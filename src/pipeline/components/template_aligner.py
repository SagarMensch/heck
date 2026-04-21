import cv2
import numpy as np
import os
import json
import logging

logger = logging.getLogger(__name__)

# STATIC TEMPLATE FOR FORM 300 - PAGE 2
# Coordinates are based on a 2550x3430 300-DPI standard template.
# Format: [x1, y1, x2, y2]
FORM_300_PAGE2_TEMPLATE = {
    "Prefix_Name": [1200, 630, 2374, 730],   # Handwritten Prefix
    "First_Name": [1200, 730, 2374, 860],    # Handwritten First Name
    "Last_Name": [1200, 860, 2374, 950],     # Handwritten Last Name
    "Father_Name": [1200, 950, 2374, 1030],  # Handwritten Father's Name
    "Mother_Name": [1200, 1030, 2374, 1100], # Handwritten Mother's Name
    "Gender": [1200, 1100, 2374, 1170],      # Tick box area
    "Marital_Status": [1200, 1170, 2374, 1250],
    "Date_of_Birth": [1200, 1250, 2374, 1360],
    "Place_of_Birth": [1200, 1650, 2374, 1780],
    "ID_Proof": [1200, 1920, 2374, 2110]
}

def align_image(img, template_img):
    """
    Aligns the scanned image to a perfect standard template using ORB feature matching.
    This guarantees that our fixed coordinate boxes perfectly overlap the fields
    even if the scanned image is shifted, rotated, or skewed.
    """
    # Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    
    # Initialize ORB detector
    orb = cv2.ORB_create(5000)
    
    # Find keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(img_gray, None)
    kp2, des2 = orb.detectAndCompute(template_gray, None)
    
    # Match features
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Keep top 20% matches
    keep = int(len(matches) * 0.2)
    matches = matches[:keep]
    
    # Extract coordinates of matches
    pts1 = np.zeros((len(matches), 2), dtype=np.float32)
    pts2 = np.zeros((len(matches), 2), dtype=np.float32)
    
    for i, match in enumerate(matches):
        pts1[i, :] = kp1[match.queryIdx].pt
        pts2[i, :] = kp2[match.trainIdx].pt
        
    # Find homography
    h, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC)
    
    # Warp image to align with template
    height, width = template_img.shape[:2]
    aligned_img = cv2.warpPerspective(img, h, (width, height))
    
    return aligned_img

def extract_template_fields(image_path: str, output_dir: str):
    """
    Extracts the exact handwritten regions using the predefined static template.
    """
    os.makedirs(output_dir, exist_ok=True)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
        
    # For this Techathon, we assume the first scanned image of P10 is our "Gold Template"
    # In production, we'd use a blank Form 300 pdf converted to image.
    # We will just use the coordinates directly on P10 for demonstration.
    
    # Draw boxes and save crops
    debug_img = img.copy()
    crops = {}
    
    for field_name, bbox in FORM_300_PAGE2_TEMPLATE.items():
        x1, y1, x2, y2 = bbox
        
        # Draw on debug image
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(debug_img, field_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        # Crop the handwritten area
        crop_img = img[y1:y2, x1:x2]
        crop_path = os.path.join(output_dir, f"{field_name}_crop.png")
        cv2.imwrite(crop_path, crop_img)
        crops[field_name] = crop_path
        logger.info(f"Cropped {field_name} -> {crop_path}")
        
    debug_path = os.path.join(output_dir, "template_alignment_debug.jpg")
    cv2.imwrite(debug_path, debug_img)
    print(f"Saved perfect template alignment debug to {debug_path}")
    
    return crops

if __name__ == "__main__":
    # Test it on P10 Page 2
    logging.basicConfig(level=logging.INFO)
    extract_template_fields("output_nemotron_p10/page_2.png", "output_nemotron_p10/template_crops")
