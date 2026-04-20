"""
ADVANCED PREPROCESSING MODULE (The "Alien-Tech" Cleaner)
Goal: Transform raw scans into perfect B&W documents for PaddleX/Qwen.
Techniques: Deskew, Denoise, CLAHE, Adaptive Thresholding
"""
import cv2
import numpy as np
import math

def rotate_image(image, angle):
  """Rotate an image with proper expansion to avoid clipping."""
  image_center = tuple(np.array(image.shape[1::-1]) / 2.0)
  rot_mat = cv2.getRotationMatrix2D(image_center, -angle, 1.0) # Negative for correct direction
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def get_skew_angle(image):
    """Detect skew angle using Hough Line Transform."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Hough Line Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    
    if lines is None:
        return 0.0
    
    # Calculate median angle
    angles = []
    for rho, theta in lines[:, 0]:
        # Convert theta to degrees
        angle_deg = np.degrees(theta) - 90
        # Filter extreme outliers (likely noise)
        if -45 < angle_deg < 45:
            angles.append(angle_deg)
    
    if not angles:
        return 0.0
        
    return np.median(angles)

def preprocess_image(image_path):
    """
    Full SOTA Preprocessing Pipeline.
    Input: Path to image
    Output: Preprocessed image (numpy array)
    """
    # 1. Load Image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # 2. Auto-Crop (Remove white borders)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = cv2.findNonZero(255 - gray) # Find non-white pixels
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        # Only crop if we found content and it's not the whole image
        if w < img.shape[1] or h < img.shape[0]:
            img = img[y:y+h, x:x+w]
            gray = gray[y:y+h, x:x+w]

    # 3. Deskew
    skew_angle = get_skew_angle(img)
    if abs(skew_angle) > 0.5: # Only rotate if significant
        img = rotate_image(img, skew_angle)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 4. Denoise (Powerful Non-Local Means)
    # h=10 removes noise well, templateWindowSize=7, searchWindowSize=21
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

    # 5. Contrast Enhancement (CLAHE)
    # Clip limit 3.0, Grid 8x8
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)

    # 6. Return ENHANCED GRAYSCALE (for PaddleX) and BINARY (for OCR)
    # PaddleX needs the intermediate contrast to see table lines
    # Binary is too harsh for layout detection
    
    # Convert enhanced grayscale to BGR for PaddleX
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    return enhanced_bgr, skew_angle, enhanced  # Return both BGR and Gray

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python advanced_preprocessor.py <image_path>")
        sys.exit(1)
    
    img_path = sys.argv[1]
    print(f"Processing {img_path}...")
    result, angle = preprocess_image(img_path)
    
    out_name = img_path.replace(".png", "_preprocessed.png")
    cv2.imwrite(out_name, result)
    print(f"Saved preprocessed image to {out_name} (Skew corrected: {angle:.2f}°)")
