"""
Adaptive Preprocessor
======================
Image preprocessing optimized for form documents.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Dict, List
import logging

logger = logging.getLogger(__name__)


class AdaptivePreprocessor:
    """
    Preprocess images for optimal OCR/VLM extraction.
    
    Pipeline:
        1. Shadow removal
        2. Denoising
        3. Deskew
        4. Contrast enhancement
    """
    
    def __init__(self, config):
        self.config = config
        
    def process(self, pil_img: Image.Image) -> Tuple[Image.Image, Dict]:
        """
        Process image with adaptive pipeline.
        
        Returns:
            Tuple of (processed_image, metadata)
        """
        img = np.array(pil_img)
        original_img = img.copy()
        
        steps = []
        
        # Convert to BGR for OpenCV
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Step 1: Remove shadows
        if self.config.remove_shadows:
            img = self._remove_shadows(img)
            steps.append("shadow_removal")
        
        # Step 2: Denoise
        if self.config.denoise:
            img = self._denoise(img)
            steps.append("denoise")
        
        # Step 3: Deskew
        if self._needs_deskew(img):
            img, angle = self._deskew(img)
            if abs(angle) > 0.5:
                steps.append(f"deskew_{angle:.1f}deg")
        
        # Step 4: Enhance contrast
        if self.config.enhance_contrast:
            img = self._enhance_contrast(img)
            steps.append("clahe")
        
        # Calculate quality score
        quality = self._calculate_quality(img)
        
        # Convert back to PIL
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_out = Image.fromarray(img_rgb)
        
        metadata = {
            "steps": steps,
            "quality_score": quality,
            "original_size": original_img.shape[:2],
            "processed_size": img.shape[:2]
        }
        
        return pil_out, metadata
    
    def _remove_shadows(self, img: np.ndarray) -> np.ndarray:
        """Remove shadows using morphological operations."""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Dilate to get background
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
            dilated = cv2.dilate(gray, kernel)
            
            # Gaussian blur
            bg = cv2.medianBlur(dilated, 21)
            
            # Difference
            diff = 255 - cv2.subtract(bg, gray)
            
            # Normalize
            norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
            
            # Apply to each channel
            result = img.copy()
            for c in range(3):
                result[:, :, c] = cv2.bitwise_and(img[:, :, c], norm)
            
            return result
            
        except Exception as e:
            logger.warning(f"Shadow removal failed: {e}")
            return img
    
    def _denoise(self, img: np.ndarray) -> np.ndarray:
        """Fast non-local means denoising."""
        try:
            return cv2.fastNlMeansDenoisingColored(
                img, None, 
                h=10, hColor=10,
                templateWindowSize=7,
                searchWindowSize=21
            )
        except Exception as e:
            logger.warning(f"Denoising failed: {e}")
            return img
    
    def _needs_deskew(self, img: np.ndarray) -> bool:
        """Check if image needs deskewing."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Check if significant text lines exist
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, 
                               minLineLength=100, maxLineGap=10)
        
        return lines is not None and len(lines) > 5
    
    def _deskew(self, img: np.ndarray) -> Tuple[np.ndarray, float]:
        """Deskew image."""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect edges and lines
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100,
                                   minLineLength=100, maxLineGap=10)
            
            if lines is None or len(lines) < 5:
                return img, 0.0
            
            # Calculate angles
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 != x1:
                    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                    # Filter horizontal lines
                    if abs(angle) < 45:
                        angles.append(angle)
            
            if not angles:
                return img, 0.0
            
            # Get median angle
            median_angle = np.median(angles)
            
            if abs(median_angle) < 0.5:
                return img, 0.0
            
            # Rotate
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            rotated = cv2.warpAffine(img, M, (w, h),
                                    flags=cv2.INTER_CUBIC,
                                    borderMode=cv2.BORDER_REPLICATE)
            
            return rotated, median_angle
            
        except Exception as e:
            logger.warning(f"Deskew failed: {e}")
            return img, 0.0
    
    def _enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        """CLAHE contrast enhancement."""
        try:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            lab = cv2.merge([l, a, b])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
        except Exception as e:
            logger.warning(f"Contrast enhancement failed: {e}")
            return img
    
    def _calculate_quality(self, img: np.ndarray) -> float:
        """Calculate image quality score."""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Laplacian variance (sharpness)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = laplacian.var()
            
            # Contrast
            contrast = gray.std()
            
            # Normalize to 0-100
            score = min(100, (sharpness / 500 * 30) + (contrast / 128 * 70))
            return score
            
        except Exception:
            return 50.0
