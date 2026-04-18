"""
Image Preprocessing Pipeline
=============================
Handles: PDF→Image, deskew, denoise, binarize, enhance, quality scoring.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging
from PIL import Image
import io

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Full preprocessing pipeline for scanned proposal forms."""

    def __init__(self, target_dpi: int = 300, quality_threshold: float = 50.0):
        self.target_dpi = target_dpi
        self.quality_threshold = quality_threshold

    def process_file(self, file_path: str) -> List[Dict]:
        """Process a PDF or image file, return list of preprocessed page images."""
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            pages = self._pdf_to_images(file_path)
        elif suffix in {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}:
            img = cv2.imread(str(file_path))
            if img is None:
                raise ValueError(f"Cannot read image: {file_path}")
            pages = [img]
        else:
            raise ValueError(f"Unsupported format: {suffix}")

        results = []
        for i, page_img in enumerate(pages):
            result = self.preprocess_single(page_img, page_num=i + 1)
            result["source_file"] = str(file_path)
            results.append(result)

        return results

    def preprocess_single(self, image: np.ndarray, page_num: int = 1) -> Dict:
        """Full preprocessing pipeline on a single page image."""
        original = image.copy()
        h, w = image.shape[:2]

        # Step 1: Quality assessment on original
        quality_score = self._compute_quality_score(image)
        is_usable = quality_score >= self.quality_threshold

        if not is_usable:
            return {
                "page_num": page_num,
                "original": original,
                "preprocessed": None,
                "quality_score": quality_score,
                "is_usable": False,
                "rejection_reason": f"Quality score {quality_score:.1f} below threshold {self.quality_threshold}",
                "preprocessing_steps": ["quality_check_failed"],
            }

        steps = []

        # Step 2: Resize if too large
        from src.config import MAX_IMAGE_DIMENSION
        image = self._resize_if_needed(image, max_dim=MAX_IMAGE_DIMENSION)
        steps.append("resize_check")

        # Step 3: Orientation detection & correction
        image = self._fix_orientation(image)
        steps.append("orientation_fix")

        # Step 4: Deskew
        image, skew_angle = self._deskew(image)
        steps.append(f"deskew(angle={skew_angle:.2f})")

        # Step 5: Convert to grayscale for processing
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Step 6: Shadow removal
        gray = self._remove_shadows(gray)
        steps.append("shadow_removal")

        # Step 7: Noise reduction (bilateral filter preserves edges)
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        steps.append("bilateral_denoise")

        # Step 8: Contrast enhancement (CLAHE)
        enhanced = self._enhance_contrast(denoised)
        steps.append("clahe_contrast")

        # Step 9: Adaptive binarization
        binary = self._adaptive_binarize(enhanced)
        steps.append("adaptive_binarize")

        # Step 10: Handwriting stroke enhancement
        stroke_enhanced = self._enhance_strokes(binary)
        steps.append("stroke_enhancement")

        # Keep color version for VLM (Qwen needs RGB)
        if len(image.shape) == 3:
            color_enhanced = self._enhance_color_for_vlm(image)
        else:
            color_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        return {
            "page_num": page_num,
            "original": original,
            "preprocessed_gray": enhanced,
            "preprocessed_binary": stroke_enhanced,
            "preprocessed_color": color_enhanced,  # For VLM input
            "quality_score": quality_score,
            "is_usable": True,
            "rejection_reason": None,
            "preprocessing_steps": steps,
            "dimensions": image.shape[:2],
        }

    def _pdf_to_images(self, pdf_path: Path) -> List[np.ndarray]:
        """Convert PDF pages to images."""
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(str(pdf_path))
            images = []
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                # Render at target DPI
                zoom = self.target_dpi / 72.0
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)
                img_data = np.frombuffer(pix.samples, dtype=np.uint8)
                img_data = img_data.reshape(pix.height, pix.width, pix.n)
                if pix.n == 4:  # RGBA
                    img_data = cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR)
                elif pix.n == 3:  # RGB
                    img_data = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
                images.append(img_data)
            doc.close()
            return images
        except ImportError:
            # Fallback to pdf2image
            from pdf2image import convert_from_path
            pil_images = convert_from_path(str(pdf_path), dpi=self.target_dpi)
            return [cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) for img in pil_images]

    def _compute_quality_score(self, image: np.ndarray) -> float:
        """Compute image quality score using Laplacian variance (blur detection)."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return float(laplacian_var)

    def _resize_if_needed(self, image: np.ndarray, max_dim: int = 4096) -> np.ndarray:
        """Resize image if any dimension exceeds max_dim."""
        h, w = image.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return image

    def _fix_orientation(self, image: np.ndarray) -> np.ndarray:
        """Detect and fix image orientation using PaddleOCR or heuristics."""
        # Simple orientation check based on text line detection
        # For production, use PaddleOCR's orientation classifier
        h, w = image.shape[:2]
        if h < w * 0.5:  # Likely landscape, might need rotation
            # Check if text runs horizontally
            pass  # Keep as-is for now, PaddleOCR handles this
        return image

    def _deskew(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Deskew image using Hough line detection."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Hough lines
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                                minLineLength=100, maxLineGap=10)

        if lines is None:
            return image, 0.0

        # Calculate median angle
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            if abs(angle) < 15:  # Only near-horizontal lines
                angles.append(angle)

        if not angles:
            return image, 0.0

        median_angle = np.median(angles)

        if abs(median_angle) < 0.3:  # Skip tiny corrections
            return image, median_angle

        # Rotate to correct skew
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h),
                                 flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)
        return rotated, median_angle

    def _remove_shadows(self, gray: np.ndarray) -> np.ndarray:
        """Remove shadows and uneven illumination."""
        # Morphological background estimation
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        bg = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel)
        # Divide to normalize illumination
        shadow_free = cv2.divide(gray, bg, scale=255)
        return shadow_free

    def _enhance_contrast(self, gray: np.ndarray) -> np.ndarray:
        """Apply CLAHE for contrast enhancement."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return enhanced

    def _adaptive_binarize(self, gray: np.ndarray) -> np.ndarray:
        """Adaptive thresholding for binarization."""
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, blockSize=21, C=10
        )
        return binary

    def _enhance_strokes(self, binary: np.ndarray) -> np.ndarray:
        """Enhance handwriting strokes for better recognition."""
        # Light morphological closing to connect broken strokes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        enhanced = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return enhanced

    def _enhance_color_for_vlm(self, image: np.ndarray) -> np.ndarray:
        """Enhance color image specifically for VLM input (Qwen2.5-VL)."""
        # Convert to LAB color space for better processing
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)

        # CLAHE on L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_channel)

        # Merge back
        enhanced_lab = cv2.merge([l_enhanced, a, b])
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        # Sharpen
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced_bgr, -1, kernel)

        # Blend original with sharpened (70% sharp, 30% original)
        blended = cv2.addWeighted(sharpened, 0.7, enhanced_bgr, 0.3, 0)
        return blended

    def numpy_to_pil(self, image: np.ndarray) -> Image.Image:
        """Convert OpenCV BGR numpy array to PIL Image."""
        if len(image.shape) == 2:  # Grayscale
            return Image.fromarray(image)
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
