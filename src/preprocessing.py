"""Image Preprocessing Pipeline
=============================
Handles: PDF->Image, deskew, denoise, binarize, enhance, quality scoring,
page routing (only process pages with target fields).
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging
from PIL import Image

from src.config import MAX_IMAGE_DIMENSION, TARGET_DPI, QUALITY_SCORE_THRESHOLD
from src.form300_templates import PAGES_WITH_TARGET_FIELDS

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Full preprocessing pipeline for scanned proposal forms."""

    def __init__(self, target_dpi: int = 300, quality_threshold: float = QUALITY_SCORE_THRESHOLD):
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

        quality_score = self._compute_quality_score(image)
        is_usable = quality_score >= self.quality_threshold

        is_target_page = page_num in PAGES_WITH_TARGET_FIELDS

        if not is_usable:
            return {
                "page_num": page_num,
                "original": original,
                "preprocessed": None,
                "quality_score": quality_score,
                "is_usable": False,
                "is_target_page": is_target_page,
                "rejection_reason": f"Quality score {quality_score:.1f} below threshold {self.quality_threshold}",
                "preprocessing_steps": ["quality_check_failed"],
            }

        steps = []

        image = self._resize_if_needed(image, max_dim=MAX_IMAGE_DIMENSION)
        steps.append("resize_check")

        image = self._fix_orientation(image)
        steps.append("orientation_fix")

        image, skew_angle = self._deskew(image)
        steps.append("deskew(angle={:.2f})".format(skew_angle))

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        gray = self._remove_shadows(gray)
        steps.append("shadow_removal")

        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        steps.append("bilateral_denoise")

        enhanced = self._enhance_contrast(denoised)
        steps.append("clahe_contrast")

        binary = self._adaptive_binarize(enhanced)
        steps.append("adaptive_binarize")

        stroke_enhanced = self._enhance_strokes(binary)
        steps.append("stroke_enhancement")

        if len(image.shape) == 3:
            color_enhanced = self._enhance_color_for_vlm(image)
        else:
            color_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        return {
            "page_num": page_num,
            "original": original,
            "preprocessed_gray": enhanced,
            "preprocessed_binary": stroke_enhanced,
            "preprocessed_color": color_enhanced,
            "quality_score": quality_score,
            "is_usable": True,
            "is_target_page": is_target_page,
            "rejection_reason": None,
            "preprocessing_steps": steps,
            "dimensions": image.shape[:2],
        }

    def _pdf_to_images(self, pdf_path: Path) -> List[np.ndarray]:
        try:
            import fitz
            doc = fitz.open(str(pdf_path))
            images = []
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                zoom = self.target_dpi / 72.0
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)
                img_data = np.frombuffer(pix.samples, dtype=np.uint8)
                img_data = img_data.reshape(pix.height, pix.width, pix.n)
                if pix.n == 4:
                    img_data = cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR)
                elif pix.n == 3:
                    img_data = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
                images.append(img_data)
            doc.close()
            return images
        except ImportError:
            from pdf2image import convert_from_path
            pil_images = convert_from_path(str(pdf_path), dpi=self.target_dpi)
            return [cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) for img in pil_images]

    def _compute_quality_score(self, image: np.ndarray) -> float:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return float(laplacian_var)

    def _resize_if_needed(self, image: np.ndarray, max_dim: int = 4096) -> np.ndarray:
        h, w = image.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return image

    def _fix_orientation(self, image: np.ndarray) -> np.ndarray:
        return image

    def _deskew(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

        if lines is None:
            return image, 0.0

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            if abs(angle) < 15:
                angles.append(angle)

        if not angles:
            return image, 0.0

        median_angle = np.median(angles)

        if abs(median_angle) < 0.3:
            return image, median_angle

        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated, median_angle

    def _remove_shadows(self, gray: np.ndarray) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        bg = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel)
        shadow_free = cv2.divide(gray, bg, scale=255)
        return shadow_free

    def _enhance_contrast(self, gray: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return enhanced

    def _adaptive_binarize(self, gray: np.ndarray) -> np.ndarray:
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=21, C=10)
        return binary

    def _enhance_strokes(self, binary: np.ndarray) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        enhanced = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return enhanced

    def _enhance_color_for_vlm(self, image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_channel)
        enhanced_lab = cv2.merge([l_enhanced, a, b])
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced_bgr, -1, kernel)
        blended = cv2.addWeighted(sharpened, 0.7, enhanced_bgr, 0.3, 0)
        return blended

    def numpy_to_pil(self, image: np.ndarray) -> Image.Image:
        if len(image.shape) == 2:
            return Image.fromarray(image)
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
