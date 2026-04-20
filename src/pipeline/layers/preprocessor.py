"""
Layer 1: Advanced Image Preprocessor
=====================================
Conditions page images for maximum downstream model accuracy.
Outputs BGR 3-channel images (required by PaddleOCR) and RGB PIL images (for VLM).

Pipeline:
  1. Shadow removal (morphological)
  2. Denoising (fastNlMeans)
  3. Deskewing (minAreaRect)
  4. CLAHE contrast enhancement
  5. Quality scoring (blur detection, contrast ratio)
"""

import logging
import numpy as np
import cv2
from typing import List, Tuple, Optional
from PIL import Image

from src.pipeline.models.schemas import PageImage

logger = logging.getLogger(__name__)


class Preprocessor:

    def process_page(self, page_num: int, pil_img: Image.Image, dpi: int = 150) -> PageImage:
        img_rgb = np.array(pil_img)
        if img_rgb.ndim == 2:
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        h, w = img_bgr.shape[:2]

        steps = []
        quality = self._quality_score(img_bgr)

        img_bgr, step = self._remove_shadows(img_bgr)
        if step:
            steps.append(step)

        img_bgr, step = self._denoise(img_bgr)
        if step:
            steps.append(step)

        img_bgr, step = self._deskew(img_bgr)
        if step:
            steps.append(step)

        img_bgr, step = self._enhance_contrast(img_bgr)
        if step:
            steps.append(step)

        quality_after = self._quality_score(img_bgr)
        pil_out = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

        return PageImage(
            page_num=page_num,
            pil_image=pil_out,
            cv_image=img_bgr,
            width=w,
            height=h,
            dpi=dpi,
            quality_score=quality_after,
            preprocessing_applied=steps,
        )

    def _remove_shadows(self, img: np.ndarray) -> Tuple[np.ndarray, str]:
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            dilated = cv2.dilate(gray, np.ones((7, 7), np.uint8), iterations=2)
            bg = cv2.medianBlur(dilated, 21)
            diff = 255 - cv2.absdiff(gray, bg)
            norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
            img_out = img.copy()
            for c in range(3):
                img_out[:, :, c] = cv2.bitwise_and(img[:, :, c], norm)
            return img_out, "shadow_removal"
        except Exception:
            return img, ""

    def _denoise(self, img: np.ndarray) -> Tuple[np.ndarray, str]:
        try:
            out = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
            return out, "denoise"
        except Exception:
            return img, ""

    def _deskew(self, img: np.ndarray) -> Tuple[np.ndarray, str]:
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            coords = np.column_stack(np.where(gray > 0))
            if len(coords) < 100:
                return img, ""
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            if abs(angle) < 0.3:
                return img, ""
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return out, f"deskew_{angle:.1f}deg"
        except Exception:
            return img, ""

    def _enhance_contrast(self, img: np.ndarray) -> Tuple[np.ndarray, str]:
        try:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            return out, "clahe"
        except Exception:
            return img, ""

    def _quality_score(self, img: np.ndarray) -> float:
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            contrast = float(np.std(gray))
            return min(100.0, (blur_score / 100.0 * 50.0) + (contrast / 128.0 * 50.0))
        except Exception:
            return 50.0

    def create_tiles(self, page: PageImage, cols: int = 2, rows: int = 3, overlap: int = 50) -> List[Tuple[Image.Image, str]]:
        pil = page.pil_image
        w, h = pil.size
        tw, th = w // cols, h // rows
        tiles = []

        for r in range(rows):
            for c in range(cols):
                left = max(0, c * tw - overlap)
                upper = max(0, r * th - overlap)
                right = min(w, (c + 1) * tw + overlap)
                lower = min(h, (r + 1) * th + overlap)
                tile = pil.crop((left, upper, right, lower))
                tiles.append((tile, f"tile_r{r}_c{c}"))

        tiles.append((pil.crop((0, 0, w, h // 2 + overlap)), "half_top"))
        tiles.append((pil.crop((0, max(0, h // 2 - overlap), w, h)), "half_bottom"))

        return tiles
