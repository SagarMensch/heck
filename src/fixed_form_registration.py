"""Production-grade registration for fixed-template LIC Form 300 pages.

This module does three things:
1. Builds reference pages from a local sample PDF bundled with the repo.
2. Registers an incoming scanned page to the reference geometry.
3. Lets downstream extractors crop fields in aligned reference space.

The core design is deterministic and self-contained. It does not rely on any
external template service or third-party geometry annotations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from src.config import DATA_DIR, PROJECT_ROOT, TARGET_DPI

logger = logging.getLogger(__name__)


@dataclass
class RegistrationResult:
    success: bool
    page_num: int
    aligned_bgr: np.ndarray
    reference_size: Tuple[int, int]
    target_to_reference: np.ndarray
    reference_to_target: np.ndarray
    method: str
    match_count: int = 0
    inlier_count: int = 0
    inlier_ratio: float = 0.0
    mean_reprojection_error: float = 0.0
    ecc_correlation: float = 0.0
    structure_overlap: float = 0.0
    failure_reason: str = ""
    metadata: Dict[str, float] = field(default_factory=dict)


class ReferencePageStore:
    """Renders and caches local reference pages from bundled sample PDFs."""

    def __init__(
        self,
        reference_pdf: Optional[Path] = None,
        render_dpi: int = TARGET_DPI,
        cache_dir: Optional[Path] = None,
    ):
        self.render_dpi = render_dpi
        self.cache_dir = cache_dir or (DATA_DIR / "reference_pages")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.reference_pdf = reference_pdf or self._resolve_reference_pdf()
        self._page_cache: Dict[int, np.ndarray] = {}

    def _resolve_reference_pdf(self) -> Path:
        candidates = [
            PROJECT_ROOT / "Scanned_Proposal_Form_No_300_Sample_Only (2).pdf",
            PROJECT_ROOT / "Techathon_Samples" / "P02.pdf",
        ]
        for candidate in candidates:
            if candidate.exists():
                logger.info("Using reference PDF: %s", candidate)
                return candidate
        raise FileNotFoundError(
            "Could not find a local reference PDF. Expected one of: "
            + ", ".join(str(p) for p in candidates)
        )

    def get_page(self, page_num: int) -> np.ndarray:
        if page_num in self._page_cache:
            return self._page_cache[page_num].copy()

        cached_png = self.cache_dir / f"page_{page_num}_{self.render_dpi}dpi.png"
        if cached_png.exists():
            img = cv2.imread(str(cached_png), cv2.IMREAD_COLOR)
            if img is not None:
                self._page_cache[page_num] = img
                return img.copy()

        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError("PyMuPDF (fitz) is required to render reference pages") from exc

        doc = fitz.open(str(self.reference_pdf))
        if page_num < 1 or page_num > doc.page_count:
            doc.close()
            raise ValueError(
                f"Reference PDF {self.reference_pdf} does not contain page {page_num}"
            )

        page = doc.load_page(page_num - 1)
        pix = page.get_pixmap(dpi=self.render_dpi, alpha=False)
        doc.close()

        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cv2.imwrite(str(cached_png), img)
        self._page_cache[page_num] = img
        return img.copy()


class FixedFormRegistrar:
    """Registers target pages into the reference page coordinate system."""

    def __init__(
        self,
        reference_store: Optional[ReferencePageStore] = None,
        match_max_dim: int = 1800,
        min_good_matches: int = 40,
        min_inliers: int = 25,
        min_inlier_ratio: float = 0.25,
        max_mean_error: float = 8.0,
        min_structure_overlap: float = 0.22,
    ):
        self.reference_store = reference_store or ReferencePageStore()
        self.match_max_dim = match_max_dim
        self.min_good_matches = min_good_matches
        self.min_inliers = min_inliers
        self.min_inlier_ratio = min_inlier_ratio
        self.max_mean_error = max_mean_error
        self.min_structure_overlap = min_structure_overlap

    def register(self, page_bgr: np.ndarray, page_num: int) -> RegistrationResult:
        reference_bgr = self.reference_store.get_page(page_num)
        ref_h, ref_w = reference_bgr.shape[:2]

        h_target = self._estimate_homography_orb(page_bgr, reference_bgr)
        method = "orb_homography"

        if h_target is None:
            h_target = self._estimate_homography_content_box(page_bgr, reference_bgr)
            method = "content_box_fallback"

        if h_target is None:
            identity = np.eye(3, dtype=np.float32)
            aligned = cv2.resize(
                page_bgr,
                (ref_w, ref_h),
                interpolation=cv2.INTER_CUBIC,
            )
            return RegistrationResult(
                success=False,
                page_num=page_num,
                aligned_bgr=aligned,
                reference_size=(ref_w, ref_h),
                target_to_reference=identity,
                reference_to_target=identity,
                method="resize_fallback",
                failure_reason="Could not estimate registration transform",
            )

        aligned = cv2.warpPerspective(
            page_bgr,
            h_target,
            (ref_w, ref_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

        ecc_correlation = 0.0
        ecc_delta = self._refine_with_ecc(aligned, reference_bgr)
        if ecc_delta is not None:
            h_target = ecc_delta @ h_target
            aligned = cv2.warpPerspective(
                page_bgr,
                h_target,
                (ref_w, ref_h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255),
            )
            ecc_correlation = float(self._estimate_correlation(aligned, reference_bgr))
            method = f"{method}+ecc"

        try:
            h_inverse = np.linalg.inv(h_target)
        except np.linalg.LinAlgError:
            h_inverse = np.eye(3, dtype=np.float32)

        stats = self._evaluate_alignment(page_bgr, reference_bgr, h_target, aligned, method)

        success = (
            (stats["inlier_count"] >= self.min_inliers)
            and (stats["inlier_ratio"] >= self.min_inlier_ratio)
            and (stats["mean_reprojection_error"] <= self.max_mean_error)
            and (stats["structure_overlap"] >= self.min_structure_overlap)
        )

        if stats["method"] == "content_box_fallback":
            success = stats["structure_overlap"] >= (self.min_structure_overlap * 0.8)

        return RegistrationResult(
            success=success,
            page_num=page_num,
            aligned_bgr=aligned,
            reference_size=(ref_w, ref_h),
            target_to_reference=h_target,
            reference_to_target=h_inverse,
            method=method if success else f"{method}_low_quality",
            match_count=stats["match_count"],
            inlier_count=stats["inlier_count"],
            inlier_ratio=stats["inlier_ratio"],
            mean_reprojection_error=stats["mean_reprojection_error"],
            ecc_correlation=ecc_correlation,
            structure_overlap=stats["structure_overlap"],
            failure_reason="" if success else stats["failure_reason"],
            metadata=stats,
        )

    def aligned_bbox_to_source_bbox(
        self,
        bbox: Tuple[int, int, int, int],
        registration: RegistrationResult,
        source_shape: Tuple[int, int],
    ) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        quad = np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        mapped = cv2.perspectiveTransform(quad, registration.reference_to_target).reshape(-1, 2)
        h_src, w_src = source_shape[:2]
        min_x = int(max(0, np.floor(mapped[:, 0].min())))
        min_y = int(max(0, np.floor(mapped[:, 1].min())))
        max_x = int(min(w_src, np.ceil(mapped[:, 0].max())))
        max_y = int(min(h_src, np.ceil(mapped[:, 1].max())))
        return min_x, min_y, max_x, max_y

    def get_reference_page(self, page_num: int) -> np.ndarray:
        return self.reference_store.get_page(page_num)

    def subtract_reference_form(
        self,
        aligned_bgr: np.ndarray,
        page_num: int,
        threshold: int = 28,
    ) -> np.ndarray:
        """Suppress fixed printed template content using the local blank reference."""
        reference_bgr = self.reference_store.get_page(page_num)
        if aligned_bgr.shape[:2] != reference_bgr.shape[:2]:
            reference_bgr = cv2.resize(
                reference_bgr,
                (aligned_bgr.shape[1], aligned_bgr.shape[0]),
                interpolation=cv2.INTER_CUBIC,
            )

        gray_aligned = self._prepare_gray(aligned_bgr)
        gray_reference = self._prepare_gray(reference_bgr)
        diff = cv2.absdiff(gray_aligned, gray_reference)

        # Handwritten/filled content appears as structural residual over the blank form.
        residual_mask = (diff > threshold).astype(np.uint8) * 255
        residual_mask = cv2.medianBlur(residual_mask, 3)
        residual_mask = cv2.morphologyEx(
            residual_mask,
            cv2.MORPH_CLOSE,
            np.ones((3, 3), np.uint8),
            iterations=1,
        )

        # Keep dark residual ink from the aligned image; set everything else to white.
        ink_source = cv2.cvtColor(gray_aligned, cv2.COLOR_GRAY2BGR)
        cleaned = np.full_like(ink_source, 255)
        cleaned[residual_mask > 0] = ink_source[residual_mask > 0]
        return cleaned

    def draw_template_overlay(self, aligned_bgr: np.ndarray, template) -> np.ndarray:
        from src.form300_templates import bbox_to_pixels

        overlay = aligned_bgr.copy()
        h_img, w_img = overlay.shape[:2]
        for field in template.fields:
            x1, y1, x2, y2 = bbox_to_pixels(field.bbox_norm, w_img, h_img, pad_norm=0.0)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                overlay,
                field.name,
                (x1, max(12, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
        return overlay

    def _estimate_homography_orb(
        self,
        target_bgr: np.ndarray,
        reference_bgr: np.ndarray,
    ) -> Optional[np.ndarray]:
        target_small, sx_t, sy_t = self._resize_for_matching(target_bgr)
        reference_small, sx_r, sy_r = self._resize_for_matching(reference_bgr)

        gray_t = self._prepare_gray(target_small)
        gray_r = self._prepare_gray(reference_small)

        orb = cv2.ORB_create(
            nfeatures=10000,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            patchSize=31,
            fastThreshold=5,
        )

        kp_t, des_t = orb.detectAndCompute(gray_t, None)
        kp_r, des_r = orb.detectAndCompute(gray_r, None)

        if des_t is None or des_r is None or not kp_t or not kp_r:
            return None

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        knn_matches = matcher.knnMatch(des_t, des_r, k=2)

        good = []
        for pair in knn_matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

        if len(good) < self.min_good_matches:
            return None

        src_small = np.float32([kp_t[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_small = np.float32([kp_r[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        h_small, mask = cv2.findHomography(src_small, dst_small, cv2.RANSAC, 5.0)
        if h_small is None:
            return None

        scale_target = np.array(
            [[sx_t, 0.0, 0.0], [0.0, sy_t, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        scale_reference = np.array(
            [[sx_r, 0.0, 0.0], [0.0, sy_r, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        h_full = np.linalg.inv(scale_reference) @ h_small @ scale_target
        return h_full.astype(np.float32)

    def _estimate_homography_content_box(
        self,
        target_bgr: np.ndarray,
        reference_bgr: np.ndarray,
    ) -> Optional[np.ndarray]:
        gray_t = self._prepare_gray(target_bgr)
        gray_r = self._prepare_gray(reference_bgr)

        quad_t = self._content_quad(gray_t)
        quad_r = self._content_quad(gray_r)
        if quad_t is None or quad_r is None:
            return None

        h_box = cv2.getPerspectiveTransform(quad_t, quad_r)
        if h_box is None:
            return None
        return h_box.astype(np.float32)

    def _content_quad(self, gray: np.ndarray) -> Optional[np.ndarray]:
        mask = (gray < 245).astype(np.uint8) * 255
        points = cv2.findNonZero(mask)
        if points is None:
            h_img, w_img = gray.shape[:2]
            return np.array(
                [[0, 0], [w_img - 1, 0], [w_img - 1, h_img - 1], [0, h_img - 1]],
                dtype=np.float32,
            )
        x, y, w_box, h_box = cv2.boundingRect(points)
        return np.array(
            [[x, y], [x + w_box, y], [x + w_box, y + h_box], [x, y + h_box]],
            dtype=np.float32,
        )

    def _refine_with_ecc(
        self,
        aligned_bgr: np.ndarray,
        reference_bgr: np.ndarray,
    ) -> Optional[np.ndarray]:
        aligned_small, sx_a, sy_a = self._resize_for_matching(aligned_bgr, max_dim=1400)
        reference_small, sx_r, sy_r = self._resize_for_matching(reference_bgr, max_dim=1400)

        gray_a = self._prepare_gray(aligned_small).astype(np.float32) / 255.0
        gray_r = self._prepare_gray(reference_small).astype(np.float32) / 255.0

        if gray_a.shape != gray_r.shape:
            return None

        warp_small = np.eye(2, 3, dtype=np.float32)
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            80,
            1e-5,
        )
        try:
            cv2.findTransformECC(
                gray_r,
                gray_a,
                warp_small,
                cv2.MOTION_AFFINE,
                criteria,
                None,
                1,
            )
        except cv2.error:
            return None

        scale_small = np.array(
            [[sx_a, 0.0, 0.0], [0.0, sy_a, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        scale_ref = np.array(
            [[sx_r, 0.0, 0.0], [0.0, sy_r, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        affine_small = np.eye(3, dtype=np.float32)
        affine_small[:2, :] = warp_small
        affine_full = np.linalg.inv(scale_ref) @ affine_small @ scale_small
        return affine_full.astype(np.float32)

    def _evaluate_alignment(
        self,
        target_bgr: np.ndarray,
        reference_bgr: np.ndarray,
        h_target: np.ndarray,
        aligned_bgr: np.ndarray,
        method: str,
    ) -> Dict[str, float]:
        target_small, sx_t, sy_t = self._resize_for_matching(target_bgr)
        reference_small, sx_r, sy_r = self._resize_for_matching(reference_bgr)
        gray_t = self._prepare_gray(target_small)
        gray_r = self._prepare_gray(reference_small)

        orb = cv2.ORB_create(nfeatures=6000, fastThreshold=8)
        kp_t, des_t = orb.detectAndCompute(gray_t, None)
        kp_r, des_r = orb.detectAndCompute(gray_r, None)

        stats = {
            "match_count": 0,
            "inlier_count": 0,
            "inlier_ratio": 0.0,
            "mean_reprojection_error": 9999.0,
            "structure_overlap": self._structure_overlap(aligned_bgr, reference_bgr),
            "failure_reason": "",
            "method": method,
        }

        if des_t is None or des_r is None or not kp_t or not kp_r:
            stats["failure_reason"] = "No feature descriptors for alignment validation"
            return stats

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        knn_matches = matcher.knnMatch(des_t, des_r, k=2)
        good = []
        for pair in knn_matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

        stats["match_count"] = len(good)
        if len(good) < 4:
            stats["failure_reason"] = "Too few feature matches after registration"
            return stats

        src = np.float32([kp_t[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp_r[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        scale_target = np.array(
            [[sx_t, 0.0, 0.0], [0.0, sy_t, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        scale_reference = np.array(
            [[sx_r, 0.0, 0.0], [0.0, sy_r, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        h_small = scale_reference @ h_target @ np.linalg.inv(scale_target)

        pred = cv2.perspectiveTransform(src, h_small)
        errors = np.linalg.norm(pred.reshape(-1, 2) - dst.reshape(-1, 2), axis=1)
        inliers = int(np.sum(errors < 6.0))

        stats["inlier_count"] = inliers
        stats["inlier_ratio"] = inliers / max(len(errors), 1)
        stats["mean_reprojection_error"] = float(np.median(errors)) if len(errors) else 9999.0

        if stats["inlier_count"] < self.min_inliers:
            stats["failure_reason"] = "Low feature inlier count"
        elif stats["inlier_ratio"] < self.min_inlier_ratio:
            stats["failure_reason"] = "Low feature inlier ratio"
        elif stats["mean_reprojection_error"] > self.max_mean_error:
            stats["failure_reason"] = "High reprojection error"
        elif stats["structure_overlap"] < self.min_structure_overlap:
            stats["failure_reason"] = "Low line-structure overlap after alignment"
        return stats

    def _resize_for_matching(
        self,
        image: np.ndarray,
        max_dim: Optional[int] = None,
    ) -> Tuple[np.ndarray, float, float]:
        limit = max_dim or self.match_max_dim
        h_img, w_img = image.shape[:2]
        scale = min(1.0, float(limit) / float(max(h_img, w_img)))
        if scale >= 1.0:
            return image.copy(), 1.0, 1.0
        resized = cv2.resize(
            image,
            (max(1, int(round(w_img * scale))), max(1, int(round(h_img * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        return resized, scale, scale

    def _prepare_gray(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def _structure_overlap(self, aligned_bgr: np.ndarray, reference_bgr: np.ndarray) -> float:
        gray_a = self._prepare_gray(aligned_bgr)
        gray_r = self._prepare_gray(reference_bgr)

        mask_a = self._structure_mask(gray_a)
        mask_r = self._structure_mask(gray_r)

        union = np.logical_or(mask_a, mask_r).sum()
        if union == 0:
            return 0.0
        inter = np.logical_and(mask_a, mask_r).sum()
        return float(inter / union)

    def _structure_mask(self, gray: np.ndarray) -> np.ndarray:
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            11,
        )
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 45))
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
        structure = cv2.add(horizontal, vertical)
        structure = cv2.dilate(structure, np.ones((3, 3), np.uint8), iterations=1)
        return structure > 0

    def _estimate_correlation(self, aligned_bgr: np.ndarray, reference_bgr: np.ndarray) -> float:
        gray_a = self._prepare_gray(aligned_bgr).astype(np.float32)
        gray_r = self._prepare_gray(reference_bgr).astype(np.float32)
        if gray_a.shape != gray_r.shape:
            return 0.0
        gray_a -= gray_a.mean()
        gray_r -= gray_r.mean()
        denom = float(np.linalg.norm(gray_a) * np.linalg.norm(gray_r))
        if denom == 0.0:
            return 0.0
        return float(np.sum(gray_a * gray_r) / denom)
