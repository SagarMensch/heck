"""Main Pipeline Orchestrator
============================
PaddleOCR-first hybrid pipeline:
  Phase 1: PaddleOCR (Hindi+English) on full pages → proximity-match to canonical bboxes
  Phase 2: Qwen-VL CoT + self-consistency for low-confidence fields
  Phase 3: Dual-model consensus for confidence calibration
  Phase 4: Validation + hallucination detection + semantic checks
"""

import json
import logging
import time
import uuid
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image

from src.config import (
    FORMS_INPUT_DIR, FORMS_OUTPUT_DIR,
    SUPPORTED_FORMATS, MAX_FORMS, FIELD_NAMES,
)
from src.preprocessing import ImagePreprocessor
from src.paddle_ocr_extractor import HybridExtractor
from src.validators import ExtractionResultBuilder
from src.form300_templates import PAGES_WITH_TARGET_FIELDS

logger = logging.getLogger(__name__)


def compute_cer(predicted: str, reference: str) -> float:
    if not reference and not predicted:
        return 0.0
    if not reference:
        return 1.0 if predicted else 0.0
    pred_chars = list(predicted)
    ref_chars = list(reference)
    n = len(ref_chars)
    m = len(pred_chars)
    if n == 0:
        return 1.0 if m > 0 else 0.0
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_chars[i - 1] == pred_chars[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[n][m] / n


def compute_field_accuracy(predicted_fields: Dict, ground_truth: Optional[Dict] = None) -> Dict:
    if ground_truth is None:
        return {"field_accuracy_exact": None, "avg_cer": None, "note": "No ground truth"}
    correct = 0
    total = 0
    cer_values = []
    for gt_field, gt_value in ground_truth.items():
        if not gt_value:
            continue
        total += 1
        pred_value = predicted_fields.get(gt_field, {}).get("value", "")
        if pred_value and pred_value.strip().lower() == str(gt_value).strip().lower():
            correct += 1
        cer_values.append(compute_cer(str(pred_value).strip(), str(gt_value).strip()))
    return {
        "field_accuracy_exact": correct / max(total, 1),
        "fields_correct": correct,
        "fields_total": total,
        "avg_cer": sum(cer_values) / max(len(cer_values), 1),
        "min_cer": min(cer_values) if cer_values else None,
        "max_cer": max(cer_values) if cer_values else None,
    }


class LICExtractionPipeline:
    """End-to-end pipeline: PaddleOCR → Qwen CoT → Consensus → Validate."""

    def __init__(self, use_qwen_fallback: bool = True, qwen_votes: int = 3):
        self.preprocessor = ImagePreprocessor()
        self.extractor = HybridExtractor(
            use_consensus=True,
            qwen_votes=qwen_votes,
            qwen_fallback_threshold=0.70,
            use_qwen_fallback=use_qwen_fallback,
        )
        self.result_builder = ExtractionResultBuilder()
        self.results = {}
        self.processing_status = {}
        self.use_qwen_fallback = use_qwen_fallback

    def process_folder(self, folder_path: str = None, ground_truth: Optional[Dict] = None) -> Dict:
        folder = Path(folder_path) if folder_path else FORMS_INPUT_DIR
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")

        files = []
        for ext in SUPPORTED_FORMATS:
            files.extend(folder.glob(f"*{ext}"))
            files.extend(folder.glob(f"*{ext.upper()}"))
        files = sorted(set(files))[:MAX_FORMS]

        logger.info(f"Found {len(files)} forms to process in {folder}")

        start_time = time.time()
        batch_results = []

        for i, file_path in enumerate(files):
            form_id = "FORM_{:03d}_{}".format(i + 1, file_path.stem)
            self.processing_status[form_id] = "processing"
            logger.info(f"[{i+1}/{len(files)}] Processing: {file_path.name}")

            try:
                result = self.process_single_form(str(file_path), form_id, ground_truth=ground_truth)
                self.results[form_id] = result
                self.processing_status[form_id] = result.get("form_status", "done")
                batch_results.append(result)
            except Exception as e:
                logger.error(f" -> FAILED: {e}")
                import traceback
                traceback.print_exc()
                self.processing_status[form_id] = "failed"
                self.results[form_id] = {
                    "form_id": form_id,
                    "form_status": "Failed",
                    "error": str(e),
                    "source_file": str(file_path),
                }
                batch_results.append(self.results[form_id])

        total_time = time.time() - start_time
        batch_kpis = self._compute_batch_kpis(batch_results, total_time, ground_truth)
        self._save_results(batch_results, batch_kpis)

        return {
            "batch_kpis": batch_kpis,
            "forms": batch_results,
            "processing_time_seconds": total_time,
        }

    def process_single_form(self, file_path: str, form_id: str = None,
                             ground_truth: Optional[Dict] = None) -> Dict:
        if form_id is None:
            form_id = "FORM_{}".format(uuid.uuid4().hex[:8])

        start = time.time()

        logger.info(f" Step 1: Preprocessing {Path(file_path).name}...")
        preprocessed_pages = self.preprocessor.process_file(file_path)

        usable_pages = [p for p in preprocessed_pages if p["is_usable"]]
        target_pages = [p for p in usable_pages if p.get("is_target_page", True)]

        if not target_pages:
            target_pages = usable_pages[:5]

        rejected_pages = [p for p in preprocessed_pages if not p["is_usable"]]
        skipped_pages = len(usable_pages) - len(target_pages)
        logger.info(f" Routed: {len(target_pages)} target pages, {skipped_pages} skipped")

        if not target_pages:
            return {
                "form_id": form_id,
                "form_status": "Rejected",
                "rejection_reason": "All pages failed quality check",
                "pages_rejected": len(rejected_pages),
                "source_file": file_path,
            }

        np_images = []
        page_num_map = []
        for page in target_pages:
            color_img = page.get("preprocessed_color", page.get("original"))
            if isinstance(color_img, Image.Image):
                color_img = np.array(color_img)
            np_images.append(color_img)
            page_num_map.append(page["page_num"])

        logger.info(f" Step 2: PaddleOCR + Qwen extraction ({len(np_images)} pages)...")
        raw_extraction = self.extractor.extract(np_images, page_num_map)

        logger.info(" Step 3: Validating and building result...")
        preprocessing_info = [
            {
                "page_num": p["page_num"],
                "quality_score": p["quality_score"],
                "is_usable": p["is_usable"],
                "is_target_page": p.get("is_target_page", True),
                "steps": p.get("preprocessing_steps", []),
            }
            for p in preprocessed_pages
        ]

        result = self.result_builder.build_result(
            form_id, raw_extraction, preprocessing_info
        )
        result["source_file"] = file_path
        result["processing_time_seconds"] = time.time() - start
        result["pages_processed"] = len(target_pages)
        result["pages_skipped"] = skipped_pages

        if ground_truth:
            predicted_fields = result.get("fields", {})
            gt_for_form = ground_truth.get(form_id, ground_truth)
            result["accuracy_metrics"] = compute_field_accuracy(predicted_fields, gt_for_form)
        else:
            result["accuracy_metrics"] = compute_field_accuracy(result.get("fields", {}), None)

        return result

    def _compute_batch_kpis(self, results: List[Dict], total_time: float,
                             ground_truth: Optional[Dict] = None) -> Dict:
        total_forms = len(results)
        processed = sum(1 for r in results if r.get("form_status") not in ("Failed", "Rejected"))
        rejected = sum(1 for r in results if r.get("form_status") == "Rejected")
        failed = sum(1 for r in results if r.get("form_status") == "Failed")
        needs_review = sum(1 for r in results if r.get("form_status") == "Needs Review")

        total_fields = 0
        extracted_fields = 0
        all_confidences = []
        all_cers = []

        for r in results:
            kpis = r.get("kpis", {})
            total_fields += kpis.get("total_fields_expected", 0)
            extracted_fields += kpis.get("fields_extracted", 0)
            if kpis.get("overall_confidence"):
                all_confidences.append(kpis["overall_confidence"])
            acc = r.get("accuracy_metrics", {})
            if acc.get("avg_cer") is not None:
                all_cers.append(acc["avg_cer"])

        field_coverage = extracted_fields / max(total_fields, 1)
        avg_confidence = float(sum(all_confidences) / max(len(all_confidences), 1))
        rejection_rate = rejected / max(total_forms, 1)
        manual_correction_rate = needs_review / max(total_forms, 1)
        avg_cer = float(sum(all_cers) / max(len(all_cers), 1)) if all_cers else None

        kpis = {
            "total_forms": total_forms,
            "forms_processed": processed,
            "forms_rejected": rejected,
            "forms_failed": failed,
            "forms_needs_review": needs_review,
            "field_coverage": round(field_coverage, 4),
            "avg_confidence": round(avg_confidence, 4),
            "rejection_rate": round(rejection_rate, 4),
            "manual_correction_rate": round(manual_correction_rate, 4),
            "total_processing_time_seconds": round(total_time, 2),
            "avg_time_per_form_seconds": round(total_time / max(total_forms, 1), 2),
            "throughput_forms_per_minute": round(total_forms / max(total_time / 60, 0.01), 2),
        }
        if avg_cer is not None:
            kpis["avg_cer"] = round(avg_cer, 4)
        return kpis

    def _save_results(self, results: List[Dict], batch_kpis: Dict):
        output_dir = FORMS_OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        for result in results:
            form_id = result.get("form_id", "unknown")
            out_file = output_dir / f"{form_id}_result.json"
            clean = self._make_serializable(result)
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(clean, f, indent=2, ensure_ascii=False)

        summary_file = output_dir / "batch_summary.json"
        summary = {
            "batch_kpis": batch_kpis,
            "form_count": len(results),
            "form_ids": [r.get("form_id") for r in results],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Results saved to {output_dir}")

    def _make_serializable(self, obj):
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()
                    if not isinstance(v, (np.ndarray, Image.Image))}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return None
        return obj

    def cleanup(self):
        self.extractor.cleanup()
