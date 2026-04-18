"""
Main Pipeline Orchestrator
============================
Ties everything together: ingest → preprocess → extract → validate → output.
"""

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

from src.config import (
    FORMS_INPUT_DIR, FORMS_OUTPUT_DIR, LOGS_DIR,
    SUPPORTED_FORMATS, MAX_FORMS, FIELD_NAMES,
)
from src.preprocessing import ImagePreprocessor
from src.extractor import DualModelExtractor, QwenExtractor, ModelManager
from src.validators import ExtractionResultBuilder

logger = logging.getLogger(__name__)


class LICExtractionPipeline:
    """End-to-end pipeline for LIC Proposal Form data extraction."""

    def __init__(self):
        self.preprocessor = ImagePreprocessor()
        self.model_manager = ModelManager()
        self.extractor = DualModelExtractor()
        self.result_builder = ExtractionResultBuilder()
        self.results = {}  # form_id -> result
        self.processing_status = {}  # form_id -> status

    def process_folder(self, folder_path: str = None) -> Dict:
        """Process all forms in a folder. Main entry point."""
        folder = Path(folder_path) if folder_path else FORMS_INPUT_DIR

        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")

        # Find all supported files
        files = []
        for ext in SUPPORTED_FORMATS:
            files.extend(folder.glob(f"*{ext}"))
            files.extend(folder.glob(f"*{ext.upper()}"))
        files = sorted(set(files))[:MAX_FORMS]

        logger.info(f"Found {len(files)} forms to process in {folder}")

        start_time = time.time()
        batch_results = []

        for i, file_path in enumerate(files):
            form_id = f"FORM_{i+1:03d}_{file_path.stem}"
            self.processing_status[form_id] = "processing"
            logger.info(f"[{i+1}/{len(files)}] Processing: {file_path.name}")

            try:
                result = self.process_single_form(str(file_path), form_id)
                self.results[form_id] = result
                self.processing_status[form_id] = result.get("form_status", "done")
                batch_results.append(result)
                logger.info(f"  → Status: {result.get('form_status')}, "
                           f"Confidence: {result.get('kpis', {}).get('overall_confidence', 0):.2%}")
            except Exception as e:
                logger.error(f"  → FAILED: {e}")
                self.processing_status[form_id] = "failed"
                self.results[form_id] = {
                    "form_id": form_id,
                    "form_status": "Failed",
                    "error": str(e),
                    "source_file": str(file_path),
                }
                batch_results.append(self.results[form_id])

        total_time = time.time() - start_time

        # Compute batch KPIs
        batch_kpis = self._compute_batch_kpis(batch_results, total_time)

        # Save results
        self._save_results(batch_results, batch_kpis)

        return {
            "batch_kpis": batch_kpis,
            "forms": batch_results,
            "processing_time_seconds": total_time,
        }

    def process_single_form(self, file_path: str, form_id: str = None) -> Dict:
        """Process a single form file through the full pipeline."""
        if form_id is None:
            form_id = f"FORM_{uuid.uuid4().hex[:8]}"

        start = time.time()

        # Step 1: Preprocess
        logger.info(f"  Step 1: Preprocessing {Path(file_path).name}...")
        preprocessed_pages = self.preprocessor.process_file(file_path)

        # Filter usable pages
        usable_pages = [p for p in preprocessed_pages if p["is_usable"]]
        rejected_pages = [p for p in preprocessed_pages if not p["is_usable"]]

        if not usable_pages:
            return {
                "form_id": form_id,
                "form_status": "Rejected",
                "rejection_reason": "All pages failed quality check",
                "pages_rejected": len(rejected_pages),
                "source_file": file_path,
            }

        # Step 2: Convert to PIL for VLM
        pil_images = []
        for page in usable_pages:
            # Use color-enhanced version for Qwen VLM
            color_img = page.get("preprocessed_color", page.get("original"))
            pil_img = self.preprocessor.numpy_to_pil(color_img)
            pil_images.append(pil_img)

        # Step 3: Extract with dual-model engine
        logger.info(f"  Step 2: Extracting fields ({len(pil_images)} pages)...")
        raw_extraction = self.extractor.extract(pil_images)

        # Step 4: Validate and build result
        logger.info(f"  Step 3: Validating and building result...")
        preprocessing_info = [
            {
                "page_num": p["page_num"],
                "quality_score": p["quality_score"],
                "is_usable": p["is_usable"],
                "steps": p.get("preprocessing_steps", []),
            }
            for p in preprocessed_pages
        ]

        result = self.result_builder.build_result(
            form_id, raw_extraction, preprocessing_info
        )
        result["source_file"] = file_path
        result["processing_time_seconds"] = time.time() - start

        return result

    def _compute_batch_kpis(self, results: List[Dict], total_time: float) -> Dict:
        """Compute aggregate KPIs across all processed forms."""
        total_forms = len(results)
        processed = sum(1 for r in results if r.get("form_status") not in ("Failed", "Rejected"))
        rejected = sum(1 for r in results if r.get("form_status") == "Rejected")
        failed = sum(1 for r in results if r.get("form_status") == "Failed")
        needs_review = sum(1 for r in results if r.get("form_status") == "Needs Review")

        # Field-level metrics
        total_fields = 0
        correct_fields = 0
        total_chars = 0
        correct_chars = 0  # Would need ground truth for actual computation
        all_confidences = []

        for r in results:
            kpis = r.get("kpis", {})
            total_fields += kpis.get("total_fields_expected", 0)
            correct_fields += kpis.get("fields_extracted", 0)
            if kpis.get("overall_confidence"):
                all_confidences.append(kpis["overall_confidence"])

        field_level_accuracy = correct_fields / max(total_fields, 1)
        avg_confidence = float(sum(all_confidences) / max(len(all_confidences), 1))
        rejection_rate = rejected / max(total_forms, 1)
        manual_correction_rate = needs_review / max(total_forms, 1)

        return {
            "total_forms": total_forms,
            "forms_processed": processed,
            "forms_rejected": rejected,
            "forms_failed": failed,
            "forms_needs_review": needs_review,
            "field_level_accuracy": round(field_level_accuracy, 4),
            "avg_confidence": round(avg_confidence, 4),
            "rejection_rate": round(rejection_rate, 4),
            "manual_correction_rate": round(manual_correction_rate, 4),
            "total_processing_time_seconds": round(total_time, 2),
            "avg_time_per_form_seconds": round(total_time / max(total_forms, 1), 2),
            "throughput_forms_per_minute": round(total_forms / max(total_time / 60, 0.01), 2),
        }

    def _save_results(self, results: List[Dict], batch_kpis: Dict):
        """Save extraction results to JSON files."""
        output_dir = FORMS_OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save individual form results
        for result in results:
            form_id = result.get("form_id", "unknown")
            out_file = output_dir / f"{form_id}_result.json"
            # Remove numpy/image objects for JSON serialization
            clean = self._make_serializable(result)
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(clean, f, indent=2, ensure_ascii=False)

        # Save batch summary
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
        """Remove non-JSON-serializable objects."""
        import numpy as np
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

    def get_dashboard_data(self) -> Dict:
        """Get current dashboard state."""
        return {
            "processing_status": self.processing_status,
            "results": {fid: self._make_serializable(r)
                       for fid, r in self.results.items()},
            "total_forms": len(self.processing_status),
            "completed": sum(1 for s in self.processing_status.values()
                           if s not in ("processing", "queued")),
        }

    def cleanup(self):
        """Release resources."""
        self.extractor.cleanup()
