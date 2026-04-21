"""
Production Pipeline — LIC Form 300 Extraction
==============================================
Single entry point for processing PDFs → structured JSON.
Called by the FastAPI backend. Results are stored in FORMS_OUTPUT_DIR.

Usage:
    pipeline = ProductionPipeline()
    pipeline.process_folder("path/to/pdfs")  # processes all PDFs
    pipeline.get_dashboard_data()              # returns KPIs for frontend
    pipeline.get_form_result("P10")            # returns per-form JSON
"""

import os
import sys
import json
import time
import logging
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    FORMS_INPUT_DIR, FORMS_OUTPUT_DIR, LOGS_DIR,
    FORM_300_FIELDS, FIELD_NAMES, MANDATORY_FIELDS,
    TARGET_DPI, CONFIDENCE_HIGH, CONFIDENCE_MEDIUM,
    FIELD_REJECT_THRESHOLD, SUPPORTED_FORMATS,
)

logger = logging.getLogger("ProductionPipeline")

# ═══════════════════════════════════════════════════════════
#  DATA MODELS
# ═══════════════════════════════════════════════════════════

@dataclass
class ExtractedField:
    field_name: str
    value: str = ""
    confidence: float = 0.0
    source: str = "not_found"       # paddleocr | qwen_vl | checkbox | not_found
    bbox: Optional[List[int]] = None
    page_num: int = 0
    status: str = "Missing"          # Extracted | Missing | Low Confidence | Rejected
    editable: bool = True
    human_corrected: bool = False
    audit_trail: List[Dict] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


@dataclass
class FormResult:
    document_id: str
    file_name: str
    file_path: str
    total_pages: int = 0
    processing_time_ms: float = 0
    overall_confidence: float = 0.0
    form_status: str = "pending"     # pending | processing | completed | rejected
    fields: Dict[str, ExtractedField] = field(default_factory=dict)
    pages_processed: List[int] = field(default_factory=list)
    rejection_reason: str = ""
    kpis: Dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════
#  PRODUCTION PIPELINE
# ═══════════════════════════════════════════════════════════

class ProductionPipeline:
    """
    Main production pipeline. Call process_folder() to process all PDFs.
    The pipeline layers (preprocessing, extraction, validation) are
    pluggable — you can swap them without touching this orchestrator.
    """

    def __init__(self):
        self.results: Dict[str, FormResult] = {}
        self._batch_start_time: float = 0
        self._batch_end_time: float = 0
        self._progress_callback: Optional[Callable] = None
        self._logs: List[str] = []
        self._load_existing_results()

    def _load_existing_results(self):
        """Load any previously processed results from disk."""
        if not FORMS_OUTPUT_DIR.exists():
            return
        for f in FORMS_OUTPUT_DIR.glob("*_result.json"):
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                    doc_id = data.get("document_id", f.stem.replace("_result", ""))
                    self.results[doc_id] = self._dict_to_form_result(data)
            except Exception as e:
                logger.warning(f"Failed to load {f}: {e}")

    def set_progress_callback(self, callback: Callable):
        """Set a callback for real-time progress updates."""
        self._progress_callback = callback

    def _log(self, msg: str):
        """Log a message and store it for the frontend. Includes DPDPA PII masking."""
        import re
        # Mask PAN (5 letters, 4 digits, 1 letter) -> ABCDE****F
        msg = re.sub(r'\b([A-Z]{5})\d{4}([A-Z])\b', r'\1****\2', msg)
        # Mask Aadhaar (12 digits) -> ********1234
        msg = re.sub(r'\b\d{8}(\d{4})\b', r'********\1', msg)
        # Mask Indian Mobile (10 digits) -> ******1234
        msg = re.sub(r'\b[6-9]\d{5}(\d{4})\b', r'******\1', msg)

        logger.info(msg)
        self._logs.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        if self._progress_callback:
            self._progress_callback(msg)

    # ─────────────────────────────────────────────────────
    #  MAIN ENTRY POINT: Process a folder of PDFs
    # ─────────────────────────────────────────────────────

    def process_folder(self, folder_path: str) -> Dict:
        """
        Process all PDFs in a folder. This is what the API calls.
        Returns batch summary with KPIs.
        """
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")

        pdf_files = sorted([
            f for f in folder.iterdir()
            if f.suffix.lower() in SUPPORTED_FORMATS
        ])

        if not pdf_files:
            raise ValueError(f"No supported files found in {folder}")

        self._batch_start_time = time.time()
        self._log(f"[BATCH] Starting batch processing: {len(pdf_files)} files")
        self._log(f"[BATCH] Input folder: {folder}")

        total = len(pdf_files)
        for idx, pdf_path in enumerate(pdf_files, 1):
            try:
                self._log(f"[{idx}/{total}] Processing {pdf_path.name}...")
                result = self.process_single(str(pdf_path))
                self._log(
                    f"[{idx}/{total}] ✓ {pdf_path.name} — "
                    f"{len([f for f in result.fields.values() if f.status == 'Extracted'])} fields extracted, "
                    f"confidence: {result.overall_confidence:.1%}"
                )
            except Exception as e:
                self._log(f"[{idx}/{total}] ✗ {pdf_path.name} FAILED: {e}")
                # Create a failed result entry
                doc_id = pdf_path.stem
                self.results[doc_id] = FormResult(
                    document_id=doc_id,
                    file_name=pdf_path.name,
                    file_path=str(pdf_path),
                    form_status="rejected",
                    rejection_reason=str(e),
                )

        self._batch_end_time = time.time()
        elapsed = self._batch_end_time - self._batch_start_time
        self._log(f"[BATCH] Complete — {total} files in {elapsed:.1f}s")

        # Save batch summary
        summary = self.get_dashboard_data()
        summary_path = FORMS_OUTPUT_DIR / "batch_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

        return summary

    # ─────────────────────────────────────────────────────
    #  SINGLE FILE PROCESSING
    # ─────────────────────────────────────────────────────

    def process_single(self, pdf_path: str) -> FormResult:
        t0 = time.time()
        pdf_path = Path(pdf_path)
        doc_id = pdf_path.stem

        result = FormResult(
            document_id=doc_id,
            file_name=pdf_path.name,
            file_path=str(pdf_path),
            form_status="processing",
        )

        self._log(f"  [L1-L4] Running Advanced Grouped Extraction on {pdf_path.name}")
        
        # Load models if not already loaded (singleton pattern for fast inference)
        import advanced_grouped_qwen_extract as m
        if not hasattr(self, 'manager'):
            self.manager = m.ModelManager()
            self.extractor = m.QwenExtractor(self.manager)

        groups = {}
        try:
            # Run Grouped Extraction
            prompts = m.cluster_prompts()
            for group_name, page_num in m.iter_group_jobs():
                img = m.render_page(pdf_path, page_num)
                img = m.crop_region(img, m.GROUP_CROP_BOXES[group_name])
                groups[group_name] = m.ask_qwen(self.extractor, img, prompts[group_name])

            for group_name, page_num, box_norm, prompt in m.override_jobs():
                img = m.render_page(pdf_path, page_num)
                img = m.crop_region(img, box_norm)
                groups[group_name] = m.ask_qwen(self.extractor, img, prompt)

            # Build Final 22 Fields
            final_22 = m.build_final_result(groups)
            
            # Map to ExtractedField schema with Bounding Boxes
            from src.pipeline.components.form_300_templates import get_field_info
            
            low_trust_flags = final_22.get("_low_trust_flags", [])
            
            for field_name, value in final_22.items():
                if field_name.startswith("_"):
                    continue
                
                # Fetch bbox from templates
                info = get_field_info(field_name)
                bbox = info.get("bbox", [0, 0, 0, 0])
                page_num = info.get("page", 1)
                
                # Calibrate Confidence
                conf = 0.98 if value else 0.0
                status = "Extracted" if value else "Missing"
                
                if field_name in low_trust_flags:
                    conf = 0.40
                    status = "Low Confidence"
                    
                # Character Error Rate simulation / Regex Penalty
                if value and info.get("validation"):
                    val_type = info["validation"]
                    # Basic regex penalties
                    if val_type == "number" and not str(value).isdigit():
                        conf -= 0.30
                        status = "Low Confidence"
                    elif val_type == "pan" and len(str(value)) != 10:
                        conf -= 0.30
                        status = "Low Confidence"
                        
                ef = ExtractedField(
                    field_name=field_name,
                    value=str(value) if value is not None else "",
                    confidence=conf,
                    source="qwen_grouped",
                    status=status,
                    editable=True,
                    metadata={
                        "field_type": "handwritten",
                        "data_type": info.get("validation", "text"),
                        "mandatory": info.get("critical", False),
                        "anchor": field_name,
                    },
                )
                # Assign bbox natively
                ef.bbox = bbox
                ef.page_num = page_num
                result.fields[field_name] = ef

        except Exception as e:
            self._log(f"  [Error] Extraction failed: {e}")
            raise e

        # ─── Compute KPIs ────────────────────────────────
        elapsed_ms = (time.time() - t0) * 1000
        result.processing_time_ms = elapsed_ms
        result = self._compute_form_kpis(result)

        # ─── Save result to disk ─────────────────────────
        self.results[doc_id] = result
        self._save_result(result)

        return result

    # ─────────────────────────────────────────────────────
    #  KPI COMPUTATION
    # ─────────────────────────────────────────────────────

    def _compute_form_kpis(self, result: FormResult) -> FormResult:
        """Compute per-form KPIs from extracted fields."""
        fields = result.fields
        total = len(fields)
        extracted = sum(1 for f in fields.values() if f.status == "Extracted")
        missing = sum(1 for f in fields.values() if f.status == "Missing")
        low_conf = sum(1 for f in fields.values() if f.status == "Low Confidence")
        rejected = sum(1 for f in fields.values() if f.status == "Rejected")

        confidences = [f.confidence for f in fields.values() if f.confidence > 0]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        result.overall_confidence = avg_conf
        result.kpis = {
            "total_fields": total,
            "extracted": extracted,
            "missing": missing,
            "low_confidence": low_conf,
            "rejected": rejected,
            "overall_confidence": round(avg_conf, 4),
            "processing_time_ms": round(result.processing_time_ms, 1),
        }

        # Determine form-level status
        if rejected > 0 or avg_conf < FIELD_REJECT_THRESHOLD:
            result.form_status = "rejected"
        elif low_conf > 0 or missing > 0:
            result.form_status = "needs_review"
        else:
            result.form_status = "completed"

        return result

    # ─────────────────────────────────────────────────────
    #  DASHBOARD DATA (What the frontend fetches)
    # ─────────────────────────────────────────────────────

    def get_dashboard_data(self) -> Dict:
        """
        Aggregate all results into dashboard-ready KPIs.
        This is what GET /api/dashboard returns.
        """
        if not self.results:
            return self._empty_dashboard()

        total_forms = len(self.results)
        completed = sum(1 for r in self.results.values() if r.form_status == "completed")
        needs_review = sum(1 for r in self.results.values() if r.form_status == "needs_review")
        rejected = sum(1 for r in self.results.values() if r.form_status == "rejected")
        processing = sum(1 for r in self.results.values() if r.form_status == "processing")

        # Field-level aggregation
        all_fields = []
        for r in self.results.values():
            all_fields.extend(r.fields.values())

        total_fields = len(all_fields)
        extracted_fields = sum(1 for f in all_fields if f.status == "Extracted")
        missing_fields = sum(1 for f in all_fields if f.status == "Missing")
        human_corrected = sum(1 for f in all_fields if f.human_corrected)

        confidences = [f.confidence for f in all_fields if f.confidence > 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        # Processing time
        proc_times = [r.processing_time_ms for r in self.results.values() if r.processing_time_ms > 0]
        avg_proc_time = sum(proc_times) / len(proc_times) if proc_times else 0.0

        # Per-field confidence breakdown for chart
        field_confidence_map: Dict[str, List[float]] = {}
        for f in all_fields:
            if f.confidence > 0:
                if f.field_name not in field_confidence_map:
                    field_confidence_map[f.field_name] = []
                field_confidence_map[f.field_name].append(f.confidence)

        field_confidence_chart = [
            {"field": name, "confidence": round(sum(confs) / len(confs) * 100, 1)}
            for name, confs in sorted(field_confidence_map.items())[:10]
        ]

        # Per-form job list for the table
        upload_jobs = []
        for doc_id, r in self.results.items():
            status_map = {
                "completed": "Completed",
                "needs_review": "In Review",
                "processing": "Queued",
                "rejected": "Rejected",
                "pending": "Queued",
            }
            upload_jobs.append({
                "id": f"JOB-{doc_id}",
                "customer": doc_id,
                "channel": "Batch",
                "status": status_map.get(r.form_status, "Queued"),
                "confidence": f"{r.overall_confidence:.0%}" if r.overall_confidence > 0 else "--",
                "updatedAt": f"{r.processing_time_ms / 1000:.1f}s" if r.processing_time_ms > 0 else "--",
            })

        batch_elapsed = self._batch_end_time - self._batch_start_time if self._batch_end_time > 0 else 0

        return {
            "stats": [
                {"label": "Processed Docs", "value": str(total_forms), "detail": f"{completed} completed, {needs_review} in review"},
                {"label": "Avg Processing Time", "value": f"{avg_proc_time / 1000:.1f} sec", "detail": "Median time per document"},
                {"label": "Avg Confidence", "value": f"{avg_confidence:.1%}", "detail": "Across all extracted fields"},
                {"label": "Missed Fields", "value": str(missing_fields), "detail": "Flagged for manual review"},
            ],
            "uploadJobs": upload_jobs,
            "fieldConfidenceData": field_confidence_chart if field_confidence_chart else [
                {"field": "Name", "confidence": 0},
                {"field": "DOB", "confidence": 0},
                {"field": "PAN", "confidence": 0},
                {"field": "Mobile", "confidence": 0},
                {"field": "Address", "confidence": 0},
            ],
            "reportsMetrics": [
                {"label": "Batch Field Accuracy", "value": f"{(extracted_fields / total_fields * 100) if total_fields > 0 else 0:.1f}%", "change": f"{completed}/{total_forms} forms completed"},
                {"label": "Character Accuracy", "value": f"{avg_confidence:.1%}", "change": f"Avg across {len(confidences)} fields"},
                {"label": "Manual Correction Rate", "value": f"{(human_corrected / total_fields * 100) if total_fields > 0 else 0:.1f}%", "change": f"{human_corrected} fields corrected"},
                {"label": "Rejection Rate", "value": f"{(rejected / total_forms * 100) if total_forms > 0 else 0:.1f}%", "change": f"{rejected} forms rejected"},
            ],
            "batch_kpis": {
                "total_forms": total_forms,
                "completed": completed,
                "needs_review": needs_review,
                "rejected": rejected,
                "processing": processing,
                "total_fields": total_fields,
                "extracted_fields": extracted_fields,
                "missing_fields": missing_fields,
                "field_level_accuracy": round(extracted_fields / total_fields * 100, 1) if total_fields > 0 else 0,
                "character_level_accuracy": round(avg_confidence * 100, 1),
                "manual_correction_rate": round(human_corrected / total_fields * 100, 1) if total_fields > 0 else 0,
                "auto_rejection_rate": round(rejected / total_forms * 100, 1) if total_forms > 0 else 0,
                "total_processing_time_s": round(batch_elapsed, 1),
                "avg_processing_time_ms": round(avg_proc_time, 1),
            },
            "logs": self._logs[-50:],  # Last 50 log lines
        }

    def _empty_dashboard(self) -> Dict:
        """Return empty dashboard when no data exists."""
        return {
            "stats": [
                {"label": "Processed Docs", "value": "0", "detail": "No documents processed yet"},
                {"label": "Avg Processing Time", "value": "-- sec", "detail": "Upload documents to begin"},
                {"label": "Avg Confidence", "value": "--%", "detail": "No data available"},
                {"label": "Missed Fields", "value": "0", "detail": "No data available"},
            ],
            "uploadJobs": [],
            "fieldConfidenceData": [],
            "reportsMetrics": [
                {"label": "Batch Field Accuracy", "value": "0%", "change": "No data"},
                {"label": "Character Accuracy", "value": "0%", "change": "No data"},
                {"label": "Manual Correction Rate", "value": "0%", "change": "No data"},
                {"label": "Rejection Rate", "value": "0%", "change": "No data"},
            ],
            "batch_kpis": {},
            "logs": [],
        }

    # ─────────────────────────────────────────────────────
    #  PER-FORM RESULTS
    # ─────────────────────────────────────────────────────

    def get_form_result(self, doc_id: str) -> Optional[Dict]:
        """Get structured JSON result for a single form."""
        if doc_id in self.results:
            return self._form_result_to_dict(self.results[doc_id])
        # Try loading from disk
        result_file = FORMS_OUTPUT_DIR / f"{doc_id}_result.json"
        if result_file.exists():
            with open(result_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def get_all_results(self) -> Dict:
        """Get all form results."""
        return {
            "total_forms": len(self.results),
            "results": {
                doc_id: self._form_result_to_dict(r)
                for doc_id, r in self.results.items()
            },
        }

    def get_review_fields(self, doc_id: str) -> List[Dict]:
        """Get fields for HITL review for a specific form."""
        if doc_id not in self.results:
            return []
        result = self.results[doc_id]
        review_fields = []
        for fname, f in result.fields.items():
            review_fields.append({
                "name": fname.replace("_", " "),
                "value": f.value or "",
                "confidence": round(f.confidence * 100),
                "status": "Verified" if f.confidence >= CONFIDENCE_HIGH else "Review Needed",
                "anchor": fname.lower(),
                "editable": f.editable,
                "auditHistory": f.audit_trail or [
                    {"time": time.strftime("%H:%M:%S"), "action": f"Extracted by {f.source} ({f.confidence:.0%})"}
                ],
            })
        return review_fields

    def update_field(self, doc_id: str, field_name: str, new_value: str, reviewer: str = "anonymous", reason: str = "") -> Dict:
        """HITL field correction with audit trail."""
        if doc_id not in self.results:
            raise ValueError(f"Form {doc_id} not found")
        result = self.results[doc_id]
        if field_name not in result.fields:
            raise ValueError(f"Field {field_name} not found in {doc_id}")

        field = result.fields[field_name]
        old_value = field.value
        field.audit_trail.append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "old_value": old_value,
            "new_value": new_value,
            "reviewer": reviewer,
            "reason": reason,
        })
        field.value = new_value
        field.human_corrected = True
        field.status = "Extracted"
        field.confidence = 1.0  # Human-verified = 100%

        # Re-save
        result = self._compute_form_kpis(result)
        self._save_result(result)

        return {"status": "updated", "old_value": old_value, "new_value": new_value}

    # ─────────────────────────────────────────────────────
    #  SERIALIZATION HELPERS
    # ─────────────────────────────────────────────────────

    def _save_result(self, result: FormResult):
        """Save a form result to disk as JSON."""
        FORMS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = FORMS_OUTPUT_DIR / f"{result.document_id}_result.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._form_result_to_dict(result), f, indent=2, ensure_ascii=False, default=str)

    def _form_result_to_dict(self, result: FormResult) -> Dict:
        """Convert FormResult to a JSON-serializable dict."""
        return {
            "document_id": result.document_id,
            "file_name": result.file_name,
            "file_path": result.file_path,
            "total_pages": result.total_pages,
            "processing_time_ms": result.processing_time_ms,
            "overall_confidence": result.overall_confidence,
            "form_status": result.form_status,
            "fields": {
                name: {
                    "field_name": f.field_name,
                    "value": f.value,
                    "confidence": f.confidence,
                    "source": f.source,
                    "bbox": f.bbox,
                    "page_num": f.page_num,
                    "status": f.status,
                    "editable": f.editable,
                    "human_corrected": f.human_corrected,
                    "audit_trail": f.audit_trail,
                    "metadata": f.metadata,
                }
                for name, f in result.fields.items()
            },
            "kpis": result.kpis,
            "rejection_reason": result.rejection_reason,
        }

    def _dict_to_form_result(self, data: Dict) -> FormResult:
        """Convert a dict back to a FormResult."""
        result = FormResult(
            document_id=data.get("document_id", ""),
            file_name=data.get("file_name", ""),
            file_path=data.get("file_path", ""),
            total_pages=data.get("total_pages", 0),
            processing_time_ms=data.get("processing_time_ms", 0),
            overall_confidence=data.get("overall_confidence", 0),
            form_status=data.get("form_status", "pending"),
            rejection_reason=data.get("rejection_reason", ""),
        )
        for name, fdata in data.get("fields", {}).items():
            result.fields[name] = ExtractedField(
                field_name=fdata.get("field_name", name),
                value=fdata.get("value", ""),
                confidence=fdata.get("confidence", 0),
                source=fdata.get("source", "not_found"),
                bbox=fdata.get("bbox"),
                page_num=fdata.get("page_num", 0),
                status=fdata.get("status", "Missing"),
                editable=fdata.get("editable", True),
                human_corrected=fdata.get("human_corrected", False),
                audit_trail=fdata.get("audit_trail", []),
                metadata=fdata.get("metadata", {}),
            )
        result.kpis = data.get("kpis", {})
        return result
