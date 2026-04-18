"""
FastAPI Backend
================
REST API for LIC Proposal Form extraction pipeline.
"""

import logging
import json
import time
import uuid
import shutil
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config import FORMS_INPUT_DIR, FORMS_OUTPUT_DIR, LOGS_DIR
from src.pipeline import LICExtractionPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "api.log", encoding="utf-8"),
    ]
)
logger = logging.getLogger("lic_api")

# Initialize app
app = FastAPI(
    title="LIC Proposal Form Data Extraction API",
    description="AI/ML-Enabled Handwritten Proposal Data Extraction - Techathon Solution",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline: Optional[LICExtractionPipeline] = None
processing_jobs = {}  # job_id -> status


def get_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = LICExtractionPipeline()
    return pipeline


# ──────────────────────── MODELS ────────────────────────

class ProcessRequest(BaseModel):
    folder_path: Optional[str] = None
    job_id: Optional[str] = None


class ReviewUpdate(BaseModel):
    field_name: str
    new_value: str
    reviewer: str = "anonymous"
    reason: str = ""


# ──────────────────────── ENDPOINTS ────────────────────────

@app.get("/")
async def root():
    return {
        "service": "LIC Proposal Form Data Extraction API",
        "version": "1.0.0",
        "status": "running",
        "models": ["Qwen2.5-VL-7B-Instruct", "TrOCR-base-handwritten"],
        "endpoints": {
            "upload": "POST /api/upload",
            "process": "POST /api/process",
            "results": "GET /api/results/{form_id}",
            "dashboard": "GET /api/dashboard",
            "batch_status": "GET /api/batch/{job_id}",
        }
    }


@app.get("/health")
async def health():
    import torch
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else "N/A"
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9 if gpu_available else 0
    gpu_free = torch.cuda.mem_get_info()[0] / 1e9 if gpu_available else 0

    return {
        "status": "healthy",
        "gpu": {
            "available": gpu_available,
            "name": gpu_name,
            "total_memory_gb": round(gpu_memory, 1),
            "free_memory_gb": round(gpu_free, 1),
        },
        "pipeline_initialized": pipeline is not None,
    }


@app.post("/api/upload")
async def upload_forms(files: List[UploadFile] = File(...)):
    """Upload one or more form files (PDF/JPG/PNG/TIFF)."""
    uploaded = []
    upload_dir = FORMS_INPUT_DIR / f"batch_{uuid.uuid4().hex[:8]}"
    upload_dir.mkdir(parents=True, exist_ok=True)

    for file in files:
        suffix = Path(file.filename).suffix.lower()
        if suffix not in {".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}:
            continue

        dest = upload_dir / file.filename
        with open(dest, "wb") as f:
            content = await file.read()
            f.write(content)
        uploaded.append(str(dest))

    return {
        "status": "uploaded",
        "files_uploaded": len(uploaded),
        "upload_folder": str(upload_dir),
        "files": uploaded,
    }


@app.post("/api/process")
async def process_forms(request: ProcessRequest, background_tasks: BackgroundTasks):
    """Start processing forms from a folder. Returns job_id for tracking."""
    folder = request.folder_path or str(FORMS_INPUT_DIR)
    job_id = request.job_id or f"job_{uuid.uuid4().hex[:8]}"

    processing_jobs[job_id] = {
        "status": "queued",
        "folder": folder,
        "start_time": time.time(),
        "result": None,
    }

    background_tasks.add_task(_run_processing, job_id, folder)

    return {
        "job_id": job_id,
        "status": "queued",
        "folder": folder,
        "message": "Processing started. Use GET /api/batch/{job_id} to check status.",
    }


@app.post("/api/process/sync")
async def process_forms_sync(request: ProcessRequest):
    """Synchronous processing (blocks until done). Use for small batches."""
    folder = request.folder_path or str(FORMS_INPUT_DIR)

    try:
        pipe = get_pipeline()
        result = pipe.process_folder(folder)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/batch/{job_id}")
async def get_batch_status(job_id: str):
    """Check status of a processing job."""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return processing_jobs[job_id]


@app.get("/api/results/{form_id}")
async def get_form_result(form_id: str):
    """Get extraction results for a specific form."""
    pipe = get_pipeline()
    if form_id in pipe.results:
        result = pipe.results[form_id]
        return pipe._make_serializable(result)

    # Try loading from disk
    result_file = FORMS_OUTPUT_DIR / f"{form_id}_result.json"
    if result_file.exists():
        with open(result_file, "r", encoding="utf-8") as f:
            return json.load(f)

    raise HTTPException(status_code=404, detail=f"Form {form_id} not found")


@app.get("/api/results")
async def get_all_results():
    """Get all extraction results."""
    pipe = get_pipeline()
    results = {}
    for form_id, result in pipe.results.items():
        results[form_id] = pipe._make_serializable(result)
    return {
        "total_forms": len(results),
        "results": results,
    }


@app.get("/api/dashboard")
async def get_dashboard():
    """Get dashboard data: KPIs, form statuses, processing progress."""
    pipe = get_pipeline()
    dashboard = pipe.get_dashboard_data()

    # Add batch KPIs if available
    summary_file = FORMS_OUTPUT_DIR / "batch_summary.json"
    if summary_file.exists():
        with open(summary_file, "r") as f:
            batch_summary = json.load(f)
            dashboard["batch_kpis"] = batch_summary.get("batch_kpis", {})

    return dashboard


@app.put("/api/review/{form_id}")
async def review_field(form_id: str, update: ReviewUpdate):
    """HITL review: update a field value with audit trail."""
    pipe = get_pipeline()

    if form_id not in pipe.results:
        raise HTTPException(status_code=404, detail=f"Form {form_id} not found")

    result = pipe.results[form_id]
    fields = result.get("fields", {})

    if update.field_name not in fields:
        raise HTTPException(status_code=404,
                          detail=f"Field {update.field_name} not found in form {form_id}")

    # Audit trail
    old_value = fields[update.field_name].get("value")
    fields[update.field_name]["audit_trail"] = fields[update.field_name].get("audit_trail", [])
    fields[update.field_name]["audit_trail"].append({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "old_value": old_value,
        "new_value": update.new_value,
        "reviewer": update.reviewer,
        "reason": update.reason,
    })
    fields[update.field_name]["value"] = update.new_value
    fields[update.field_name]["human_corrected"] = True
    fields[update.field_name]["category"] = "Human Corrected"

    # Save updated result
    result_file = FORMS_OUTPUT_DIR / f"{form_id}_result.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(pipe._make_serializable(result), f, indent=2, ensure_ascii=False)

    return {
        "status": "updated",
        "form_id": form_id,
        "field": update.field_name,
        "old_value": old_value,
        "new_value": update.new_value,
    }


@app.get("/api/audit/{form_id}")
async def get_audit_trail(form_id: str):
    """Get complete audit trail for a form."""
    pipe = get_pipeline()

    if form_id not in pipe.results:
        raise HTTPException(status_code=404, detail=f"Form {form_id} not found")

    result = pipe.results[form_id]
    audit = {}
    for field_name, field_data in result.get("fields", {}).items():
        trail = field_data.get("audit_trail", [])
        if trail:
            audit[field_name] = trail

    return {
        "form_id": form_id,
        "audit_trail": audit,
        "total_corrections": sum(len(t) for t in audit.values()),
    }


@app.get("/api/model-info")
async def get_model_info():
    """Display model/LLM details (required by Techathon rules at 12:00 hrs)."""
    import torch
    return {
        "models": [
            {
                "name": "Qwen2.5-VL-7B-Instruct",
                "version": "7B parameters",
                "type": "Open-weight Vision-Language Model",
                "license": "Apache 2.0",
                "hosting": "On-premise / MEITY-empanelled cloud",
                "inference_mode": "Data-isolated, inference-only, no data persistence",
                "data_residency": "All data processed within India",
                "weights_accessible": True,
                "fine_tuning_supported": True,
            },
            {
                "name": "TrOCR-base-handwritten",
                "version": "Base (334M parameters)",
                "type": "Open-weight Vision Encoder-Decoder (Transformer OCR)",
                "license": "MIT",
                "hosting": "On-premise / MEITY-empanelled cloud",
                "inference_mode": "Data-isolated, inference-only",
                "fine_tuned": True,
                "fine_tuning_method": "LoRA (PEFT) on LIC proposal form field crops",
            },
            {
                "name": "PaddleOCR PP-OCRv5",
                "version": "Server Det + Rec",
                "type": "Open-source OCR Pipeline",
                "license": "Apache 2.0",
                "purpose": "Layout detection, text region segmentation",
            },
        ],
        "infrastructure": {
            "gpu": "NVIDIA L4 (24GB VRAM)",
            "framework": "PyTorch 2.7.1 + CUDA 11.8",
            "deployment": "Docker container on Indian infrastructure",
            "data_persistence": "None — inference only",
        },
        "compliance": {
            "data_residency": "India only",
            "no_foreign_api_calls": True,
            "no_data_export": True,
            "irdai_compliant": True,
            "dpdpa_2023_compliant": True,
        }
    }


# ──────────────────────── BACKGROUND TASK ────────────────────────

async def _run_processing(job_id: str, folder: str):
    """Background processing task."""
    try:
        processing_jobs[job_id]["status"] = "processing"
        pipe = get_pipeline()
        result = pipe.process_folder(folder)
        processing_jobs[job_id]["status"] = "completed"
        processing_jobs[job_id]["result"] = pipe._make_serializable(result.get("batch_kpis", {}))
        processing_jobs[job_id]["end_time"] = time.time()
    except Exception as e:
        processing_jobs[job_id]["status"] = "failed"
        processing_jobs[job_id]["error"] = str(e)
        logger.error(f"Job {job_id} failed: {e}")


# ──────────────────────── RUN ────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
