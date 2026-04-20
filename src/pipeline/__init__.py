"""
LIC Extraction Pipeline v3 — Production SOTA
=============================================
5-Layer VLM-First Architecture

Layer 1: Preprocessor  — Image conditioning for downstream models
Layer 2: VLM Extractor — Qwen2.5-VL direct form understanding
Layer 3: OCR Verifier  — PaddleOCR bilingual cross-check + bbox localization
Layer 4: Validation+KB — Regex, taxonomy, Verhoeff, fuzzy encyclopedia
Layer 5: Confidence    — Multi-source scoring, human review flag, click-to-source overlay
"""

from src.pipeline.layers.preprocessor import Preprocessor
from src.pipeline.layers.vlm_extractor import VLMExtractor
from src.pipeline.layers.ocr_verifier import OCRVerifier
from src.pipeline.layers.validation_kb import ValidationKB
from src.pipeline.layers.confidence_scorer import ConfidenceScorer
from src.pipeline.orchestrator import PipelineOrchestrator

__all__ = [
    "Preprocessor",
    "VLMExtractor",
    "OCRVerifier",
    "ValidationKB",
    "ConfidenceScorer",
    "PipelineOrchestrator",
]
