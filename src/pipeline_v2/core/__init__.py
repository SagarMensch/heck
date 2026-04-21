"""
LIC Proposal Form Extraction Pipeline - Core Module
====================================================
Production-grade pipeline for handwritten form extraction.

Architecture:
    Preprocessor → LayoutDetector → OCR → VLM Fallback → Validator
"""

from .pipeline import ExtractionPipeline
from .models import Field, PageResult, DocumentResult, BoundingBox
from .interfaces import IPreprocessor, ILayoutDetector, IOCR, IVLM, IValidator

__all__ = [
    'ExtractionPipeline',
    'Field',
    'PageResult',
    'DocumentResult',
    'BoundingBox',
    'IPreprocessor',
    'ILayoutDetector',
    'IOCR',
    'IVLM',
    'IValidator',
]
