"""
LIC Proposal Form Extraction Pipeline v2
========================================
Production-grade multimodal extraction pipeline using:
- Nemotron for table structure detection
- PaddleOCR for fast text extraction  
- Qwen-VL for handwritten fallback
- Comprehensive validation layer

Example:
    >>> from pipeline_v2 import ExtractionPipeline
    >>> pipeline = ExtractionPipeline()
    >>> result = pipeline.process("form.pdf")
"""

__version__ = "2.0.0"
__author__ = "AI Lab"

from .core.pipeline import ExtractionPipeline
from .core.models import DocumentResult, Field, BoundingBox

__all__ = ["ExtractionPipeline", "DocumentResult", "Field", "BoundingBox"]
