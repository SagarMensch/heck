"""
Data Models - Canonical data structures for the pipeline
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
import json


@dataclass(frozen=True)
class BoundingBox:
    """Immutable bounding box with coordinate conversion."""
    x1: int
    y1: int
    x2: int
    y2: int
    page_width: int = 0
    page_height: int = 0
    
    def __post_init__(self):
        # Normalize coordinates
        if self.x1 > self.x2:
            object.__setattr__(self, 'x1', self.x2)
            object.__setattr__(self, 'x2', self.x1)
        if self.y1 > self.y2:
            object.__setattr__(self, 'y1', self.y2)
            object.__setattr__(self, 'y2', self.y1)
    
    @property
    def width(self) -> int:
        return abs(self.x2 - self.x1)
    
    @property
    def height(self) -> int:
        return abs(self.y2 - self.y1)
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)
    
    def to_normalized(self) -> Tuple[float, float, float, float]:
        """Return normalized coordinates [0-1]."""
        if self.page_width == 0 or self.page_height == 0:
            return (0.0, 0.0, 0.0, 0.0)
        return (
            self.x1 / self.page_width,
            self.y1 / self.page_height,
            self.x2 / self.page_width,
            self.y2 / self.page_height
        )
    
    def to_coco(self) -> Tuple[int, int, int, int]:
        """Return COCO format [x, y, width, height]."""
        return (self.x1, self.y1, self.width, self.height)
    
    def expand(self, margin: int) -> "BoundingBox":
        """Expand bbox by margin pixels."""
        return BoundingBox(
            max(0, self.x1 - margin),
            max(0, self.y1 - margin),
            self.x2 + margin,
            self.y2 + margin,
            self.page_width,
            self.page_height
        )
    
    def contains(self, point: Tuple[int, int]) -> bool:
        """Check if point is inside bbox."""
        x, y = point
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2
    
    def iou(self, other: "BoundingBox") -> float:
        """Calculate IoU with another bbox."""
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        
        if x2 < x1 or y2 < y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        union = self.area + other.area - intersection
        
        return intersection / union if union > 0 else 0.0


@dataclass
class Field:
    """Extracted field with full provenance."""
    name: str
    value: str = ""
    confidence: float = 0.0
    bbox: Optional[BoundingBox] = None
    source: str = "unknown"
    
    # Validation
    validation_status: str = "pending"
    corrected_value: str = ""
    validation_issues: List[str] = field(default_factory=list)
    needs_human_review: bool = False
    
    # Metadata
    extraction_time_ms: float = 0.0
    ocr_confidence: float = 0.0
    vlm_confidence: float = 0.0
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    
    def update_from_validation(self, is_valid: bool, corrected: str = "", 
                               issues: Optional[List[str]] = None):
        """Update field after validation."""
        self.validation_status = "valid" if is_valid else "invalid"
        if corrected:
            self.corrected_value = corrected
        if issues:
            self.validation_issues.extend(issues)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "confidence": round(self.confidence, 4),
            "source": self.source,
            "validation_status": self.validation_status,
            "needs_review": self.needs_human_review,
            "bbox": self.bbox.to_normalized() if self.bbox else None,
        }


@dataclass
class PageResult:
    """Extraction result for single page."""
    page_num: int
    fields: List[Field] = field(default_factory=list)
    image_dimensions: Tuple[int, int] = (0, 0)
    processing_time_ms: float = 0.0
    
    # Layer metadata
    preprocessing_applied: List[str] = field(default_factory=list)
    layout_detection_method: str = ""
    ocr_regions_detected: int = 0
    vlm_calls_made: int = 0
    
    def get_field(self, name: str) -> Optional[Field]:
        """Get field by name."""
        return next((f for f in self.fields if f.name == name), None)
    
    @property
    def extracted_count(self) -> int:
        return len([f for f in self.fields if f.value])
    
    @property
    def avg_confidence(self) -> float:
        found = [f.confidence for f in self.fields if f.value]
        return sum(found) / len(found) if found else 0.0


@dataclass
class DocumentResult:
    """Complete document extraction result."""
    document_id: str
    file_path: str
    pages: List[PageResult] = field(default_factory=list)
    total_processing_time_ms: float = 0.0
    
    # Summary
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    form_status: str = "pending"
    
    def __post_init__(self):
        if not self.document_id:
            import uuid
            self.document_id = str(uuid.uuid4())
    
    @property
    def total_pages(self) -> int:
        return len(self.pages)
    
    @property
    def all_fields(self) -> List[Field]:
        """Flatten all fields from all pages."""
        return [f for page in self.pages for f in page.fields]
    
    @property
    def extracted_fields(self) -> List[Field]:
        return [f for f in self.all_fields if f.value]
    
    @property
    def missing_fields(self) -> List[Field]:
        return [f for f in self.all_fields if not f.value]
    
    @property
    def overall_confidence(self) -> float:
        found = self.extracted_fields
        return sum(f.confidence for f in found) / len(found) if found else 0.0
    
    @property
    def needs_review_count(self) -> int:
        return len([f for f in self.extracted_fields if f.needs_human_review])
    
    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON."""
        return json.dumps({
            "document_id": self.document_id,
            "file_path": self.file_path,
            "total_pages": self.total_pages,
            "total_time_ms": self.total_processing_time_ms,
            "overall_confidence": round(self.overall_confidence, 4),
            "form_status": self.form_status,
            "pages": [
                {
                    "page_num": p.page_num,
                    "fields": [f.to_dict() for f in p.fields],
                    "processing_time_ms": p.processing_time_ms
                }
                for p in self.pages
            ],
            "summary": {
                "total_fields": len(self.all_fields),
                "extracted": len(self.extracted_fields),
                "missing": len(self.missing_fields),
                "needs_review": self.needs_review_count
            }
        }, indent=indent, ensure_ascii=False)


@dataclass
class ProcessingConfig:
    """Pipeline configuration."""
    
    # Preprocessing
    dpi: int = 150
    enhance_contrast: bool = True
    remove_shadows: bool = True
    denoise: bool = True
    
    # Layout Detection
    layout_model: str = "nemotron-table-structure-v1"
    use_cached_structure: bool = True
    
    # OCR
    ocr_lang: str = "en"
    ocr_batch_size: int = 16
    
    # VLM Fallback
    vlm_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    vlm_confidence_threshold: float = 0.7
    max_vlm_calls_per_page: int = 5
    
    # Validation
    strict_validation: bool = True
    cross_field_validation: bool = True
