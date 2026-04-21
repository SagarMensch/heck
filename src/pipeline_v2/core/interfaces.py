"""
Core Interfaces - Abstractions for pipeline components
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import numpy as np


class IPreprocessor(ABC):
    """Interface for image preprocessing."""
    
    @abstractmethod
    def process(self, image: Image.Image) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Preprocess image for extraction.
        
        Returns:
            Tuple of (processed_image, metadata)
        """
        pass


class ILayoutDetector(ABC):
    """Interface for table/cell layout detection."""
    
    @abstractmethod
    def detect(self, image: Image.Image, page_num: int = 0) -> "TableStructure":
        """
        Detect table structure.
        
        Returns:
            TableStructure with cells and field mappings
        """
        pass
    
    @abstractmethod
    def get_field_region(self, field_name: str) -> Optional["BoundingBox"]:
        """Get bounding box for specific field."""
        pass


class IOCR(ABC):
    """Interface for OCR engines."""
    
    @abstractmethod
    def recognize(self, image: Image.Image, bbox: Optional["BoundingBox"] = None) -> "OCRResult":
        """
        Perform OCR on image or region.
        
        Returns:
            OCRResult with text and confidence
        """
        pass
    
    @abstractmethod
    def recognize_batch(self, regions: List[Tuple[Image.Image, "BoundingBox"]]) -> List["OCRResult"]:
        """Batch OCR for multiple regions."""
        pass


class IVLM(ABC):
    """Interface for Vision-Language Models."""
    
    @abstractmethod
    def extract_field(self, image: Image.Image, field_name: str, hint: str = "") -> "ExtractionResult":
        """
        Extract specific field using VLM.
        
        Args:
            image: Full page image
            field_name: Field to extract
            hint: Additional context
            
        Returns:
            ExtractionResult with value and confidence
        """
        pass
    
    @abstractmethod
    def extract_region(self, image: Image.Image, bbox: "BoundingBox", field_name: str) -> "ExtractionResult":
        """Extract from specific region."""
        pass


class IValidator(ABC):
    """Interface for field validation."""
    
    @abstractmethod
    def validate(self, field: "Field") -> "ValidationResult":
        """
        Validate extracted field.
        
        Returns:
            ValidationResult with status and corrections
        """
        pass
    
    @abstractmethod
    def validate_document(self, fields: List["Field"]) -> "DocumentValidationResult":
        """Cross-field validation."""
        pass


class BoundingBox:
    """Immutable bounding box."""
    
    def __init__(self, x1: int, y1: int, x2: int, y2: int, 
                 page_width: int = 0, page_height: int = 0):
        self.x1 = min(x1, x2)
        self.y1 = min(y1, y2)
        self.x2 = max(x1, x2)
        self.y2 = max(y1, y2)
        self.page_width = page_width
        self.page_height = page_height
    
    @property
    def width(self) -> int:
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        return self.y2 - self.y1
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)
    
    def to_list(self) -> List[int]:
        return [self.x1, self.y1, self.x2, self.y2]
    
    def normalize(self) -> List[float]:
        if self.page_width == 0 or self.page_height == 0:
            return [0.0, 0.0, 0.0, 0.0]
        return [
            self.x1 / self.page_width,
            self.y1 / self.page_height,
            self.x2 / self.page_width,
            self.y2 / self.page_height
        ]
    
    def crop(self, image: Image.Image) -> Image.Image:
        return image.crop((self.x1, self.y1, self.x2, self.y2))
    
    def __repr__(self) -> str:
        return f"BoundingBox({self.x1}, {self.y1}, {self.x2}, {self.y2})"


class OCRResult:
    """OCR recognition result."""
    
    def __init__(self, text: str, confidence: float, 
                 bbox: Optional[BoundingBox] = None,
                 language: str = "unknown"):
        self.text = text
        self.confidence = confidence
        self.bbox = bbox
        self.language = language


class ExtractionResult:
    """VLM extraction result."""
    
    def __init__(self, value: str, confidence: float,
                 source: str = "vlm", raw_output: str = ""):
        self.value = value
        self.confidence = confidence
        self.source = source
        self.raw_output = raw_output


class ValidationResult:
    """Field validation result."""
    
    def __init__(self, is_valid: bool, corrected_value: str = "",
                 issues: Optional[List[str]] = None, 
                 confidence_adjustment: float = 0.0):
        self.is_valid = is_valid
        self.corrected_value = corrected_value
        self.issues = issues or []
        self.confidence_adjustment = confidence_adjustment


class TableStructure:
    """Detected table structure."""
    
    def __init__(self, cells: List[Dict[str, Any]], 
                 method: str = "unknown",
                 confidence: float = 0.0):
        self.cells = cells
        self.method = method
        self.confidence = confidence
        self._field_map = {}
    
    def add_field_mapping(self, field_name: str, cell_idx: int):
        """Map field name to cell index."""
        self._field_map[field_name] = cell_idx
    
    def get_field_bbox(self, field_name: str) -> Optional[BoundingBox]:
        """Get bounding box for field."""
        if field_name in self._field_map:
            cell = self.cells[self._field_map[field_name]]
            return BoundingBox(*cell["bbox"])
        return None


class Field:
    """Extracted field."""
    
    def __init__(self, name: str, value: str = "", 
                 confidence: float = 0.0,
                 bbox: Optional[BoundingBox] = None,
                 source: str = "unknown"):
        self.name = name
        self.value = value
        self.confidence = confidence
        self.bbox = bbox
        self.source = source
        self.validation_status = "pending"
        self.corrected_value = ""
        self.issues = []
        self.needs_review = False
    
    def update_validation(self, result: ValidationResult):
        """Update field with validation results."""
        if result.corrected_value:
            self.corrected_value = result.corrected_value
        self.validation_status = "valid" if result.is_valid else "invalid"
        self.issues.extend(result.issues)
        self.confidence += result.confidence_adjustment
        self.confidence = max(0.0, min(1.0, self.confidence))


class PageResult:
    """Extraction result for single page."""
    
    def __init__(self, page_num: int, fields: Optional[List[Field]] = None,
                 processing_time_ms: float = 0.0):
        self.page_num = page_num
        self.fields = fields or []
        self.processing_time_ms = processing_time_ms
        self.metadata = {}
    
    def get_field(self, name: str) -> Optional[Field]:
        """Get field by name."""
        for f in self.fields:
            if f.name == name:
                return f
        return None


class DocumentResult:
    """Final extraction result."""
    
    def __init__(self, document_id: str, 
                 pages: Optional[List[PageResult]] = None,
                 total_processing_time_ms: float = 0.0):
        self.document_id = document_id
        self.pages = pages or []
        self.total_processing_time_ms = total_processing_time_ms
        self.metadata = {}
        self.overall_confidence = 0.0
    
    def all_fields(self) -> List[Field]:
        """Get all fields from all pages."""
        fields = []
        for page in self.pages:
            fields.extend(page.fields)
        return fields
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "document_id": self.document_id,
            "pages": [
                {
                    "page_num": p.page_num,
                    "fields": [
                        {
                            "name": f.name,
                            "value": f.value,
                            "confidence": f.confidence,
                            "source": f.source,
                            "validation": f.validation_status
                        }
                        for f in p.fields
                    ]
                }
                for p in self.pages
            ],
            "total_time_ms": self.total_processing_time_ms,
            "overall_confidence": self.overall_confidence
        }
