"""
Nemotron Layout Detector
=========================
Uses NVIDIA Nemotron-table-structure-v1 for table detection.
"""

import logging
from typing import Dict, List, Optional
from PIL import Image
import torch
import numpy as np

from ..core.interfaces import ILayoutDetector, BoundingBox, TableStructure

logger = logging.getLogger(__name__)


class NemotronLayoutDetector(ILayoutDetector):
    """
    Table structure detection using Nemotron model.
    
    Falls back to rule-based detection if model unavailable.
    """
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.processor = None
        self._loaded = False
        self._cache = {}
        
    def load(self):
        """Load Nemotron model."""
        if self._loaded:
            return
            
        try:
            from transformers import AutoModelForObjectDetection, AutoProcessor
            
            model_name = self.config.layout_model
            logger.info(f"Loading Nemotron: {model_name}")
            
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            self.model = AutoModelForObjectDetection.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            self._loaded = True
            logger.info("Nemotron loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Nemotron: {e}")
            logger.warning("Will use fallback detection")
            self._loaded = False
    
    def detect(self, image: Image.Image, page_num: int = 0) -> TableStructure:
        """
        Detect table structure.
        
        Returns:
            TableStructure with cells
        """
        # Check cache
        cache_key = f"page_{page_num}"
        if self.config.use_cached_structure and cache_key in self._cache:
            return self._cache[cache_key]
        
        # Load model if needed
        if not self._loaded:
            self.load()
        
        # Detect
        if self._loaded:
            result = self._detect_with_nemotron(image)
        else:
            result = self._detect_fallback(image)
        
        # Cache
        if self.config.use_cached_structure:
            self._cache[cache_key] = result
        
        return result
    
    def _detect_with_nemotron(self, image: Image.Image) -> TableStructure:
        """Detect using Nemotron model."""
        try:
            # Process
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Post-process
            target_sizes = torch.tensor([image.size[::-1]])
            results = self.processor.post_process_object_detection(
                outputs,
                threshold=0.3,
                target_sizes=target_sizes
            )[0]
            
            # Extract cells
            cells = []
            for score, label, box in zip(
                results["scores"], 
                results["labels"], 
                results["boxes"]
            ):
                x1, y1, x2, y2 = box.tolist()
                
                cells.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "score": float(score),
                    "label": int(label),
                    "row": -1,  # Will be assigned
                    "col": -1
                })
            
            # Organize into rows/cols
            cells = self._organize_cells(cells, image.size)
            
            return TableStructure(
                cells=cells,
                method="nemotron",
                confidence=float(results["scores"].mean()) if len(results["scores"]) > 0 else 0.5
            )
            
        except Exception as e:
            logger.error(f"Nemotron detection failed: {e}")
            return self._detect_fallback(image)
    
    def _detect_fallback(self, image: Image.Image) -> TableStructure:
        """Fallback: Rule-based table detection."""
        import cv2
        
        img = np.array(image)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Detect lines
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate to connect line segments
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours (cells)
        contours, _ = cv2.findContours(
            edges, 
            cv2.RETR_TREE, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        cells = []
        h, w = img.shape[:2]
        
        min_area = (w * h) * 0.001  # 0.1% of image
        max_area = (w * h) * 0.1   # 10% of image
        
        for i, cnt in enumerate(contours):
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cw * ch
            
            if min_area < area < max_area:
                cells.append({
                    "bbox": [x, y, x + cw, y + ch],
                    "score": 0.5,
                    "row": -1,
                    "col": -1
                })
        
        # Organize
        cells = self._organize_cells(cells, image.size)
        
        return TableStructure(
            cells=cells,
            method="fallback",
            confidence=0.5
        )
    
    def _organize_cells(self, cells: List[Dict], img_size: tuple) -> List[Dict]:
        """Organize cells into rows and columns."""
        if not cells:
            return cells
        
        w, h = img_size
        
        # Sort by Y position
        cells_sorted = sorted(cells, key=lambda c: c["bbox"][1])
        
        # Group into rows
        rows = []
        current_row = []
        prev_y = cells_sorted[0]["bbox"][1]
        
        for cell in cells_sorted:
            y = cell["bbox"][1]
            h_cell = cell["bbox"][3] - cell["bbox"][1]
            
            # New row if Y changed significantly
            if abs(y - prev_y) > max(h_cell * 0.5, h * 0.02):
                if current_row:
                    rows.append(sorted(current_row, key=lambda c: c["bbox"][0]))
                current_row = [cell]
                prev_y = y
            else:
                current_row.append(cell)
        
        if current_row:
            rows.append(sorted(current_row, key=lambda c: c["bbox"][0]))
        
        # Assign row/col
        organized = []
        for row_idx, row in enumerate(rows):
            for col_idx, cell in enumerate(row):
                cell["row"] = row_idx
                cell["col"] = col_idx
                organized.append(cell)
        
        return organized
    
    def get_field_region(self, field_name: str) -> Optional[BoundingBox]:
        """Get region for specific field."""
        # Map field to cell
        from ..utils.field_mappings import LIC_FIELD_MAP
        
        field_map = LIC_FIELD_MAP.get("form_300", {})
        if field_name not in field_map:
            return None
        
        row, col = field_map[field_name]
        
        # Find matching cell
        for cache_key, structure in self._cache.items():
            for cell in structure.cells:
                if cell.get("row") == row and cell.get("col") == col:
                    return BoundingBox(*cell["bbox"])
        
        return None
    
    def clear_cache(self):
        """Clear detection cache."""
        self._cache = {}
