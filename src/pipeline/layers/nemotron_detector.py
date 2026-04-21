"""
Nemotron Table Structure Detector
==================================
Uses NVIDIA Nemotron-table-structure-v1 for table/cell detection.
Provides accurate bounding boxes for form fields.

Download: huggingface.co/nvidia/nemotron-table-structure-v1
"""

import os
import logging
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from PIL import Image
import torch

logger = logging.getLogger(__name__)


class NemotronTableDetector:
    """
    Detects table structure using Nemotron model.
    Outputs cell bounding boxes for downstream OCR.
    """
    
    def __init__(self, model_name: str = "nvidia/nemotron-table-structure-v1"):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self._loaded = False
        
        # Cache for table structure (same form layout)
        self._structure_cache = {}
        
    def load(self):
        """Load Nemotron model."""
        if self._loaded:
            return
            
        logger.info(f"Loading Nemotron: {self.model_name}...")
        
        try:
            from transformers import AutoModelForObjectDetection, AutoProcessor
            
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            self.model = AutoModelForObjectDetection.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            self._loaded = True
            logger.info("Nemotron loaded on GPU")
            
        except Exception as e:
            logger.error(f"Failed to load Nemotron: {e}")
            # Fallback to manual detection
            self._loaded = False
            
    def detect_table(self, pil_img: Image.Image, form_id: str = "default") -> Dict:
        """
        Detect table structure.
        
        Returns:
            Dictionary with cells: [{"bbox": [x1,y1,x2,y2], "row": i, "col": j}]
        """
        # Check cache
        if form_id in self._structure_cache:
            return self._structure_cache[form_id]
            
        if not self._loaded:
            self.load()
            
        if not self._loaded:
            # Fallback: use manual grid detection
            return self._fallback_detection(pil_img)
            
        try:
            # Process with Nemotron
            inputs = self.processor(images=pil_img, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Post-process
            target_sizes = torch.tensor([pil_img.size[::-1]])
            results = self.processor.post_process_object_detection(
                outputs, 
                threshold=0.5,
                target_sizes=target_sizes
            )[0]
            
            # Extract cells
            cells = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                x1, y1, x2, y2 = box.tolist()
                cells.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "score": float(score),
                    "label": int(label)
                })
                
            # Sort into rows/cols
            cells = self._organize_cells(cells)
            
            result = {"cells": cells, "method": "nemotron"}
            
            # Cache it
            self._structure_cache[form_id] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Nemotron detection failed: {e}")
            return self._fallback_detection(pil_img)
            
    def _fallback_detection(self, pil_img: Image.Image) -> Dict:
        """Fallback: simple grid-based detection."""
        import cv2
        
        w, h = pil_img.size
        img = np.array(pil_img)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Detect lines
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        
        cells = []
        
        if lines is not None:
            # Group horizontal and vertical lines
            h_lines = []
            v_lines = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(x2-x1) > abs(y2-y1):
                    h_lines.append((y1+y2)//2)
                else:
                    v_lines.append((x1+x2)//2)
                    
            # Sort and deduplicate
            h_lines = sorted(list(set([int(y) for y in h_lines])))
            v_lines = sorted(list(set([int(x) for x in v_lines])))
            
            # Create cells
            for i in range(len(h_lines)-1):
                for j in range(len(v_lines)-1):
                    cells.append({
                        "bbox": [v_lines[j], h_lines[i], v_lines[j+1], h_lines[i+1]],
                        "row": i,
                        "col": j,
                        "score": 0.5
                    })
                    
        return {"cells": cells, "method": "fallback"}
        
    def _organize_cells(self, cells: List[Dict]) -> List[Dict]:
        """Organize cells into row/col structure."""
        if not cells:
            return cells
            
        # Group by Y position (rows)
        cells_sorted = sorted(cells, key=lambda c: c["bbox"][1])
        
        # Simple row assignment
        rows = []
        current_row = []
        prev_y = cells_sorted[0]["bbox"][1]
        
        for cell in cells_sorted:
            y = cell["bbox"][1]
            if abs(y - prev_y) > 30:  # New row
                if current_row:
                    rows.append(sorted(current_row, key=lambda c: c["bbox"][0]))
                current_row = [cell]
                prev_y = y
            else:
                current_row.append(cell)
                
        if current_row:
            rows.append(sorted(current_row, key=lambda c: c["bbox"][0]))
            
        # Assign row/col indices
        organized = []
        for row_idx, row in enumerate(rows):
            for col_idx, cell in enumerate(row):
                cell["row"] = row_idx
                cell["col"] = col_idx
                organized.append(cell)
                
        return organized
        
    def get_field_regions(self, table_result: Dict, field_mapping: Dict) -> Dict:
        """
        Map detected cells to known fields.
        
        Args:
            table_result: Output from detect_table
            field_mapping: {field_name: (row, col)} mapping
            
        Returns:
            {field_name: {"bbox": [x1,y1,x2,y2]}}
        """
        cells = table_result.get("cells", [])
        field_regions = {}
        
        for field_name, (row, col) in field_mapping.items():
            # Find matching cell
            for cell in cells:
                if cell.get("row") == row and cell.get("col") == col:
                    field_regions[field_name] = {
                        "bbox": cell["bbox"],
                        "row": row,
                        "col": col
                    }
                    break
                    
        return field_regions
        
    def clear_cache(self):
        """Clear structure cache."""
        self._structure_cache = {}
