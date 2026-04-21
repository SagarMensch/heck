"""
PaddleOCR Engine
================
Optimized OCR using PaddleOCR for form fields.
"""

import os
import logging
from typing import List, Tuple, Optional
from PIL import Image
import numpy as np

from ..core.interfaces import IOCR, OCRResult, BoundingBox

logger = logging.getLogger(__name__)

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "1"


class PaddleOCREngine(IOCR):
    """
    PaddleOCR-based text recognition.
    """
    
    def __init__(self, config):
        self.config = config
        self.ocr_en = None
        self.ocr_hi = None
        self._loaded = False
        
    def load(self):
        """Load PaddleOCR models."""
        if self._loaded:
            return
            
        try:
            from paddleocr import PaddleOCR
            
            logger.info("Loading PaddleOCR...")
            
            # English OCR
            self.ocr_en = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                use_gpu=True,
                show_log=False
            )
            
            # Hindi OCR
            self.ocr_hi = PaddleOCR(
                use_angle_cls=True,
                lang='hi',
                use_gpu=True,
                show_log=False
            )
            
            self._loaded = True
            logger.info("PaddleOCR loaded (EN+HI)")
            
        except Exception as e:
            logger.error(f"Failed to load PaddleOCR: {e}")
            raise
    
    def recognize(self, image: Image.Image, 
                  bbox: Optional[BoundingBox] = None) -> OCRResult:
        """
        Recognize text in image or region.
        
        Args:
            image: PIL Image
            bbox: Optional region to crop
            
        Returns:
            OCRResult with text and confidence
        """
        if not self._loaded:
            self.load()
        
        # Crop if bbox provided
        if bbox:
            image = bbox.crop(image)
        
        img_array = np.array(image)
        
        # Run OCR
        try:
            # Try English first
            result = self.ocr_en.ocr(img_array, cls=True)
            
            if not result or not result[0]:
                # Try Hindi
                result = self.ocr_hi.ocr(img_array, cls=True)
                lang = "hi"
            else:
                lang = "en"
            
            if result and result[0]:
                lines = result[0]
                texts = []
                confidences = []
                
                for line in lines:
                    if line:
                        text = line[1][0]
                        conf = line[1][1]
                        texts.append(text)
                        confidences.append(conf)
                
                full_text = " ".join(texts).strip()
                avg_conf = np.mean(confidences) if confidences else 0.0
                
                return OCRResult(
                    text=full_text,
                    confidence=float(avg_conf),
                    bbox=bbox,
                    language=lang
                )
            
            return OCRResult(text="", confidence=0.0, bbox=bbox)
            
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return OCRResult(text="", confidence=0.0, bbox=bbox)
    
    def recognize_batch(self, regions: List[Tuple[Image.Image, Optional[BoundingBox]]]) -> List[OCRResult]:
        """
        Batch OCR for multiple regions.
        
        Note: PaddleOCR doesn't support true batching, so we process sequentially.
        """
        results = []
        for image, bbox in regions:
            result = self.recognize(image, bbox)
            results.append(result)
        return results
