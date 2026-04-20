"""
PaddleOCR Advanced Engine
Bilingual (Hindi+English) OCR with PaddleX integration
"""
import os
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class OCRResult:
    text: str
    confidence: float
    bbox: List[float]
    language: str


class PaddleOCREngine:
    """
    Advanced PaddleOCR engine with:
    - Bilingual (Hindi + English) support
    - Handwriting optimization
    - PaddleX layout integration
    """
    
    def __init__(self):
        self.ocr_en = None
        self.ocr_hi = None
        self._loaded = False
    
    def load(self):
        """Lazy load PaddleOCR models"""
        if self._loaded:
            return
        
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "1"
        
        from paddleocr import PaddleOCR
        
        # English OCR - optimized for handwriting
        self.ocr_en = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            use_gpu=True,
            show_log=False,
            det_model_dir=None,  # Use default
            rec_model_dir=None,
            cls_model_dir=None,
            det_db_thresh=0.3,
            det_db_box_thresh=0.5,
            rec_batch_num=8,
            max_text_length=100,
            use_space_char=True,
            drop_score=0.5,
            det_db_unclip_ratio=1.6,
        )
        
        # Hindi OCR - for Devanagari script
        self.ocr_hi = PaddleOCR(
            use_angle_cls=True,
            lang='hi',
            use_gpu=True,
            show_log=False,
            det_model_dir=None,
            rec_model_dir=None,
            cls_model_dir=None,
            det_db_thresh=0.3,
            det_db_box_thresh=0.5,
            rec_batch_num=8,
            max_text_length=100,
            use_space_char=True,
            drop_score=0.5,
        )
        
        self._loaded = True
    
    def ocr_bilingual(self, img: np.ndarray) -> List[OCRResult]:
        """
        Run bilingual OCR (Hindi + English) on image
        Returns combined results with language tagging
        """
        self.load()
        
        results = []
        
        # Run English OCR
        en_result = self.ocr_en.ocr(img, cls=True)
        if en_result and en_result[0]:
            for line in en_result[0]:
                bbox_flat = line[0]
                text = line[1][0]
                conf = line[1][1]
                results.append(OCRResult(
                    text=text,
                    confidence=conf,
                    bbox=bbox_flat,
                    language='en'
                ))
        
        # Run Hindi OCR
        hi_result = self.ocr_hi.ocr(img, cls=True)
        if hi_result and hi_result[0]:
            for line in hi_result[0]:
                bbox_flat = line[0]
                text = line[1][0]
                conf = line[1][1]
                results.append(OCRResult(
                    text=text,
                    confidence=conf,
                    bbox=bbox_flat,
                    language='hi'
                ))
        
        return results
    
    def ocr_on_region(self, img: np.ndarray, bbox: List[int], language: str = 'en') -> str:
        """
        Extract text from a specific region (bbox)
        """
        self.load()
        
        x1, y1, x2, y2 = map(int, bbox)
        cropped = img[y1:y2, x1:x2]
        
        if cropped.size == 0:
            return ""
        
        ocr_engine = self.ocr_hi if language == 'hi' else self.ocr_en
        result = ocr_engine.ocr(cropped, cls=True)
        
        texts = []
        if result and result[0]:
            for line in result[0]:
                texts.append(line[1][0])
        
        return ' '.join(texts)


# Global instance
ocr_engine = PaddleOCREngine()
