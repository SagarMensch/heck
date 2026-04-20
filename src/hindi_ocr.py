"""
Hindi OCR Engine with PaddleOCR
Enables bilingual (Hindi + English) OCR for LIC forms
"""
import os
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class BilingualOCRResult:
    text: str
    confidence: float
    bbox: List[float]
    language: str
    is_handwritten: bool


class BilingualOCREngine:
    """
    Advanced bilingual OCR with:
    - Hindi (Devanagari) support
    - English support
    - Handwriting detection
    - Confidence scoring
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
        
        print("Loading English OCR model...")
        self.ocr_en = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            use_gpu=True,
            show_log=False,
            det_db_thresh=0.3,
            det_db_box_thresh=0.5,
            rec_batch_num=8,
            max_text_length=250,
            use_space_char=True,
            drop_score=0.5,
        )
        
        print("Loading Hindi OCR model...")
        self.ocr_hi = PaddleOCR(
            use_angle_cls=True,
            lang='hi',
            use_gpu=True,
            show_log=False,
            det_db_thresh=0.3,
            det_db_box_thresh=0.5,
            rec_batch_num=8,
            max_text_length=250,
            use_space_char=True,
            drop_score=0.5,
        )
        
        self._loaded = True
        print("Bilingual OCR loaded successfully")
    
    def ocr_bilingual(self, img: np.ndarray) -> List[BilingualOCRResult]:
        """Run bilingual OCR on image"""
        self.load()
        
        all_results = []
        
        try:
            en_result = self.ocr_en.ocr(img, cls=True)
            if en_result and en_result[0]:
                for line in en_result[0]:
                    bbox = line[0]
                    text = line[1][0]
                    conf = line[1][1]
                    all_results.append(BilingualOCRResult(
                        text=text, confidence=conf, bbox=bbox,
                        language='en', is_handwritten=self._detect_handwriting(text)
                    ))
        except Exception as e:
            print(f"English OCR error: {e}")
        
        try:
            hi_result = self.ocr_hi.ocr(img, cls=True)
            if hi_result and hi_result[0]:
                for line in hi_result[0]:
                    bbox = line[0]
                    text = line[1][0]
                    conf = line[1][1]
                    all_results.append(BilingualOCRResult(
                        text=text, confidence=conf, bbox=bbox,
                        language='hi', is_handwritten=self._detect_handwriting(text)
                    ))
        except Exception as e:
            print(f"Hindi OCR error: {e}")
        
        all_results.sort(key=lambda x: x.confidence, reverse=True)
        return all_results
    
    def _detect_handwriting(self, text: str) -> bool:
        if not text:
            return False
        text_upper = text.upper()
        confusion_patterns = ['0O', 'O0', '1l', 'l1', '5S', 'S5', '8B', 'B8']
        for pattern in confusion_patterns:
            if pattern in text_upper:
                return True
        if '  ' in text or text.startswith(' ') or text.endswith(' '):
            return True
        return False
    
    def ocr_on_region(self, img: np.ndarray, bbox: List[int]) -> str:
        """Extract text from a specific region"""
        self.load()
        
        x1, y1, x2, y2 = map(int, bbox)
        cropped = img[y1:y2, x1:x2]
        
        if cropped.size == 0:
            return ""
        
        try:
            hi_result = self.ocr_hi.ocr(cropped, cls=True)
            if hi_result and hi_result[0]:
                texts = [line[1][0] for line in hi_result[0]]
                if texts:
                    return ' '.join(texts)
        except:
            pass
        
        try:
            en_result = self.ocr_en.ocr(cropped, cls=True)
            if en_result and en_result[0]:
                texts = [line[1][0] for line in en_result[0]]
                return ' '.join(texts)
        except:
            pass
        
        return ""


bilingual_ocr = BilingualOCREngine()
