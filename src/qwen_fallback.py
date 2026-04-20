"""
Qwen2.5-VL Fallback Module
For low-confidence fields only
"""
import os
import logging
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class QwenResult:
    value: str
    confidence: float
    reasoning: str
    model: str


class QwenFallback:
    """
    Qwen2.5-VL fallback for ambiguous fields
    Only called when confidence < threshold
    """
    
    def __init__(self, model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
        self.model_path = model_path
        self.model = None
        self.processor = None
        self._loaded = False
        self.device = "cuda"
    
    def load(self):
        """Lazy load Qwen model"""
        if self._loaded:
            return
        
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForVision2Seq
            
            logger.info(f"Loading Qwen2.5-VL from {self.model_path}...")
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self._loaded = True
            logger.info("Qwen2.5-VL loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load Qwen: {e}")
            self._loaded = False
    
    def query(self, image: Any, question: str, context: str = "") -> QwenResult:
        """
        Query Qwen about a specific field
        
        Args:
            image: PIL Image or numpy array
            question: Question to ask about the image
            context: Additional context
        
        Returns:
            QwenResult with value and confidence
        """
        self.load()
        
        if not self._loaded:
            return QwenResult(
                value="",
                confidence=0.0,
                reasoning="Model not loaded",
                model="none"
            )
        
        try:
            # Prepare prompt
            if context:
                prompt = f"{context}\n\nQuestion: {question}\nAnswer concisely:"
            else:
                prompt = f"Question: {question}\nAnswer concisely:"
            
            # Process
            inputs = self.processor(
                images=image,
                text=prompt,
                return_tensors="pt"
            ).to(self.model.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.3
                )
            
            # Decode
            result_text = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Extract answer (remove prompt)
            answer = result_text.replace(prompt, "").strip()
            
            return QwenResult(
                value=answer,
                confidence=0.85,  # Estimated
                reasoning="Qwen extraction",
                model="Qwen2.5-VL-7B"
            )
            
        except Exception as e:
            logger.warning(f"Qwen query failed: {e}")
            return QwenResult(
                value="",
                confidence=0.0,
                reasoning=f"Error: {str(e)}",
                model="none"
            )
    
    def extract_field(self, cropped_image: Any, field_name: str, field_type: str) -> QwenResult:
        """
        Extract a specific field from cropped image
        
        Args:
            cropped_image: Cropped image of the field region
            field_name: Name of the field (e.g., "Date of Birth")
            field_type: Type of field (e.g., "date", "name", "number")
        
        Returns:
            QwenResult
        """
        question = f"What is the handwritten value in the '{field_name}' field? "
        
        if field_type == 'date':
            question += "Extract the date in DD/MM/YYYY format if possible."
        elif field_type == 'name':
            question += "Extract the person's name exactly as written."
        elif field_type == 'number' or field_type == 'money':
            question += "Extract only the numeric value."
        elif field_type == 'choice':
            question += "Select the most appropriate option."
        else:
            question += "Extract the handwritten text exactly as written."
        
        return self.query(cropped_image, question)
    
    def batch_extract(self, images_and_fields: List[Dict]) -> List[QwenResult]:
        """
        Batch extract multiple fields (more efficient than individual calls)
        
        Args:
            images_and_fields: List of dicts with 'image', 'field_name', 'field_type'
        
        Returns:
            List of QwenResult
        """
        # For now, just call individually
        # Could be optimized with batch inference
        results = []
        for item in images_and_fields:
            result = self.extract_field(
                item['image'],
                item['field_name'],
                item['field_type']
            )
            results.append(result)
        
        return results


def extract_with_qwen(image: Any, field_name: str, field_type: str = "text") -> QwenResult:
    """Convenience function for single field extraction"""
    fallback = QwenFallback()
    return fallback.extract_field(image, field_name, field_type)
