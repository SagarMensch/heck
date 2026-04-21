"""
Qwen-VL Fallback
================
VLM-based extraction for difficult handwritten fields.
Uses existing VLMExtractor from pipeline v1.
"""

import logging
from PIL import Image
import sys
import os

from ..core.interfaces import IVLM, ExtractionResult

logger = logging.getLogger(__name__)


class QwenVLFallback(IVLM):
    """
    VLM fallback using Qwen2.5-VL.
    Wraps existing VLMExtractor from pipeline v1.
    """
    
    def __init__(self, config):
        self.config = config
        self._extractor = None
        self._loaded = False
        
    def load(self):
        """Load VLM model."""
        if self._loaded:
            return
            
        try:
            # Import from existing pipeline
            from src.pipeline.layers.vlm_extractor import VLMExtractor
            
            self._extractor = VLMExtractor(
                model_name=self.config.vlm_model
            )
            
            self._loaded = True
            logger.info(f"VLM fallback loaded: {self.config.vlm_model}")
            
        except Exception as e:
            logger.error(f"Failed to load VLM: {e}")
            self._loaded = False
    
    def extract_field(self, image: Image.Image, field_name: str, 
                    hint: str = "") -> ExtractionResult:
        """
        Extract single field using VLM.
        
        Args:
            image: Full page image
            field_name: Field to extract
            hint: Context hint
            
        Returns:
            ExtractionResult with value and confidence
        """
        if not self._loaded:
            self.load()
        
        if not self._loaded:
            return ExtractionResult(value="", confidence=0.0, source="failed")
        
        try:
            # Create focused prompt
            prompt = self._create_field_prompt(field_name, hint)
            
            # Extract using existing VLM
            result = self._extractor.extract_tile(image, prompt)
            
            value = result.get(field_name, "")
            
            # Calculate confidence based on VLM output
            confidence = 0.85 if value else 0.0
            
            return ExtractionResult(
                value=value if value else "",
                confidence=confidence,
                source="vlm",
                raw_output=str(result)
            )
            
        except Exception as e:
            logger.error(f"VLM extraction failed for {field_name}: {e}")
            return ExtractionResult(value="", confidence=0.0, source="error")
    
    def extract_region(self, image: Image.Image, bbox, field_name: str) -> ExtractionResult:
        """Extract from specific region."""
        # Crop region
        region = image.crop(bbox.to_list())
        return self.extract_field(region, field_name)
    
    def _create_field_prompt(self, field_name: str, hint: str) -> str:
        """Create extraction prompt for field."""
        field_descriptions = {
            "Proposer_Name": "Extract the full name of the proposer.",
            "Proposer_Date_of_Birth": "Extract date of birth in DD/MM/YYYY format.",
            "Proposer_PAN": "Extract PAN number (format: AAAAA9999A).",
            "Proposer_Aadhaar": "Extract Aadhaar number (12 digits).",
            "Sum_Assured": "Extract sum assured amount.",
            "Premium_Amount": "Extract premium amount.",
        }
        
        desc = field_descriptions.get(field_name, f"Extract {field_name}")
        
        prompt = f"""Extract only the {field_name} from this form field.
{desc}

Rules:
- Focus only on handwritten text
- Do not include printed labels
- Return empty if not visible
- Be precise with characters

{hint}"""
        
        return prompt
