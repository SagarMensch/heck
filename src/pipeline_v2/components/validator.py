"""
Field Validator
===============
Validates extracted fields using regex, checksums, and cross-field checks.
Wraps existing ValidationKB from pipeline v1.
"""

import logging
from typing import List, Dict, Optional

from ..core.interfaces import IValidator, Field, ValidationResult

logger = logging.getLogger(__name__)


class FieldValidator(IValidator):
    """
    Field validation using existing ValidationKB.
    """
    
    def __init__(self, config):
        self.config = config
        self._validator = None
        self._loaded = False
        
    def load(self):
        """Load validation engine."""
        if self._loaded:
            return
            
        try:
            from src.pipeline.layers.validation_kb import ValidationKB
            
            self._validator = ValidationKB()
            self._loaded = True
            logger.info("Validator loaded")
            
        except Exception as e:
            logger.error(f"Failed to load validator: {e}")
            self._loaded = False
    
    def validate(self, field: Field) -> ValidationResult:
        """
        Validate single field.
        
        Args:
            field: Field to validate
            
        Returns:
            ValidationResult with status and corrections
        """
        if not self._loaded:
            self.load()
        
        if not self._loaded or not field.value:
            return ValidationResult(
                is_valid=True,
                corrected_value=field.value,
                confidence_adjustment=0.0
            )
        
        # Use existing validation if available
        if self._loaded:
            try:
                # Convert to format expected by ValidationKB
                from src.pipeline.models.schemas import ExtractedField, ValidationStatus
                
                ef = ExtractedField(
                    field_name=field.name,
                    value=field.value,
                    confidence=field.confidence,
                    source="ocr"
                )
                
                # Validate
                validated = self._validator.validate_field(ef)
                
                # Update field
                field.validation_status = validated.validation_status
                field.corrected_value = validated.value if validated.kb_corrected else ""
                field.validation_issues = validated.cross_field_issues
                
                # Calculate confidence adjustment
                conf_adj = 0.0
                if validated.kb_corrected:
                    conf_adj = 0.05
                if validated.validation_status == ValidationStatus.INVALID.value:
                    conf_adj = -0.3
                
                return ValidationResult(
                    is_valid=validated.validation_status != ValidationStatus.INVALID.value,
                    corrected_value=validated.value if validated.kb_corrected else field.value,
                    issues=validated.cross_field_issues,
                    confidence_adjustment=conf_adj
                )
                
            except Exception as e:
                logger.warning(f"Validation failed: {e}")
        
        # Fallback: basic validation
        return self._basic_validate(field)
    
    def _basic_validate(self, field: Field) -> ValidationResult:
        """Basic validation without ValidationKB."""
        import re
        
        value = field.value.strip()
        issues = []
        conf_adj = 0.0
        is_valid = True
        
        # Field-specific validation
        validators = {
            "Proposer_PAN": (r"^[A-Z]{5}\d{4}[A-Z]$", "PAN format"),
            "Proposer_Aadhaar": (r"^\d{12}$", "12 digits"),
            "Proposer_Pincode": (r"^\d{6}$", "6 digits"),
            "Proposer_Mobile": (r"^[6-9]\d{9}$", "10 digits"),
            "Bank_IFSC": (r"^[A-Z]{4}0[A-Z0-9]{6}$", "IFSC format"),
        }
        
        if field.name in validators:
            pattern, desc = validators[field.name]
            if not re.match(pattern, value.upper().replace(" ", "")):
                is_valid = False
                issues.append(f"Invalid {desc}")
                conf_adj = -0.3
        
        return ValidationResult(
            is_valid=is_valid,
            corrected_value=value,
            issues=issues,
            confidence_adjustment=conf_adj
        )
    
    def validate_cross_fields(self, fields: List[Field]) -> None:
        """Cross-field validation (e.g., DOB vs Age)."""
        if not self._loaded:
            return
            
        try:
            # Find related fields
            dob_field = next((f for f in fields if "DOB" in f.name), None)
            age_field = next((f for f in fields if "Age" in f.name and "LA" not in f.name), None)
            
            if dob_field and age_field and dob_field.value and age_field.value:
                # Validate DOB matches Age
                from datetime import datetime
                
                try:
                    dob_str = dob_field.value.replace("-", "/").replace(".", "/")
                    dob = datetime.strptime(dob_str, "%d/%m/%Y")
                    computed_age = int((datetime.now() - dob).days / 365.25)
                    stated_age = int(''.join(filter(str.isdigit, age_field.value)))
                    
                    if abs(computed_age - stated_age) > 2:
                        age_field.validation_issues.append(
                            f"Age mismatch: DOB suggests {computed_age}, stated {stated_age}"
                        )
                        age_field.needs_human_review = True
                        
                except Exception:
                    pass
                    
        except Exception as e:
            logger.warning(f"Cross-field validation failed: {e}")
    
    def validate_document(self, fields: List[Field]) -> Dict:
        """Validate entire document."""
        for field in fields:
            self.validate(field)
        
        self.validate_cross_fields(fields)
        
        return {
            "total": len(fields),
            "valid": len([f for f in fields if f.validation_status == "valid"]),
            "invalid": len([f for f in fields if f.validation_status == "invalid"]),
            "needs_review": len([f for f in fields if f.needs_human_review])
        }
