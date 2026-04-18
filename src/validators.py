"""
Post-Processing Validators
============================
Field-level regex, cross-field logic, OCR error correction, confidence categorization.
"""

import re
import logging
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.config import (
    VALIDATION_RULES, FORM_300_FIELDS, MANDATORY_FIELDS,
    CONFIDENCE_HIGH, CONFIDENCE_MEDIUM, CONFIDENCE_LOW,
    FIELD_REJECT_THRESHOLD,
)

logger = logging.getLogger(__name__)

# Common OCR confusion pairs for auto-correction
OCR_CONFUSIONS = {
    "0": "O", "O": "0",
    "1": "I", "I": "1", "l": "1",
    "5": "S", "S": "5",
    "8": "B", "B": "8",
    "2": "Z", "Z": "2",
    "6": "G", "G": "6",
}

# PAN position-specific corrections
PAN_LETTER_POSITIONS = {0, 1, 2, 3, 4, 9}  # Positions that must be letters
PAN_DIGIT_POSITIONS = {5, 6, 7, 8}  # Positions that must be digits


class FieldValidator:
    """Validates and corrects extracted field values."""

    def validate_all(self, fields: Dict) -> Dict:
        """Run all validations on extracted fields."""
        validated = {}

        for field_name, field_data in fields.items():
            value = field_data.get("value")
            confidence = field_data.get("confidence", 0)

            # Skip empty/null fields
            if not value:
                validated[field_name] = {
                    **field_data,
                    "validation_status": "missing",
                    "category": "Missing",
                }
                continue

            # Run field-specific validation
            val_result = self._validate_field(field_name, str(value))

            # Apply OCR corrections if validation failed
            corrected_value = value
            if not val_result["is_valid"] and val_result.get("correctable"):
                corrected_value = val_result["corrected_value"]
                val_result = self._validate_field(field_name, str(corrected_value))
                field_data["original_value"] = value
                field_data["auto_corrected"] = True

            # Categorize field
            category = self._categorize_field(confidence, val_result["is_valid"])

            validated[field_name] = {
                **field_data,
                "value": corrected_value,
                "validation_status": "valid" if val_result["is_valid"] else "invalid",
                "validation_message": val_result.get("message", ""),
                "category": category,
            }

        # Cross-field validations
        validated = self._cross_field_validate(validated)

        return validated

    def _validate_field(self, field_name: str, value: str) -> Dict:
        """Validate a single field against its rules."""
        result = {"is_valid": True, "message": "OK", "correctable": False}

        # Check regex rules
        if field_name in VALIDATION_RULES:
            rule = VALIDATION_RULES[field_name]
            pattern = rule["regex"]
            clean_value = value.strip().upper().replace(" ", "")

            if not re.match(pattern, clean_value):
                # Attempt OCR correction
                corrected = self._try_ocr_correction(field_name, clean_value)
                if corrected and re.match(pattern, corrected):
                    result["is_valid"] = False
                    result["correctable"] = True
                    result["corrected_value"] = corrected
                    result["message"] = f"Auto-corrected: '{value}' → '{corrected}'"
                else:
                    result["is_valid"] = False
                    result["message"] = f"Failed regex: {rule['description']}"
            else:
                result["corrected_value"] = clean_value

        # Date-specific validation
        if "Date" in field_name or "DOB" in field_name:
            date_result = self._validate_date(value, field_name)
            if not date_result["is_valid"]:
                result = date_result

        # Numeric range checks
        if field_name in ("Proposer_Age", "LA_Age", "Nominee_Age"):
            result = self._validate_age(value, field_name)

        return result

    def _try_ocr_correction(self, field_name: str, value: str) -> Optional[str]:
        """Attempt to fix common OCR errors."""
        if "PAN" in field_name:
            return self._correct_pan(value)
        elif "IFSC" in field_name:
            return self._correct_ifsc(value)
        elif "Mobile" in field_name:
            return self._correct_mobile(value)
        elif "Pincode" in field_name:
            return self._correct_numeric_only(value, length=6)
        elif "Aadhaar" in field_name:
            return self._correct_numeric_only(value, length=12)
        return None

    def _correct_pan(self, value: str) -> str:
        """Correct common OCR errors in PAN numbers."""
        if len(value) != 10:
            return value

        corrected = list(value)
        for i in range(10):
            char = corrected[i]
            if i in PAN_LETTER_POSITIONS:
                # Should be a letter
                if char.isdigit():
                    if char in OCR_CONFUSIONS:
                        corrected[i] = OCR_CONFUSIONS[char]
            elif i in PAN_DIGIT_POSITIONS:
                # Should be a digit
                if char.isalpha():
                    if char in OCR_CONFUSIONS:
                        corrected[i] = OCR_CONFUSIONS[char]

        return "".join(corrected)

    def _correct_ifsc(self, value: str) -> str:
        """Correct IFSC code errors."""
        if len(value) != 11:
            return value
        corrected = list(value.upper())
        # First 4 must be letters
        for i in range(4):
            if corrected[i].isdigit() and corrected[i] in OCR_CONFUSIONS:
                corrected[i] = OCR_CONFUSIONS[corrected[i]]
        # 5th must be 0
        corrected[4] = "0"
        return "".join(corrected)

    def _correct_mobile(self, value: str) -> str:
        """Correct mobile number errors."""
        # Remove non-digit characters
        digits = re.sub(r'\D', '', value)
        # Remove country code if present
        if len(digits) == 12 and digits.startswith("91"):
            digits = digits[2:]
        elif len(digits) == 11 and digits.startswith("0"):
            digits = digits[1:]
        return digits

    def _correct_numeric_only(self, value: str, length: int) -> str:
        """Extract only digits from a value."""
        digits = re.sub(r'\D', '', value)
        return digits[:length] if len(digits) >= length else digits

    def _validate_date(self, value: str, field_name: str) -> Dict:
        """Validate date values."""
        clean = value.strip().replace(" ", "")
        # Try common formats
        for fmt in ["%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%Y-%m-%d", "%d/%m/%y"]:
            try:
                parsed = datetime.strptime(clean, fmt)
                # Sanity checks
                if parsed.year < 1900 or parsed.year > 2030:
                    return {"is_valid": False, "message": f"Year {parsed.year} out of range"}

                # DOB-specific checks
                if "Birth" in field_name or "DOB" in field_name:
                    age = (datetime.now() - parsed).days / 365.25
                    if age < 0 or age > 120:
                        return {"is_valid": False, "message": f"Age {age:.0f} unreasonable"}
                    if "Proposer" in field_name and age < 18:
                        return {"is_valid": False, "message": f"Proposer age {age:.0f} below 18"}

                return {"is_valid": True, "message": "OK",
                        "corrected_value": parsed.strftime("%d/%m/%Y")}
            except ValueError:
                continue

        return {"is_valid": False, "message": f"Cannot parse date: {value}"}

    def _validate_age(self, value: str, field_name: str) -> Dict:
        """Validate age values."""
        try:
            age = int(re.sub(r'\D', '', str(value)))
            if "Proposer" in field_name and not (18 <= age <= 100):
                return {"is_valid": False, "message": f"Proposer age {age} out of range 18-100"}
            if "Nominee" in field_name and not (0 <= age <= 120):
                return {"is_valid": False, "message": f"Nominee age {age} out of range 0-120"}
            return {"is_valid": True, "message": "OK"}
        except (ValueError, TypeError):
            return {"is_valid": False, "message": f"Cannot parse age: {value}"}

    def _cross_field_validate(self, fields: Dict) -> Dict:
        """Cross-field consistency checks."""
        issues = []

        # Age vs DOB consistency
        dob_field = fields.get("Proposer_Date_of_Birth", {})
        age_field = fields.get("Proposer_Age", {})
        if dob_field.get("value") and age_field.get("value"):
            try:
                dob_str = dob_field["value"]
                for fmt in ["%d/%m/%Y", "%d-%m-%Y"]:
                    try:
                        dob = datetime.strptime(dob_str, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    dob = None

                if dob:
                    computed_age = int((datetime.now() - dob).days / 365.25)
                    stated_age = int(re.sub(r'\D', '', str(age_field["value"])))
                    if abs(computed_age - stated_age) > 1:
                        issues.append(f"Age mismatch: DOB gives {computed_age}, stated {stated_age}")
                        fields["Proposer_Age"]["cross_field_issue"] = True
                        fields["Proposer_Date_of_Birth"]["cross_field_issue"] = True
            except Exception as e:
                logger.debug(f"Cross-field age check failed: {e}")

        # Mandatory field completeness
        missing_mandatory = []
        for field_name in MANDATORY_FIELDS:
            if field_name not in fields or not fields[field_name].get("value"):
                missing_mandatory.append(field_name)

        if missing_mandatory:
            issues.append(f"Missing mandatory fields: {missing_mandatory}")

        # Store cross-field results
        for field_name in fields:
            if "cross_field_issues" not in fields[field_name]:
                fields[field_name]["cross_field_issues"] = issues

        return fields

    def _categorize_field(self, confidence: float, is_valid: bool) -> str:
        """Categorize field into Extracted/Missing/Low Confidence/Rejected."""
        if confidence < FIELD_REJECT_THRESHOLD:
            return "Rejected"
        elif confidence < CONFIDENCE_LOW:
            return "Low Confidence"
        elif confidence < CONFIDENCE_MEDIUM:
            return "Needs Review"
        elif not is_valid:
            return "Validation Failed"
        else:
            return "Extracted"


class ExtractionResultBuilder:
    """Builds the final structured output for a form."""

    def __init__(self):
        self.validator = FieldValidator()

    def build_result(self, form_id: str, raw_extraction: Dict,
                     preprocessing_info: List[Dict] = None) -> Dict:
        """Build the complete extraction result for a form."""
        fields = raw_extraction.get("fields", {})

        # Validate all fields
        validated_fields = self.validator.validate_all(fields)

        # Compute KPIs
        total_expected = len(MANDATORY_FIELDS)
        extracted = sum(1 for f in validated_fields.values()
                       if f.get("category") in ("Extracted", "Needs Review"))
        missing = sum(1 for f in validated_fields.values()
                     if f.get("category") == "Missing")
        low_conf = sum(1 for f in validated_fields.values()
                      if f.get("category") == "Low Confidence")
        rejected = sum(1 for f in validated_fields.values()
                      if f.get("category") == "Rejected")
        validation_failed = sum(1 for f in validated_fields.values()
                               if f.get("category") == "Validation Failed")

        confidences = [f.get("confidence", 0) for f in validated_fields.values()
                      if f.get("value")]
        overall_confidence = float(np.mean(confidences)) if confidences else 0.0

        # Determine form status
        mandatory_present = sum(1 for fn in MANDATORY_FIELDS
                               if fn in validated_fields and
                               validated_fields[fn].get("category") in ("Extracted", "Needs Review"))
        form_completion = mandatory_present / max(total_expected, 1)

        if overall_confidence < 0.5 or form_completion < 0.5:
            form_status = "Rejected"
        elif low_conf + validation_failed > extracted * 0.3:
            form_status = "Needs Review"
        else:
            form_status = "Processed"

        return {
            "form_id": form_id,
            "form_type": "Proposal Form 300",
            "form_status": form_status,
            "fields": validated_fields,
            "kpis": {
                "total_fields_expected": total_expected,
                "fields_extracted": extracted,
                "fields_missing": missing,
                "fields_low_confidence": low_conf,
                "fields_rejected": rejected,
                "fields_validation_failed": validation_failed,
                "overall_confidence": round(overall_confidence, 4),
                "form_completion_rate": round(form_completion, 4),
            },
            "models_used": raw_extraction.get("models_used", []),
            "total_pages": raw_extraction.get("total_pages", 0),
            "preprocessing": preprocessing_info,
        }
