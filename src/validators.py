"""
Post-Processing Validators
===========================
Field-level regex, cross-field logic, OCR error correction,
hallucination detection, repetitive output detection, semantic validation.
"""

import re
import logging
import string
from collections import Counter
from datetime import datetime, date
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from src.config import (
    VALIDATION_RULES, FORM_300_FIELDS, MANDATORY_FIELDS,
    CONFIDENCE_HIGH, CONFIDENCE_MEDIUM, CONFIDENCE_LOW,
    FIELD_REJECT_THRESHOLD, REVERSE_FIELD_NAME_MAP,
)

logger = logging.getLogger(__name__)

OCR_CONFUSIONS = {
    "0": "O", "O": "0",
    "1": "I", "I": "1", "l": "1",
    "5": "S", "S": "5",
    "8": "B", "B": "8",
    "2": "Z", "Z": "2",
    "6": "G", "G": "6",
}

PAN_LETTER_POSITIONS = {0, 1, 2, 3, 4, 9}
PAN_DIGIT_POSITIONS = {5, 6, 7, 8}

TROCR_GIBBERISH_PATTERNS = [
    re.compile(r"^[^a-zA-Z0-9\s]+$"),
    re.compile(r"(.)\1{4,}"),
    re.compile(r"^[bcdfghjklmnpqrstvwxyz]{5,}$", re.IGNORECASE),
    re.compile(r"^[aeiou]{5,}$", re.IGNORECASE),
    re.compile(r"^\W+$"),
]

KNOWN_TROCR_HALLUCINATIONS = {
    "avelandian", "Linsularate", "thelasthe", "thelasthe",
    "INDIA", "lhe", "lhe_", "Tha", "Dhe", "Fhe",
    "Tand", "tand", "tion", "tion_", "ing", "ing_",
    "NaN", "nan", "None", "null", "NULL",
}

INDIAN_STATES = {
    "andhra pradesh", "arunachal pradesh", "assam", "bihar",
    "chhattisgarh", "goa", "gujarat", "haryana", "himachal pradesh",
    "jharkhand", "karnataka", "kerala", "madhya pradesh", "maharashtra",
    "manipur", "meghalaya", "mizoram", "nagaland", "odisha",
    "punjab", "rajasthan", "sikkim", "tamil nadu", "telangana",
    "tripura", "uttar pradesh", "uttarakhand", "west bengal",
    "delhi", "chandigarh", "puducherry", "jammu and kashmir",
    "ladakh", "andaman and nicobar", "dadra and nagar haveli",
    "daman and diu", "lakshadweep",
}

LIC_PLAN_NAMES = {
    "jeevan anand", "jeevan umang", "jeevan labh", "jeevan lakshya",
    "jeevan saathi", "jeevan shanti", "jeevan sugam", "jeevan tarun",
    "jeevan pragati", "jeevan vansh", "jeevan kiran", "jeevan utsav",
    "dhan sanchay", "dhan varsha", "dhan rekha", "dhan laxmi",
    "bima jyoti", "bima ratna", "bima gold", "bima shree",
    "tech term", "sarb jeevan", "new endowment", "amulya jeevan",
    "anmol jeevan", "new jeevan anand",
}

FAMILY_RELATIONSHIPS = {
    "father", "mother", "spouse", "husband", "wife",
    "son", "daughter", "brother", "sister",
    "grandfather", "grandmother", "uncle", "aunt",
    "nephew", "niece", "cousin",
    "पिता", "माता", "पति", "पत्नी", "पुत्र", "पुत्री",
    "भाई", "बहन", "बेटा", "बेटी",
}

PREMIUM_MODES = {"yearly", "half-yearly", "quarterly", "monthly", "y", "h", "q", "m",
                 "वार्षिक", "अर्धवार्षिक", "तिमाही", "मासिक"}

GENDER_VALUES = {"male", "female", "other", "m", "f", "o",
                 "पुरुष", "महिला", "अन्य"}

MARITAL_VALUES = {"single", "married", "divorced", "widowed", "unmarried",
                  "अविवाहित", "विवाहित", "तलाक", "विधवा"}


def _is_gibberish(text: str) -> bool:
    if not text or len(text) < 2:
        return True
    for pat in TROCR_GIBBERISH_PATTERNS:
        if pat.match(text):
            return True
    if text.lower() in KNOWN_TROCR_HALLUCINATIONS:
        return True
    return False


def _is_repetitive(text: str, max_repeat_ratio: float = 0.6) -> bool:
    if not text or len(text) < 3:
        return False
    if re.search(r'\d', text):
        return False
    char_counts = Counter(text.lower())
    if not char_counts:
        return False
    most_common_freq = char_counts.most_common(1)[0][1]
    ratio = most_common_freq / len(text)
    return ratio > max_repeat_ratio


def _char_variety_score(text: str) -> float:
    if not text:
        return 0.0
    unique = len(set(text.lower()) - {' '})
    total = len(text.replace(' ', ''))
    return unique / max(total, 1)


class HallucinationDetector:
    def __init__(self):
        self._page_texts: Dict[int, str] = {}

    def register_page_text(self, page_num: int, ocr_text: str):
        self._page_texts[page_num] = ocr_text

    def is_hallucination(self, value: str, field_name: str, field_family: str,
                         confidence: float) -> Tuple[bool, str]:
        if not value or not value.strip():
            return False, ""

        val_stripped = value.strip()

        if val_stripped.lower() in KNOWN_TROCR_HALLUCINATIONS:
            return True, "Known TrOCR hallucination token"

        if _is_gibberish(val_stripped):
            return True, "Gibberish pattern detected"

        if _is_repetitive(val_stripped):
            return True, "Repetitive character pattern"

        if field_family in ("name_text", "short_text") and len(val_stripped) > 2:
            variety = _char_variety_score(val_stripped)
            if variety < 0.2 and confidence < 0.7:
                return True, f"Low character variety ({variety:.2f}) for text field"

        if field_family == "numeric" and not re.search(r'\d', val_stripped):
            return True, "Numeric field has no digits"

        if field_family == "amount" and not re.search(r'\d', val_stripped):
            return True, "Amount field has no digits"

        if field_family == "date" and not re.search(r'\d', val_stripped):
            return True, "Date field has no digits"

        if field_family == "short_id" and len(val_stripped) > 40:
            return True, f"ID field too long ({len(val_stripped)} chars)"

        if field_family in ("name_text", "short_text") and len(val_stripped) > 150:
            return True, f"Text field suspiciously long ({len(val_stripped)} chars)"

        if field_family == "binary_mark" and val_stripped.lower() not in (
            "yes", "no", "true", "false", "1", "0", "x", "✓", "tick", "checked",
            "unchecked", "marked", "unmarked",
        ):
            if len(val_stripped) > 5:
                return True, f"Binary mark field has unexpected value: {val_stripped[:20]}"

        return False, ""


class SemanticValidator:
    def validate(self, field_name: str, value: str, all_fields: Dict) -> Tuple[bool, Optional[str]]:
        if not value or not value.strip():
            return True, None

        val = value.strip()
        config_name = field_name

        if "State" in config_name or "state" in config_name:
            val_lower = val.lower()
            if val_lower not in INDIAN_STATES and len(val) < 3:
                return False, f"Unrecognized Indian state: {val}"

        if "Plan_Name" in config_name or "proposed_plan_name" in config_name:
            val_lower = val.lower()
            if any(p in val_lower for p in LIC_PLAN_NAMES):
                return True, None
            if len(val) < 3:
                return False, "Plan name too short"

        if "Relationship" in config_name or "nominee_relationship" in config_name:
            val_lower = val.lower()
            if val_lower in FAMILY_RELATIONSHIPS:
                return True, None

        if "Premium_Mode" in config_name or "proposed_premium_mode" in config_name:
            val_lower = val.lower()
            if val_lower in PREMIUM_MODES:
                return True, None
            if len(val) <= 2 or val_lower not in PREMIUM_MODES:
                return False, f"Unrecognized premium mode: {val}"

        if "Gender" in config_name or "gender_mark" in config_name:
            val_lower = val.lower()
            if val_lower in GENDER_VALUES:
                return True, None

        if "Marital" in config_name or "marital_status" in config_name:
            val_lower = val.lower()
            if val_lower in MARITAL_VALUES:
                return True, None

        if "Email" in config_name or "email" in config_name:
            if "@" not in val or "." not in val.split("@")[-1]:
                return False, "Invalid email format"

        if "Mobile" in config_name or "mobile_number" in config_name:
            digits = re.sub(r'\D', '', val)
            if len(digits) != 10 or digits[0] not in "6789":
                return False, f"Invalid Indian mobile: {val}"

        return True, None


class FieldValidator:
    """Validates and corrects extracted field values."""

    def __init__(self):
        self.hallucination_detector = HallucinationDetector()
        self.semantic_validator = SemanticValidator()

    def validate_all(self, fields: Dict) -> Dict:
        validated = {}

        for field_name, field_data in fields.items():
            value = field_data.get("value")
            confidence = field_data.get("confidence", 0)
            field_family = field_data.get("field_family", "text")

            if not value:
                validated[field_name] = {
                    **field_data,
                    "validation_status": "missing",
                    "category": "Missing",
                }
                continue

            str_value = str(value)

            is_halluc, halluc_reason = self.hallucination_detector.is_hallucination(
                str_value, field_name, field_family, confidence
            )
            if is_halluc:
                logger.warning(f"Hallucination detected for {field_name}: '{str_value}' ({halluc_reason})")
                validated[field_name] = {
                    **field_data,
                    "value": "",
                    "validation_status": "hallucination",
                    "hallucination_reason": halluc_reason,
                    "original_hallucinated_value": str_value,
                    "category": "Rejected",
                }
                continue

            val_result = self._validate_field(field_name, str_value)

            corrected_value = value
            if not val_result["is_valid"] and val_result.get("correctable"):
                corrected_value = val_result["corrected_value"]
                val_result = self._validate_field(field_name, str(corrected_value))
                field_data["original_value"] = value
                field_data["auto_corrected"] = True

            sem_ok, sem_msg = self.semantic_validator.validate(
                field_name, str(corrected_value), validated
            )
            if not sem_ok:
                val_result["is_valid"] = False
                val_result["message"] = sem_msg

            category = self._categorize_field(confidence, val_result["is_valid"])

            validated[field_name] = {
                **field_data,
                "value": corrected_value,
                "validation_status": "valid" if val_result["is_valid"] else "invalid",
                "validation_message": val_result.get("message", ""),
                "category": category,
            }

        validated = self._cross_field_validate(validated)

        return validated

    def _validate_field(self, field_name: str, value: str) -> Dict:
        result = {"is_valid": True, "message": "OK", "correctable": False}

        if field_name in VALIDATION_RULES:
            rule = VALIDATION_RULES[field_name]
            pattern = rule["regex"]
            clean_value = value.strip().upper().replace(" ", "")

            if not re.match(pattern, clean_value):
                corrected = self._try_ocr_correction(field_name, clean_value)
                if corrected and re.match(pattern, corrected):
                    result["is_valid"] = False
                    result["correctable"] = True
                    result["corrected_value"] = corrected
                    result["message"] = f"Auto-corrected: '{value}' -> '{corrected}'"
                else:
                    result["is_valid"] = False
                    result["message"] = f"Failed regex: {rule['description']}"
            else:
                result["corrected_value"] = clean_value

        if "Date" in field_name or "DOB" in field_name:
            date_result = self._validate_date(value, field_name)
            if not date_result["is_valid"]:
                result = date_result

        if field_name in ("Proposer_Age", "LA_Age", "Nominee_Age"):
            result = self._validate_age(value, field_name)

        return result

    def _try_ocr_correction(self, field_name: str, value: str) -> Optional[str]:
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
        if len(value) != 10:
            return value
        corrected = list(value)
        for i in range(10):
            char = corrected[i]
            if i in PAN_LETTER_POSITIONS:
                if char.isdigit():
                    if char in OCR_CONFUSIONS:
                        corrected[i] = OCR_CONFUSIONS[char]
            elif i in PAN_DIGIT_POSITIONS:
                if char.isalpha():
                    if char in OCR_CONFUSIONS:
                        corrected[i] = OCR_CONFUSIONS[char]
        return "".join(corrected)

    def _correct_ifsc(self, value: str) -> str:
        if len(value) != 11:
            return value
        corrected = list(value.upper())
        for i in range(4):
            if corrected[i].isdigit() and corrected[i] in OCR_CONFUSIONS:
                corrected[i] = OCR_CONFUSIONS[corrected[i]]
        corrected[4] = "0"
        return "".join(corrected)

    def _correct_mobile(self, value: str) -> str:
        digits = re.sub(r'\D', '', value)
        if len(digits) == 12 and digits.startswith("91"):
            digits = digits[2:]
        elif len(digits) == 11 and digits.startswith("0"):
            digits = digits[1:]
        return digits

    def _correct_numeric_only(self, value: str, length: int) -> str:
        digits = re.sub(r'\D', '', value)
        return digits[:length] if len(digits) >= length else digits

    def _validate_date(self, value: str, field_name: str) -> Dict:
        clean = value.strip().replace(" ", "")
        for fmt in ["%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%Y-%m-%d", "%d/%m/%y"]:
            try:
                parsed = datetime.strptime(clean, fmt)
                if parsed.year < 1900 or parsed.year > 2030:
                    return {"is_valid": False, "message": f"Year {parsed.year} out of range"}
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
        issues = []

        dob_field = fields.get("Proposer_Date_of_Birth", {})
        age_field = fields.get("Proposer_Age", {})
        if dob_field.get("value") and age_field.get("value"):
            try:
                dob_str = str(dob_field["value"])
                dob = None
                for fmt in ["%d/%m/%Y", "%d-%m-%Y"]:
                    try:
                        dob = datetime.strptime(dob_str, fmt)
                        break
                    except ValueError:
                        continue

                if dob:
                    computed_age = int((datetime.now() - dob).days / 365.25)
                    stated_age = int(re.sub(r'\D', '', str(age_field["value"])))
                    if abs(computed_age - stated_age) > 1:
                        issues.append(f"Age mismatch: DOB gives {computed_age}, stated {stated_age}")
                        fields["Proposer_Age"]["cross_field_issue"] = True
                        fields["Proposer_Date_of_Birth"]["cross_field_issue"] = True
            except Exception as e:
                logger.debug(f"Cross-field age check failed: {e}")

        sum_assured = fields.get("Sum_Assured", {}).get("value")
        premium = fields.get("Premium_Amount", {}).get("value")
        if sum_assured and premium:
            try:
                sa_num = float(re.sub(r'[^\d.]', '', str(sum_assured)))
                prem_num = float(re.sub(r'[^\d.]', '', str(premium)))
                if prem_num > 0 and sa_num > 0:
                    ratio = sa_num / prem_num
                    if ratio < 1:
                        issues.append("Premium exceeds sum assured — likely swapped")
                        fields["Sum_Assured"]["cross_field_issue"] = True
                        fields["Premium_Amount"]["cross_field_issue"] = True
            except (ValueError, ZeroDivisionError):
                pass

        missing_mandatory = []
        for field_name in MANDATORY_FIELDS:
            if field_name not in fields or not fields[field_name].get("value"):
                missing_mandatory.append(field_name)
        if missing_mandatory:
            issues.append(f"Missing mandatory fields: {missing_mandatory}")

        for field_name in fields:
            if "cross_field_issues" not in fields[field_name]:
                fields[field_name]["cross_field_issues"] = issues

        return fields

    def _categorize_field(self, confidence: float, is_valid: bool) -> str:
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
        fields = raw_extraction.get("fields", {})

        validated_fields = self.validator.validate_all(fields)

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
        hallucinated = sum(1 for f in validated_fields.values()
                          if f.get("validation_status") == "hallucination")

        confidences = [f.get("confidence", 0) for f in validated_fields.values()
                      if f.get("value")]
        overall_confidence = float(np.mean(confidences)) if confidences else 0.0

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
                "fields_hallucinated": hallucinated,
                "overall_confidence": round(overall_confidence, 4),
                "form_completion_rate": round(form_completion, 4),
            },
            "models_used": raw_extraction.get("models_used", []),
            "total_pages": raw_extraction.get("total_pages", 0),
            "preprocessing": preprocessing_info,
        }
