"""
Field-aware normalization and validation rules for polished labeling.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.form300_templates import FORM300_FIELD_INDEX


YES_MARKERS = {"Y", "YES", "✓", "/", "TRUE"}
NO_MARKERS = {"N", "NO", "X", "FALSE"}


@dataclass
class ValidationResult:
    field_name: str
    field_family: str
    raw_text: str
    cleaned_text: str
    canonical_text: str
    is_valid: bool
    score: float
    issues: List[str] = field(default_factory=list)


def _collapse_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _upper_alnum(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9/.-]", "", text).upper()


def normalize_field_value(field_name: str, raw_text: str) -> ValidationResult:
    field = FORM300_FIELD_INDEX[field_name]
    family = field.family
    cleaned = _collapse_spaces(str(raw_text or ""))
    issues: List[str] = []
    score = 0.0
    canonical = cleaned

    if not cleaned:
        return ValidationResult(field_name, family, raw_text, "", "", False, 0.0, ["empty"])

    if family == "binary_mark":
        upper = cleaned.upper().replace(".", "")
        if upper in YES_MARKERS:
            canonical = "YES"
            score = 0.98
        elif upper in NO_MARKERS:
            canonical = "NO"
            score = 0.98
        else:
            issues.append("unknown_mark")
            score = 0.15
        return ValidationResult(field_name, family, raw_text, cleaned, canonical, not issues, score, issues)

    if family == "date":
        compact = cleaned.replace(" ", "")
        if re.match(r"^\d{2}[\/\-.]\d{2}[\/\-.]\d{4}$", compact):
            canonical = compact.replace("-", "/").replace(".", "/")
            score = 0.98
        elif re.match(r"^(NA|N/A)$", compact.upper()):
            canonical = compact.upper()
            score = 0.90
        else:
            issues.append("bad_date_shape")
            score = 0.20
        return ValidationResult(field_name, family, raw_text, cleaned, canonical, not issues, score, issues)

    if family == "amount":
        compact = cleaned.replace(" ", "")
        numeric = compact.replace(",", "")
        if re.match(r"^\d{1,12}$", numeric):
            canonical = numeric
            score = 0.97
        else:
            issues.append("bad_amount_shape")
            score = 0.20
        return ValidationResult(field_name, family, raw_text, cleaned, canonical, not issues, score, issues)

    if family == "numeric":
        compact = re.sub(r"[^\d]", "", cleaned)
        if compact:
            canonical = compact
            score = 0.96
        else:
            issues.append("bad_numeric_shape")
            score = 0.15
        return ValidationResult(field_name, family, raw_text, cleaned, canonical, not issues, score, issues)

    if family == "short_id":
        compact = _upper_alnum(cleaned)
        canonical = compact
        if field_name == "pan_number":
            if re.match(r"^[A-Z]{5}\d{4}[A-Z]$", compact):
                score = 0.99
            else:
                issues.append("bad_pan_shape")
                score = 0.15
        elif field_name in {"address_pincode"}:
            if re.match(r"^\d{6}$", compact):
                score = 0.98
            else:
                issues.append("bad_pincode_shape")
                score = 0.20
        elif field_name == "aadhaar_last_or_id_number":
            if re.match(r"^\d{12}$", compact):
                score = 0.98
            else:
                issues.append("bad_aadhaar_shape")
                score = 0.20
        elif field_name == "ckyc_number":
            if re.match(r"^[A-Z]{3,6}\d{5,9}$", compact):
                score = 0.94
            else:
                issues.append("weak_ckyc_shape")
                score = 0.55
        elif field_name == "customer_id":
            if re.match(r"^[A-Z]{3,6}\d{5,9}$", compact):
                score = 0.94
            else:
                issues.append("weak_customer_id_shape")
                score = 0.55
        else:
            if re.match(r"^[A-Z0-9]{3,20}$", compact):
                score = 0.90
            else:
                issues.append("bad_short_id_shape")
                score = 0.25
        return ValidationResult(field_name, family, raw_text, cleaned, canonical, len(issues) == 0, score, issues)

    if family in {"name_text", "short_text", "long_text"}:
        compact = _collapse_spaces(cleaned.replace("|", " ").replace("_", " "))
        canonical = compact
        if len(compact) < 2:
            issues.append("too_short")
            score = 0.15
        else:
            if family == "name_text":
                if not re.match(r"^[A-Za-z .'-/]+$", compact):
                    issues.append("non_name_chars")
                    score = 0.55
                else:
                    score = 0.90
            elif family == "short_text":
                if len(compact) > 40:
                    issues.append("too_long_for_short_text")
                    score = 0.45
                else:
                    score = 0.88
            else:
                if len(compact) > 120:
                    issues.append("too_long_for_long_text")
                    score = 0.40
                else:
                    score = 0.86
        return ValidationResult(field_name, family, raw_text, cleaned, canonical, len(issues) == 0, score, issues)

    return ValidationResult(field_name, family, raw_text, cleaned, canonical, True, 0.70, issues)


def compare_normalized_labels(a: str, b: str) -> bool:
    if a is None or b is None:
        return False
    return _collapse_spaces(str(a)).upper() == _collapse_spaces(str(b)).upper()
