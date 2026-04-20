"""
Layer 4: Validation Engine + Knowledge Base Fuzzy Matching
==========================================================
The brain that catches errors, corrects OCR garble, and grounds
extractions against a comprehensive Indian data taxonomy.

Components:
  - Regex validators (PAN, Aadhaar, PIN, Mobile, IFSC, Date, Email)
  - Verhoeff algorithm for Aadhaar checksum
  - Indian data taxonomy (states, cities, PIN ranges, LIC plan names)
  - Fuzzy matching encyclopedia (OCR confusions: Nagpuo→Nagpur, Maharashts→Maharashtra)
  - Cross-field logic (DOB↔Age, Sum Assured↔Premium)
  - Hallucination detection (gibberish, repetitive, known TrOCR artifacts)
"""

import re
import logging
from datetime import datetime
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

from src.pipeline.models.schemas import ExtractedField, ValidationStatus, ReviewCategory
from src.verhoeff_validator import verhoeff_check

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# INDIAN DATA TAXONOMY (comprehensive)
# ──────────────────────────────────────────────────────────────────

INDIAN_STATES = {
    "andhra pradesh", "arunachal pradesh", "assam", "bihar",
    "chhattisgarh", "goa", "gujarat", "haryana", "himachal pradesh",
    "jharkhand", "karnataka", "kerala", "madhya pradesh", "maharashtra",
    "manipur", "meghalaya", "mizoram", "nagaland", "odisha",
    "punjab", "rajasthan", "sikkim", "tamil nadu", "telangana",
    "tripura", "uttar pradesh", "uttarakhand", "west bengal",
    "delhi", "chandigarh", "puducherry", "jammu and kashmir", "ladakh",
}

INDIAN_CITIES = {
    "mumbai", "bombay", "delhi", "new delhi", "kolkata", "calcutta", "chennai", "madras",
    "bangalore", "bengaluru", "hyderabad", "pune", "poona", "ahmedabad", "jaipur",
    "lucknow", "chandigarh", "bhopal", "patna", "nagpur", "indore", "thane",
    "kochi", "cochin", "surat", "bhubaneswar", "dehradun", "agra", "varanasi",
    "kanpur", "nashik", "faridabad", "ghaziabad", "vadodara", "rajkot",
    "gurgaon", "gurugram", "noida", "howrah", "ranchi", "coimbatore",
    "vishakhapatnam", "visakhapatnam", "madurai", "ludhiana", "amritsar",
    "jamshedpur", "allahabad", "prayagraj", "aurangabad", "solapur",
    "meerut", "jodhpur", "raipur", "guwahati", "trivandrum", "thiruvananthapuram",
    "bhiwandi", "saharanpur", "gorakhpur", "gwalior", "jabalpur", "dhanbad",
    "borivali", "boriwali", "virar", "kalyan", "dombivli", "navi mumbai",
    "navimumbai", "thane", "kalyan-dombivli", "ulhasnagar", "bhayandar",
}

PIN_PREFIX_STATE = {
    "11": "delhi", "12": "haryana", "13": "haryana", "14": "punjab",
    "15": "punjab", "16": "chandigarh", "17": "himachal pradesh",
    "18": "maharashtra", "19": "maharashtra", "20": "uttar pradesh",
    "21": "uttar pradesh", "22": "uttar pradesh", "23": "uttar pradesh",
    "24": "uttar pradesh", "25": "uttar pradesh", "26": "uttar pradesh",
    "27": "uttar pradesh", "28": "uttar pradesh",
    "30": "rajasthan", "31": "rajasthan", "32": "rajasthan", "33": "rajasthan", "34": "rajasthan",
    "36": "gujarat", "37": "gujarat", "38": "gujarat", "39": "gujarat",
    "40": "maharashtra", "41": "maharashtra", "42": "maharashtra", "43": "maharashtra", "44": "maharashtra",
    "45": "madhya pradesh", "46": "madhya pradesh", "47": "madhya pradesh", "48": "madhya pradesh", "49": "madhya pradesh",
    "50": "telangana", "51": "telangana", "52": "telangana", "53": "telangana",
    "56": "karnataka", "57": "karnataka", "58": "karnataka", "59": "karnataka",
    "60": "tamil nadu", "61": "tamil nadu", "62": "tamil nadu", "63": "tamil nadu", "64": "tamil nadu",
    "67": "kerala", "68": "kerala", "69": "kerala",
    "70": "west bengal", "71": "west bengal", "72": "west bengal", "73": "west bengal", "74": "west bengal",
    "75": "odisha", "76": "odisha", "77": "odisha",
    "78": "assam", "79": "assam",
    "80": "bihar", "81": "bihar", "82": "bihar", "83": "bihar", "84": "bihar", "85": "bihar",
    "90": "andhra pradesh", "91": "andhra pradesh", "92": "andhra pradesh", "93": "andhra pradesh",
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

GENDER_VALUES = {"male", "female", "transgender", "other", "m", "f", "o"}
MARITAL_VALUES = {"single", "married", "divorced", "widowed", "unmarried"}
PREMIUM_MODES = {"yearly", "half-yearly", "quarterly", "monthly", "y", "h", "q", "m"}
RELATIONSHIPS = {
    "father", "mother", "spouse", "husband", "wife", "son", "daughter",
    "brother", "sister", "grandfather", "grandmother", "uncle", "aunt",
}

# ──────────────────────────────────────────────────────────────────
# OCR CONFUSION MAPS (comprehensive fuzzy encyclopedia)
# ──────────────────────────────────────────────────────────────────

OCR_CONFUSIONS = {
    '0': 'O', 'O': '0', '1': 'I', 'I': '1', 'l': '1',
    '5': 'S', 'S': '5', '8': 'B', 'B': '8', '2': 'Z',
    '6': 'G', 'G': '6', 'o': '0', 'i': '1', 'z': '2',
}

CITY_FUZZY_MAP = {
    "nagpuo": "Nagpur", "nagpuo": "Nagpur", "nagpru": "Nagpur",
    "mumbai": "Mumbai", "munbai": "Mumbai", "bombay": "Mumbai",
    "pune": "Pune", "poona": "Pune",
    "boriwali": "Borivali", "borivale": "Borivali", "borivali": "Borivali",
    "bengaluru": "Bengaluru", "bangalore": "Bengaluru",
    "chennai": "Chennai", "madras": "Chennai",
    "kolkata": "Kolkata", "calcutta": "Kolkata",
    "thiruvananthapuram": "Thiruvananthapuram", "trivandrum": "Thiruvananthapuram",
    "prayagraj": "Prayagraj", "allahabad": "Prayagraj",
}

STATE_FUZZY_MAP = {
    "maharashts": "Maharashtra", "maharash": "Maharashtra", "maharastra": "Maharashtra",
    "mah": "Maharashtra", "maha": "Maharashtra",
    "gujrat": "Gujarat", "guj": "Gujarat",
    "rajsthan": "Rajasthan", "raj": "Rajasthan",
    "karnatka": "Karnataka", "karnatka": "Karnataka",
    "tamilnadu": "Tamil Nadu", "tamilnad": "Tamil Nadu",
    "westbengal": "West Bengal", "w bengal": "West Bengal",
    "up": "Uttar Pradesh", "uttarpradesh": "Uttar Pradesh",
    "mp": "Madhya Pradesh", "madhyapratnesh": "Madhya Pradesh",
    "chattisgarh": "Chhattisgarh", "chattisgadh": "Chhattisgarh",
}

# ──────────────────────────────────────────────────────────────────
# VALIDATION RULES (regex patterns)
# ──────────────────────────────────────────────────────────────────

REGEX_RULES = {
    "Proposer_PAN": (r"^[A-Z]{5}\d{4}[A-Z]$", "5 letters + 4 digits + 1 letter"),
    "Proposer_Aadhaar": (r"^\d{12}$", "exactly 12 digits"),
    "Proposer_Pincode": (r"^\d{6}$", "exactly 6 digits"),
    "Proposer_Mobile_Number": (r"^[6-9]\d{9}$", "10 digits starting 6-9"),
    "Bank_IFSC": (r"^[A-Z]{4}0[A-Z0-9]{6}$", "4 letters + 0 + 6 alphanumeric"),
    "Proposer_Date_of_Birth": (r"^\d{2}[/\-\.]\d{2}[/\-\.]\d{4}$", "DD/MM/YYYY"),
    "LA_Date_of_Birth": (r"^\d{2}[/\-\.]\d{2}[/\-\.]\d{4}$", "DD/MM/YYYY"),
    "Date_of_Proposal": (r"^\d{2}[/\-\.]\d{2}[/\-\.]\d{4}$", "DD/MM/YYYY"),
    "Bank_Account_Number": (r"^\d{8,18}$", "8-18 digits"),
    "Proposer_Email": (r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$", "valid email"),
}

# ──────────────────────────────────────────────────────────────────
# HALLUCINATION DETECTION
# ──────────────────────────────────────────────────────────────────

GIBBERISH_PATTERNS = [
    re.compile(r"^[^a-zA-Z0-9\s]+$"),
    re.compile(r"(.)\1{4,}"),
    re.compile(r"^[bcdfghjklmnpqrstvwxyz]{5,}$", re.IGNORECASE),
    re.compile(r"^[aeiou]{5,}$", re.IGNORECASE),
]

KNOWN_HALLUCINATIONS = {
    "avelandian", "linsularate", "thelasthe", "india", "lhe", "lhe_",
    "tha", "dhe", "fhe", "tand", "tion", "tion_", "ing", "ing_",
}


class ValidationKB:

    def __init__(self):
        self.fuzzy_city_map = {k.lower(): v for k, v in CITY_FUZZY_MAP.items()}
        self.fuzzy_state_map = {k.lower(): v for k, v in STATE_FUZZY_MAP.items()}

    def validate_field(self, field: ExtractedField) -> ExtractedField:
        if not field.value or not field.value.strip():
            field.validation_status = ValidationStatus.MISSING.value
            field.review_category = ReviewCategory.MISSING.value
            return field

        value = field.value.strip()

        # 1. Hallucination check
        if self._is_hallucination(value):
            field.hallucination_flag = True
            field.hallucination_reason = "Gibberish or known hallucination pattern"
            field.validation_status = ValidationStatus.HALLUCINATION.value
            field.review_category = ReviewCategory.REJECTED.value
            field.confidence = 0.0
            return field

        # 2. KB fuzzy correction (before regex — correct garble first)
        corrected, reason = self._kb_fuzzy_correct(field.field_name, value)
        if corrected != value:
            field.kb_corrected = True
            field.kb_original_value = value
            field.kb_correction_reason = reason
            field.value = corrected
            value = corrected
            field.validation_status = ValidationStatus.CORRECTED.value

        # 3. Regex validation
        if field.field_name in REGEX_RULES:
            pattern, desc = REGEX_RULES[field.field_name]
            clean = self._clean_for_regex(field.field_name, value)
            if not re.match(pattern, clean):
                # Try OCR confusion correction
                ocr_fixed = self._ocr_confusion_correct(field.field_name, clean)
                if ocr_fixed and re.match(pattern, ocr_fixed):
                    field.kb_corrected = True
                    if not field.kb_original_value:
                        field.kb_original_value = value
                    field.value = ocr_fixed
                    field.kb_correction_reason = f"OCR confusion corrected: {value} → {ocr_fixed}"
                    field.validation_status = ValidationStatus.CORRECTED.value
                    value = ocr_fixed
                else:
                    field.validation_status = ValidationStatus.INVALID.value
                    field.cross_field_issues.append(f"Regex fail: {desc}")
                    field.needs_human_review = True

        # 4. Aadhaar Verhoeff check
        if "Aadhaar" in field.field_name and value:
            digits = re.sub(r'\D', '', value)
            if len(digits) == 12:
                if not verhoeff_check(digits):
                    field.validation_status = ValidationStatus.INVALID.value
                    field.cross_field_issues.append("Aadhaar Verhoeff checksum failed")
                    field.needs_human_review = True

        # 5. PIN-State cross-check
        if "Pincode" in field.field_name and value:
            digits = re.sub(r'\D', '', value)
            if len(digits) == 6 and digits[:2] in PIN_PREFIX_STATE:
                pin_state = PIN_PREFIX_STATE[digits[:2]]
                # Will be validated against Proposer_State in cross_field_validate

        # 6. Date validation
        if any(k in field.field_name for k in ("Date_of_Birth", "Date_of_Proposal", "DOB")):
            self._validate_date_field(field, value)

        # 7. Age validation
        if "Age" in field.field_name and value:
            self._validate_age_field(field, value)

        if field.validation_status not in (ValidationStatus.INVALID.value, ValidationStatus.HALLUCINATION.value):
            if field.validation_status != ValidationStatus.CORRECTED.value:
                field.validation_status = ValidationStatus.VALID.value

        return field

    def cross_field_validate(self, fields: List[ExtractedField]) -> List[ExtractedField]:
        field_map = {f.field_name: f for f in fields}

        # DOB ↔ Age cross-check
        dob_f = field_map.get("Proposer_Date_of_Birth")
        age_f = field_map.get("Proposer_Age")
        if dob_f and dob_f.value and age_f and age_f.value:
            try:
                dob_str = re.sub(r'[^\d/.\-]', '', dob_f.value)
                for fmt in ["%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y"]:
                    try:
                        dob = datetime.strptime(dob_str, fmt)
                        computed_age = int((datetime.now() - dob).days / 365.25)
                        stated_age = int(re.sub(r'\D', '', age_f.value))
                        if abs(computed_age - stated_age) > 1:
                            age_f.cross_field_issues.append(
                                f"DOB gives age {computed_age}, stated {stated_age}")
                            dob_f.cross_field_issues.append(
                                f"DOB gives age {computed_age}, stated {stated_age}")
                            age_f.needs_human_review = True
                            dob_f.needs_human_review = True
                        break
                    except ValueError:
                        continue
            except Exception:
                pass

        # PIN ↔ State cross-check
        pin_f = field_map.get("Proposer_Pincode")
        state_f = field_map.get("Proposer_State")
        if pin_f and pin_f.value and state_f and state_f.value:
            digits = re.sub(r'\D', '', pin_f.value)
            if len(digits) == 6 and digits[:2] in PIN_PREFIX_STATE:
                pin_state = PIN_PREFIX_STATE[digits[:2]]
                stated_state = state_f.value.lower().strip()
                if pin_state not in stated_state and stated_state not in pin_state:
                    pin_f.cross_field_issues.append(
                        f"PIN {digits} suggests {pin_state.title()}, but state is {state_f.value}")
                    state_f.cross_field_issues.append(
                        f"PIN {digits} suggests {pin_state.title()}, but state is {state_f.value}")
                    pin_f.needs_human_review = True
                    state_f.needs_human_review = True

        # Sum Assured > Premium check
        sa_f = field_map.get("Sum_Assured")
        prem_f = field_map.get("Premium_Amount")
        if sa_f and sa_f.value and prem_f and prem_f.value:
            try:
                sa = float(re.sub(r'[^\d.]', '', sa_f.value))
                prem = float(re.sub(r'[^\d.]', '', prem_f.value))
                if prem > sa:
                    sa_f.cross_field_issues.append("Premium exceeds sum assured — likely swapped")
                    prem_f.cross_field_issues.append("Premium exceeds sum assured — likely swapped")
                    sa_f.needs_human_review = True
                    prem_f.needs_human_review = True
            except (ValueError, ZeroDivisionError):
                pass

        return fields

    def _kb_fuzzy_correct(self, field_name: str, value: str) -> Tuple[str, str]:
        val_lower = value.lower().strip()

        if "City" in field_name:
            if val_lower in self.fuzzy_city_map:
                return self.fuzzy_city_map[val_lower], f"City fuzzy match: {value} → {self.fuzzy_city_map[val_lower]}"
            for garbled, correct in self.fuzzy_city_map.items():
                if SequenceMatcher(None, val_lower, garbled).ratio() > 0.85:
                    return correct, f"City fuzzy match (similarity): {value} → {correct}"

        if "State" in field_name:
            if val_lower in self.fuzzy_state_map:
                return self.fuzzy_state_map[val_lower], f"State fuzzy match: {value} → {self.fuzzy_state_map[val_lower]}"
            if val_lower in INDIAN_STATES:
                return value.title(), "State canonical match"
            for garbled, correct in self.fuzzy_state_map.items():
                if SequenceMatcher(None, val_lower, garbled).ratio() > 0.85:
                    return correct, f"State fuzzy match (similarity): {value} → {correct}"

        if "Gender" in field_name:
            for valid in GENDER_VALUES:
                if valid in val_lower:
                    return valid.title(), f"Gender normalized: {value} → {valid.title()}"

        if "Marital" in field_name:
            for valid in MARITAL_VALUES:
                if valid in val_lower:
                    return valid.title(), f"Marital status normalized: {value} → {valid.title()}"

        if "Premium_Mode" in field_name:
            for valid in PREMIUM_MODES:
                if valid in val_lower:
                    return valid.title(), f"Premium mode normalized: {value} → {valid.title()}"

        if "Relationship" in field_name or "Nominee_Relationship" in field_name:
            for valid in RELATIONSHIPS:
                if valid in val_lower:
                    return valid.title(), f"Relationship normalized: {value} → {valid.title()}"

        if "Plan_Name" in field_name:
            for valid in LIC_PLAN_NAMES:
                if valid in val_lower:
                    return valid.title(), f"LIC plan recognized: {value} → {valid.title()}"

        return value, ""

    def _ocr_confusion_correct(self, field_name: str, value: str) -> Optional[str]:
        if "PAN" in field_name:
            return self._correct_pan(value)
        elif "IFSC" in field_name:
            return self._correct_ifsc(value)
        elif "Mobile" in field_name:
            return self._correct_mobile(value)
        elif "Pincode" in field_name:
            digits = re.sub(r'\D', '', value)
            return digits[:6] if len(digits) >= 6 else None
        elif "Aadhaar" in field_name:
            digits = re.sub(r'\D', '', value)
            return digits[:12] if len(digits) >= 12 else None
        return None

    def _correct_pan(self, value: str) -> Optional[str]:
        if len(value) != 10:
            return None
        corrected = list(value.upper())
        letter_positions = {0, 1, 2, 3, 4, 9}
        digit_positions = {5, 6, 7, 8}
        for i in range(10):
            if i in letter_positions and corrected[i].isdigit():
                inv = {v: k for k, v in OCR_CONFUSIONS.items()}
                if corrected[i] in inv:
                    corrected[i] = inv[corrected[i]]
            elif i in digit_positions and corrected[i].isalpha():
                if corrected[i] in OCR_CONFUSIONS:
                    corrected[i] = OCR_CONFUSIONS[corrected[i]]
        return "".join(corrected)

    def _correct_ifsc(self, value: str) -> Optional[str]:
        if len(value) != 11:
            return None
        corrected = list(value.upper())
        corrected[4] = "0"
        return "".join(corrected)

    def _correct_mobile(self, value: str) -> Optional[str]:
        digits = re.sub(r'\D', '', value)
        if len(digits) == 12 and digits.startswith("91"):
            return digits[2:]
        elif len(digits) == 11 and digits.startswith("0"):
            return digits[1:]
        return digits if len(digits) == 10 else None

    def _clean_for_regex(self, field_name: str, value: str) -> str:
        if "PAN" in field_name or "IFSC" in field_name:
            return value.strip().upper().replace(" ", "")
        if "Aadhaar" in field_name or "Pincode" in field_name or "Mobile" in field_name:
            return re.sub(r'\D', '', value)
        return value.strip()

    def _is_hallucination(self, text: str) -> bool:
        if not text or len(text) < 2:
            return True
        for pat in GIBBERISH_PATTERNS:
            if pat.match(text):
                return True
        if text.lower() in KNOWN_HALLUCINATIONS:
            return True
        return False

    def _validate_date_field(self, field: ExtractedField, value: str):
        clean = re.sub(r'[^\d/.\-]', '', value)
        for fmt in ["%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y"]:
            try:
                parsed = datetime.strptime(clean, fmt)
                if parsed.year < 1900 or parsed.year > 2030:
                    field.cross_field_issues.append(f"Year {parsed.year} out of range")
                    field.needs_human_review = True
                if "Birth" in field.field_name or "DOB" in field.field_name:
                    age = (datetime.now() - parsed).days / 365.25
                    if age < 0 or age > 120:
                        field.cross_field_issues.append(f"Age from DOB: {age:.0f} — unreasonable")
                        field.needs_human_review = True
                return
            except ValueError:
                continue
        field.cross_field_issues.append(f"Cannot parse date: {value}")
        field.needs_human_review = True

    def _validate_age_field(self, field: ExtractedField, value: str):
        try:
            age = int(re.sub(r'\D', '', str(value)))
            if "Proposer" in field.field_name and not (18 <= age <= 100):
                field.cross_field_issues.append(f"Proposer age {age} out of range 18-100")
                field.needs_human_review = True
        except (ValueError, TypeError):
            field.cross_field_issues.append(f"Cannot parse age: {value}")
            field.needs_human_review = True
