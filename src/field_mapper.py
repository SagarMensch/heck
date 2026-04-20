"""
Field Mapping Module - Maps extracted regions to canonical LIC Form 300 fields
Uses bbox overlap, semantic matching, and fuzzy logic
"""
import re
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
import logging

from .form300_templates import FORM300_PAGE_TEMPLATES as PAGE_TEMPLATES

# Field name mapping (reverse lookup)
FIELD_NAME_MAP = {
    'name': 'Proposer_Full_Name',
    'father': 'Proposer_Father_Name',
    'mother': 'Proposer_Mother_Name',
    'dob': 'Proposer_DOB',
    'age': 'Proposer_Age',
    'gender': 'Proposer_Gender',
    'marital': 'Proposer_Marital_Status',
    'spouse': 'Proposer_Spouse_Name',
    'birth_place': 'Proposer_Birth_Place',
    'nationality': 'Proposer_Nationality',
    'citizenship': 'Proposer_Citizenship',
    'address': 'Proposer_Permanent_Address',
    'pin': 'Proposer_PIN',
    'phone': 'Proposer_Phone',
    'email': 'Proposer_Email',
    'occupation': 'Proposer_Occupation',
    'income': 'Proposer_Income',
    'pan': 'Proposer_PAN',
    'aadhaar': 'Proposer_Aadhaar',
}

logger = logging.getLogger(__name__)


@dataclass
class MappedField:
    """A field mapped to canonical schema"""
    field_name: str
    value: str
    raw_text: str
    confidence: float
    source_bbox: List[float]
    page_num: int
    validation_status: str = "pending"
    validation_details: Dict = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


class FieldMapper:
    """Map extracted text regions to canonical LIC fields"""
    
    # Canonical field definitions with expected patterns
    FIELD_PATTERNS = {
        'Proposer_Full_Name': {'type': 'name', 'required': True},
        'Proposer_Father_Name': {'type': 'name'},
        'Proposer_Mother_Name': {'type': 'name'},
        'Proposer_DOB': {'type': 'date', 'pattern': r'\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}'},
        'Proposer_Age': {'type': 'number', 'pattern': r'\d{1,3}'},
        'Proposer_Gender': {'type': 'choice', 'choices': ['Male', 'Female', 'Transgender']},
        'Proposer_Marital_Status': {'type': 'choice', 'choices': ['Married', 'Unmarried', 'Divorced', 'Widowed']},
        'Proposer_Spouse_Name': {'type': 'name'},
        'Proposer_Birth_Place': {'type': 'text'},
        'Proposer_Nationality': {'type': 'text', 'default': 'Indian'},
        'Proposer_Citizenship': {'type': 'text', 'default': 'Indian'},
        'Proposer_Permanent_Address': {'type': 'address'},
        'Proposer_PIN': {'type': 'pincode', 'pattern': r'\d{6}'},
        'Proposer_Phone': {'type': 'phone', 'pattern': r'[\d\s\-+]{8,15}'},
        'Proposer_Email': {'type': 'email', 'pattern': r'[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}'},
        'Proposer_Occupation': {'type': 'text'},
        'Proposer_Income': {'type': 'money'},
        'Proposer_PAN': {'type': 'pan', 'pattern': r'[A-Z]{5}\d{4}[A-Z]'},
        'Proposer_Aadhaar': {'type': 'aadhaar', 'pattern': r'\d{12}'},
        'LA_Full_Name': {'type': 'name'},
        'LA_Father_Name': {'type': 'name'},
        'LA_DOB': {'type': 'date'},
        'LA_Age': {'type': 'number'},
        'LA_Gender': {'type': 'choice'},
        'Policy_No': {'type': 'alphanumeric'},
        'Sum_Assured': {'type': 'money'},
        'Premium_Amount': {'type': 'money'},
        'Premium_Mode': {'type': 'choice'},
        'Branch_Name': {'type': 'text'},
        'Agent_Code': {'type': 'alphanumeric'},
    }
    
    # Keyword mappings for semantic matching
    KEYWORD_MAP = {
        'name': ['name', 'full name', 'proposer name'],
        'father': ["father's name", 'father name', 's/o', 'son of'],
        'mother': ["mother's name", 'mother name', 'd/o'],
        'dob': ['date of birth', 'dob', 'birth date', 'born on'],
        'age': ['age', 'years old', 'yrs'],
        'gender': ['gender', 'sex', 'male', 'female'],
        'marital': ['marital status', 'married', 'unmarried'],
        'spouse': ['spouse name', 'husband name', 'wife name'],
        'birth_place': ['place of birth', 'born at', 'birth city'],
        'nationality': ['nationality', 'citizen of'],
        'address': ['address', 'residence', 'permanent address'],
        'pin': ['pin code', 'pincode', 'postal code', 'zip'],
        'phone': ['phone', 'mobile', 'telephone', 'tel no'],
        'email': ['email', 'e-mail', 'mail id'],
        'occupation': ['occupation', 'profession', 'job', 'business'],
        'income': ['income', 'annual income', 'salary'],
        'pan': ['pan', 'pan no', 'permanent account number'],
        'aadhaar': ['aadhaar', 'aadhar', 'uidai'],
        'policy': ['policy no', 'policy number', 'policy no.'],
        'sum_assured': ['sum assured', 'sum assured (', 'coverage'],
        'premium': ['premium', 'premium amount', 'rs.'],
        'branch': ['branch', 'branch name', 'branch office'],
        'agent': ['agent', 'agent code', 'agent no'],
    }
    
    def __init__(self):
        self.mapped_fields = []
    
    def normalize_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        # Remove extra whitespace, normalize unicode
        text = re.sub(r'\s+', ' ', text).strip()
        text = text.replace('l', '1').replace('O', '0').replace('o', '0')
        return text
    
    def fuzzy_match(self, s1: str, s2: str) -> float:
        """Fuzzy string matching score (0-1)"""
        if not s1 or not s2:
            return 0.0
        return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()
    
    def jaro_winkler(self, s1: str, s2: str) -> float:
        """Jaro-Winkler similarity (favors prefix matches)"""
        if not s1 or not s2:
            return 0.0
        
        # Simple implementation
        if s1 == s2:
            return 1.0
        
        # Jaro distance
        match_window = max(len(s1) // 2 - 1, 0)
        matches = 0
        s1_matches = [False] * len(s1)
        s2_matches = [False] * len(s2)
        
        for i in range(len(s1)):
            start = max(0, i - match_window)
            end = min(i + match_window + 1, len(s2))
            for j in range(start, end):
                if s1[i] == s2[j] and not s2_matches[j]:
                    s1_matches[i] = True
                    s2_matches[j] = True
                    matches += 1
                    break
        
        if matches == 0:
            return 0.0
        
        jaro = (matches / len(s1) + matches / len(s2) + matches / matches) / 3
        
        # Winkler modification
        prefix = 0
        for i in range(min(4, len(s1), len(s2))):
            if s1[i] == s2[i]:
                prefix += 1
            else:
                break
        
        jw = jaro + 0.1 * prefix * (1 - jaro)
        return min(jw, 1.0)
    
    def match_field_by_keywords(self, text: str, label: str) -> Optional[str]:
        """Match text to field using keyword heuristics"""
        text_lower = text.lower()
        label_lower = label.lower()
        
        best_match = None
        best_score = 0.5  # Threshold
        
        for field_key, keywords in self.KEYWORD_MAP.items():
            for keyword in keywords:
                # Exact keyword match in label
                if keyword in label_lower:
                    score = 0.9
                    if score > best_score:
                        best_score = score
                        best_match = field_key
                    continue
                
                # Fuzzy match
                score = self.jaro_winkler(keyword, label_lower)
                if score > 0.85:
                    if score > best_score:
                        best_score = score
                        best_match = field_key
        
        return best_match
    
    def extract_value(self, text: str, field_name: str) -> str:
        """Extract clean value from text"""
        # Remove common noise patterns
        text = re.sub(r'^[:\-\*\.]\s*', '', text)  # Leading punctuation
        text = re.sub(r'\s*[:\-\*\.]$', '', text)  # Trailing punctuation
        text = text.strip()
        
        # Field-specific cleaning
        if 'name' in field_name.lower():
            # Name: keep alphabetic, spaces, dots
            text = re.sub(r'[^\w\s\.]', '', text)
        elif 'date' in field_name.lower() or 'dob' in field_name.lower():
            # Date: extract digits and separators
            match = re.search(r'[\d\-\.\/]+', text)
            if match:
                text = match.group()
        elif 'pin' in field_name.lower():
            # PIN: extract 6 digits
            match = re.search(r'\d{6}', text)
            if match:
                text = match.group()
        elif 'aadhaar' in field_name.lower():
            # Aadhaar: extract 12 digits
            match = re.search(r'\d{12}', text)
            if match:
                text = match.group()
        elif 'pan' in field_name.lower():
            # PAN: extract ABCDE1234F pattern
            match = re.search(r'[A-Z]{5}\d{4}[A-Z]', text.upper())
            if match:
                text = match.group()
        
        return text.strip()
    
    def map_regions_to_fields(self, regions: List[Any], page_num: int) -> List[MappedField]:
        """
        Map extracted regions to canonical fields
        
        Args:
            regions: List of TextRegion objects from layout extractor
            page_num: Page number
        
        Returns:
            List of MappedField objects
        """
        mapped = []
        
        for region in regions:
            # Try to match to known field
            field_name = self.match_field_by_keywords(region.text, region.label)
            
            if not field_name:
                # Use generic field name
                field_name = f"unknown_{region.label}_{region.page_num}"
            
            # Extract and clean value
            value = self.extract_value(region.text, field_name)
            
            if value:  # Only map if we have a value
                mapped.append(MappedField(
                    field_name=field_name,
                    value=value,
                    raw_text=region.text,
                    confidence=region.confidence,
                    source_bbox=region.bbox,
                    page_num=page_num
                ))
        
        return mapped
    
    def validate_field(self, field: MappedField) -> Tuple[str, Dict]:
        """
        Validate field value based on type
        
        Returns:
            (status, details)
        """
        field_def = self.FIELD_PATTERNS.get(field.field_name, {})
        field_type = field_def.get('type', 'text')
        pattern = field_def.get('pattern')
        
        details = {'field_type': field_type}
        
        # Pattern matching
        if pattern:
            if re.search(pattern, field.value):
                return 'valid', details
            else:
                details['error'] = f'Pattern mismatch: expected {pattern}'
                return 'invalid', details
        
        # Type-specific validation
        if field_type == 'date':
            if re.match(r'\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}', field.value):
                return 'valid', details
            details['error'] = 'Invalid date format'
            return 'invalid', details
        
        if field_type == 'pincode':
            if re.match(r'\d{6}', field.value):
                return 'valid', details
            details['error'] = 'PIN should be 6 digits'
            return 'invalid', details
        
        if field_type == 'aadhaar':
            if re.match(r'\d{12}', field.value):
                # Verhoeff check
                if self.verhoeff_check(field.value):
                    return 'valid', details
                details['error'] = 'Verhoeff check failed'
                return 'invalid', details
            details['error'] = 'Aadhaar should be 12 digits'
            return 'invalid', details
        
        if field_type == 'pan':
            if re.match(r'[A-Z]{5}\d{4}[A-Z]', field.value.upper()):
                return 'valid', details
            details['error'] = 'Invalid PAN format'
            return 'invalid', details
        
        return 'valid', details
    
    def verhoeff_check(self, aadhaar: str) -> bool:
        """
        Verhoeff algorithm for Aadhaar validation
        Returns True if checksum is valid
        """
        if len(aadhaar) != 12 or not aadhaar.isdigit():
            return False
        
        # Verhoeff tables
        d = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
            [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
            [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
            [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
            [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
            [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
            [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
            [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
            [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        ]
        
        p = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 7, 2, 5, 3, 8, 9, 0, 4, 6],
            [2, 4, 1, 9, 7, 6, 3, 8, 5, 0],
            [3, 8, 5, 0, 1, 2, 4, 9, 6, 7],
            [4, 3, 9, 2, 6, 0, 7, 1, 5, 8],
            [5, 0, 6, 8, 4, 1, 3, 7, 2, 9],
            [6, 9, 3, 1, 8, 5, 2, 4, 0, 7],
            [7, 2, 8, 6, 9, 4, 0, 5, 1, 3],
            [8, 5, 4, 3, 0, 9, 1, 6, 7, 2],
            [9, 6, 7, 4, 2, 8, 5, 3, 0, 1]
        ]
        
        c = 0
        for i, digit in enumerate(reversed(aadhaar)):
            c = d[c][p[(i % 8)][int(digit)]]
        
        return c == 0


def map_fields(regions: List, page_num: int) -> List[MappedField]:
    """Convenience function to map regions to fields"""
    mapper = FieldMapper()
    return mapper.map_regions_to_fields(regions, page_num)
