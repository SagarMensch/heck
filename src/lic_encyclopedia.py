"""
LIC Form 300 Encyclopedia & Fuzzy Logic
Corrects OCR errors using known dictionaries, phonetics, and fuzzy matching.
"""
import re
from difflib import SequenceMatcher
from typing import List, Dict, Optional, Tuple

# --- 1. KNOWLEDGE BASE (The "Encyclopedia") ---
GENDER_CHOICES = ['MALE', 'FEMALE', 'TRANSGENDER', 'OTHER']
MARITAL_CHOICES = ['MARRIED', 'UNMARRIED', 'DIVORCED', 'WIDOWED', 'SINGLE']
OCCUPATION_MAP = {
    'SALESPERSON': 'SALES', 'SALESMAN': 'SALES', 'SASE': 'SALES', 'BUSINESS': 'BUSINESS',
    'SERVICE': 'SERVICE', 'PROFESSIONAL': 'PROFESSIONAL', 'EMPLOYEE': 'EMPLOYEE',
    'HOUSEWIFE': 'HOUSEWIFE', 'STUDENT': 'STUDENT', 'RETIRED': 'RETIRED'
}
CITY_MAP = {
    'MUMBAI': 'MUMBAI', 'BOMBAY': 'MUMBAI', 'MUNBAI': 'MUMBAI', 'MUNBAI': 'MUMBAI',
    'NAGPUR': 'NAGPUR', 'NAGPUO': 'NAGPUR', 'NAGPUR': 'NAGPUR',
    'PUNE': 'PUNE', 'POONA': 'PUNE', 'DELHI': 'DELHI', 'KOLKATA': 'KOLKATA',
    'CHENNAI': 'CHENNAI', 'MADRAS': 'CHENNAI', 'BANGALORE': 'BENGALURU', 'BENGALURU': 'BENGALURU'
}
STATE_MAP = {
    'MAHARASHTRA': 'MAHARASHTRA', 'MAHARASH': 'MAHARASHTRA', 'MAHARASHHS': 'MAHARASHTRA',
    'MAH': 'MAHARASHTRA', 'GUJARAT': 'GUJARAT', 'RAJASTHAN': 'RAJASTHAN',
    'TAMIL NADU': 'TAMIL NADU', 'KARNATAKA': 'KARNATAKA', 'WEST BENGAL': 'WEST BENGAL'
}

# Common OCR Confusions
OCR_CONFUSIONS = {
    '0': 'O', '1': 'I', '5': 'S', '8': 'B', '6': 'G',
    'l': '1', 'o': '0', 'i': '1', 'z': '2'
}

# --- 2. FUZZY MATCHING ALGORITHMS ---

def similarity_ratio(s1: str, s2: str) -> float:
    """Levenshtein-based similarity (0.0 to 1.0)"""
    return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()

def jaro_winkler_similarity(s1: str, s2: str) -> float:
    """Jaro-Winkler similarity (favors prefix matches)"""
    # Simple implementation for demo; use `jellyfish` lib for production
    if s1 == s2: return 1.0
    # Approximation logic would go here, but we'll use SequenceMatcher for now
    base = similarity_ratio(s1, s2)
    # Boost for common prefix
    prefix_len = 0
    for i in range(min(4, len(s1), len(s2))):
        if s1[i] == s2[i]: prefix_len += 1
        else: break
    return min(base + 0.1 * prefix_len, 1.0)

# --- 3. CLEANING & CORRECTION ENGINE ---

class LicCleaner:
    def __init__(self):
        self.gender_cache = None
        self.city_cache = None
    
    def clean_text(self, text: str) -> str:
        """Basic text cleaning"""
        if not text: return ""
        # Remove non-printable chars, normalize spaces
        text = re.sub(r'[^\w\s/\.(\)-]', '', text)
        return re.sub(r'\s+', ' ', text).strip().upper()
    
    def correct_gender(self, raw: str) -> Optional[str]:
        """Extract gender from garbled text"""
        if not raw: return None
        clean = self.clean_text(raw)
        for choice in GENDER_CHOICES:
            if choice in clean or similarity_ratio(clean, choice) > 0.8:
                return choice.capitalize()
        # Heuristics
        if 'MALE' in clean or 'M' in clean[:2]: return 'Male'
        if 'FEMALE' in clean or 'F' in clean[:2]: return 'Female'
        return None

    def correct_marital_status(self, raw: str) -> Optional[str]:
        """Extract marital status"""
        if not raw: return None
        clean = self.clean_text(raw)
        for choice in MARITAL_CHOICES:
            if choice in clean or similarity_ratio(clean, choice) > 0.8:
                return choice.capitalize()
        if 'MARR' in clean: return 'Married'
        if 'UNMARR' in clean: return 'Unmarried'
        return None

    def correct_city(self, raw: str) -> Optional[str]:
        if not raw: return None
        clean = self.clean_text(raw)
        # Exact match
        if clean in CITY_MAP: return CITY_MAP[clean].capitalize()
        # Fuzzy match
        best_match = None
        best_score = 0.8
        for key, val in CITY_MAP.items():
            score = jaro_winkler_similarity(clean, key)
            if score > best_score:
                best_score = score
                best_match = val
        return best_match.capitalize() if best_match else None

    def correct_state(self, raw: str) -> Optional[str]:
        if not raw: return None
        clean = self.clean_text(raw)
        if clean in STATE_MAP: return STATE_MAP[clean].capitalize()
        # Fuzzy
        best_match = None
        best_score = 0.7
        for key, val in STATE_MAP.items():
            if jaro_winkler_similarity(clean, key) > best_score:
                best_match = val
                best_score = jaro_winkler_similarity(clean, key)
        return best_match.capitalize() if best_match else None

    def correct_occupation(self, raw: str) -> Optional[str]:
        if not raw: return None
        clean = self.clean_text(raw)
        for key, val in OCCUPATION_MAP.items():
            if key in clean or similarity_ratio(clean, key) > 0.7:
                return val
        return clean.capitalize()

    def extract_pincode(self, raw: str) -> Optional[str]:
        """Extract 6-digit PIN from garbage"""
        if not raw: return None
        match = re.search(r'\d{6}', raw.replace(' ', '').replace('.', ''))
        if match:
            return match.group()
        # Try to find 6 consecutive digits
        digits = re.sub(r'\D', '', raw)
        if len(digits) >= 6:
            return digits[:6]
        return None

    def extract_pan(self, raw: str) -> Optional[str]:
        """Extract PAN (ABCDE1234F)"""
        if not raw: return None
        # Pattern: 3 letters, 1 letter (A/B/F/G/H/J/L/P/T), 1 letter, 4 digits, 1 letter
        # Simplified: [A-Z]{5}\d{4}[A-Z]
        clean = raw.upper().replace(' ', '')
        match = re.search(r'[A-Z]{5}\d{4}[A-Z]', clean)
        if match:
            return match.group()
        # Fuzzy: look for 5 chars + 4 digits + 1 char
        return None

    def normalize_date(self, raw: str) -> Optional[str]:
        """Normalize date to DD/MM/YYYY"""
        if not raw: return None
        # Find digits
        nums = re.findall(r'\d+', raw)
        if len(nums) >= 3:
            d, m, y = nums[0], nums[1], nums[2]
            if len(y) == 2: y = '20' + y if int(y) < 50 else '19' + y
            elif len(y) == 3: y = '200' + y # Handle OCR error like "2005" -> "2005"
            elif len(y) == 1: y = '200' + y
            return f"{d.zfill(2)}/{m.zfill(2)}/{y}"
        return None

# Global instance
cleaner = LicCleaner()
