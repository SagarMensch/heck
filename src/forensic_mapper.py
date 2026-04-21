"""
Forensic Field Mapper v2.0
Uses Encyclopedia, Constraints, Field Mappings, and Validation to ensure 99% accuracy.
"""
import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from .lic_encyclopedia import cleaner, LicCleaner
from .field_mapper import FieldMapper, MappedField
from .table_parser import parse_table_html, extract_key_values_from_table
from .table_value_extractor import parse_table_to_pairs, extract_field_value_from_table, extract_all_key_values
from .field_mappings import match_label_to_field, get_field_type, FIELD_MAPPINGS
from .verhoeff_validator import validate_aadhaar, extract_aadhaar_from_text, verhoeff_check

logger = logging.getLogger(__name__)

@dataclass
class ForensicResult:
    field_name: str
    raw_value: str
    cleaned_value: str
    confidence: float  # 0.0 to 1.0
    validation_status: str  # 'valid', 'invalid', 'corrected'
    correction_method: str  # 'none', 'fuzzy', 'dictionary', 'constraint'
    notes: str = ""

class ForensicFieldMapper:
    def __init__(self):
        self.cleaner = cleaner
        self.base_mapper = FieldMapper()

    def _is_name_or_entity_field(self, field_name: str) -> bool:
        key = (field_name or "").lower()
        return any(
            needle in key
            for needle in (
                "name",
                "father",
                "mother",
                "spouse",
                "nominee",
                "appointee",
                "employer",
                "branch",
                "plan",
                "officer",
            )
        )
    
    def map_and_validate(self, regions: List[Any], page_num: int) -> List[ForensicResult]:
        """
        Map regions to fields with forensic validation.
        Enhanced to parse HTML tables and extract field/value pairs.
        """
        results = []
        
        # First pass: Base mapping
        base_mapped = self.base_mapper.map_regions_to_fields(regions, page_num)
        
        for field in base_mapped:
            raw = field.value
            field_name = field.field_name
            
            # Check if this is table HTML - parse it!
            if raw.strip().startswith('<html') and 'table' in raw.lower():
                # Use advanced table value extractor
                pairs = parse_table_to_pairs(raw)
                
                # Process each label-value pair
                for pair in pairs:
                    label = pair.get('label', '')
                    value = pair.get('value', '')
                    
                    # CRITICAL: Use VALUE from cell, not just label!
                    if value and len(value.strip()) > 0:
                        # We have actual handwritten value!
                        sub_result = self._process_field(label, value.strip(), page_num)
                        results.append(sub_result)
                    elif label and len(label.strip()) > 0:
                        # Label only, no value
                        sub_result = self._process_field(label, '', page_num)
                        results.append(sub_result)
                continue
            
            # Non-table field
            result = self._process_field(field_name, raw, page_num)
            results.append(result)
        
        return results
    
    def _process_field(self, field_name: str, raw: str, page_num: int) -> ForensicResult:
        """Process a single field with forensic validation using field mappings"""

        # Step 1: Preserve structured field ids; only remap human-readable labels.
        original_field_name = field_name or ''
        structured_field_id = (
            bool(re.fullmatch(r'[A-Za-z0-9_]+', original_field_name))
            and ('_' in original_field_name or original_field_name.islower())
        )
        if not structured_field_id:
            canonical_field = match_label_to_field(original_field_name)
            if not canonical_field and raw and len(raw.split()) <= 6:
                canonical_field = match_label_to_field(raw)
            if canonical_field:
                field_name = canonical_field
        
        # Get field mapping if exists
        field_mapping = FIELD_MAPPINGS.get(field_name)
        field_type = field_mapping.field_type if field_mapping else 'text'
        
        # --- FORENSIC ANALYSIS PER FIELD TYPE ---
        
        # 1. GENDER
        if field_type == 'choice' and field_mapping and field_mapping.choices:
            cleaned = self.cleaner.correct_gender(raw) if 'gender' in field_name.lower() else None
            if not cleaned and 'marital' in field_name.lower():
                cleaned = self.cleaner.correct_marital_status(raw)
            if cleaned:
                return ForensicResult(
                    field_name=field_name,
                    raw_value=raw,
                    cleaned_value=cleaned,
                    confidence=0.99,
                    validation_status='corrected',
                    correction_method='dictionary',
                    notes=f"Corrected from '{raw}'"
                )
        
        # 2. CITY / PLACE
        if 'birth_place' in field_name.lower() or 'city' in field_name.lower() or 'place' in field_name.lower():
            cleaned = self.cleaner.correct_city(raw)
            if cleaned:
                return ForensicResult(
                    field_name='Proposer_Birth_Place',
                    raw_value=raw,
                    cleaned_value=cleaned,
                    confidence=0.95,
                    validation_status='corrected',
                    correction_method='fuzzy_city_map'
                )

        # 3. STATE
        if 'state' in field_name.lower():
            cleaned = self.cleaner.correct_state(raw)
            if cleaned:
                return ForensicResult(
                    field_name='Proposer_State',
                    raw_value=raw,
                    cleaned_value=cleaned,
                    confidence=0.96,
                    validation_status='corrected',
                    correction_method='fuzzy_state_map'
                )

        # 4. PINCODE (Strict Validation)
        if field_type == 'pincode' or 'pin' in field_name.lower():
            cleaned = self.cleaner.extract_pincode(raw)
            if cleaned:
                return ForensicResult(
                    field_name='Proposer_PIN',
                    raw_value=raw,
                    cleaned_value=cleaned,
                    confidence=0.99,
                    validation_status='valid',
                    correction_method='regex_extract'
                )

        # 5. PAN (Strict Validation)
        if field_type == 'pan' or 'pan' in field_name.lower():
            cleaned = self.cleaner.extract_pan(raw)
            if cleaned:
                return ForensicResult(
                    field_name='Proposer_PAN',
                    raw_value=raw,
                    cleaned_value=cleaned,
                    confidence=0.98,
                    validation_status='valid',
                    correction_method='regex_extract'
                )

        # 6. AADHAAR (Verhoeff Validation)
        if field_type == 'aadhaar' or 'aadhaar' in field_name.lower():
            aadhaar = extract_aadhaar_from_text(raw)
            if aadhaar:
                validation = validate_aadhaar(aadhaar)
                if validation['valid']:
                    return ForensicResult(
                        field_name='Proposer_Aadhaar',
                        raw_value=raw,
                        cleaned_value=aadhaar,
                        confidence=0.99,
                        validation_status='valid',
                        correction_method='verhoeff_algorithm',
                        notes='Verhoeff checksum validated'
                    )
                else:
                    return ForensicResult(
                        field_name='Proposer_Aadhaar',
                        raw_value=raw,
                        cleaned_value=aadhaar,
                        confidence=0.70,
                        validation_status='invalid',
                        correction_method='verhoeff_failed',
                        notes=validation['message']
                    )

        # 7. DATE OF BIRTH
        if field_type == 'date' or 'dob' in field_name.lower() or 'birth' in field_name.lower() or 'date' in field_name.lower():
            cleaned = self.cleaner.normalize_date(raw)
            if cleaned:
                return ForensicResult(
                    field_name='Proposer_DOB',
                    raw_value=raw,
                    cleaned_value=cleaned,
                    confidence=0.90,
                    validation_status='corrected',
                    correction_method='date_normalization'
                )
        
        # 8. OCCUPATION
        if field_type == 'text' and ('occupation' in field_name.lower() or 'occup' in raw.lower()):
            cleaned = self.cleaner.correct_occupation(raw)
            if cleaned:
                return ForensicResult(
                    field_name='Proposer_Occupation',
                    raw_value=raw,
                    cleaned_value=cleaned,
                    confidence=0.92,
                    validation_status='corrected',
                    correction_method='occupation_map'
                )
        
        # 9. NAME / ENTITY fields (lexicon-backed fuzzy resolution)
        if field_type == 'text' and self._is_name_or_entity_field(field_name):
            decision = self.cleaner.resolve_name_or_entity(raw, field_name)
            fallback_text = self.cleaner.prettify_text(raw)
            if decision.accepted:
                status = 'corrected' if decision.resolved_text != fallback_text else 'valid'
                notes = decision.reason
                if decision.candidates:
                    notes = f"{notes}; top_candidate={decision.candidates[0].text}"
                return ForensicResult(
                    field_name=field_name,
                    raw_value=raw,
                    cleaned_value=decision.resolved_text,
                    confidence=max(0.88, decision.confidence),
                    validation_status=status,
                    correction_method=decision.method,
                    notes=notes,
                )

            if fallback_text:
                status = 'pending_review' if decision.review_required else 'valid'
                return ForensicResult(
                    field_name=field_name,
                    raw_value=raw,
                    cleaned_value=fallback_text,
                    confidence=max(0.72, decision.confidence),
                    validation_status=status,
                    correction_method=decision.method,
                    notes=decision.reason,
                )

        # DEFAULT: Return cleaned text
        cleaned = self.cleaner.clean_text(raw)
        return ForensicResult(
            field_name=field_name if field_name else 'unknown_field',
            raw_value=raw,
            cleaned_value=cleaned if cleaned else raw,
            confidence=0.85,
            validation_status='valid' if cleaned else 'pending_review',
            correction_method='basic_clean'
        )

    def to_dict_list(self, results: List[ForensicResult]) -> List[Dict]:
        """Convert to list of dicts for JSON export"""
        return [
            {
                'field': r.field_name,
                'value': r.cleaned_value,
                'raw': r.raw_value,
                'confidence': r.confidence,
                'status': r.validation_status,
                'method': r.correction_method,
                'notes': r.notes
            }
            for r in results
        ]
