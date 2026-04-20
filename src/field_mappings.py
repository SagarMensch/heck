"""
LIC Form 300 Field Mappings
Maps table labels to canonical field names with validation rules
"""
import re
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass

@dataclass
class FieldMapping:
    canonical_name: str
    label_patterns: List[str]  # Regex patterns to match labels
    field_type: str  # text, date, number, choice, pan, aadhaar, pincode
    validation_fn: Optional[Callable] = None
    choices: Optional[List[str]] = None  # For choice fields
    required: bool = False
    default: Optional[str] = None


# Complete field mappings for LIC Form 300
FIELD_MAPPINGS: Dict[str, FieldMapping] = {
    # === PROPOSER DETAILS ===
    'Proposer_Full_Name': FieldMapping(
        canonical_name='Proposer_Full_Name',
        label_patterns=[r'name', r'full name', r'proposer name', r'life to be assured'],
        field_type='text',
        required=True
    ),
    'Proposer_Father_Name': FieldMapping(
        canonical_name='Proposer_Father_Name',
        label_patterns=[r"father", r"father's name", r'mother name', r"mother's name", r's/o', r'd/o'],
        field_type='text',
        required=True
    ),
    'Proposer_DOB': FieldMapping(
        canonical_name='Proposer_DOB',
        label_patterns=[r'date of birth', r'dob', r'birth date', r'born'],
        field_type='date',
        required=True
    ),
    'Proposer_Age': FieldMapping(
        canonical_name='Proposer_Age',
        label_patterns=[r'\bage\b', r'years', r'age last', r'age nearer'],
        field_type='number',
        required=True
    ),
    'Proposer_Gender': FieldMapping(
        canonical_name='Proposer_Gender',
        label_patterns=[r'gender', r'sex', r'male', r'female', r'transgender'],
        field_type='choice',
        choices=['Male', 'Female', 'Transgender'],
        required=True
    ),
    'Proposer_Marital_Status': FieldMapping(
        canonical_name='Proposer_Marital_Status',
        label_patterns=[r'marital', r'married', r'unmarried', r'spouse'],
        field_type='choice',
        choices=['Married', 'Unmarried', 'Divorced', 'Widowed'],
        required=True
    ),
    'Proposer_Spouse_Name': FieldMapping(
        canonical_name='Proposer_Spouse_Name',
        label_patterns=[r'spouse', r'husband', r'wife', r'married'],
        field_type='text',
        required=False
    ),
    'Proposer_Birth_Place': FieldMapping(
        canonical_name='Proposer_Birth_Place',
        label_patterns=[r'place of birth', r'birth place', r'born at', r'birth city'],
        field_type='text',
        required=True
    ),
    'Proposer_Nationality': FieldMapping(
        canonical_name='Proposer_Nationality',
        label_patterns=[r'nationality'],
        field_type='text',
        default='Indian'
    ),
    'Proposer_Citizenship': FieldMapping(
        canonical_name='Proposer_Citizenship',
        label_patterns=[r'citizenship'],
        field_type='text',
        default='Indian'
    ),
    'Proposer_Permanent_Address': FieldMapping(
        canonical_name='Proposer_Permanent_Address',
        label_patterns=[r'permanent address', r'address', r'residence', r'correspondence address'],
        field_type='address',
        required=True
    ),
    'Proposer_PIN': FieldMapping(
        canonical_name='Proposer_PIN',
        label_patterns=[r'pin', r'pin code', r'postal code', r'zip'],
        field_type='pincode',
        required=True
    ),
    'Proposer_Phone': FieldMapping(
        canonical_name='Proposer_Phone',
        label_patterns=[r'phone', r'mobile', r'tel', r'telephone', r'std'],
        field_type='phone',
        required=False
    ),
    'Proposer_Email': FieldMapping(
        canonical_name='Proposer_Email',
        label_patterns=[r'email', r'e-mail', r'mail'],
        field_type='email',
        required=False
    ),
    'Proposer_Occupation': FieldMapping(
        canonical_name='Proposer_Occupation',
        label_patterns=[r'occupation', r'profession', r'job', r'business', r'service'],
        field_type='text',
        required=True
    ),
    'Proposer_Income': FieldMapping(
        canonical_name='Proposer_Income',
        label_patterns=[r'income', r'annual income', r'salary', r'rs\.', r'rupees'],
        field_type='money',
        required=False
    ),
    'Proposer_PAN': FieldMapping(
        canonical_name='Proposer_PAN',
        label_patterns=[r'pan', r'permanent account', r'income tax'],
        field_type='pan',
        required=True
    ),
    'Proposer_Aadhaar': FieldMapping(
        canonical_name='Proposer_Aadhaar',
        label_patterns=[r'aadhaar', r'aadhar', r'uidai', r'uid'],
        field_type='aadhaar',
        required=False
    ),
    
    # === KYC & DOCUMENTS ===
    'Customer_ID': FieldMapping(
        canonical_name='Customer_ID',
        label_patterns=[r'customer id', r'client id', r'proposer id'],
        field_type='text',
        required=False
    ),
    'KYC_Number': FieldMapping(
        canonical_name='KYC_Number',
        label_patterns=[r'kyc', r'kyc number', r'kyc registry'],
        field_type='text',
        required=False
    ),
    
    # === POLICY DETAILS ===
    'Plan_Name': FieldMapping(
        canonical_name='Plan_Name',
        label_patterns=[r'plan', r'plan name', r'product', r'scheme'],
        field_type='text',
        required=True
    ),
    'Policy_Term': FieldMapping(
        canonical_name='Policy_Term',
        label_patterns=[r'term', r'policy term', r'years'],
        field_type='number',
        required=True
    ),
    'Sum_Assured': FieldMapping(
        canonical_name='Sum_Assured',
        label_patterns=[r'sum assured', r'sum assured \(', r'coverage', r'sa'],
        field_type='money',
        required=True
    ),
    'Premium_Amount': FieldMapping(
        canonical_name='Premium_Amount',
        label_patterns=[r'premium', r'premium amount', r'rs\.', r'rupees'],
        field_type='money',
        required=True
    ),
    'Premium_Mode': FieldMapping(
        canonical_name='Premium_Mode',
        label_patterns=[r'mode', r'premium mode', r'frequency', r'yearly', r'half yearly', r'monthly'],
        field_type='choice',
        choices=['Yearly', 'Half-Yearly', 'Quarterly', 'Monthly'],
        required=True
    ),
    
    # === NOMINEE DETAILS ===
    'Nominee_Name': FieldMapping(
        canonical_name='Nominee_Name',
        label_patterns=[r'nominee', r'nominee name', r'nomination'],
        field_type='text',
        required=False
    ),
    'Nominee_Relationship': FieldMapping(
        canonical_name='Nominee_Relationship',
        label_patterns=[r'relationship', r'relation', r'nominee relation'],
        field_type='text',
        required=False
    ),
    
    # === AGENT DETAILS ===
    'Agent_Code': FieldMapping(
        canonical_name='Agent_Code',
        label_patterns=[r'agent', r'agent code', r'agent no', r'agent number'],
        field_type='alphanumeric',
        required=True
    ),
    'Agent_Name': FieldMapping(
        canonical_name='Agent_Name',
        label_patterns=[r'agent name', r'agent/intermediary'],
        field_type='text',
        required=False
    ),
    'Branch_Name': FieldMapping(
        canonical_name='Branch_Name',
        label_patterns=[r'branch', r'branch name', r'branch office', r'branch code'],
        field_type='text',
        required=True
    ),
    
    # === MEDICAL DETAILS ===
    'Height': FieldMapping(
        canonical_name='Height',
        label_patterns=[r'height', r'ht\.', r'cms', r'feet'],
        field_type='text',
        required=False
    ),
    'Weight': FieldMapping(
        canonical_name='Weight',
        label_patterns=[r'weight', r'wt\.', r'kgs', r'kg'],
        field_type='text',
        required=False
    ),
}


def match_label_to_field(label: str) -> Optional[str]:
    """
    Match a label string to a canonical field name
    Returns the canonical field name or None
    """
    label_lower = label.lower().strip()
    
    for field_name, mapping in FIELD_MAPPINGS.items():
        for pattern in mapping.label_patterns:
            if re.search(pattern, label_lower, re.IGNORECASE):
                return field_name
    
    return None


def get_field_type(field_name: str) -> str:
    """Get the field type for a canonical field name"""
    if field_name in FIELD_MAPPINGS:
        return FIELD_MAPPINGS[field_name].field_type
    return 'text'


def get_validation_fn(field_name: str) -> Optional[Callable]:
    """Get validation function for a field"""
    if field_name in FIELD_MAPPINGS:
        return FIELD_MAPPINGS[field_name].validation_fn
    return None
