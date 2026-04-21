"""
Field Mappings for LIC Proposal Form 300
==========================================
Maps logical field names to table row/column positions.

Form Structure:
- Page 2 (Personal Details): Rows 1-16
- Page 3 (Policy Details): Rows 1-15  
- Page 4 (Medical/Nominee): Rows 1-12
"""

from typing import Dict, Tuple, List

# Field mapping: (row, col) in table
# Row is 0-indexed from top
# Col: 0=label, 1=value

FORM_300_PAGE2_FIELDS: Dict[str, Tuple[int, int]] = {
    # Row 1: Customer ID
    "Proposer_Customer_ID": (0, 1),
    
    # Row 2: CKYC
    "Proposer_CKYC": (1, 1),
    
    # Row 3: Name (split across 3 sub-columns)
    "Proposer_First_Name": (2, 1),
    "Proposer_Middle_Name": (2, 2),
    "Proposer_Last_Name": (2, 3),
    "Proposer_Prefix": (2, 0),  # Shri/Smt/etc in label column
    
    # Row 4: Father/Husband
    "Proposer_Father_Husband_Name": (3, 1),
    
    # Row 5: Mother
    "Proposer_Mother_Name": (4, 1),
    
    # Row 6: Gender
    "Proposer_Gender": (5, 1),
    
    # Row 7: Marital Status
    "Proposer_Marital_Status": (6, 1),
    
    # Row 8: Spouse Name
    "Proposer_Spouse_Name": (7, 1),
    
    # Row 9: DOB
    "Proposer_Date_of_Birth": (8, 1),
    
    # Row 10: Age
    "Proposer_Age": (9, 1),
    
    # Row 11: Birth Place
    "Proposer_Birth_Place": (10, 1),
    
    # Row 12: Age Proof
    "Proposer_Age_Proof": (11, 1),
    
    # Row 13: Nationality
    "Proposer_Nationality": (12, 1),
    
    # Row 14: Citizenship
    "Proposer_Citizenship": (13, 1),
    
    # Row 15: Permanent Address (nested structure)
    "Proposer_Address_Line1": (14, 1),  # House/Street
    "Proposer_Address_Line2": (14, 1),  # Town/Village
    "Proposer_City": (14, 1),           # City
    "Proposer_State": (14, 1),          # State
    "Proposer_Pincode": (14, 1),        # PIN
    "Proposer_Mobile_Number": (14, 1),  # Phone
    
    # Row 16: Correspondence Address
    "Correspondence_Address": (15, 1),
}

FORM_300_PAGE3_FIELDS: Dict[str, Tuple[int, int]] = {
    # Policy Details
    "Plan_Name": (0, 1),
    "Plan_Number": (1, 1),
    "Policy_Term": (2, 1),
    "Premium_Paying_Term": (3, 1),
    "Sum_Assured": (4, 1),
    "Premium_Amount": (5, 1),
    "Premium_Mode": (6, 1),
    "Date_of_Proposal": (7, 1),
    "Place_of_Signing": (8, 1),
    
    # Bank Details
    "Bank_Name": (9, 1),
    "Bank_Branch": (10, 1),
    "Bank_Account_Number": (11, 1),
    "Bank_IFSC": (12, 1),
    
    # Agent Details
    "Agent_Code": (13, 1),
    "Agent_Name": (14, 1),
}

FORM_300_PAGE4_FIELDS: Dict[str, Tuple[int, int]] = {
    # Nominee Details
    "Nominee_Name": (0, 1),
    "Nominee_Relationship": (1, 1),
    "Nominee_Age": (2, 1),
    "Nominee_Address": (3, 1),
    
    # Medical Questions (Yes/No)
    "Medical_Q1": (4, 1),
    "Medical_Q2": (5, 1),
    "Medical_Q3": (6, 1),
    
    # Previous Policy
    "Previous_Policy_Number": (7, 1),
    "Previous_Policy_Sum_Assured": (8, 1),
    
    # Documents
    "PAN": (9, 1),
    "Aadhaar": (10, 1),
    "Email": (11, 1),
}

# All fields combined
ALL_LIC_FIELDS: List[str] = [
    # Personal Details (Page 2)
    "Proposer_Customer_ID", "Proposer_CKYC",
    "Proposer_Prefix", "Proposer_First_Name", "Proposer_Middle_Name", "Proposer_Last_Name",
    "Proposer_Father_Husband_Name", "Proposer_Mother_Name",
    "Proposer_Gender", "Proposer_Marital_Status", "Proposer_Spouse_Name",
    "Proposer_Date_of_Birth", "Proposer_Age", "Proposer_Birth_Place",
    "Proposer_Nationality", "Proposer_Citizenship",
    "Proposer_Address_Line1", "Proposer_Address_Line2",
    "Proposer_City", "Proposer_State", "Proposer_Pincode",
    "Proposer_Mobile_Number", "Proposer_Email",
    "Proposer_PAN", "Proposer_Aadhaar",
    
    # Policy Details (Page 3)
    "Plan_Name", "Plan_Number", "Policy_Term", "Premium_Paying_Term",
    "Sum_Assured", "Premium_Amount", "Premium_Mode",
    "Date_of_Proposal", "Place_of_Signing",
    "Bank_Name", "Bank_Branch", "Bank_Account_Number", "Bank_IFSC",
    "Agent_Code", "Agent_Name",
    
    # Nominee (Page 4)
    "Nominee_Name", "Nominee_Relationship", "Nominee_Age", "Nominee_Address",
    
    # LA (Life Assured) - if different from proposer
    "LA_Full_Name", "LA_Date_of_Birth", "LA_Age", "LA_Relationship",
]

# Field metadata for validation
FIELD_METADATA = {
    "Proposer_PAN": {
        "type": "text",
        "regex": r"^[A-Z]{5}\d{4}[A-Z]$",
        "mandatory": True,
        "description": "Permanent Account Number"
    },
    "Proposer_Aadhaar": {
        "type": "number",
        "regex": r"^\d{12}$",
        "mandatory": True,
        "description": "Aadhaar Number",
        "checksum": "verhoeff"
    },
    "Proposer_Pincode": {
        "type": "number",
        "regex": r"^\d{6}$",
        "mandatory": True,
        "description": "6-digit PIN code"
    },
    "Proposer_Mobile_Number": {
        "type": "text",
        "regex": r"^[6-9]\d{9}$",
        "mandatory": True,
        "description": "10-digit mobile"
    },
    "Proposer_Date_of_Birth": {
        "type": "date",
        "format": "%d/%m/%Y",
        "mandatory": True,
        "description": "DD/MM/YYYY format"
    },
    "Proposer_Age": {
        "type": "number",
        "min": 18,
        "max": 100,
        "mandatory": True
    },
    "Sum_Assured": {
        "type": "currency",
        "min": 50000,
        "mandatory": True
    },
    "Premium_Amount": {
        "type": "currency",
        "mandatory": True
    },
    "Bank_Account_Number": {
        "type": "text",
        "regex": r"^\d{8,18}$",
        "mandatory": False
    },
    "Bank_IFSC": {
        "type": "text",
        "regex": r"^[A-Z]{4}0[A-Z0-9]{6}$",
        "mandatory": False
    },
    "Proposer_Email": {
        "type": "email",
        "regex": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
        "mandatory": False
    }
}

# Critical fields (must have high accuracy)
CRITICAL_FIELDS = [
    "Proposer_PAN",
    "Proposer_Aadhaar",
    "Proposer_Date_of_Birth",
    "Proposer_Mobile_Number",
    "Sum_Assured",
    "Premium_Amount",
    "Proposer_Pincode"
]

# Mapping for all form types
LIC_FIELD_MAP = {
    "form_300": FORM_300_PAGE2_FIELDS,
    "form_300_page2": FORM_300_PAGE2_FIELDS,
    "form_300_page3": FORM_300_PAGE3_FIELDS,
    "form_300_page4": FORM_300_PAGE4_FIELDS,
}


def get_field_metadata(field_name: str) -> dict:
    """Get metadata for a field."""
    return FIELD_METADATA.get(field_name, {
        "type": "text",
        "mandatory": False
    })


def is_critical_field(field_name: str) -> bool:
    """Check if field is critical."""
    return field_name in CRITICAL_FIELDS


def get_page_for_field(field_name: str) -> int:
    """Determine which page a field belongs to."""
    if field_name in FORM_300_PAGE2_FIELDS:
        return 2
    elif field_name in FORM_300_PAGE3_FIELDS:
        return 3
    elif field_name in FORM_300_PAGE4_FIELDS:
        return 4
    return 2  # Default
