"""
Complete Form 300 Templates - All Pages
=========================================
Canonical templates for all pages of LIC Proposal Form 300.
Each page has specific fields at fixed coordinates.
"""

# PAGE 2 - Personal Details (Most important page)
FORM_300_PAGE2_TEMPLATE = {
    # Row 1-2: IDs (usually empty or printed)
    "Customer_ID": {"bbox": [1200, 500, 2374, 560], "type": "printed", "editable": False},
    "CKYC_Number": {"bbox": [1200, 560, 2374, 630], "type": "printed", "editable": False},
    
    # Row 3: Name (split into 3 sub-fields)
    "Prefix": {"bbox": [1200, 630, 1400, 730], "type": "handwritten", "editable": True, "critical": True},
    "First_Name": {"bbox": [1400, 730, 1900, 860], "type": "handwritten", "editable": True, "critical": True},
    "Middle_Name": {"bbox": [1900, 730, 2100, 860], "type": "handwritten", "editable": True, "critical": False},
    "Last_Name": {"bbox": [2100, 860, 2374, 950], "type": "handwritten", "editable": True, "critical": True},
    
    # Row 4-5: Parents
    "Father_Husband_Name": {"bbox": [1200, 950, 2374, 1030], "type": "handwritten", "editable": True, "critical": False},
    "Mother_Name": {"bbox": [1200, 1030, 2374, 1100], "type": "handwritten", "editable": True, "critical": False},
    
    # Row 6: Gender (checkbox)
    "Gender_Male": {"bbox": [1400, 1100, 1460, 1160], "type": "checkbox", "editable": True, "critical": True},
    "Gender_Female": {"bbox": [1700, 1100, 1760, 1160], "type": "checkbox", "editable": True, "critical": True},
    "Gender_Transgender": {"bbox": [2000, 1100, 2060, 1160], "type": "checkbox", "editable": True, "critical": True},
    
    # Row 7: Marital Status
    "Marital_Status": {"bbox": [1200, 1170, 2374, 1250], "type": "handwritten", "editable": True, "critical": True},
    
    # Row 8: Spouse
    "Spouse_Name": {"bbox": [1200, 1250, 2374, 1360], "type": "handwritten", "editable": True, "critical": False},
    
    # Row 9-10: DOB and Age
    "Date_of_Birth": {"bbox": [1200, 1360, 1800, 1430], "type": "handwritten", "editable": True, "critical": True, "validation": "date"},
    "Age": {"bbox": [1800, 1360, 2100, 1430], "type": "handwritten", "editable": True, "critical": True, "validation": "number"},
    
    # Row 11-12: Birth details
    "Place_of_Birth": {"bbox": [1200, 1650, 2374, 1780], "type": "handwritten", "editable": True, "critical": False},
    "Age_Proof": {"bbox": [1200, 1780, 2374, 1850], "type": "handwritten", "editable": True, "critical": False},
    
    # Row 13-14: Nationality
    "Nationality": {"bbox": [1200, 1850, 2374, 1920], "type": "handwritten", "editable": True, "critical": True},
    "Citizenship": {"bbox": [1200, 1920, 2374, 2000], "type": "handwritten", "editable": True, "critical": True},
    
    # Row 15: Permanent Address (multi-line)
    "Address_Line1": {"bbox": [1200, 2100, 2374, 2200], "type": "handwritten", "editable": True, "critical": True},
    "Address_Line2": {"bbox": [1200, 2200, 2374, 2280], "type": "handwritten", "editable": True, "critical": False},
    "City": {"bbox": [1200, 2280, 1800, 2360], "type": "handwritten", "editable": True, "critical": True, "fuzzy": True},
    "State": {"bbox": [1800, 2280, 2374, 2360], "type": "handwritten", "editable": True, "critical": True, "fuzzy": True},
    "Pincode": {"bbox": [1200, 2360, 1600, 2430], "type": "handwritten", "editable": True, "critical": True, "validation": "pincode"},
    "STD_Code": {"bbox": [1600, 2360, 1800, 2430], "type": "handwritten", "editable": True, "critical": False},
    "Phone": {"bbox": [1800, 2360, 2374, 2430], "type": "handwritten", "editable": True, "critical": True, "validation": "mobile"},
    
    # Row 16: Correspondence Address (if different)
    "Corr_Address_Line1": {"bbox": [1200, 2500, 2374, 2580], "type": "handwritten", "editable": True, "critical": False},
    "Corr_Address_Line2": {"bbox": [1200, 2580, 2374, 2650], "type": "handwritten", "editable": True, "critical": False},
    "Corr_City": {"bbox": [1200, 2650, 1800, 2720], "type": "handwritten", "editable": True, "critical": False},
    "Corr_State": {"bbox": [1800, 2650, 2374, 2720], "type": "handwritten", "editable": True, "critical": False},
    "Corr_Pincode": {"bbox": [1200, 2720, 1600, 2780], "type": "handwritten", "editable": True, "critical": False},
    
    # Critical IDs at bottom
    "PAN": {"bbox": [1200, 2800, 1800, 2870], "type": "handwritten", "editable": True, "critical": True, "validation": "pan"},
    "Aadhaar": {"bbox": [1800, 2800, 2374, 2870], "type": "handwritten", "editable": True, "critical": True, "validation": "aadhaar"},
    "Email": {"bbox": [1200, 2870, 2374, 2940], "type": "handwritten", "editable": True, "critical": False, "validation": "email"},
}

# PAGE 3 - Policy Details
FORM_300_PAGE3_TEMPLATE = {
    # Plan Details
    "Plan_Name": {"bbox": [1200, 600, 2374, 700], "type": "handwritten", "editable": True, "critical": True, "fuzzy": True},
    "Plan_Number": {"bbox": [1200, 700, 2374, 800], "type": "handwritten", "editable": True, "critical": True},
    
    # Terms
    "Policy_Term": {"bbox": [1200, 800, 1800, 900], "type": "handwritten", "editable": True, "critical": True, "validation": "number"},
    "Premium_Paying_Term": {"bbox": [1800, 800, 2374, 900], "type": "handwritten", "editable": True, "critical": True, "validation": "number"},
    
    # Financial (CRITICAL)
    "Sum_Assured": {"bbox": [1200, 1000, 2374, 1100], "type": "handwritten", "editable": True, "critical": True, "validation": "currency"},
    "Premium_Amount": {"bbox": [1200, 1100, 1800, 1200], "type": "handwritten", "editable": True, "critical": True, "validation": "currency"},
    "Premium_Mode": {"bbox": [1800, 1100, 2374, 1200], "type": "handwritten", "editable": True, "critical": True, "fuzzy": True},  # Yearly/Half-yearly/etc
    
    # Proposal details
    "Date_of_Proposal": {"bbox": [1200, 1300, 1800, 1400], "type": "handwritten", "editable": True, "critical": True, "validation": "date"},
    "Place_of_Proposal": {"bbox": [1800, 1300, 2374, 1400], "type": "handwritten", "editable": True, "critical": True},
    
    # Banking (CRITICAL)
    "Bank_Name": {"bbox": [1200, 1600, 2374, 1700], "type": "handwritten", "editable": True, "critical": True, "fuzzy": True},
    "Bank_Branch": {"bbox": [1200, 1700, 2374, 1800], "type": "handwritten", "editable": True, "critical": True},
    "Bank_Account_Number": {"bbox": [1200, 1800, 2374, 1900], "type": "handwritten", "editable": True, "critical": True, "validation": "account"},
    "Bank_IFSC": {"bbox": [1200, 1900, 2374, 2000], "type": "handwritten", "editable": True, "critical": True, "validation": "ifsc"},
    "MICR_Code": {"bbox": [1200, 2000, 2374, 2100], "type": "handwritten", "editable": True, "critical": False},
    
    # Agent Details
    "Agent_Code": {"bbox": [1200, 2300, 1800, 2400], "type": "handwritten", "editable": True, "critical": False},
    "Agent_Name": {"bbox": [1800, 2300, 2374, 2400], "type": "handwritten", "editable": True, "critical": False},
    "Branch_Code": {"bbox": [1200, 2400, 2374, 2500], "type": "handwritten", "editable": True, "critical": False},
}

# PAGE 4 - Nominee and Medical
FORM_300_PAGE4_TEMPLATE = {
    # Nominee (Important)
    "Nominee_Name": {"bbox": [1200, 600, 2374, 700], "type": "handwritten", "editable": True, "critical": True},
    "Nominee_Relationship": {"bbox": [1200, 700, 1800, 800], "type": "handwritten", "editable": True, "critical": True, "fuzzy": True},
    "Nominee_Date_of_Birth": {"bbox": [1800, 700, 2374, 800], "type": "handwritten", "editable": True, "critical": True, "validation": "date"},
    "Nominee_Age": {"bbox": [1200, 800, 1800, 900], "type": "handwritten", "editable": True, "critical": True, "validation": "number"},
    "Nominee_Address": {"bbox": [1200, 900, 2374, 1100], "type": "handwritten", "editable": True, "critical": True},
    
    # Appointee (if nominee minor)
    "Appointee_Name": {"bbox": [1200, 1200, 2374, 1300], "type": "handwritten", "editable": True, "critical": False},
    "Appointee_Relationship": {"bbox": [1200, 1300, 2374, 1400], "type": "handwritten", "editable": True, "critical": False},
    
    # Previous Policy
    "Previous_Policy_Number": {"bbox": [1200, 1600, 2374, 1700], "type": "handwritten", "editable": True, "critical": False},
    "Previous_Policy_SA": {"bbox": [1200, 1700, 2374, 1800], "type": "handwritten", "editable": True, "critical": False},
    
    # Medical Questions (Yes/No checkboxes)
    "Med_Q1_Heart": {"bbox": [1400, 2000, 1460, 2060], "type": "checkbox", "editable": True, "critical": True},
    "Med_Q2_BP": {"bbox": [1400, 2100, 1460, 2160], "type": "checkbox", "editable": True, "critical": True},
    "Med_Q3_Diabetes": {"bbox": [1400, 2200, 1460, 2260], "type": "checkbox", "editable": True, "critical": True},
}

# Map page numbers to templates
PAGE_TEMPLATES = {
    2: FORM_300_PAGE2_TEMPLATE,
    3: FORM_300_PAGE3_TEMPLATE,
    4: FORM_300_PAGE4_TEMPLATE,
}

# All extractable fields across all pages
ALL_FIELDS = {}
for page_num, template in PAGE_TEMPLATES.items():
    for field_name, field_info in template.items():
        ALL_FIELDS[field_name] = {
            **field_info,
            "page": page_num
        }

def get_template_for_page(page_num: int) -> dict:
    """Get field template for specific page."""
    return PAGE_TEMPLATES.get(page_num, {})

def get_all_critical_fields() -> list:
    """Get list of all critical fields."""
    return [name for name, info in ALL_FIELDS.items() if info.get("critical", False)]

def get_field_info(field_name: str) -> dict:
    """Get complete info for a field."""
    return ALL_FIELDS.get(field_name, {})

def get_page_for_field(field_name: str) -> int:
    """Get which page a field belongs to."""
    info = get_field_info(field_name)
    return info.get("page", -1)

# For RFP Techathon - Expected JSON Output Format
def create_field_output(field_name: str, value: str, confidence: float, 
                       bbox: list, page_num: int, status: str = "Verified") -> dict:
    """Create standardized field output for ReviewWorkspace UI."""
    field_info = get_field_info(field_name)
    
    return {
        "value": value,
        "confidence": round(confidence, 2),
        "bbox": bbox,
        "page_num": page_num,
        "status": status,
        "editable": field_info.get("editable", True),
        "anchor": field_name.lower(),
        "metadata": {
            "field_type": field_info.get("type", "handwritten"),
            "expected_length": "1-50",
            "mandatory": field_info.get("critical", False),
            "data_type": "Alphabetic" if field_info.get("type") == "handwritten" else "AlphaNumeric",
            "validation_rule": field_info.get("validation", None),
            "fuzzy_match": field_info.get("fuzzy", False)
        }
    }

if __name__ == "__main__":
    print("Form 300 Templates Loaded")
    print(f"Total fields defined: {len(ALL_FIELDS)}")
    print(f"Critical fields: {len(get_all_critical_fields())}")
    print(f"Pages covered: {list(PAGE_TEMPLATES.keys())}")
