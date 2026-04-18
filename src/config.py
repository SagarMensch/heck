"""
LIC Techathon Pipeline Configuration
=====================================
Central config for all pipeline components.
"""

import os
from pathlib import Path

# ──────────────────────── PATHS ────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
FORMS_INPUT_DIR = DATA_DIR / "input_forms"
FORMS_OUTPUT_DIR = DATA_DIR / "output"
MODELS_DIR = PROJECT_ROOT / "models"
TROCR_FT_DIR = MODELS_DIR / "trocr-finetuned"
SYNTH_DATA_DIR = DATA_DIR / "synthetic"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create dirs
for d in [DATA_DIR, FORMS_INPUT_DIR, FORMS_OUTPUT_DIR, MODELS_DIR,
          TROCR_FT_DIR, SYNTH_DATA_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────── GPU / DEVICE ────────────────────────
DEVICE = "cuda"
TORCH_DTYPE = "auto"  # auto selects best for GPU

# ──────────────────────── MODEL CONFIG ────────────────────────
# Primary VLM: Qwen2.5-VL-3B (Fits in 24GB VRAM with high res eager attention)
QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
QWEN_MAX_NEW_TOKENS = 4096
QWEN_TEMPERATURE = 0.1

# Field-level HTR: TrOCR
TROCR_MODEL_ID = "microsoft/trocr-base-handwritten"
TROCR_LARGE_MODEL_ID = "microsoft/trocr-large-handwritten"

# Layout Detection: PaddleOCR
PADDLE_DET_MODEL = "PP-OCRv5_server_det"
PADDLE_REC_MODEL = "PP-OCRv5_server_rec"
PADDLE_LANG = "en"

# ──────────────────────── CONFIDENCE THRESHOLDS ────────────────────────
CONFIDENCE_HIGH = 0.90      # Auto-accept
CONFIDENCE_MEDIUM = 0.70    # Flag for review
CONFIDENCE_LOW = 0.50       # Trigger VLM fallback / reject
FIELD_REJECT_THRESHOLD = 0.30  # Below this = missing

# ──────────────────────── IMAGE PREPROCESSING ────────────────────────
TARGET_DPI = 300
MAX_IMAGE_DIMENSION = 1536
QUALITY_SCORE_THRESHOLD = 50.0  # Laplacian variance below = too blurry
SUPPORTED_FORMATS = {".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}

# ──────────────────────── PROCESSING ────────────────────────
MAX_BATCH_SIZE = 4  # Forms processed in parallel
PROCESSING_TIMEOUT = 120  # seconds per form
MAX_FORMS = 100  # Max forms in single batch

# ──────────────────────── FIELD DEFINITIONS (Form 300) ────────────────────────
# Each field: (field_name, field_type, data_type, expected_length, mandatory, validation_regex)
FORM_300_FIELDS = [
    # Section 1: Proposer Details
    {"field_name": "Proposer_Full_Name", "field_type": "Handwritten_Text", "data_type": "Alphabetic", "expected_length": "1-100", "mandatory": True, "section": "Proposer Details"},
    {"field_name": "Proposer_Date_of_Birth", "field_type": "Handwritten_Date", "data_type": "Date", "expected_length": "10", "mandatory": True, "section": "Proposer Details"},
    {"field_name": "Proposer_Age", "field_type": "Handwritten_Numeric", "data_type": "Numeric", "expected_length": "1-3", "mandatory": True, "section": "Proposer Details"},
    {"field_name": "Proposer_Gender", "field_type": "Checkbox_or_Text", "data_type": "Alphabetic", "expected_length": "1-10", "mandatory": True, "section": "Proposer Details"},
    {"field_name": "Proposer_Marital_Status", "field_type": "Checkbox_or_Text", "data_type": "Alphabetic", "expected_length": "1-15", "mandatory": False, "section": "Proposer Details"},
    {"field_name": "Proposer_Father_Husband_Name", "field_type": "Handwritten_Text", "data_type": "Alphabetic", "expected_length": "1-100", "mandatory": True, "section": "Proposer Details"},
    {"field_name": "Proposer_Address_Line1", "field_type": "Handwritten_Text", "data_type": "Alphanumeric", "expected_length": "1-200", "mandatory": True, "section": "Proposer Details"},
    {"field_name": "Proposer_Address_Line2", "field_type": "Handwritten_Text", "data_type": "Alphanumeric", "expected_length": "0-200", "mandatory": False, "section": "Proposer Details"},
    {"field_name": "Proposer_City", "field_type": "Handwritten_Text", "data_type": "Alphabetic", "expected_length": "1-50", "mandatory": True, "section": "Proposer Details"},
    {"field_name": "Proposer_State", "field_type": "Handwritten_Text", "data_type": "Alphabetic", "expected_length": "1-30", "mandatory": True, "section": "Proposer Details"},
    {"field_name": "Proposer_Pincode", "field_type": "Handwritten_Numeric", "data_type": "Numeric", "expected_length": "6", "mandatory": True, "section": "Proposer Details"},
    {"field_name": "Proposer_Mobile_Number", "field_type": "Handwritten_Numeric", "data_type": "Numeric", "expected_length": "10", "mandatory": True, "section": "Proposer Details"},
    {"field_name": "Proposer_Email", "field_type": "Handwritten_Text", "data_type": "Alphanumeric", "expected_length": "1-100", "mandatory": False, "section": "Proposer Details"},
    {"field_name": "Proposer_PAN", "field_type": "Handwritten_Alphanumeric", "data_type": "Alphanumeric", "expected_length": "10", "mandatory": True, "section": "Proposer Details"},
    {"field_name": "Proposer_Aadhaar", "field_type": "Handwritten_Numeric", "data_type": "Numeric", "expected_length": "12", "mandatory": False, "section": "Proposer Details"},
    {"field_name": "Proposer_Occupation", "field_type": "Handwritten_Text", "data_type": "Alphabetic", "expected_length": "1-50", "mandatory": True, "section": "Proposer Details"},
    {"field_name": "Proposer_Annual_Income", "field_type": "Handwritten_Numeric", "data_type": "Numeric", "expected_length": "1-15", "mandatory": True, "section": "Proposer Details"},

    # Section 2: Life Assured Details (if different from proposer)
    {"field_name": "LA_Full_Name", "field_type": "Handwritten_Text", "data_type": "Alphabetic", "expected_length": "1-100", "mandatory": False, "section": "Life Assured"},
    {"field_name": "LA_Date_of_Birth", "field_type": "Handwritten_Date", "data_type": "Date", "expected_length": "10", "mandatory": False, "section": "Life Assured"},
    {"field_name": "LA_Age", "field_type": "Handwritten_Numeric", "data_type": "Numeric", "expected_length": "1-3", "mandatory": False, "section": "Life Assured"},
    {"field_name": "LA_Relationship", "field_type": "Handwritten_Text", "data_type": "Alphabetic", "expected_length": "1-30", "mandatory": False, "section": "Life Assured"},

    # Section 3: Plan / Policy Details
    {"field_name": "Plan_Name", "field_type": "Handwritten_Text", "data_type": "Alphanumeric", "expected_length": "1-100", "mandatory": True, "section": "Plan Details"},
    {"field_name": "Plan_Number", "field_type": "Handwritten_Numeric", "data_type": "Numeric", "expected_length": "1-10", "mandatory": False, "section": "Plan Details"},
    {"field_name": "Policy_Term", "field_type": "Handwritten_Numeric", "data_type": "Numeric", "expected_length": "1-3", "mandatory": True, "section": "Plan Details"},
    {"field_name": "Premium_Paying_Term", "field_type": "Handwritten_Numeric", "data_type": "Numeric", "expected_length": "1-3", "mandatory": True, "section": "Plan Details"},
    {"field_name": "Sum_Assured", "field_type": "Handwritten_Numeric", "data_type": "Numeric", "expected_length": "1-15", "mandatory": True, "section": "Plan Details"},
    {"field_name": "Premium_Amount", "field_type": "Handwritten_Numeric", "data_type": "Numeric", "expected_length": "1-15", "mandatory": True, "section": "Plan Details"},
    {"field_name": "Premium_Mode", "field_type": "Checkbox_or_Text", "data_type": "Alphabetic", "expected_length": "1-20", "mandatory": True, "section": "Plan Details"},

    # Section 4: Nominee Details
    {"field_name": "Nominee_Name", "field_type": "Handwritten_Text", "data_type": "Alphabetic", "expected_length": "1-100", "mandatory": True, "section": "Nominee Details"},
    {"field_name": "Nominee_Relationship", "field_type": "Handwritten_Text", "data_type": "Alphabetic", "expected_length": "1-30", "mandatory": True, "section": "Nominee Details"},
    {"field_name": "Nominee_Age", "field_type": "Handwritten_Numeric", "data_type": "Numeric", "expected_length": "1-3", "mandatory": True, "section": "Nominee Details"},
    {"field_name": "Nominee_Address", "field_type": "Handwritten_Text", "data_type": "Alphanumeric", "expected_length": "1-200", "mandatory": False, "section": "Nominee Details"},

    # Section 5: Bank Details
    {"field_name": "Bank_Account_Number", "field_type": "Handwritten_Numeric", "data_type": "Numeric", "expected_length": "8-18", "mandatory": True, "section": "Bank Details"},
    {"field_name": "Bank_Name", "field_type": "Handwritten_Text", "data_type": "Alphabetic", "expected_length": "1-60", "mandatory": True, "section": "Bank Details"},
    {"field_name": "Bank_IFSC", "field_type": "Handwritten_Alphanumeric", "data_type": "Alphanumeric", "expected_length": "11", "mandatory": True, "section": "Bank Details"},
    {"field_name": "Bank_Branch", "field_type": "Handwritten_Text", "data_type": "Alphabetic", "expected_length": "1-60", "mandatory": False, "section": "Bank Details"},

    # Section 6: Agent Details
    {"field_name": "Agent_Code", "field_type": "Handwritten_Alphanumeric", "data_type": "Alphanumeric", "expected_length": "1-20", "mandatory": False, "section": "Agent Details"},
    {"field_name": "Agent_Name", "field_type": "Handwritten_Text", "data_type": "Alphabetic", "expected_length": "1-60", "mandatory": False, "section": "Agent Details"},
    {"field_name": "Branch_Code", "field_type": "Handwritten_Alphanumeric", "data_type": "Alphanumeric", "expected_length": "1-10", "mandatory": False, "section": "Agent Details"},

    # Section 7: Declaration & Signature
    {"field_name": "Date_of_Proposal", "field_type": "Handwritten_Date", "data_type": "Date", "expected_length": "10", "mandatory": True, "section": "Declaration"},
    {"field_name": "Place_of_Signing", "field_type": "Handwritten_Text", "data_type": "Alphabetic", "expected_length": "1-50", "mandatory": True, "section": "Declaration"},
    {"field_name": "Proposer_Signature", "field_type": "Signature", "data_type": "Image", "expected_length": "N/A", "mandatory": True, "section": "Declaration"},
    {"field_name": "Proposer_Photo", "field_type": "Photograph", "data_type": "Image", "expected_length": "N/A", "mandatory": True, "section": "Declaration"},
]

# Field names list for quick lookup
FIELD_NAMES = [f["field_name"] for f in FORM_300_FIELDS]
MANDATORY_FIELDS = [f["field_name"] for f in FORM_300_FIELDS if f["mandatory"]]

# ──────────────────────── VALIDATION RULES ────────────────────────
VALIDATION_RULES = {
    "Proposer_PAN": {
        "regex": r"^[A-Z]{5}\d{4}[A-Z]$",
        "description": "PAN: 5 uppercase letters + 4 digits + 1 uppercase letter"
    },
    "Proposer_Aadhaar": {
        "regex": r"^\d{12}$",
        "description": "Aadhaar: exactly 12 digits"
    },
    "Proposer_Pincode": {
        "regex": r"^\d{6}$",
        "description": "Pincode: exactly 6 digits"
    },
    "Proposer_Mobile_Number": {
        "regex": r"^[6-9]\d{9}$",
        "description": "Indian mobile: starts with 6-9, 10 digits"
    },
    "Bank_IFSC": {
        "regex": r"^[A-Z]{4}0[A-Z0-9]{6}$",
        "description": "IFSC: 4 letters + 0 + 6 alphanumeric"
    },
    "Proposer_Date_of_Birth": {
        "regex": r"^\d{2}[/\-\.]\d{2}[/\-\.]\d{4}$",
        "description": "Date: DD/MM/YYYY or DD-MM-YYYY"
    },
    "LA_Date_of_Birth": {
        "regex": r"^\d{2}[/\-\.]\d{2}[/\-\.]\d{4}$",
        "description": "Date: DD/MM/YYYY or DD-MM-YYYY"
    },
    "Date_of_Proposal": {
        "regex": r"^\d{2}[/\-\.]\d{2}[/\-\.]\d{4}$",
        "description": "Date: DD/MM/YYYY or DD-MM-YYYY"
    },
    "Bank_Account_Number": {
        "regex": r"^\d{8,18}$",
        "description": "Bank account: 8-18 digits"
    },
    "Proposer_Email": {
        "regex": r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$",
        "description": "Valid email format"
    },
}

# ──────────────────────── TrOCR FINE-TUNING CONFIG ────────────────────────
TROCR_FINETUNE_CONFIG = {
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj", "k_proj", "out_proj"],
    "learning_rate": 5e-5,
    "num_epochs": 10,
    "batch_size": 8,
    "gradient_accumulation_steps": 2,
    "fp16": True,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "max_length": 64,
    "image_size": (384, 384),
}

# ──────────────────────── SYNTHETIC DATA CONFIG ────────────────────────
SYNTH_CONFIG = {
    "num_samples_per_field": 500,
    "fonts_dir": PROJECT_ROOT / "fonts",
    "augmentation_probability": 0.8,
    "font_size_range": (28, 48),
    "rotation_range": (-5, 5),
    "blur_range": (0, 2),
    "noise_var_range": (5, 30),
}

# ──────────────────────── QWEN EXTRACTION PROMPT ────────────────────────
QWEN_SYSTEM_PROMPT = """You are an expert insurance document data extraction AI. You extract structured data from handwritten LIC (Life Insurance Corporation of India) Proposal Form No. 300.

CRITICAL INSTRUCTIONS:
1. Extract ALL fields visible in the form image
2. For each field, provide the extracted value and a confidence score (0.0 to 1.0)
3. If a field is empty or illegible, set value to null and confidence to 0.0
4. Maintain exact handwritten text as-is (preserve spelling, formatting)
5. For dates, use DD/MM/YYYY format
6. For amounts, extract numeric value without commas/symbols
7. For checkboxes, indicate the selected option
8. Output ONLY valid JSON, no explanation text"""

QWEN_EXTRACTION_PROMPT = """Extract ALL fields from this handwritten LIC Proposal Form No. 300 image.

Return a JSON object with this exact structure:
{
  "form_metadata": {
    "form_type": "Proposal Form 300",
    "total_pages": <number>,
    "scan_quality": "good|fair|poor"
  },
  "fields": {
    "<field_name>": {
      "value": "<extracted_value_or_null>",
      "confidence": <0.0_to_1.0>,
      "field_type": "handwritten|printed|checkbox|signature|photo",
      "bounding_box": [x1, y1, x2, y2] or null
    }
  },
  "extraction_summary": {
    "total_fields_found": <number>,
    "fields_extracted": <number>,
    "fields_missing": <number>,
    "fields_low_confidence": <number>,
    "overall_confidence": <0.0_to_1.0>
  }
}

Expected fields to extract (extract any additional fields found too):
""" + "\n".join([f"- {f['field_name']} ({f['data_type']}, {'mandatory' if f['mandatory'] else 'optional'})" for f in FORM_300_FIELDS])
