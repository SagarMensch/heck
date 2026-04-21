"""Grouped Qwen extraction for the 22 supported surgical fields.

This is a broader strategy than one-crop-per-field:
- Extract logically related fields together from the same page/section.
- Use constrained prompts per cluster.
- Derive final 22-field output from the grouped responses.

It is intended for review/debug and can be used to compare against the
taxonomy-driven crop-based pass on a specific sample.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import cv2
import fitz
import numpy as np
from PIL import Image

from src.extractor import ModelManager, QwenExtractor


def render_page(pdf_path: Path, page_num: int, dpi: int = 160) -> Image.Image:
    doc = fitz.open(str(pdf_path))
    page = doc.load_page(page_num - 1)
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    doc.close()
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
    if pix.n == 4:
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    else:
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def crop_region(image: Image.Image, box_norm: Tuple[float, float, float, float]) -> Image.Image:
    width, height = image.size
    x1n, y1n, x2n, y2n = box_norm
    x1 = max(0, min(width - 1, int(round(x1n * width))))
    y1 = max(0, min(height - 1, int(round(y1n * height))))
    x2 = max(x1 + 1, min(width, int(round(x2n * width))))
    y2 = max(y1 + 1, min(height, int(round(y2n * height))))
    return image.crop((x1, y1, x2, y2))


GROUP_CROP_BOXES: Dict[str, Tuple[float, float, float, float]] = {
    "page2_identity": (0.02, 0.08, 0.98, 0.58),
    "page2_address": (0.02, 0.54, 0.98, 0.96),
    "page3_kyc": (0.02, 0.28, 0.98, 0.98),
    "page6_nominee": (0.02, 0.02, 0.98, 0.42),
    "page7_plan": (0.02, 0.40, 0.98, 0.86),
    "page10_medical": (0.58, 0.46, 0.99, 0.90),
    "page14_declaration": (0.20, 0.72, 0.95, 0.98),
}


def ask_qwen(extractor: QwenExtractor, image: Image.Image, prompt: str) -> Dict:
    import torch
    import gc
    result = extractor.extract_from_image(image, custom_prompt=prompt)
    
    # Force VRAM cleanup after every crop
    torch.cuda.empty_cache()
    gc.collect()
    
    if isinstance(result, dict) and "fields" in result and isinstance(result["fields"], dict):
        # Normalization for full-page parser fallback
        flat = {}
        for key, value in result["fields"].items():
            if isinstance(value, dict):
                flat[key] = value.get("value")
            else:
                flat[key] = value
        return flat
    return result if isinstance(result, dict) else {}


def normalize_text(value) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    low = text.lower()
    if low in {"null", "none", "not found", "unknown", "n/a", "na"}:
        return None
    return text


def normalize_number(value) -> Optional[str]:
    text = normalize_text(value)
    if text is None:
        return None
    digits = re.sub(r"[^\d]", "", text)
    return digits or None


def normalize_pan(value) -> Optional[str]:
    text = normalize_text(value)
    if text is None:
        return None
    text = re.sub(r"[^A-Za-z0-9]", "", text).upper()
    return text or None


def normalize_gender(value) -> Optional[str]:
    text = normalize_text(value)
    if text is None:
        return None
    low = text.lower()
    if "male" in low:
        return "Male"
    if "female" in low:
        return "Female"
    if "trans" in low:
        return "Transgender"
    return text


def cluster_prompts() -> Dict[str, str]:
    return {
        "page2_identity": """Read this LIC Form 300 page and return ONLY valid JSON.
Extract these fields exactly as written by hand:
{
  "first_name": "...",
  "middle_name": "...",
  "last_name": "...",
  "father_full_name": "...",
  "mother_full_name": "...",
  "gender": "Male/Female/Transgender/null",
  "marital_status": "...",
  "spouse_full_name": "...",
  "date_of_birth": "DD/MM/YYYY or visible handwritten form",
  "age_years": "...",
  "place_of_birth": "...",
  "nature_of_age_proof": "...",
  "nationality": "...",
  "citizenship": "..."
}
Rules:
- Use row context, not label text.
- For gender, answer the selected option from the marked row only.
- For marital_status, return the handwritten category only.
- If a field is blank, return null.""",
        "page2_address": """Read this LIC Form 300 page and return ONLY valid JSON.
Extract these fields exactly as written by hand:
{
  "address_line1": "...",
  "address_city": "...",
  "address_state_country": "...",
  "address_pincode": "...",
  "mobile_number": "...",
  "email": "...",
  "current_address_same": "same as above/different/null"
}
Rules:
- Return handwritten values only.
- For pincode and mobile, preserve only the visible handwritten digits.
- If email is blank on this page, return null.""",
        "page3_kyc": """Read this LIC Form 300 KYC page and return ONLY valid JSON.
Extract:
{
  "pan_number": "...",
  "proof_of_identity_selected": "Aadhar/Driving Licence/Voter ID/Passport/null",
  "id_number_value": "...",
  "id_number_is_last4_only": true,
  "education": "...",
  "present_occupation": "...",
  "source_of_income": "...",
  "employer_name": "...",
  "nature_of_duties": "...",
  "length_of_service": "...",
  "annual_income": "..."
}
Rules:
- PAN is from the PAN row only, not from the ID row.
- Determine which proof-of-identity option is selected.
- If Aadhar is selected and only last 4 digits are written, return those digits exactly and set id_number_is_last4_only=true.
- If a field is blank, return null.""",
        "page6_nominee": """Read this LIC Form 300 nominee page and return ONLY valid JSON.
Extract:
{
  "nominee_name": "...",
  "nominee_share": "...",
  "nominee_age": "...",
  "nominee_relationship": "...",
  "appointee_name": "...",
  "appointee_id_number": "...",
  "mobile_of_life_assured": "...",
  "email_of_life_assured": "..."
}
Rules:
- Use the nominee row for nominee fields.
- If a value is blank, return null.
- Do not copy printed labels.""",
        "page7_plan": """Read this LIC Form 300 proposed plan page and return ONLY valid JSON.
Extract:
{
  "plan_name": "...",
  "policy_term": "...",
  "sum_assured": "...",
  "premium_amount": "..."
}
Rules:
- Use handwritten entries only from the plan details table.
- If the table cells are blank, return null.
- Do not return printed section headers like LIC or Proposed Plan Details as values.""",
        "page10_medical": """Read this LIC Form 300 medical page and return ONLY valid JSON.
Extract:
{
  "med_q1_heart": "Yes/No/null"
}
Rules:
- Base the answer on the cardiovascular/heart disease mark only.
- If no clear selection is visible, return null.""",
        "page14_declaration": """Read this LIC Form 300 declaration page and return ONLY valid JSON.
Extract:
{
  "date_of_proposal": "DD/MM/YYYY or visible handwritten form"
}
Rules:
- Return only the handwritten date value.
- If blank, return null.""",
    }


def override_jobs() -> Iterable[tuple[str, int, Tuple[float, float, float, float], str]]:
    yield (
        "page2_gender_override",
        2,
        (0.04, 0.23, 0.98, 0.39),
        """Read this LIC Form 300 crop and return ONLY valid JSON:
{
  "gender_selected": "Male/Female/Transgender/null",
  "marital_status": "..."
}
Rules:
- Gender must come from the selected mark in the gender row only.
- Marital status must come from the handwritten value in the row below.
- If unclear, return null.""",
    )
    yield (
        "page3_pan_override",
        3,
        (0.45, 0.37, 0.98, 0.50),
        """Read this LIC Form 300 crop and return ONLY valid JSON:
{
  "pan_number": "..."
}
Rules:
- Read the handwritten PAN only from the PAN row.
- Preserve letters and digits exactly.
- If unclear, return null.""",
    )
    yield (
        "page3_proof_override",
        3,
        (0.32, 0.48, 0.98, 0.74),
        """Read this LIC Form 300 crop and return ONLY valid JSON:
{
  "proof_of_identity_selected": "Aadhar/Driving Licence/Voter ID/Passport/null",
  "id_number_value": "...",
  "id_number_is_last4_only": true
}
Rules:
- Return the selected proof option only, not all printed options.
- Return the handwritten ID number exactly as visible.
- If only the last 4 digits are written, set id_number_is_last4_only=true.""",
    )


def build_final_result(groups: Dict[str, Dict]) -> Dict[str, object]:
    p2_id = groups.get("page2_identity", {})
    p2_addr = groups.get("page2_address", {})
    p3 = groups.get("page3_kyc", {})
    p6 = groups.get("page6_nominee", {})
    p7 = groups.get("page7_plan", {})
    p10 = groups.get("page10_medical", {})
    p14 = groups.get("page14_declaration", {})
    p2_override = groups.get("page2_gender_override", {})
    p3_pan_override = groups.get("page3_pan_override", {})
    p3_proof_override = groups.get("page3_proof_override", {})

    gender = normalize_gender(p2_override.get("gender_selected")) or normalize_gender(p2_id.get("gender"))
    proof_selected = normalize_text(p3_proof_override.get("proof_of_identity_selected")) or normalize_text(p3.get("proof_of_identity_selected"))
    marital_status = normalize_text(p2_override.get("marital_status")) or normalize_text(p2_id.get("marital_status"))
    pan_value = normalize_pan(p3_pan_override.get("pan_number")) or normalize_pan(p3.get("pan_number"))
    aadhaar_value = normalize_number(p3_proof_override.get("id_number_value")) or normalize_number(p3.get("id_number_value"))
    aadhaar_last4 = p3_proof_override.get("id_number_is_last4_only")
    if aadhaar_last4 is None:
        aadhaar_last4 = p3.get("id_number_is_last4_only")

    # Anti-hallucination gating for Plan Details (Page 7 is sparse)
    plan_name = normalize_text(p7.get("plan_name"))
    sum_assured = normalize_text(p7.get("sum_assured"))
    premium = normalize_text(p7.get("premium_amount"))
    
    # If Plan Name is exactly "LIC" or "Proposed Plan", it's a hallucination
    if plan_name and plan_name.upper() in ["LIC", "PROPOSED PLAN DETAILS", "PROPOSED PLAN"]:
        plan_name = None
    if sum_assured and sum_assured == "0":
        sum_assured = None
        
    low_trust = []
    if not plan_name: low_trust.append("Plan_Name")
    if not sum_assured: low_trust.append("Sum_Assured")
    if not normalize_text(p14.get("date_of_proposal")): low_trust.append("Date_of_Proposal")
    if not normalize_number(p2_addr.get("mobile_number")): low_trust.append("Phone")
    if not normalize_text(p2_addr.get("email")): low_trust.append("Email")
    if p10.get("med_q1_heart") is None: low_trust.append("Med_Q1_Heart")

    return {
        "First_Name": normalize_text(p2_id.get("first_name")),
        "Last_Name": normalize_text(p2_id.get("last_name")),
        "Date_of_Birth": normalize_text(p2_id.get("date_of_birth")),
        "Age": normalize_number(p2_id.get("age_years")),
        "Gender_Male": True if gender == "Male" else False if gender else None,
        "Gender_Female": True if gender == "Female" else False if gender else None,
        "Marital_Status": marital_status,
        "Address_Line1": normalize_text(p2_addr.get("address_line1")),
        "Pincode": normalize_number(p2_addr.get("address_pincode")),
        "Phone": normalize_number(p2_addr.get("mobile_number")),
        "PAN": pan_value,
        "Aadhaar": aadhaar_value if proof_selected and "aad" in proof_selected.lower() else None,
        "Email": normalize_text(p2_addr.get("email")) or normalize_text(p6.get("email_of_life_assured")),
        "Plan_Name": plan_name,
        "Policy_Term": normalize_number(p7.get("policy_term")),
        "Sum_Assured": sum_assured,
        "Premium_Amount": premium,
        "Date_of_Proposal": normalize_text(p14.get("date_of_proposal")),
        "Nominee_Name": normalize_text(p6.get("nominee_name")),
        "Nominee_Relationship": normalize_text(p6.get("nominee_relationship")),
        "Nominee_Age": normalize_number(p6.get("nominee_age")),
        "Med_Q1_Heart": normalize_text(p10.get("med_q1_heart")),
        "_low_trust_flags": low_trust,
        "_derived_context": {
            "gender": gender,
            "proof_of_identity_selected": proof_selected,
            "id_number_is_last4_only": aadhaar_last4,
            "middle_name": normalize_text(p2_id.get("middle_name")),
            "father_full_name": normalize_text(p2_id.get("father_full_name")),
            "mother_full_name": normalize_text(p2_id.get("mother_full_name")),
            "spouse_full_name": normalize_text(p2_id.get("spouse_full_name")),
            "place_of_birth": normalize_text(p2_id.get("place_of_birth")),
            "nature_of_age_proof": normalize_text(p2_id.get("nature_of_age_proof")),
            "nationality": normalize_text(p2_id.get("nationality")),
            "citizenship": normalize_text(p2_id.get("citizenship")),
            "address_city": normalize_text(p2_addr.get("address_city")),
            "address_state_country": normalize_text(p2_addr.get("address_state_country")),
            "current_address_same": normalize_text(p2_addr.get("current_address_same")),
            "education": normalize_text(p3.get("education")),
            "present_occupation": normalize_text(p3.get("present_occupation")),
            "source_of_income": normalize_text(p3.get("source_of_income")),
            "employer_name": normalize_text(p3.get("employer_name")),
            "nature_of_duties": normalize_text(p3.get("nature_of_duties")),
            "length_of_service": normalize_text(p3.get("length_of_service")),
            "annual_income": normalize_text(p3.get("annual_income")),
            "nominee_share": normalize_text(p6.get("nominee_share")),
            "appointee_name": normalize_text(p6.get("appointee_name")),
            "appointee_id_number": normalize_text(p6.get("appointee_id_number")),
            "mobile_of_life_assured": normalize_text(p6.get("mobile_of_life_assured")),
            "email_of_life_assured": normalize_text(p6.get("email_of_life_assured")),
        },
    }


def iter_group_jobs() -> Iterable[tuple[str, int]]:
    yield "page2_identity", 2
    yield "page2_address", 2
    yield "page3_kyc", 3
    yield "page6_nominee", 6
    yield "page7_plan", 7
    yield "page10_medical", 10
    yield "page14_declaration", 14


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    output_path = Path(args.output) if args.output else pdf_path.with_name(f"{pdf_path.stem}_advanced_grouped_qwen_22.json")

    prompts = cluster_prompts()
    manager = ModelManager()
    extractor = QwenExtractor(manager)
    groups: Dict[str, Dict] = {}

    try:
        for group_name, page_num in iter_group_jobs():
            print(f"Running {group_name} on page {page_num}...")
            image = render_page(pdf_path, page_num)
            image = crop_region(image, GROUP_CROP_BOXES[group_name])
            groups[group_name] = ask_qwen(extractor, image, prompts[group_name])
            print(json.dumps(groups[group_name], ensure_ascii=False))

        for group_name, page_num, box_norm, prompt in override_jobs():
            print(f"Running {group_name} on page {page_num}...")
            image = render_page(pdf_path, page_num)
            image = crop_region(image, box_norm)
            groups[group_name] = ask_qwen(extractor, image, prompt)
            print(json.dumps(groups[group_name], ensure_ascii=False))

        final_result = build_final_result(groups)
        payload = {
            "pdf": pdf_path.name,
            "groups": groups,
            "final_22": final_result,
        }
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved to: {output_path}")
    finally:
        manager.unload_all()


if __name__ == "__main__":
    main()
