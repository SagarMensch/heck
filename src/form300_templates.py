"""Canonical LIC Form 300 page templates.

Complete page taxonomy for the 28-page LIC Proposal Form No. 300 (Rv.2023).
Bilingual Hindi+English printed template with handwritten fills.

Template geometry is highly stable across files, making normalized field boxes
a reliable CPU-side bootstrap for crop extraction.

Pages with target extraction fields:
  1  = cover_page           (office use: proposal no, receipt date, deposit)
  2  = proposer_details     (name, dob, age, gender, address, parents)
  3  = kyc_occupation       (PAN, Aadhaar, occupation, income)
  5  = existing_policies    (policy_number, insurer, plan_term, sum_assured, premium)
  6  = nominee_details      (nominee name, relationship, age, address, appointee)
  7  = plan_details         (plan name, term, sum assured, premium, mode)
 10  = health_habits        (height, weight, disease checkboxes)
 16  = agent_details        (agent code, agent name, branch code)
 28  = suitability_last     (preferred plan, term, SA, mode, premium)

Other pages (4,8,9,11-15,17-27) contain medical declarations, legal text,
plan-specific addenda, and supplementary info — mostly checkboxes and boilerplate.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


CANONICAL_PAGE_SIZE = (1024, 1400)


@dataclass(frozen=True)
class FieldTemplate:
    name: str
    family: str
    page_type: str
    page_ref: str
    bbox_norm: Tuple[float, float, float, float]
    multiline: bool = False
    max_lines: int = 1
    required: bool = False
    renderer: str = "text"
    notes: str = ""


@dataclass(frozen=True)
class PageTemplate:
    page_type: str
    page_ref: str
    fields: Tuple[FieldTemplate, ...]


def _f(
    name: str,
    family: str,
    page_type: str,
    page_ref: str,
    bbox_norm: Tuple[float, float, float, float],
    multiline: bool = False,
    max_lines: int = 1,
    required: bool = False,
    renderer: str = "text",
    notes: str = "",
) -> FieldTemplate:
    return FieldTemplate(
        name=name,
        family=family,
        page_type=page_type,
        page_ref=page_ref,
        bbox_norm=bbox_norm,
        multiline=multiline,
        max_lines=max_lines,
        required=required,
        renderer=renderer,
        notes=notes,
    )


# ──────────────────────────── PAGE 1: COVER ────────────────────────────

COVER_PAGE_FIELDS = (
    _f("office_inward_no", "proposal_number", "cover_page", "1", (0.12, 0.905, 0.31, 0.94)),
    _f("office_proposal_no", "proposal_number", "cover_page", "1", (0.19, 0.932, 0.36, 0.967), required=True),
    _f("office_receipt_date", "date", "cover_page", "1", (0.70, 0.905, 0.82, 0.94)),
    _f("office_deposit_amount", "amount", "cover_page", "1", (0.84, 0.932, 0.985, 0.967), required=True),
)


# ──────────────────────────── PAGE 2: PROPOSER DETAILS ────────────────────────────

PROPOSER_DETAILS_FIELDS = (
    _f("customer_id", "short_id", "proposer_details", "2", (0.4839, 0.1425, 0.9573, 0.1596), required=True),
    _f("ckyc_number", "short_id", "proposer_details", "2", (0.4839, 0.1596, 0.9573, 0.1795), required=True),
    _f("first_name", "name_text", "proposer_details", "2", (0.5645, 0.2081, 0.7661, 0.2337), required=True),
    _f("middle_name", "name_text", "proposer_details", "2", (0.7661, 0.2081, 0.8468, 0.2337)),
    _f("last_name", "name_text", "proposer_details", "2", (0.8468, 0.2081, 0.9573, 0.2337), required=True),
    _f("father_full_name", "name_text", "proposer_details", "2", (0.4839, 0.2337, 0.9573, 0.2559), required=True),
    _f("mother_full_name", "name_text", "proposer_details", "2", (0.4839, 0.2770, 0.9573, 0.2935)),
    _f("gender_mark", "binary_mark", "proposer_details", "2", (0.5645, 0.2981, 0.8306, 0.3135), renderer="mark", required=True),
    _f("marital_status", "short_text", "proposer_details", "2", (0.4839, 0.3163, 0.9573, 0.3340)),
    _f("spouse_full_name", "name_text", "proposer_details", "2", (0.4839, 0.3363, 0.9573, 0.3539)),
    _f("date_of_birth", "date", "proposer_details", "2", (0.4839, 0.3562, 0.7258, 0.3750), required=True),
    _f("age_years", "numeric", "proposer_details", "2", (0.7258, 0.3562, 0.8468, 0.3750), required=True),
    _f("place_of_birth", "short_text", "proposer_details", "2", (0.4839, 0.4275, 0.9573, 0.4457)),
    _f("nature_of_age_proof", "short_text", "proposer_details", "2", (0.4839, 0.4469, 0.9573, 0.4788)),
    _f("nationality", "short_text", "proposer_details", "2", (0.4839, 0.4816, 0.9573, 0.4984), required=True),
    _f("citizenship", "short_text", "proposer_details", "2", (0.4839, 0.5001, 0.9573, 0.5172), required=True),
    _f("address_line1", "long_text", "proposer_details", "2", (0.4839, 0.5986, 0.9573, 0.6271), multiline=True, max_lines=1, required=True),
    _f("address_line2", "long_text", "proposer_details", "2", (0.4839, 0.6271, 0.9573, 0.6499), multiline=True, max_lines=1),
    _f("address_city", "short_text", "proposer_details", "2", (0.4839, 0.6499, 0.7258, 0.6727), required=True),
    _f("address_state_country", "short_text", "proposer_details", "2", (0.7258, 0.6499, 0.9573, 0.6727), required=True),
    _f("address_pincode", "numeric", "proposer_details", "2", (0.4839, 0.6727, 0.6452, 0.6927), required=True),
    _f("current_address_same", "short_text", "proposer_details", "2", (0.4839, 0.7119, 0.9573, 0.7310)),
    _f("mobile_number", "numeric", "proposer_details", "2", (0.7258, 0.6727, 0.9573, 0.6927), required=True),
    _f("email", "short_id", "proposer_details", "2", (0.4839, 0.8181, 0.9573, 0.8381)),
)


# ──────────────────────────── PAGE 3: KYC / OCCUPATION ────────────────────────────

KYC_OCCUPATION_FIELDS = (
    _f("residential_status_mark", "binary_mark", "kyc_occupation", "3", (0.91, 0.114, 0.98, 0.146), renderer="mark"),
    _f("income_tax_assessee_mark", "binary_mark", "kyc_occupation", "3", (0.91, 0.283, 0.98, 0.319), renderer="mark"),
    _f("pan_number", "short_id", "kyc_occupation", "3", (0.56, 0.365, 0.92, 0.430), required=True),
    _f("gst_registered_mark", "binary_mark", "kyc_occupation", "3", (0.60, 0.425, 0.84, 0.485), renderer="mark"),
    _f("aadhaar_last_or_id_number", "short_id", "kyc_occupation", "3", (0.55, 0.516, 0.92, 0.584), required=True),
    _f("id_expiry_or_na", "date", "kyc_occupation", "3", (0.60, 0.575, 0.92, 0.627)),
    _f("address_proof_submitted_mark", "binary_mark", "kyc_occupation", "3", (0.60, 0.620, 0.86, 0.668), renderer="mark"),
    _f("education", "short_text", "kyc_occupation", "3", (0.60, 0.720, 0.92, 0.755)),
    _f("present_occupation", "short_text", "kyc_occupation", "3", (0.60, 0.752, 0.92, 0.786), required=True),
    _f("source_of_income", "short_text", "kyc_occupation", "3", (0.60, 0.784, 0.92, 0.818)),
    _f("employer_name", "short_text", "kyc_occupation", "3", (0.60, 0.815, 0.92, 0.850)),
    _f("nature_of_duties", "short_text", "kyc_occupation", "3", (0.60, 0.845, 0.92, 0.882)),
    _f("length_of_service", "numeric", "kyc_occupation", "3", (0.60, 0.878, 0.78, 0.914)),
    _f("annual_income", "amount", "kyc_occupation", "3", (0.60, 0.911, 0.86, 0.946), required=True),
)


# ──────────────────────────── PAGE 5: EXISTING POLICIES ────────────────────────────

EXISTING_POLICIES_FIELDS = (
    _f("existing_policy_number", "short_id", "existing_policies", "5", (0.4899, 0.3876, 0.5798, 0.4389), required=False),
    _f("existing_insurer_name", "short_text", "existing_policies", "5", (0.5839, 0.3876, 0.6742, 0.4389)),
    _f("existing_plan_term", "short_text", "existing_policies", "5", (0.6782, 0.3876, 0.7710, 0.4389)),
    _f("existing_sum_assured", "amount", "existing_policies", "5", (0.7750, 0.3876, 0.8661, 0.4389)),
    _f("existing_premium", "amount", "existing_policies", "5", (0.8710, 0.3876, 0.9597, 0.4389)),
)


# ──────────────────────────── PAGE 6: NOMINEE DETAILS ────────────────────────────

NOMINEE_DETAILS_FIELDS = (
    _f("nominee_name", "name_text", "nominee_details", "6", (0.4871, 0.3528, 0.6403, 0.3705), required=True),
    _f("nominee_relationship", "short_text", "nominee_details", "6", (0.6444, 0.3528, 0.7903, 0.3705), required=True),
    _f("nominee_age", "numeric", "nominee_details", "6", (0.7952, 0.3528, 0.8823, 0.3705), required=True),
    _f("nominee_address", "long_text", "nominee_details", "6", (0.4871, 0.3853, 0.8823, 0.4315), multiline=True, max_lines=2),
    _f("appointee_name", "name_text", "nominee_details", "6", (0.4871, 0.4554, 0.6403, 0.4816)),
    _f("appointee_relationship", "short_text", "nominee_details", "6", (0.6444, 0.4554, 0.7903, 0.4816)),
)


# ──────────────────────────── PAGE 7: PLAN DETAILS ────────────────────────────

PLAN_DETAILS_FIELDS = (
    _f("proposed_plan_name", "short_text", "plan_details", "7", (0.4532, 0.3545, 0.7129, 0.4047), required=True),
    _f("proposed_plan_term", "numeric", "plan_details", "7", (0.7177, 0.3545, 0.8395, 0.4047), required=True),
    _f("proposed_sum_assured", "amount", "plan_details", "7", (0.4532, 0.4246, 0.7129, 0.4959), required=True),
    _f("proposed_premium_paying_term", "numeric", "plan_details", "7", (0.7177, 0.4246, 0.8395, 0.4959), required=True),
    _f("proposed_premium_amount", "amount", "plan_details", "7", (0.4532, 0.5130, 0.7129, 0.5392), required=True),
    _f("proposed_premium_mode", "short_text", "plan_details", "7", (0.7177, 0.5130, 0.8395, 0.5392), required=True),
    _f("objective_of_insurance", "short_text", "plan_details", "7", (0.4532, 0.5420, 0.9016, 0.6025)),
)


# ──────────────────────────── PAGE 10: HEALTH / HABITS ────────────────────────────

HEALTH_FIELDS = (
    _f("height_cm", "numeric", "health_habits", "10", (0.83, 0.040, 0.90, 0.083), required=True),
    _f("weight_kg", "numeric", "health_habits", "10", (0.92, 0.040, 0.99, 0.083), required=True),
    _f("medical_consultation_mark", "binary_mark", "health_habits", "10", (0.92, 0.150, 0.985, 0.196), renderer="mark"),
    _f("hospital_admission_mark", "binary_mark", "health_habits", "10", (0.92, 0.252, 0.985, 0.300), renderer="mark"),
    _f("health_absence_mark", "binary_mark", "health_habits", "10", (0.92, 0.352, 0.985, 0.399), renderer="mark"),
    _f("respiratory_disease_mark", "binary_mark", "health_habits", "10", (0.435, 0.622, 0.505, 0.714), renderer="mark"),
    _f("cardio_disease_mark", "binary_mark", "health_habits", "10", (0.925, 0.622, 0.992, 0.714), renderer="mark"),
    _f("digestive_disease_mark", "binary_mark", "health_habits", "10", (0.435, 0.714, 0.505, 0.805), renderer="mark"),
    _f("urinary_disease_mark", "binary_mark", "health_habits", "10", (0.925, 0.714, 0.992, 0.805), renderer="mark"),
    _f("neuro_disease_mark", "binary_mark", "health_habits", "10", (0.435, 0.805, 0.505, 0.924), renderer="mark"),
    _f("venereal_disease_mark", "binary_mark", "health_habits", "10", (0.925, 0.805, 0.992, 0.924), renderer="mark"),
)


# ──────────────────────────── PAGE 16: AGENT DETAILS ────────────────────────────

AGENT_DETAILS_FIELDS = (
    _f("agent_code", "short_id", "agent_details", "16", (0.50, 0.04, 0.78, 0.10)),
    _f("agent_name", "name_text", "agent_details", "16", (0.50, 0.12, 0.85, 0.18)),
    _f("branch_code", "short_id", "agent_details", "16", (0.50, 0.20, 0.78, 0.26)),
    _f("branch_name", "short_text", "agent_details", "16", (0.50, 0.28, 0.85, 0.34)),
    _f("development_officer_name", "name_text", "agent_details", "16", (0.50, 0.40, 0.85, 0.46)),
)


# ──────────────────────────── PAGE 28: SUITABILITY LAST ────────────────────────────

SUITABILITY_LAST_FIELDS = (
    _f("preferred_plan_name", "short_text", "suitability_last", "last", (0.634, 0.425, 0.818, 0.486)),
    _f("preferred_plan_term", "numeric", "suitability_last", "last", (0.818, 0.425, 0.953, 0.486)),
    _f("preferred_sum_assured", "amount", "suitability_last", "last", (0.634, 0.486, 0.738, 0.545)),
    _f("preferred_mode", "short_text", "suitability_last", "last", (0.738, 0.486, 0.836, 0.545)),
    _f("preferred_premium", "amount", "suitability_last", "last", (0.836, 0.486, 0.953, 0.545)),
)


# ──────────────────────────── PAGE 14: DECLARATION ────────────────────────────

DECLARATION_FIELDS = (
    _f("date_of_proposal", "date", "declaration_main", "14", (0.55, 0.82, 0.78, 0.87), required=True),
    _f("place_of_signing", "short_text", "declaration_main", "14", (0.30, 0.82, 0.55, 0.87), required=True),
    _f("proposer_signature_present", "signature_presence", "declaration_main", "14", (0.60, 0.88, 0.80, 0.96), renderer="mark", required=True),
)


# ──────────────────────────── PAGE 15: DECLARANT ────────────────────────────

DECLARANT_FIELDS = (
    _f("declarant_name", "name_text", "declaration_declarant", "15", (0.30, 0.40, 0.70, 0.48)),
    _f("declarant_address", "long_text", "declaration_declarant", "15", (0.30, 0.50, 0.70, 0.58)),
)


# ──────────────────────────── PAGE 13: SPOUSE / FEMALE HEALTH ────────────────────────────

SPOUSE_GYNEC_FIELDS = (
    _f("husband_full_name", "name_text", "spouse_gynec", "13", (0.50, 0.70, 0.88, 0.76)),
    _f("husband_occupation", "short_text", "spouse_gynec", "13", (0.50, 0.76, 0.88, 0.82)),
)


# ──────────────────────────── PAGE 18: SETTLEMENT ADDENDUM ────────────────────────────

SETTLEMENT_ADDENDUM_FIELDS = (
    _f("settlement_option_yn", "binary_mark", "settlement_addendum", "18", (0.60, 0.14, 0.70, 0.20), renderer="mark"),
    _f("settlement_period_years", "numeric", "settlement_addendum", "18", (0.60, 0.28, 0.80, 0.34)),
    _f("settlement_full_or_part", "short_text", "settlement_addendum", "18", (0.60, 0.40, 0.80, 0.46)),
    _f("settlement_instalment_mode", "short_text", "settlement_addendum", "18", (0.60, 0.54, 0.80, 0.60)),
)


# ──────────────────────────── MASTER REGISTRY ────────────────────────────

FORM300_PAGE_TEMPLATES: Dict[str, PageTemplate] = {
    "cover_page": PageTemplate("cover_page", "1", COVER_PAGE_FIELDS),
    "proposer_details": PageTemplate("proposer_details", "2", PROPOSER_DETAILS_FIELDS),
    "kyc_occupation": PageTemplate("kyc_occupation", "3", KYC_OCCUPATION_FIELDS),
    "existing_policies": PageTemplate("existing_policies", "5", EXISTING_POLICIES_FIELDS),
    "nominee_details": PageTemplate("nominee_details", "6", NOMINEE_DETAILS_FIELDS),
    "plan_details": PageTemplate("plan_details", "7", PLAN_DETAILS_FIELDS),
    "health_habits": PageTemplate("health_habits", "10", HEALTH_FIELDS),
    "spouse_gynec": PageTemplate("spouse_gynec", "13", SPOUSE_GYNEC_FIELDS),
    "declaration_main": PageTemplate("declaration_main", "14", DECLARATION_FIELDS),
    "declaration_declarant": PageTemplate("declaration_declarant", "15", DECLARANT_FIELDS),
    "agent_details": PageTemplate("agent_details", "16", AGENT_DETAILS_FIELDS),
    "settlement_addendum": PageTemplate("settlement_addendum", "18", SETTLEMENT_ADDENDUM_FIELDS),
    "suitability_last": PageTemplate("suitability_last", "last", SUITABILITY_LAST_FIELDS),
}

FORM300_FIELD_INDEX: Dict[str, FieldTemplate] = {
    field.name: field
    for template in FORM300_PAGE_TEMPLATES.values()
    for field in template.fields
}

PAGE_TYPE_TO_PAGE_NUM: Dict[str, int] = {
    "cover_page": 1,
    "proposer_details": 2,
    "kyc_occupation": 3,
    "existing_policies": 5,
    "nominee_details": 6,
    "plan_details": 7,
    "health_habits": 10,
    "spouse_gynec": 13,
    "declaration_main": 14,
    "declaration_declarant": 15,
    "agent_details": 16,
    "settlement_addendum": 18,
    "suitability_last": 28,
}

PAGES_WITH_TARGET_FIELDS = {1, 2, 3, 5, 6, 7, 10, 13, 14, 15, 16, 28}


def iter_page_templates() -> Iterable[PageTemplate]:
    return FORM300_PAGE_TEMPLATES.values()


def iter_fields() -> Iterable[FieldTemplate]:
    for template in iter_page_templates():
        yield from template.fields


def resolve_page_index(num_pages: int, page_ref: str) -> int:
    if page_ref == "last":
        return max(0, num_pages - 1)
    page_num = int(page_ref)
    return max(0, min(num_pages - 1, page_num - 1))


def bbox_to_pixels(
    bbox_norm: Tuple[float, float, float, float],
    width: int,
    height: int,
    pad_norm: float = 0.006,
) -> Tuple[int, int, int, int]:
    x1n, y1n, x2n, y2n = bbox_norm
    x_pad = int(width * pad_norm)
    y_pad = int(height * pad_norm)
    x1 = max(0, int(x1n * width) - x_pad)
    y1 = max(0, int(y1n * height) - y_pad)
    x2 = min(width, int(x2n * width) + x_pad)
    y2 = min(height, int(y2n * height) + y_pad)
    return x1, y1, x2, y2


def page_template_summary() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for template in iter_page_templates():
        rows.append(
            {
                "page_type": template.page_type,
                "page_ref": template.page_ref,
                "field_count": len(template.fields),
                "fields": [field.name for field in template.fields],
            }
        )
    return rows
