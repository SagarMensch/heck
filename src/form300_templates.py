"""
Canonical LIC Form 300 page templates.

The sample packs are visually stable enough that normalized field boxes are a
useful CPU-side bootstrap for:

- real crop harvesting
- synthetic crop generation
- full-page synthetic projection

These templates intentionally focus on the highest-value handwritten fields for
the Techathon rather than attempting to annotate every printed region.
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


COVER_PAGE_FIELDS = (
    _f("office_inward_no", "proposal_number", "cover_office_use", "1", (0.12, 0.905, 0.31, 0.94)),
    _f("office_proposal_no", "proposal_number", "cover_office_use", "1", (0.19, 0.932, 0.36, 0.967), required=True),
    _f("office_receipt_date", "date", "cover_office_use", "1", (0.70, 0.905, 0.82, 0.94)),
    _f("office_deposit_amount", "amount", "cover_office_use", "1", (0.84, 0.932, 0.985, 0.967), required=True),
)


PROPOSER_DETAILS_FIELDS = (
    _f("customer_id", "short_id", "proposer_details", "2", (0.50, 0.107, 0.89, 0.138), required=True),
    _f("ckyc_number", "short_id", "proposer_details", "2", (0.50, 0.140, 0.89, 0.173), required=True),
    _f("first_name", "name_text", "proposer_details", "2", (0.51, 0.181, 0.63, 0.233), required=True),
    _f("middle_name", "name_text", "proposer_details", "2", (0.63, 0.181, 0.79, 0.233)),
    _f("last_name", "name_text", "proposer_details", "2", (0.79, 0.181, 0.93, 0.233), required=True),
    _f("father_full_name", "name_text", "proposer_details", "2", (0.51, 0.236, 0.92, 0.273), required=True),
    _f("mother_full_name", "name_text", "proposer_details", "2", (0.51, 0.273, 0.92, 0.309)),
    _f("gender_mark", "binary_mark", "proposer_details", "2", (0.82, 0.307, 0.93, 0.346), renderer="mark", required=True),
    _f("marital_status", "short_text", "proposer_details", "2", (0.59, 0.344, 0.92, 0.383)),
    _f("spouse_full_name", "name_text", "proposer_details", "2", (0.58, 0.383, 0.92, 0.421)),
    _f("date_of_birth", "date", "proposer_details", "2", (0.64, 0.422, 0.92, 0.459), required=True),
    _f("age_years", "numeric", "proposer_details", "2", (0.69, 0.460, 0.81, 0.497), required=True),
    _f("place_of_birth", "short_text", "proposer_details", "2", (0.62, 0.551, 0.92, 0.588)),
    _f("nature_of_age_proof", "short_text", "proposer_details", "2", (0.61, 0.588, 0.92, 0.626)),
    _f("nationality", "short_text", "proposer_details", "2", (0.63, 0.628, 0.92, 0.664), required=True),
    _f("citizenship", "short_text", "proposer_details", "2", (0.63, 0.663, 0.92, 0.701), required=True),
    _f("address_line1", "long_text", "proposer_details", "2", (0.56, 0.788, 0.92, 0.822), multiline=True, max_lines=1, required=True),
    _f("address_line2", "long_text", "proposer_details", "2", (0.56, 0.818, 0.92, 0.849), multiline=True, max_lines=1),
    _f("address_city", "short_text", "proposer_details", "2", (0.56, 0.845, 0.92, 0.878), required=True),
    _f("address_state_country", "short_text", "proposer_details", "2", (0.56, 0.872, 0.92, 0.908), required=True),
    _f("address_pincode", "numeric", "proposer_details", "2", (0.68, 0.899, 0.92, 0.930), required=True),
    _f("current_address_same", "short_text", "proposer_details", "2", (0.72, 0.932, 0.92, 0.968)),
)


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


HEALTH_FIELDS = (
    _f("height_cm", "numeric", "health_details", "10", (0.83, 0.040, 0.90, 0.083), required=True),
    _f("weight_kg", "numeric", "health_details", "10", (0.92, 0.040, 0.99, 0.083), required=True),
    _f("medical_consultation_mark", "binary_mark", "health_details", "10", (0.92, 0.150, 0.985, 0.196), renderer="mark"),
    _f("hospital_admission_mark", "binary_mark", "health_details", "10", (0.92, 0.252, 0.985, 0.300), renderer="mark"),
    _f("health_absence_mark", "binary_mark", "health_details", "10", (0.92, 0.352, 0.985, 0.399), renderer="mark"),
    _f("respiratory_disease_mark", "binary_mark", "health_details", "10", (0.435, 0.622, 0.505, 0.714), renderer="mark"),
    _f("cardio_disease_mark", "binary_mark", "health_details", "10", (0.925, 0.622, 0.992, 0.714), renderer="mark"),
    _f("digestive_disease_mark", "binary_mark", "health_details", "10", (0.435, 0.714, 0.505, 0.805), renderer="mark"),
    _f("urinary_disease_mark", "binary_mark", "health_details", "10", (0.925, 0.714, 0.992, 0.805), renderer="mark"),
    _f("neuro_disease_mark", "binary_mark", "health_details", "10", (0.435, 0.805, 0.505, 0.924), renderer="mark"),
    _f("venereal_disease_mark", "binary_mark", "health_details", "10", (0.925, 0.805, 0.992, 0.924), renderer="mark"),
)


SUITABILITY_LAST_FIELDS = (
    _f("preferred_plan_name", "short_text", "suitability_last", "last", (0.634, 0.425, 0.818, 0.486)),
    _f("preferred_plan_term", "numeric", "suitability_last", "last", (0.818, 0.425, 0.953, 0.486)),
    _f("preferred_sum_assured", "amount", "suitability_last", "last", (0.634, 0.486, 0.738, 0.545)),
    _f("preferred_mode", "short_text", "suitability_last", "last", (0.738, 0.486, 0.836, 0.545)),
    _f("preferred_premium", "amount", "suitability_last", "last", (0.836, 0.486, 0.953, 0.545)),
)


FORM300_PAGE_TEMPLATES: Dict[str, PageTemplate] = {
    "cover_office_use": PageTemplate("cover_office_use", "1", COVER_PAGE_FIELDS),
    "proposer_details": PageTemplate("proposer_details", "2", PROPOSER_DETAILS_FIELDS),
    "kyc_occupation": PageTemplate("kyc_occupation", "3", KYC_OCCUPATION_FIELDS),
    "health_details": PageTemplate("health_details", "10", HEALTH_FIELDS),
    "suitability_last": PageTemplate("suitability_last", "last", SUITABILITY_LAST_FIELDS),
}


FORM300_FIELD_INDEX: Dict[str, FieldTemplate] = {
    field.name: field
    for template in FORM300_PAGE_TEMPLATES.values()
    for field in template.fields
}


def iter_page_templates() -> Iterable[PageTemplate]:
    return FORM300_PAGE_TEMPLATES.values()


def iter_fields() -> Iterable[FieldTemplate]:
    for template in iter_page_templates():
        yield from template.fields


def resolve_page_index(num_pages: int, page_ref: str) -> int:
    """
    Resolve a page reference to a zero-based page index.

    Supported values:
    - "1", "2", ...
    - "last"
    """
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
    """
    Convert normalized bbox to padded pixel bbox.
    """
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
