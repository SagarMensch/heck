"""Focused field taxonomy for hackathon-style routing and stable zone generation.

This module intentionally narrows the problem to a small, high-value field set.
It also keeps page routing honest: fields are mapped to the pages/templates that
exist in this codebase's canonical Form 300 templates, not to aspirational
groupings that the current repo cannot yet support.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class FocusFieldSpec:
    public_id: str
    template_field: Optional[str]
    actual_page_num: Optional[int]
    route_group: str
    field_type: str
    use_case: str
    supported: bool = True
    notes: str = ""
    aliases: Tuple[str, ...] = ()
    prompt_variants: Tuple[str, ...] = ()
    search_pad: Tuple[int, int, int] = (120, 120, 180)  # x, top, bottom
    min_size: Tuple[int, int] = (220, 72)  # width, height


SURGICAL_25_FIELDS: Tuple[FocusFieldSpec, ...] = (
    FocusFieldSpec(
        public_id="First_Name",
        template_field="first_name",
        actual_page_num=2,
        route_group="identity_kyc",
        field_type="name_text",
        use_case="Core handwritten text",
        aliases=("first_name", "first name", "proposer_first_name"),
        prompt_variants=(
            "Find the handwritten value for First Name only. It is in the Name row, left of Middle Name and left of Last Name. Do not include Middle Name.",
            "Locate only the first handwritten name token in this local form crop. It is in the left sub-cell of the Name row.",
            "Ground the writable region for First Name only. Exclude Middle Name, Last Name, and the Father's Name row below.",
        ),
        min_size=(260, 80),
    ),
    FocusFieldSpec(
        public_id="Last_Name",
        template_field="last_name",
        actual_page_num=2,
        route_group="identity_kyc",
        field_type="name_text",
        use_case="Core handwritten text",
        aliases=("last_name", "last name", "surname", "proposer_last_name"),
        prompt_variants=(
            "Find the handwritten value for Last Name only. It is on the right side of the Name row, right of Middle Name.",
            "Locate only the far-right Last Name writable area in the Name row. Do not include Middle Name.",
            "Ground the surname area only. It is the rightmost sub-cell in the Name row, above Father's Full Name.",
        ),
        min_size=(220, 80),
    ),
    FocusFieldSpec(
        public_id="Date_of_Birth",
        template_field="date_of_birth",
        actual_page_num=2,
        route_group="identity_kyc",
        field_type="date",
        use_case="Strict date formatting DD/MM/YYYY",
        aliases=("date_of_birth", "date of birth", "dob", "proposer_dob"),
        prompt_variants=(
            "Find the handwritten Date of Birth only. It is left of Age/Years in the Date of Birth row.",
            "Locate the handwritten date with digits and slashes in the Date of Birth row only. Exclude the Age row.",
            "Ground the DOB writable area only. It sits above Age/Years and below Spouse Full Name.",
        ),
        min_size=(260, 76),
    ),
    FocusFieldSpec(
        public_id="Age",
        template_field="age_years",
        actual_page_num=2,
        route_group="identity_kyc",
        field_type="numeric",
        use_case="Numerical extraction",
        aliases=("age", "age_years", "age years"),
        prompt_variants=(
            "Find the handwritten Age only. It is in the Age/Years cell to the right of Date of Birth.",
            "Locate the small numeric age field in the row below Date of Birth.",
        ),
        min_size=(170, 72),
    ),
    FocusFieldSpec(
        public_id="Gender_Male",
        template_field="gender_mark",
        actual_page_num=2,
        route_group="identity_kyc",
        field_type="checkbox",
        use_case="Checkbox detection",
        aliases=("gender_male", "male", "gender male"),
        prompt_variants=(
            "Find the selected gender mark area in the Gender row. Include the check or handwritten mark only.",
            "Locate the marked selection in the Gender row. Do not include the Marital Status row below.",
        ),
        min_size=(220, 64),
    ),
    FocusFieldSpec(
        public_id="Gender_Female",
        template_field="gender_mark",
        actual_page_num=2,
        route_group="identity_kyc",
        field_type="checkbox",
        use_case="Checkbox detection",
        aliases=("gender_female", "female", "gender female"),
        prompt_variants=(
            "Find the selected gender mark area in the Gender row. Include the check or handwritten mark only.",
            "Locate the marked selection in the Gender row. Do not include the Marital Status row below.",
        ),
        min_size=(220, 64),
    ),
    FocusFieldSpec(
        public_id="Marital_Status",
        template_field="marital_status",
        actual_page_num=2,
        route_group="identity_kyc",
        field_type="short_text",
        use_case="Categorical handwritten text",
        aliases=("marital_status", "marital status"),
        prompt_variants=(
            "Find the handwritten Marital Status only. It is in the marital status row, below Gender and above Spouse Full Name.",
            "Locate only the Marital Status answer area in its own row. Exclude the Gender row above and Spouse row below.",
            "Ground the marital status writable region only. It should contain a short handwritten word, not the gender marks.",
        ),
        min_size=(320, 72),
    ),
    FocusFieldSpec(
        public_id="Address_Line1",
        template_field="address_line1",
        actual_page_num=2,
        route_group="identity_kyc",
        field_type="long_text",
        use_case="Long-form alphanumeric text",
        aliases=("address_line1", "address line1", "address line 1"),
        prompt_variants=(
            "Find the handwritten Address Line 1 only. It is the first address row in the Permanent Address section.",
            "Locate the full writable area for the first address line. Exclude City, State, and PIN rows below.",
        ),
        search_pad=(140, 140, 160),
        min_size=(700, 90),
    ),
    FocusFieldSpec(
        public_id="Pincode",
        template_field="address_pincode",
        actual_page_num=2,
        route_group="identity_kyc",
        field_type="numeric",
        use_case="Strict 6-digit constraint",
        aliases=("pincode", "pin", "pin code", "address_pincode"),
        prompt_variants=(
            "Find the handwritten PIN Code only. It is the left value cell in the PIN Code row.",
            "Locate the six-digit PIN Code area only. Exclude the mobile or telephone row.",
        ),
        min_size=(220, 72),
    ),
    FocusFieldSpec(
        public_id="Phone",
        template_field="mobile_number",
        actual_page_num=2,
        route_group="identity_kyc",
        field_type="numeric",
        use_case="Strict 10-digit constraint",
        aliases=("phone", "mobile", "mobile_number", "phone number"),
        prompt_variants=(
            "Find the handwritten mobile or phone number only. It is the right-side value cell paired with PIN Code.",
            "Locate the phone number writable area only. Exclude the PIN code cell on the left.",
        ),
        min_size=(260, 72),
    ),
    FocusFieldSpec(
        public_id="PAN",
        template_field="pan_number",
        actual_page_num=3,
        route_group="identity_kyc",
        field_type="short_id",
        use_case="Regex [A-Z]{5}[0-9]{4}[A-Z]",
        aliases=("pan", "pan_number", "pan number"),
        prompt_variants=(
            "Find the PAN Number only. It is an alphanumeric identifier in the KYC page.",
            "Locate the PAN writable area only. Expect a short uppercase alphanumeric code.",
        ),
        min_size=(280, 84),
    ),
    FocusFieldSpec(
        public_id="Aadhaar",
        template_field="aadhaar_last_or_id_number",
        actual_page_num=3,
        route_group="identity_kyc",
        field_type="short_id",
        use_case="Numeric regex [0-9]{12}",
        aliases=("aadhaar", "aadhaar number", "aadhaar_number", "uid"),
        prompt_variants=(
            "Find the Aadhaar or ID Number only on this KYC page.",
            "Locate the Aadhaar writable area only. Expect a numeric identifier.",
        ),
        min_size=(280, 84),
    ),
    FocusFieldSpec(
        public_id="Email",
        template_field="email",
        actual_page_num=2,
        route_group="identity_kyc",
        field_type="short_id",
        use_case="Email constraint",
        aliases=("email", "email_id", "email address"),
        prompt_variants=(
            "Find the Email only. It is the email row near the lower part of page 2.",
            "Locate the email writable area only. Expect letters, numbers, @ and dots.",
        ),
        search_pad=(120, 100, 140),
        min_size=(420, 78),
    ),
    FocusFieldSpec(
        public_id="Plan_Name",
        template_field="proposed_plan_name",
        actual_page_num=7,
        route_group="financial",
        field_type="short_text",
        use_case="Fuzzy dictionary match against LIC product catalog",
        aliases=("plan_name", "plan name", "proposed_plan_name"),
        search_pad=(140, 140, 140),
        min_size=(300, 90),
    ),
    FocusFieldSpec(
        public_id="Policy_Term",
        template_field="proposed_plan_term",
        actual_page_num=7,
        route_group="financial",
        field_type="numeric",
        use_case="Small numerical constraint",
        aliases=("policy_term", "policy term", "proposed_plan_term"),
        min_size=(180, 82),
    ),
    FocusFieldSpec(
        public_id="Sum_Assured",
        template_field="proposed_sum_assured",
        actual_page_num=7,
        route_group="financial",
        field_type="amount",
        use_case="High-value currency formatting",
        aliases=("sum_assured", "sum assured", "proposed_sum_assured"),
        min_size=(260, 88),
    ),
    FocusFieldSpec(
        public_id="Premium_Amount",
        template_field="proposed_premium_amount",
        actual_page_num=7,
        route_group="financial",
        field_type="amount",
        use_case="High-value currency formatting",
        aliases=("premium_amount", "premium amount", "proposed_premium_amount"),
        min_size=(260, 88),
    ),
    FocusFieldSpec(
        public_id="Date_of_Proposal",
        template_field="date_of_proposal",
        actual_page_num=14,
        route_group="financial",
        field_type="date",
        use_case="Date extraction and validation against current year",
        aliases=("date_of_proposal", "date of proposal"),
        search_pad=(120, 120, 120),
        min_size=(260, 80),
    ),
    FocusFieldSpec(
        public_id="Bank_Name",
        template_field=None,
        actual_page_num=None,
        route_group="financial",
        field_type="short_text",
        use_case="Fuzzy match against RBI recognized bank list",
        supported=False,
        notes="Current canonical templates in this repo do not define a bank-name field zone yet.",
        aliases=("bank_name", "bank name"),
    ),
    FocusFieldSpec(
        public_id="Bank_Account_Number",
        template_field=None,
        actual_page_num=None,
        route_group="financial",
        field_type="numeric",
        use_case="Variable length numeric",
        supported=False,
        notes="Current canonical templates in this repo do not define a bank-account field zone yet.",
        aliases=("bank_account_number", "bank account number"),
    ),
    FocusFieldSpec(
        public_id="Bank_IFSC",
        template_field=None,
        actual_page_num=None,
        route_group="financial",
        field_type="short_id",
        use_case="Strict 11-char alphanumeric regex",
        supported=False,
        notes="Current canonical templates in this repo do not define an IFSC field zone yet.",
        aliases=("bank_ifsc", "ifsc", "ifsc code"),
    ),
    FocusFieldSpec(
        public_id="Nominee_Name",
        template_field="nominee_name",
        actual_page_num=6,
        route_group="nominee_underwriting",
        field_type="name_text",
        use_case="Secondary entity name",
        aliases=("nominee_name", "nominee name"),
        min_size=(260, 78),
    ),
    FocusFieldSpec(
        public_id="Nominee_Relationship",
        template_field="nominee_relationship",
        actual_page_num=6,
        route_group="nominee_underwriting",
        field_type="short_text",
        use_case="Categorical text like Son or Wife",
        aliases=("nominee_relationship", "nominee relationship"),
        min_size=(220, 78),
    ),
    FocusFieldSpec(
        public_id="Nominee_Age",
        template_field="nominee_age",
        actual_page_num=6,
        route_group="nominee_underwriting",
        field_type="numeric",
        use_case="Cross-field validation nominee age",
        aliases=("nominee_age", "nominee age"),
        min_size=(150, 74),
    ),
    FocusFieldSpec(
        public_id="Med_Q1_Heart",
        template_field="cardio_disease_mark",
        actual_page_num=10,
        route_group="nominee_underwriting",
        field_type="checkbox",
        use_case="Critical underwriting checkbox",
        aliases=("med_q1_heart", "heart", "cardio_disease_mark"),
        min_size=(120, 120),
        notes="Mapped to the current canonical cardiovascular disease mark on page 10, not page 4.",
    ),
)


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.strip().lower())


_LOOKUP: Dict[str, FocusFieldSpec] = {}
for _spec in SURGICAL_25_FIELDS:
    keys = [_spec.public_id]
    if _spec.template_field:
        keys.append(_spec.template_field)
    keys.extend(_spec.aliases)
    for _key in keys:
        _LOOKUP[_normalize_key(_key)] = _spec


def get_focus_field(name: str) -> Optional[FocusFieldSpec]:
    return _LOOKUP.get(_normalize_key(name))


def iter_focus_fields(supported_only: bool = False) -> Iterable[FocusFieldSpec]:
    for spec in SURGICAL_25_FIELDS:
        if supported_only and not spec.supported:
            continue
        yield spec


def focus_fields_for_page(page_num: int, supported_only: bool = False) -> List[FocusFieldSpec]:
    return [
        spec
        for spec in iter_focus_fields(supported_only=supported_only)
        if spec.actual_page_num == page_num
    ]


def taxonomy_summary() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for spec in SURGICAL_25_FIELDS:
        rows.append(
            {
                "public_id": spec.public_id,
                "template_field": spec.template_field,
                "actual_page_num": spec.actual_page_num,
                "route_group": spec.route_group,
                "field_type": spec.field_type,
                "supported": spec.supported,
                "notes": spec.notes,
            }
        )
    return rows


def supported_focus_public_ids() -> List[str]:
    return [spec.public_id for spec in SURGICAL_25_FIELDS if spec.supported]
