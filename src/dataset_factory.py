"""
CPU-side Form 300 dataset factory.

Builds:
- canonical background pages from real scans
- unlabeled real crops for gold / weak labeling
- synthetic crop dataset by field family
- projected synthetic full pages
- JSONL / CSV manifests for downstream GPU training

This module intentionally avoids importing heavy model inference code.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import re
import string
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import fitz
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from src.form300_templates import (
    CANONICAL_PAGE_SIZE,
    FORM300_FIELD_INDEX,
    FORM300_PAGE_TEMPLATES,
    FieldTemplate,
    bbox_to_pixels,
    iter_fields,
    iter_page_templates,
    resolve_page_index,
)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _paper_texture(height: int, width: int, base: int = 244) -> np.ndarray:
    img = np.full((height, width, 3), base, dtype=np.uint8)
    noise = np.random.normal(0, 6, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 215, 255).astype(np.uint8)
    return img


def _render_pdf_page(pdf_path: Path, page_index: int, target_width: int = CANONICAL_PAGE_SIZE[0]) -> Image.Image:
    doc = fitz.open(str(pdf_path))
    page = doc.load_page(page_index)
    pix = page.get_pixmap(matrix=fitz.Matrix(140 / 72, 140 / 72), alpha=False)
    doc.close()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    if img.width != target_width:
        target_height = int(img.height * (target_width / img.width))
        img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
    return img


def _to_gray_np(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)


def _blankness_score(img: Image.Image) -> Dict[str, float]:
    gray = _to_gray_np(img)
    dark_ratio = float((gray < 210).mean())
    std_value = float(gray.std())
    entropy = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return {
        "dark_ratio": dark_ratio,
        "std": std_value,
        "edge_var": entropy,
        "is_blank": dark_ratio < 0.015 and std_value < 20.0,
    }


def _load_font_candidates() -> List[Path]:
    fonts: List[Path] = []
    search_dirs = [
        Path(__file__).resolve().parent.parent / "fonts",
        Path("C:/Windows/Fonts"),
    ]
    keywords = ("script", "hand", "cursive", "brush", "comic", "segoesc", "lucidahand")
    for directory in search_dirs:
        if not directory.exists():
            continue
        for ext in ("*.ttf", "*.otf", "*.TTF", "*.OTF"):
            for font_path in directory.glob(ext):
                if directory.name.lower() == "fonts" or any(key in font_path.name.lower() for key in keywords):
                    fonts.append(font_path)
    deduped: Dict[str, Path] = {}
    for path in fonts:
        deduped[str(path)] = path
    return list(deduped.values())


@dataclass
class SyntheticRecord:
    record_id: str
    values: Dict[str, str]


class SyntheticRecordGenerator:
    """
    Generates coherent synthetic field values.
    """

    male_first = [
        "Amit", "Rajesh", "Suresh", "Vikram", "Nitin", "Rohit", "Pranav", "Anand",
        "Murali", "Venkatesh", "Arjun", "Sanjay", "Ashok", "Prakash", "Sunil", "Dinesh",
    ]
    female_first = [
        "Anita", "Priya", "Sunita", "Lakshmi", "Kavita", "Rekha", "Meena", "Usha",
        "Asha", "Pooja", "Savita", "Veena", "Anuradha", "Divya", "Kripa", "Nandini",
    ]
    surnames = [
        "Sharma", "Patel", "Reddy", "Gupta", "Nair", "Iyer", "Chatterji", "Joshi",
        "Verma", "Singh", "Rao", "Agarwal", "Desai", "Kumar", "Bose", "Mehta",
    ]
    occupations = [
        "Service", "Business", "Self Employed", "Engineer", "Teacher", "Doctor",
        "Accountant", "Consultant", "Manager", "Supervisor", "Technician",
    ]
    education = [
        "Graduate", "Post Graduate", "Diploma", "HSC", "SSC", "Professional",
    ]
    source_income = [
        "Salary", "Business", "Consulting", "Practice", "Commission", "Pension",
    ]
    age_proofs = [
        "Service", "Passport", "PAN", "Birth Certificate", "Aadhaar", "School Leaving",
    ]
    relationships = ["Wife", "Husband", "Son", "Daughter", "Father", "Mother", "Spouse"]
    genders = ["Male", "Female"]
    marital = ["Married", "Single"]
    locations = [
        {"city": "Mumbai", "state": "Maharashtra", "area": "Santacruz West", "pin": "400054"},
        {"city": "Hyderabad", "state": "Telangana", "area": "Jubilee Hills", "pin": "500033"},
        {"city": "Bengaluru", "state": "Karnataka", "area": "Koramangala", "pin": "560034"},
        {"city": "Chennai", "state": "Tamil Nadu", "area": "Adyar", "pin": "600020"},
        {"city": "Delhi", "state": "Delhi", "area": "Lajpat Nagar", "pin": "110024"},
        {"city": "Pune", "state": "Maharashtra", "area": "Aundh", "pin": "411007"},
        {"city": "Kolkata", "state": "West Bengal", "area": "Salt Lake", "pin": "700091"},
        {"city": "Jaipur", "state": "Rajasthan", "area": "Malviya Nagar", "pin": "302017"},
    ]
    employer_prefix = ["HCL", "Infosys", "TCS", "LIC", "Wipro", "Tech Mahindra", "State Bank", "Axis Bank"]

    def __init__(self, seed: int = 7):
        self.seed = seed
        self.rng = random.Random(seed)

    def _choice(self, items: Sequence[str]) -> str:
        return self.rng.choice(list(items))

    def _gender_bundle(self) -> Tuple[str, str]:
        gender = self._choice(self.genders)
        if gender == "Male":
            return gender, self._choice(self.male_first)
        return gender, self._choice(self.female_first)

    def _full_name(self, gender: Optional[str] = None) -> Tuple[str, str, str]:
        selected_gender, first = self._gender_bundle() if gender is None else (
            gender,
            self._choice(self.male_first if gender == "Male" else self.female_first),
        )
        middle = self._choice(self.surnames)
        last = self._choice(self.surnames)
        return first, middle, last

    def _pan(self, surname_hint: str = "A") -> str:
        first_five = "".join(self.rng.choice(string.ascii_uppercase) for _ in range(4))
        fifth = surname_hint[:1].upper() if surname_hint else self.rng.choice(string.ascii_uppercase)
        digits = "".join(self.rng.choice(string.digits) for _ in range(4))
        final = self.rng.choice(string.ascii_uppercase)
        return f"{first_five}{fifth}{digits}{final}"

    def _ckyc(self) -> str:
        prefix = self._choice(["CKYC", "CMC", "KYC", "LIC"])
        suffix = "".join(self.rng.choice(string.digits) for _ in range(7))
        return f"{prefix}{suffix}"

    def _mobile(self) -> str:
        return self.rng.choice("6789") + "".join(self.rng.choice(string.digits) for _ in range(9))

    def _proposal_no(self) -> str:
        return "".join(self.rng.choice(string.digits) for _ in range(5))

    def _aadhaar(self) -> str:
        return "".join(self.rng.choice(string.digits) for _ in range(12))

    def _date(self, year_min: int, year_max: int) -> Tuple[str, int, int, int]:
        year = self.rng.randint(year_min, year_max)
        month = self.rng.randint(1, 12)
        day = self.rng.randint(1, 28)
        return f"{day:02d}/{month:02d}/{year}", day, month, year

    def _dob_and_age(self) -> Tuple[str, str]:
        dob, day, month, year = self._date(1955, 2004)
        age = max(18, 2026 - year)
        return dob, str(age)

    def _amount(self, low: int, high: int) -> str:
        value = self.rng.randint(low, high)
        if self.rng.random() < 0.55:
            return f"{value:,}"
        return str(value)

    def _health_mark(self, positive_rate: float = 0.15) -> str:
        return "YES" if self.rng.random() < positive_rate else "NO"

    def build_record(self, record_id: str) -> SyntheticRecord:
        gender, first_name = self._gender_bundle()
        middle_name = self._choice(self.surnames)
        last_name = self._choice(self.surnames)
        father_name = " ".join(self._full_name("Male"))
        mother_name = " ".join(self._full_name("Female"))
        spouse_name = " ".join(self._full_name("Female" if gender == "Male" else "Male"))
        marital = self._choice(self.marital)
        dob, age = self._dob_and_age()
        location = self._choice(self.locations)
        house_no = self._choice(["12", "45/A", "B-302", "Flat 7", "A-12", "1-2-3", "H.No. 234"])
        street = self._choice(["MG Road", "Vikas Nagar", "Station Road", "Park Avenue", "Jeevan Vikas", "Civil Lines"])
        line1 = f"{house_no}, {street}"
        line2 = location["area"]
        employer = self._choice(self.employer_prefix)
        annual_income_int = self.rng.randint(300000, 2500000)
        annual_income = f"{annual_income_int:,}"
        premium_int = max(12000, annual_income_int // self.rng.randint(20, 40))
        sum_assured_int = premium_int * self.rng.randint(20, 60)
        values = {
            "office_inward_no": self._proposal_no(),
            "office_proposal_no": self._proposal_no(),
            "office_receipt_date": self._date(2025, 2026)[0],
            "office_deposit_amount": self._amount(5000, 500000),
            "customer_id": f"LIC{self.rng.randint(1000000, 9999999)}",
            "ckyc_number": self._ckyc(),
            "first_name": first_name,
            "middle_name": middle_name,
            "last_name": last_name,
            "father_full_name": father_name,
            "mother_full_name": mother_name,
            "gender_mark": gender.upper(),
            "marital_status": marital,
            "spouse_full_name": spouse_name if marital == "Married" else "",
            "date_of_birth": dob,
            "age_years": age,
            "place_of_birth": location["city"],
            "nature_of_age_proof": self._choice(self.age_proofs),
            "nationality": "Indian",
            "citizenship": "Indian",
            "address_line1": line1,
            "address_line2": line2,
            "address_city": location["city"],
            "address_state_country": f"{location['state']}, India",
            "address_pincode": location["pin"],
            "current_address_same": self._choice(["Same", "same", "SAME"]),
            "residential_status_mark": self._choice(["Resident Indian", "Indian"]),
            "income_tax_assessee_mark": self._choice(["YES", "NO"]),
            "pan_number": self._pan(last_name),
            "gst_registered_mark": self._choice(["NO", "YES"]),
            "aadhaar_last_or_id_number": self._aadhaar(),
            "id_expiry_or_na": self._choice(["NA", "N/A", self._date(2026, 2031)[0]]),
            "address_proof_submitted_mark": self._choice(["YES", "NO"]),
            "education": self._choice(self.education),
            "present_occupation": self._choice(self.occupations),
            "source_of_income": self._choice(self.source_income),
            "employer_name": employer,
            "nature_of_duties": self._choice(["Service", "Operations", "Accounts", "Management", "Field Work"]),
            "length_of_service": str(self.rng.randint(1, 25)),
            "annual_income": annual_income,
            "height_cm": str(self.rng.randint(145, 190)),
            "weight_kg": str(self.rng.randint(45, 105)),
            "medical_consultation_mark": self._health_mark(),
            "hospital_admission_mark": self._health_mark(0.10),
            "health_absence_mark": self._health_mark(0.10),
            "respiratory_disease_mark": self._health_mark(0.08),
            "cardio_disease_mark": self._health_mark(0.08),
            "digestive_disease_mark": self._health_mark(0.08),
            "urinary_disease_mark": self._health_mark(0.06),
            "neuro_disease_mark": self._health_mark(0.05),
            "venereal_disease_mark": self._health_mark(0.03),
            "preferred_plan_name": self._choice(["Jeevan Anand", "Jeevan Umang", "Tech Term", "Endowment Plan"]),
            "preferred_plan_term": str(self.rng.randint(10, 30)),
            "preferred_sum_assured": f"{sum_assured_int:,}",
            "preferred_mode": self._choice(["Yearly", "Half-Yearly", "Quarterly", "Monthly"]),
            "preferred_premium": f"{premium_int:,}",
        }
        return SyntheticRecord(record_id=record_id, values=values)


class HandwritingRenderer:
    """
    Render crop-sized handwriting-like images on CPU.
    """

    def __init__(self, seed: int = 7):
        self.seed = seed
        self.rng = random.Random(seed)
        self.font_paths = _load_font_candidates()

    def _font(self, size: int) -> ImageFont.ImageFont:
        if not self.font_paths:
            return ImageFont.load_default()
        try:
            font_path = self.rng.choice(self.font_paths)
            return ImageFont.truetype(str(font_path), size=size)
        except OSError:
            return ImageFont.load_default()

    def _mark_variant(self, canonical: str) -> str:
        if canonical.upper().startswith("Y"):
            return self.rng.choice(["Y", "Yes", "YES", "/", "✓"])
        return self.rng.choice(["N", "No", "NO", "x"])

    def _fit_font_size(self, draw: ImageDraw.ImageDraw, text: str, width: int, height: int, multiline: bool) -> int:
        size = max(18, min(height - 4, 44))
        while size > 12:
            font = self._font(size)
            if multiline:
                bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=4)
            else:
                bbox = draw.textbbox((0, 0), text, font=font)
            if (bbox[2] - bbox[0]) <= max(10, width - 8) and (bbox[3] - bbox[1]) <= max(10, height - 8):
                return size
            size -= 2
        return max(size, 12)

    def _augment(self, img: Image.Image) -> Image.Image:
        np_img = np.array(img)
        if self.rng.random() < 0.7:
            angle = self.rng.uniform(-3.0, 3.0)
            h, w = np_img.shape[:2]
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            np_img = cv2.warpAffine(np_img, matrix, (w, h), borderValue=(245, 245, 245))
        if self.rng.random() < 0.45:
            shear = self.rng.uniform(-0.06, 0.06)
            h, w = np_img.shape[:2]
            matrix = np.float32([[1, shear, 0], [0, 1, 0]])
            np_img = cv2.warpAffine(np_img, matrix, (w, h), borderValue=(245, 245, 245))
        if self.rng.random() < 0.40:
            noise = np.random.normal(0, self.rng.uniform(3, 12), np_img.shape)
            np_img = np.clip(np_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        if self.rng.random() < 0.35:
            kernel = self.rng.choice([3, 5])
            np_img = cv2.GaussianBlur(np_img, (kernel, kernel), 0)
        if self.rng.random() < 0.25:
            alpha = self.rng.uniform(0.85, 1.10)
            beta = self.rng.uniform(-15, 15)
            np_img = np.clip(alpha * np_img + beta, 0, 255).astype(np.uint8)
        if self.rng.random() < 0.20:
            gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
            kernel = np.ones((2, 2), np.uint8)
            gray = cv2.erode(gray, kernel, iterations=1)
            np_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(np_img)

    def render(
        self,
        value: str,
        field: FieldTemplate,
        target_size: Tuple[int, int],
    ) -> Tuple[Image.Image, Dict[str, object]]:
        width = max(target_size[0], 48)
        height = max(target_size[1], 24)
        background = _paper_texture(height, width)
        canvas = Image.fromarray(background)
        draw = ImageDraw.Draw(canvas)
        rendered_text = value
        if field.renderer == "mark":
            rendered_text = self._mark_variant(value if value else "NO")
        if field.multiline and rendered_text:
            wrap_width = max(10, int(width / 10))
            tokens = rendered_text.split()
            lines: List[str] = []
            current = ""
            for token in tokens:
                proposal = token if not current else f"{current} {token}"
                if len(proposal) <= wrap_width:
                    current = proposal
                else:
                    lines.append(current)
                    current = token
            if current:
                lines.append(current)
            rendered_text = "\n".join(lines[: field.max_lines])
        font_size = self._fit_font_size(draw, rendered_text or " ", width, height, field.multiline)
        font = self._font(font_size)
        ink = self.rng.choice([(24, 32, 96), (18, 18, 18), (28, 28, 28), (12, 52, 110)])
        if field.renderer == "mark" and rendered_text in {"✓", "/"}:
            x1 = int(width * self.rng.uniform(0.25, 0.40))
            y1 = int(height * self.rng.uniform(0.45, 0.65))
            x2 = int(width * self.rng.uniform(0.45, 0.58))
            y2 = int(height * self.rng.uniform(0.68, 0.85))
            x3 = int(width * self.rng.uniform(0.62, 0.82))
            y3 = int(height * self.rng.uniform(0.18, 0.38))
            draw.line((x1, y1, x2, y2), fill=ink, width=max(1, height // 12))
            draw.line((x2, y2, x3, y3), fill=ink, width=max(1, height // 12))
        else:
            if field.multiline:
                bbox = draw.multiline_textbbox((0, 0), rendered_text, font=font, spacing=4)
            else:
                bbox = draw.textbbox((0, 0), rendered_text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            x = max(2, int((width - tw) / 2) + self.rng.randint(-2, 2))
            y = max(2, int((height - th) / 2) + self.rng.randint(-2, 2))
            if field.multiline:
                draw.multiline_text((x, y), rendered_text, fill=ink, font=font, spacing=4, align="left")
            else:
                draw.text((x, y), rendered_text, fill=ink, font=font)
        canvas = self._augment(canvas)
        meta = {
            "rendered_text": rendered_text,
            "font_size": font_size,
            "target_size": [width, height],
        }
        return canvas, meta


class BackgroundTemplateBuilder:
    """
    Bootstraps cleaned page backgrounds from a real sample PDF.
    """

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir

    def build(self, pdf_path: Path) -> Dict[str, Path]:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        results: Dict[str, Path] = {}
        doc = fitz.open(str(pdf_path))
        num_pages = len(doc)
        doc.close()
        for template in iter_page_templates():
            page_index = resolve_page_index(num_pages, template.page_ref)
            page_img = _render_pdf_page(pdf_path, page_index)
            page_np = np.array(page_img)
            h, w = page_np.shape[:2]
            for field in template.fields:
                x1, y1, x2, y2 = bbox_to_pixels(field.bbox_norm, w, h, pad_norm=0.004)
                patch = page_np[y1:y2, x1:x2]
                if patch.size == 0:
                    continue
                fill = np.percentile(patch, 88)
                noise = np.random.normal(fill, 4, patch.shape)
                page_np[y1:y2, x1:x2] = np.clip(noise, 220, 255).astype(np.uint8)
            out_path = self.out_dir / f"{template.page_type}.png"
            Image.fromarray(page_np).save(out_path)
            results[template.page_type] = out_path
        return results


class RealCropExtractor:
    """
    Extract real field crops from sample PDFs using canonical templates.
    """

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.images_dir = self.out_dir / "images"

    def extract(self, pdf_dir: Path) -> List[Dict[str, object]]:
        self.images_dir.mkdir(parents=True, exist_ok=True)
        manifests: List[Dict[str, object]] = []
        pdf_paths = sorted(pdf_dir.glob("*.pdf"))
        for pdf_path in pdf_paths:
            doc = fitz.open(str(pdf_path))
            num_pages = len(doc)
            doc.close()
            page_cache: Dict[str, Image.Image] = {}
            for template in iter_page_templates():
                page_index = resolve_page_index(num_pages, template.page_ref)
                page_img = _render_pdf_page(pdf_path, page_index)
                page_cache[template.page_type] = page_img
                w, h = page_img.size
                for field in template.fields:
                    x1, y1, x2, y2 = bbox_to_pixels(field.bbox_norm, w, h, pad_norm=0.005)
                    crop = page_img.crop((x1, y1, x2, y2))
                    blank = _blankness_score(crop)
                    crop_id = f"{pdf_path.stem}__{field.name}"
                    image_path = self.images_dir / f"{crop_id}.png"
                    crop.save(image_path)
                    manifests.append(
                        {
                            "id": crop_id,
                            "image_path": str(image_path),
                            "source_tier": "real_unlabeled",
                            "pdf_path": str(pdf_path),
                            "page_type": field.page_type,
                            "page_ref": field.page_ref,
                            "field_name": field.name,
                            "field_family": field.family,
                            "bbox_norm": list(field.bbox_norm),
                            "is_blank": bool(blank["is_blank"]),
                            "blankness": blank,
                            "target_text": None,
                        }
                    )
        return manifests


class SyntheticDatasetGenerator:
    """
    Build synthetic field crops and projected full pages.
    """

    def __init__(self, out_dir: Path, backgrounds: Dict[str, Path], seed: int = 7):
        self.out_dir = out_dir
        self.crop_dir = out_dir / "synthetic_crops"
        self.page_dir = out_dir / "synthetic_pages"
        self.backgrounds = backgrounds
        self.record_generator = SyntheticRecordGenerator(seed=seed)
        self.renderer = HandwritingRenderer(seed=seed)
        self.seed = seed

    def _target_size(self, field: FieldTemplate) -> Tuple[int, int]:
        width, height = CANONICAL_PAGE_SIZE
        x1, y1, x2, y2 = bbox_to_pixels(field.bbox_norm, width, height, pad_norm=0.0)
        return max(48, x2 - x1), max(24, y2 - y1)

    def build_records(self, num_records: int) -> List[SyntheticRecord]:
        return [
            self.record_generator.build_record(f"REC_{idx:05d}")
            for idx in range(1, num_records + 1)
        ]

    def build_crops(self, records: Sequence[SyntheticRecord]) -> List[Dict[str, object]]:
        self.crop_dir.mkdir(parents=True, exist_ok=True)
        rows: List[Dict[str, object]] = []
        for record in records:
            for field in iter_fields():
                value = record.values.get(field.name, "")
                target_size = self._target_size(field)
                img, render_meta = self.renderer.render(value, field, target_size)
                field_dir = self.crop_dir / field.family / field.name
                field_dir.mkdir(parents=True, exist_ok=True)
                image_path = field_dir / f"{record.record_id}__{field.name}.png"
                img.save(image_path)
                rows.append(
                    {
                        "id": f"{record.record_id}__{field.name}",
                        "record_id": record.record_id,
                        "image_path": str(image_path),
                        "source_tier": "synthetic",
                        "field_name": field.name,
                        "field_family": field.family,
                        "page_type": field.page_type,
                        "page_ref": field.page_ref,
                        "target_text": value,
                        "rendered_text": render_meta["rendered_text"],
                        "bbox_norm": list(field.bbox_norm),
                    }
                )
        return rows

    def _project_one_page(self, record: SyntheticRecord, page_type: str) -> Tuple[Path, Dict[str, object]]:
        template = FORM300_PAGE_TEMPLATES[page_type]
        if page_type not in self.backgrounds:
            raise FileNotFoundError(f"Missing background for page_type={page_type}")
        background = Image.open(self.backgrounds[page_type]).convert("RGB")
        draw_page = background.copy()
        page_np = np.array(draw_page)
        h, w = page_np.shape[:2]
        page_labels: Dict[str, str] = {}
        for field in template.fields:
            x1, y1, x2, y2 = bbox_to_pixels(field.bbox_norm, w, h, pad_norm=0.004)
            patch = page_np[y1:y2, x1:x2]
            fill = np.percentile(patch, 88) if patch.size else 246
            noise = np.random.normal(fill, 4, patch.shape if patch.size else (y2 - y1, x2 - x1, 3))
            page_np[y1:y2, x1:x2] = np.clip(noise, 220, 255).astype(np.uint8)
        page_img = Image.fromarray(page_np)
        for field in template.fields:
            value = record.values.get(field.name, "")
            x1, y1, x2, y2 = bbox_to_pixels(field.bbox_norm, page_img.width, page_img.height, pad_norm=0.002)
            crop_img, _ = self.renderer.render(value, field, (x2 - x1, y2 - y1))
            crop_img = crop_img.resize((x2 - x1, y2 - y1), Image.Resampling.LANCZOS)
            page_img.paste(crop_img, (x1, y1))
            page_labels[field.name] = value
        page_dir = self.page_dir / page_type
        page_dir.mkdir(parents=True, exist_ok=True)
        out_path = page_dir / f"{record.record_id}__{page_type}.png"
        page_img.save(out_path)
        manifest = {
            "id": f"{record.record_id}__{page_type}",
            "record_id": record.record_id,
            "image_path": str(out_path),
            "source_tier": "synthetic_page",
            "page_type": page_type,
            "page_ref": template.page_ref,
            "labels": page_labels,
        }
        return out_path, manifest

    def build_projected_pages(
        self,
        records: Sequence[SyntheticRecord],
        max_records: int,
    ) -> List[Dict[str, object]]:
        self.page_dir.mkdir(parents=True, exist_ok=True)
        manifests: List[Dict[str, object]] = []
        for record in records[:max_records]:
            for page_type in FORM300_PAGE_TEMPLATES:
                _, manifest = self._project_one_page(record, page_type)
                manifests.append(manifest)
        return manifests


class CPUForm300DatasetBuilder:
    """
    End-to-end CPU dataset builder.
    """

    def __init__(self, out_dir: Path, seed: int = 7):
        self.out_dir = out_dir
        self.seed = seed
        self.manifest_dir = self.out_dir / "manifests"
        self.background_dir = self.out_dir / "backgrounds"
        self.real_dir = self.out_dir / "real_crops"

    def _gold_label_sheet(self, real_rows: Sequence[Dict[str, object]], max_per_family: int = 20) -> List[Dict[str, object]]:
        buckets: Dict[str, List[Dict[str, object]]] = defaultdict(list)
        for row in real_rows:
            if not row["is_blank"]:
                buckets[str(row["field_family"])].append(row)
        selected: List[Dict[str, object]] = []
        for family, rows in buckets.items():
            for row in rows[:max_per_family]:
                selected.append(
                    {
                        "crop_id": row["id"],
                        "image_path": row["image_path"],
                        "field_name": row["field_name"],
                        "field_family": row["field_family"],
                        "pdf_path": row["pdf_path"],
                        "suggested_text": "",
                        "gold_text": "",
                        "review_status": "pending",
                        "notes": "",
                    }
                )
        return selected

    def _weak_label_requests(self, real_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
        requests: List[Dict[str, object]] = []
        for row in real_rows:
            if row["is_blank"]:
                continue
            requests.append(
                {
                    "id": row["id"],
                    "image_path": row["image_path"],
                    "field_name": row["field_name"],
                    "field_family": row["field_family"],
                    "page_type": row["page_type"],
                    "expected_canonical_output": "YES/NO" if row["field_family"] == "binary_mark" else "TEXT",
                    "validator_hint": row["field_family"],
                }
            )
        return requests

    def _split_rows(self, rows: Sequence[Dict[str, object]], val_ratio: float = 0.1) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
        grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
        for row in rows:
            grouped[str(row["record_id"])].append(row)
        record_ids = sorted(grouped)
        rng = random.Random(self.seed)
        rng.shuffle(record_ids)
        val_count = max(1, int(len(record_ids) * val_ratio))
        val_ids = set(record_ids[:val_count])
        train_rows: List[Dict[str, object]] = []
        val_rows: List[Dict[str, object]] = []
        for row in rows:
            if row["record_id"] in val_ids:
                val_rows.append(row)
            else:
                train_rows.append(row)
        return train_rows, val_rows

    def _write_gold_csv(self, rows: Sequence[Dict[str, object]], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            return
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def build(
        self,
        pdf_dir: Optional[Path],
        num_records: int = 2000,
        num_projected_records: int = 250,
    ) -> Dict[str, object]:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        backgrounds: Dict[str, Path] = {}
        real_rows: List[Dict[str, object]] = []
        
        # Check for SKIP_PDF environment variable to skip PDF processing
        skip_pdf = os.environ.get("SKIP_PDF", "0") == "1"
        
        if pdf_dir and pdf_dir.exists() and not skip_pdf:
            try:
                pdf_paths = sorted(pdf_dir.glob("*.pdf"))
                if pdf_paths:
                    bootstrap_pdf = pdf_paths[0]
                    backgrounds = BackgroundTemplateBuilder(self.background_dir).build(bootstrap_pdf)
                    real_rows = RealCropExtractor(self.real_dir).extract(pdf_dir)
                    _write_jsonl(self.manifest_dir / "real_crops_unlabeled.jsonl", real_rows)
                    _write_jsonl(self.manifest_dir / "weak_label_requests.jsonl", self._weak_label_requests(real_rows))
                    self._write_gold_csv(self._gold_label_sheet(real_rows), self.manifest_dir / "gold_label_sheet.csv")
            except Exception as e:
                print(f"Warning: PDF processing failed: {e}")
                print("Falling back to synthetic-only mode")
        if not backgrounds:
            # Fall back to synthetic white page backgrounds if no PDFs are available.
            self.background_dir.mkdir(parents=True, exist_ok=True)
            for template in iter_page_templates():
                bg = Image.fromarray(_paper_texture(CANONICAL_PAGE_SIZE[1], CANONICAL_PAGE_SIZE[0]))
                out_path = self.background_dir / f"{template.page_type}.png"
                bg.save(out_path)
                backgrounds[template.page_type] = out_path
        synthetic = SyntheticDatasetGenerator(self.out_dir, backgrounds=backgrounds, seed=self.seed)
        records = synthetic.build_records(num_records)
        synthetic_rows = synthetic.build_crops(records)
        train_rows, val_rows = self._split_rows(synthetic_rows, val_ratio=0.1)
        projected_rows = synthetic.build_projected_pages(records, max_records=num_projected_records)
        _write_jsonl(self.manifest_dir / "synthetic_crops.jsonl", synthetic_rows)
        _write_jsonl(self.manifest_dir / "train_crops.jsonl", train_rows)
        _write_jsonl(self.manifest_dir / "val_crops.jsonl", val_rows)
        _write_jsonl(self.manifest_dir / "synthetic_pages.jsonl", projected_rows)
        summary = {
            "out_dir": str(self.out_dir),
            "background_count": len(backgrounds),
            "real_crops": len(real_rows),
            "synthetic_crops": len(synthetic_rows),
            "train_crops": len(train_rows),
            "val_crops": len(val_rows),
            "synthetic_pages": len(projected_rows),
            "page_templates": [template.page_type for template in iter_page_templates()],
            "fields": [field.name for field in iter_fields()],
        }
        with open(self.manifest_dir / "summary.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        return summary
