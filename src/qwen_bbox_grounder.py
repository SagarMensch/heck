"""Qwen-VL reference bounding-box generation for LIC Form 300.

This module bootstraps higher-quality field boxes from a blank/reference form
page using Qwen2.5-VL, then lets the extraction pipeline reuse those boxes on
aligned scans instead of the older hand-estimated template coordinates.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import torch

from src.config import DATA_DIR, PROJECT_ROOT
from src.fixed_form_registration import FixedFormRegistrar, ReferencePageStore
from src.form300_templates import FORM300_PAGE_TEMPLATES, PAGE_TYPE_TO_PAGE_NUM

logger = logging.getLogger(__name__)


QWEN_REFERENCE_BBOX_PATH = DATA_DIR / "qwen_reference_bboxes.json"
QWEN_REFERENCE_AUDIT_DIR = DATA_DIR / "qwen_reference_bbox_audit"

PAGE_NUM_TO_TYPE = {v: k for k, v in PAGE_TYPE_TO_PAGE_NUM.items()}


FIELD_DESCRIPTION_OVERRIDES = {
    "customer_id": "Customer ID",
    "ckyc_number": "CKYC Number",
    "father_full_name": "Father or Husband Full Name",
    "mother_full_name": "Mother Full Name",
    "gender_mark": "Gender checkbox selection area",
    "marital_status": "Marital Status value area",
    "spouse_full_name": "Spouse Full Name",
    "date_of_birth": "Date of Birth",
    "age_years": "Age in Years",
    "place_of_birth": "Place of Birth",
    "nature_of_age_proof": "Nature of Age Proof",
    "address_line1": "Address Line 1",
    "address_line2": "Address Line 2",
    "address_city": "City",
    "address_state_country": "State or Country",
    "address_pincode": "Pincode",
    "current_address_same": "Current Address Same As Above selection area",
    "mobile_number": "Mobile Number",
    "pan_number": "PAN Number",
    "aadhaar_last_or_id_number": "Aadhaar or ID Number",
    "id_expiry_or_na": "ID Expiry Date or NA",
    "residential_status_mark": "Residential Status checkbox selection area",
    "income_tax_assessee_mark": "Income Tax Assessee checkbox selection area",
    "gst_registered_mark": "GST Registered checkbox selection area",
    "address_proof_submitted_mark": "Address Proof Submitted checkbox selection area",
    "present_occupation": "Present Occupation",
    "source_of_income": "Source of Income",
    "employer_name": "Employer Name",
    "nature_of_duties": "Nature of Duties",
    "length_of_service": "Length of Service",
    "annual_income": "Annual Income",
    "nominee_relationship": "Nominee Relationship",
    "appointee_relationship": "Appointee Relationship",
    "proposed_plan_name": "Proposed Plan Name",
    "proposed_plan_term": "Proposed Plan Term",
    "proposed_sum_assured": "Proposed Sum Assured",
    "proposed_premium_paying_term": "Premium Paying Term",
    "proposed_premium_amount": "Premium Amount",
    "proposed_premium_mode": "Premium Mode",
    "objective_of_insurance": "Objective of Insurance",
    "medical_consultation_mark": "Medical Consultation checkbox selection area",
    "hospital_admission_mark": "Hospital Admission checkbox selection area",
    "health_absence_mark": "Health Absence checkbox selection area",
    "respiratory_disease_mark": "Respiratory Disease checkbox selection area",
    "cardio_disease_mark": "Cardiovascular Disease checkbox selection area",
    "digestive_disease_mark": "Digestive Disease checkbox selection area",
    "urinary_disease_mark": "Urinary Disease checkbox selection area",
    "neuro_disease_mark": "Neurological Disease checkbox selection area",
    "venereal_disease_mark": "Venereal Disease checkbox selection area",
    "agent_code": "Agent Code",
    "agent_name": "Agent Name",
    "branch_code": "Branch Code",
    "branch_name": "Branch Name",
    "development_officer_name": "Development Officer Name",
    "date_of_proposal": "Date of Proposal",
    "place_of_signing": "Place of Signing",
    "proposer_signature_present": "Proposer signature area",
    "declarant_name": "Declarant Name",
    "declarant_address": "Declarant Address",
    "husband_full_name": "Husband Full Name",
    "husband_occupation": "Husband Occupation",
    "settlement_option_yn": "Settlement option Yes or No checkbox area",
    "settlement_period_years": "Settlement Period in Years",
    "settlement_full_or_part": "Settlement Full or Part value area",
    "settlement_instalment_mode": "Settlement Instalment Mode",
    "preferred_plan_name": "Preferred Plan Name",
    "preferred_plan_term": "Preferred Plan Term",
    "preferred_sum_assured": "Preferred Sum Assured",
    "preferred_mode": "Preferred Mode",
    "preferred_premium": "Preferred Premium",
}


class QwenReferenceBBoxStore:
    """Persistent store for Qwen-generated reference boxes."""

    def __init__(self, path: Path = QWEN_REFERENCE_BBOX_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: Optional[Dict] = None

    def load(self) -> Dict:
        if self._data is not None:
            return self._data
        if self.path.exists():
            self._data = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self._data = {"pages": {}}
        return self._data

    def save(self) -> None:
        data = self.load()
        self.path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def get_field_bbox(
        self,
        page_num: int,
        field_name: str,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> Optional[Tuple[int, int, int, int]]:
        data = self.load()
        page_data = data.get("pages", {}).get(str(page_num))
        if not page_data:
            return None

        field_data = page_data.get("fields", {}).get(field_name)
        if not field_data:
            return None

        bbox = field_data.get("bbox")
        if not bbox or len(bbox) != 4:
            return None

        src_w = int(page_data.get("width") or 0)
        src_h = int(page_data.get("height") or 0)
        x1, y1, x2, y2 = [int(v) for v in bbox]

        if target_size and src_w > 0 and src_h > 0:
            dst_w, dst_h = target_size
            if (dst_w, dst_h) != (src_w, src_h):
                sx = dst_w / max(src_w, 1)
                sy = dst_h / max(src_h, 1)
                x1 = int(round(x1 * sx))
                y1 = int(round(y1 * sy))
                x2 = int(round(x2 * sx))
                y2 = int(round(y2 * sy))

        return x1, y1, x2, y2

    def set_field_bbox(
        self,
        page_num: int,
        page_size: Tuple[int, int],
        field_name: str,
        bbox: Tuple[int, int, int, int],
        confidence: float,
        description: str,
        renderer: str,
    ) -> None:
        data = self.load()
        pages = data.setdefault("pages", {})
        page_key = str(page_num)
        page_entry = pages.setdefault(
            page_key,
            {"width": int(page_size[0]), "height": int(page_size[1]), "fields": {}},
        )
        page_entry["width"] = int(page_size[0])
        page_entry["height"] = int(page_size[1])
        page_entry.setdefault("fields", {})[field_name] = {
            "bbox": [int(v) for v in bbox],
            "confidence": round(float(confidence), 4),
            "description": description,
            "renderer": renderer,
        }


class QwenBBoxGrounder:
    """Generate reference bounding boxes from the blank/sample LIC form."""

    def __init__(
        self,
        store: Optional[QwenReferenceBBoxStore] = None,
        reference_store: Optional[ReferencePageStore] = None,
        bootstrap_pdf_path: Optional[Path] = None,
        model_manager=None,
    ):
        self.store = store or QwenReferenceBBoxStore()
        self.reference_store = reference_store or ReferencePageStore()
        self.registrar = FixedFormRegistrar(reference_store=self.reference_store)
        self.bootstrap_pdf_path = Path(bootstrap_pdf_path) if bootstrap_pdf_path else self._default_bootstrap_pdf()
        self._model_manager = model_manager

    def _default_bootstrap_pdf(self) -> Optional[Path]:
        candidates = [
            PROJECT_ROOT / "Techathon_Samples" / "P10.pdf",
            PROJECT_ROOT / "Techathon_Samples" / "P02.pdf",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _get_qwen(self):
        if self._model_manager is None:
            from src.extractor import ModelManager

            self._model_manager = ModelManager()
        return self._model_manager.get_qwen()

    def close(self) -> None:
        if self._model_manager is not None:
            self._model_manager.unload_all()

    def bootstrap_pages(self, page_nums, force: bool = False) -> Dict:
        results = {}
        try:
            for page_num in page_nums:
                results[str(page_num)] = self.bootstrap_page(page_num, force=force)
            self.store.save()
            return results
        finally:
            self.close()

    def bootstrap_page(self, page_num: int, force: bool = False) -> Dict:
        page_type = PAGE_NUM_TO_TYPE.get(page_num)
        if page_type not in FORM300_PAGE_TEMPLATES:
            raise ValueError(f"No registered template fields for page {page_num}")

        template = FORM300_PAGE_TEMPLATES[page_type]
        page_bgr = self._get_bootstrap_page(page_num)
        page_rgb = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2RGB)
        pil_page = Image.fromarray(page_rgb)
        page_h, page_w = page_bgr.shape[:2]

        page_results = {"page_num": page_num, "page_type": page_type, "fields": {}}
        existing = self.store.load().get("pages", {}).get(str(page_num), {}).get("fields", {})

        for field in template.fields:
            if not force and field.name in existing:
                page_results["fields"][field.name] = existing[field.name]
                continue

            description = FIELD_DESCRIPTION_OVERRIDES.get(field.name, field.name.replace("_", " ").title())
            try:
                bbox, confidence = self.ground_field(
                    pil_page,
                    field_name=field.name,
                    field_description=description,
                    renderer=field.renderer,
                    page_num=page_num,
                )
            except Exception as exc:
                logger.warning(
                    "Qwen grounding skipped for page %s field %s: %s",
                    page_num,
                    field.name,
                    exc,
                )
                page_results["fields"][field.name] = {
                    "bbox": None,
                    "confidence": 0.0,
                    "description": description,
                    "renderer": field.renderer,
                    "status": "skipped",
                }
                continue

            self.store.set_field_bbox(
                page_num=page_num,
                page_size=(page_w, page_h),
                field_name=field.name,
                bbox=bbox,
                confidence=confidence,
                description=description,
                renderer=field.renderer,
            )
            self.store.save()

            page_results["fields"][field.name] = {
                "bbox": list(bbox),
                "confidence": confidence,
                "description": description,
                "renderer": field.renderer,
            }
            logger.info(
                "Qwen grounded page %s field %s -> %s (conf=%.3f)",
                page_num,
                field.name,
                bbox,
                confidence,
            )

        self._write_overlay(page_num, page_bgr)
        return page_results

    def ground_field(
        self,
        pil_image: Image.Image,
        field_name: str,
        field_description: str,
        renderer: str = "text",
        page_num: Optional[int] = None,
    ) -> Tuple[Tuple[int, int, int, int], float]:
        model, processor = self._get_qwen()
        from qwen_vl_utils import process_vision_info

        qwen_image, scale_x, scale_y = self._prepare_qwen_image(pil_image)
        value_hint = (
            "the checkbox, signature, or marked answer area"
            if renderer == "mark"
            else "the empty writable value area"
        )
        prompt = (
            "This is an LIC Form 300 page.\n"
            f"Locate {value_hint} for the field \"{field_description}\""
            f"{' on page ' + str(page_num) if page_num else ''}.\n"
            "Important rules:\n"
            "- Return the answer area only, not the printed label text.\n"
            "- Keep the box tight but include the full writable region.\n"
            "- Return only valid JSON in an array.\n"
            "- If the field is a checkbox or mark, box only the selectable region.\n"
            "JSON format: [{\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"" + field_name + "\", \"confidence\": 0.0}]\n"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": qwen_image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )

        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        output_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        bbox, confidence = self._parse_output(output_text, qwen_image.size)
        bbox = (
            int(round(bbox[0] * scale_x)),
            int(round(bbox[1] * scale_y)),
            int(round(bbox[2] * scale_x)),
            int(round(bbox[3] * scale_y)),
        )
        bbox = self._normalize_bbox(bbox, pil_image.size)
        return bbox, confidence

    def _prepare_qwen_image(
        self,
        pil_image: Image.Image,
        max_edge: int = 1280,
    ) -> Tuple[Image.Image, float, float]:
        src_w, src_h = pil_image.size
        largest = max(src_w, src_h)
        if largest <= max_edge:
            return pil_image, 1.0, 1.0

        scale = max_edge / float(largest)
        dst_w = max(32, int(round(src_w * scale)))
        dst_h = max(32, int(round(src_h * scale)))
        resized = pil_image.resize((dst_w, dst_h), Image.Resampling.BICUBIC)
        return resized, src_w / float(dst_w), src_h / float(dst_h)

    def _parse_output(
        self,
        raw_text: str,
        image_size: Tuple[int, int],
    ) -> Tuple[Tuple[int, int, int, int], float]:
        parsed = None

        stripped = raw_text.strip()
        if stripped.startswith("["):
            try:
                loaded = json.loads(stripped)
                if isinstance(loaded, list):
                    if len(loaded) == 4 and all(isinstance(v, (int, float)) for v in loaded):
                        parsed = {"bbox": loaded}
                    elif loaded and isinstance(loaded[0], dict):
                        parsed = loaded[0]
                    else:
                        parsed = {"bbox": loaded}
                elif isinstance(loaded, dict):
                    parsed = loaded
            except json.JSONDecodeError:
                parsed = None

        if parsed is None:
            json_match = re.search(r"\{[\s\S]*\}", raw_text)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                except json.JSONDecodeError:
                    parsed = None

        if parsed is None:
            list_match = re.search(r"\[(.*?)\]", raw_text)
            if list_match:
                parts = [p.strip() for p in list_match.group(1).split(",")]
                if len(parts) == 4:
                    parsed = {"bbox": [float(p) for p in parts]}

        if not isinstance(parsed, dict):
            raise ValueError(f"Could not parse structured bbox from Qwen output: {raw_text}")

        bbox_payload = parsed.get("bbox_2d", parsed.get("bbox"))
        if bbox_payload is None:
            raise ValueError(f"Could not parse bbox from Qwen output: {raw_text}")

        confidence = float(parsed.get("confidence", 0.85))
        if confidence <= 0.0:
            confidence = 0.85
        bbox = self._normalize_bbox(bbox_payload, image_size)
        return bbox, confidence

    def _normalize_bbox(
        self,
        bbox,
        image_size: Tuple[int, int],
    ) -> Tuple[int, int, int, int]:
        img_w, img_h = image_size
        if isinstance(bbox, dict):
            bbox = [bbox.get(k) for k in ("x1", "y1", "x2", "y2")]
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            numeric = []
            for item in bbox:
                if isinstance(item, (int, float)):
                    numeric.append(float(item))
                if len(numeric) == 4:
                    break
            if len(numeric) == 4:
                bbox = numeric
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError(f"Invalid bbox payload: {bbox}")

        coords = [float(v) for v in bbox]
        max_coord = max(coords)

        if max_coord <= 1.0:
            coords = [coords[0] * img_w, coords[1] * img_h, coords[2] * img_w, coords[3] * img_h]
        elif max_coord <= 1000.0 and (img_w > 1200 or img_h > 1200):
            coords = [
                coords[0] * img_w / 1000.0,
                coords[1] * img_h / 1000.0,
                coords[2] * img_w / 1000.0,
                coords[3] * img_h / 1000.0,
            ]

        x1, y1, x2, y2 = [int(round(v)) for v in coords]
        x1 = max(0, min(img_w - 1, x1))
        y1 = max(0, min(img_h - 1, y1))
        x2 = max(0, min(img_w, x2))
        y2 = max(0, min(img_h, y2))

        if x2 <= x1:
            x2 = min(img_w, x1 + 8)
        if y2 <= y1:
            y2 = min(img_h, y1 + 8)

        return x1, y1, x2, y2

    def _write_overlay(self, page_num: int, page_bgr: np.ndarray) -> None:
        QWEN_REFERENCE_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        overlay = page_bgr.copy()
        page_data = self.store.load().get("pages", {}).get(str(page_num), {})
        fields = page_data.get("fields", {})

        for field_name, field_data in fields.items():
            bbox = field_data.get("bbox") or []
            if len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 180, 0), 2)
            cv2.putText(
                overlay,
                field_name,
                (x1, max(18, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 120, 0),
                1,
                cv2.LINE_AA,
            )

        out_path = QWEN_REFERENCE_AUDIT_DIR / f"page_{page_num:02d}_qwen_overlay.png"
        cv2.imwrite(str(out_path), overlay)
        logger.info("Saved Qwen bbox audit overlay: %s", out_path)

    def _get_bootstrap_page(self, page_num: int) -> np.ndarray:
        if self.bootstrap_pdf_path and self.bootstrap_pdf_path.exists():
            try:
                rendered = self._render_pdf_page(self.bootstrap_pdf_path, page_num)
                registration = self.registrar.register(rendered, page_num)
                if registration.success or registration.method != "resize_fallback":
                    logger.info(
                        "Using aligned bootstrap page %s page %s via %s",
                        self.bootstrap_pdf_path.name,
                        page_num,
                        registration.method,
                    )
                    return registration.aligned_bgr
                logger.warning(
                    "Bootstrap page alignment low quality for %s page %s, using raw render",
                    self.bootstrap_pdf_path.name,
                    page_num,
                )
                return rendered
            except Exception as exc:
                logger.warning("Bootstrap PDF render failed for page %s: %s", page_num, exc)

        logger.info("Falling back to blank reference page for page %s", page_num)
        return self.reference_store.get_page(page_num)

    def _render_pdf_page(self, pdf_path: Path, page_num: int, dpi: int = 200) -> np.ndarray:
        import fitz

        doc = fitz.open(str(pdf_path))
        if page_num < 1 or page_num > doc.page_count:
            doc.close()
            raise ValueError(f"{pdf_path} does not contain page {page_num}")

        page = doc.load_page(page_num - 1)
        pix = page.get_pixmap(dpi=dpi, alpha=False)
        doc.close()

        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
