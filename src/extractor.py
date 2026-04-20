"""AI Extraction Engine
=====================
Template-first extraction: canonical crop -> TrOCR (family-specific) -> Qwen VLM fallback.
Memory-managed for 24GB NVIDIA L4.

Fixes from audit:
- TrOCR-large-handwritten (not base)
- Fallback threshold 0.70 (not 0.85)
- Field-family routing (not one generic adapter)
- Qwen on full-page context (not tiny crops) for fallback
- Correct model ID reporting
"""

import torch
import json
import re
import logging
import gc
import time
from typing import Dict, List, Optional, Tuple
from PIL import Image
import numpy as np

from src.config import (
    QWEN_MODEL_ID, QWEN_SYSTEM_PROMPT, QWEN_EXTRACTION_PROMPT,
    QWEN_MAX_NEW_TOKENS, QWEN_TEMPERATURE,
    TROCR_MODEL_ID, TROCR_FT_DIR,
    DEVICE, CONFIDENCE_LOW, CONFIDENCE_MEDIUM,
    FORM_300_FIELDS, FIELD_NAMES,
    TROCR_FALLBACK_THRESHOLD,
    FIELD_FAMILY_EXTRACTOR,
    REVERSE_FIELD_NAME_MAP,
)
from src.form300_templates import FORM300_FIELD_INDEX

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages GPU memory by loading/unloading models as needed."""

    def __init__(self):
        self._qwen_model = None
        self._qwen_processor = None
        self._trocr_model = None
        self._trocr_processor = None
        self._active_model = None

    def _clear_gpu(self):
        if self._active_model == "qwen" and self._qwen_model is not None:
            del self._qwen_model
            del self._qwen_processor
            self._qwen_model = None
            self._qwen_processor = None
        elif self._active_model == "trocr" and self._trocr_model is not None:
            del self._trocr_model
            del self._trocr_processor
            self._trocr_model = None
            self._trocr_processor = None
        self._active_model = None
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("GPU memory cleared")

    def get_qwen(self):
        if self._qwen_model is not None:
            return self._qwen_model, self._qwen_processor

        if self._active_model == "trocr":
            self._clear_gpu()

        logger.info(f"Loading {QWEN_MODEL_ID}...")
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        self._qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            QWEN_MODEL_ID,
            dtype=torch.float16,
            device_map="auto",
            attn_implementation="eager",
        )
        self._qwen_processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID)
        self._active_model = "qwen"

        vram_used = torch.cuda.memory_allocated() / 1e9
        logger.info(f"Qwen loaded. VRAM: {vram_used:.1f} GB")
        return self._qwen_model, self._qwen_processor

    def get_trocr(self):
        if self._trocr_model is not None:
            return self._trocr_model, self._trocr_processor

        if self._active_model == "qwen":
            self._clear_gpu()

        model_path = str(TROCR_FT_DIR) if TROCR_FT_DIR.exists() and any(TROCR_FT_DIR.iterdir()) else TROCR_MODEL_ID

        logger.info(f"Loading TrOCR from {model_path}...")
        from transformers import VisionEncoderDecoderModel, TrOCRProcessor

        self._trocr_model = VisionEncoderDecoderModel.from_pretrained(model_path)
        self._trocr_model.to(DEVICE)
        self._trocr_model.eval()

        self._trocr_processor = TrOCRProcessor.from_pretrained(TROCR_MODEL_ID)
        self._active_model = "trocr"

        vram_used = torch.cuda.memory_allocated() / 1e9
        logger.info(f"TrOCR loaded. VRAM: {vram_used:.1f} GB")
        return self._trocr_model, self._trocr_processor

    def unload_all(self):
        self._clear_gpu()


_model_manager = ModelManager()


class TrOCRExtractor:
    """Field-level HTR using TrOCR for cropped field regions."""

    def __init__(self, model_manager: ModelManager = None):
        self.manager = model_manager or _model_manager

    def recognize_field(self, field_image: Image.Image, field_family: str = "text") -> Dict:
        """Recognize text in a single cropped field image."""
        model, processor = self.manager.get_trocr()

        if field_image.mode != "RGB":
            field_image = field_image.convert("RGB")

        pixel_values = processor(images=field_image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(DEVICE)

        with torch.no_grad():
            outputs = model.generate(
                pixel_values,
                max_new_tokens=64,
                output_scores=True,
                return_dict_in_generate=True,
            )

        generated_ids = outputs.sequences
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        confidence = self._compute_confidence(outputs)

        return {
            "value": text,
            "confidence": confidence,
            "model_used": "TrOCR-large-handwritten",
            "field_family": field_family,
        }

    def recognize_batch(self, field_images: List[Image.Image],
                        field_names: List[str] = None,
                        field_families: List[str] = None) -> Dict[str, Dict]:
        """Batch recognize multiple field crops."""
        model, processor = self.manager.get_trocr()
        results = {}

        batch_size = 8
        for i in range(0, len(field_images), batch_size):
            batch_imgs = field_images[i:i + batch_size]
            batch_names = field_names[i:i + batch_size] if field_names else ["field_{}".format(i + j) for j in range(len(batch_imgs))]
            batch_families = field_families[i:i + batch_size] if field_families else ["text"] * len(batch_imgs)

            rgb_imgs = [img.convert("RGB") if img.mode != "RGB" else img for img in batch_imgs]
            pixel_values = processor(images=rgb_imgs, return_tensors="pt", padding=True).pixel_values
            pixel_values = pixel_values.to(DEVICE)

            with torch.no_grad():
                outputs = model.generate(
                    pixel_values,
                    max_new_tokens=64,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            texts = processor.batch_decode(outputs.sequences, skip_special_tokens=True)

            for j, (name, text, family) in enumerate(zip(batch_names, texts, batch_families)):
                results[name] = {
                    "value": text.strip(),
                    "confidence": self._compute_confidence_single(outputs, j),
                    "model_used": "TrOCR-large-handwritten",
                    "field_family": family,
                }

        return results

    def _compute_confidence(self, outputs) -> float:
        if not hasattr(outputs, "scores") or not outputs.scores:
            return 0.5
        confidences = []
        for score in outputs.scores:
            probs = torch.softmax(score[0], dim=-1)
            max_prob = probs.max().item()
            confidences.append(max_prob)
        return float(np.mean(confidences)) if confidences else 0.5

    def _compute_confidence_single(self, outputs, batch_idx: int) -> float:
        if not hasattr(outputs, "scores") or not outputs.scores:
            return 0.5
        confidences = []
        for score in outputs.scores:
            if batch_idx < score.shape[0]:
                probs = torch.softmax(score[batch_idx], dim=-1)
                max_prob = probs.max().item()
                confidences.append(max_prob)
        return float(np.mean(confidences)) if confidences else 0.5


class QwenExtractor:
    """VLM extraction using Qwen2.5-VL — used for full-page fallback, not tiny crops."""

    def __init__(self, model_manager: ModelManager = None):
        self.manager = model_manager or _model_manager

    def extract_from_image(self, image: Image.Image, page_num: int = 1,
                           custom_prompt: str = None) -> Dict:
        """Extract fields from a full form page image using Qwen VLM."""
        model, processor = self.manager.get_qwen()

        prompt = custom_prompt or QWEN_EXTRACTION_PROMPT

        messages = [
            {"role": "system", "content": QWEN_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(model.device)

        start_time = time.time()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=QWEN_MAX_NEW_TOKENS,
                temperature=QWEN_TEMPERATURE, do_sample=False, num_beams=1,
            )

        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        elapsed = time.time() - start_time
        logger.info(f"Qwen extraction for page {page_num} took {elapsed:.1f}s")

        result = self._parse_extraction_output(output_text)
        result["extraction_time_seconds"] = elapsed
        result["model_used"] = QWEN_MODEL_ID
        result["page_num"] = page_num

        return result

    def extract_field_fallback(self, image: Image.Image, field_name: str,
                               page_image: Image.Image = None) -> Dict:
        """Field-scoped VLM fallback: use full page for context, ask for specific field."""
        model, processor = self.manager.get_qwen()

        target_img = page_image if page_image is not None else image

        prompt = (
            "Look at this LIC Proposal Form page. "
            "Extract ONLY the value for the field '{}'. "
            "Output JSON: {{\"value\": \"extracted text\", \"confidence\": 0.95}}. "
            "If the field is empty or not found, output: {{\"value\": null, \"confidence\": 0.0}}"
        ).format(field_name)

        messages = [
            {"role": "system", "content": "You are an expert Indian handwritten form reader. Output only valid JSON."},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": target_img},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)

        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        try:
            json_match = re.search(r"\{[\s\S]*\}", output_text)
            if json_match:
                parsed = json.loads(json_match.group())
                return {
                    "value": parsed.get("value", ""),
                    "confidence": parsed.get("confidence", 0.5),
                    "model_used": QWEN_MODEL_ID,
                }
        except json.JSONDecodeError:
            pass

        return {"value": output_text, "confidence": 0.3, "model_used": QWEN_MODEL_ID}

    def _parse_extraction_output(self, output_text: str) -> Dict:
        try:
            json_match = re.search(r"\{[\s\S]*\}", output_text)
            if json_match:
                parsed = json.loads(json_match.group())
                return parsed
        except json.JSONDecodeError:
            pass

        logger.warning("Failed to parse JSON from Qwen output, attempting fallback parsing")
        return self._fallback_parse(output_text)

    def _fallback_parse(self, text: str) -> Dict:
        fields = {}
        lines = text.strip().split("\n")

        for line in lines:
            for sep in [":", "=", "\u2192", "->"]:
                if sep in line:
                    parts = line.split(sep, 1)
                    key = parts[0].strip().strip('"').strip("'")
                    value = parts[1].strip().strip('"').strip("'")
                    if key and value:
                        norm_key = key.replace(" ", "_").replace("-", "_")
                        fields[norm_key] = {
                            "value": value,
                            "confidence": 0.5,
                            "field_type": "handwritten",
                        }
                    break

        return {
            "fields": fields,
            "parse_method": "fallback",
            "extraction_summary": {
                "total_fields_found": len(fields),
                "fields_extracted": len(fields),
                "overall_confidence": 0.5,
            }
        }


class DualModelExtractor:
    """Template-first extraction: canonical crop -> TrOCR -> Qwen VLM fallback.

    Phase 1: Template matcher crops fields using canonical bboxes
    Phase 2: TrOCR runs on all crops (fast, primary)
    Phase 3: Qwen VLM on FULL PAGE for low-confidence fields (not tiny crops)
    """

    def __init__(self):
        self.manager = ModelManager()
        self.qwen = QwenExtractor(self.manager)
        self.trocr = TrOCRExtractor(self.manager)
        self._template_matcher = None

    def _get_matcher(self):
        if self._template_matcher is None:
            from src.template_matcher import FormTemplateMatcher
            self._template_matcher = FormTemplateMatcher()
        return self._template_matcher

    def extract(self, images: List[Image.Image],
                field_crops: Optional[Dict[str, Tuple[Image.Image, int]]] = None) -> Dict:
        """Full template-first extraction pipeline."""
        all_fields = {}
        matcher = self._get_matcher()

        # Phase 1: Canonical crop extraction
        logger.info("Phase 1: Canonical template cropping...")
        if field_crops is None:
            page_crops = matcher.match_all_pages(images)
        else:
            page_crops = field_crops

        if not page_crops:
            logger.warning("No fields cropped from template matcher. Running Qwen on all pages as fallback.")
            return self._full_qwen_fallback(images)

        # Phase 2: TrOCR primary extraction with field-family awareness
        logger.info("Phase 2: Running TrOCR on {} cropped fields...".format(len(page_crops)))
        crop_names = list(page_crops.keys())
        crop_imgs = [page_crops[name][0] if isinstance(page_crops[name], tuple) else page_crops[name] for name in crop_names]
        crop_pages = [page_crops[name][1] if isinstance(page_crops[name], tuple) else 1 for name in crop_names]

        field_families = []
        for name in crop_names:
            field_info = FORM300_FIELD_INDEX.get(name)
            field_families.append(field_info.family if field_info else "text")

        trocr_results = self.trocr.recognize_batch(crop_imgs, crop_names, field_families)

        low_conf_fields = []

        for i, field_name in enumerate(crop_names):
            result = trocr_results.get(field_name, {})
            conf = result.get("confidence", 0.0)
            family = field_families[i]

            field_data = {
                "value": result.get("value", ""),
                "confidence": conf,
                "field_type": "handwritten",
                "source_page": crop_pages[i],
                "selected_model": "TrOCR",
                "field_family": family,
            }

            all_fields[field_name] = field_data

            if conf < TROCR_FALLBACK_THRESHOLD or not field_data["value"].strip() or len(field_data["value"]) < 2:
                low_conf_fields.append((field_name, crop_pages[i], field_data["value"]))

        # Phase 3: VLM fallback on full pages (not tiny crops)
        if low_conf_fields:
            logger.info("Phase 3: VLM fallback for {} low-confidence fields...".format(len(low_conf_fields)))
            fallback_pages_needed = set(f[1] for f in low_conf_fields)

            page_image_cache = {}
            for page_num in fallback_pages_needed:
                idx = page_num - 1
                if 0 <= idx < len(images):
                    page_image_cache[page_num] = images[idx]

            for field_name, page_num, trocr_val in low_conf_fields:
                page_img = page_image_cache.get(page_num)
                if page_img is None:
                    continue

                qwen_res = self.qwen.extract_field_fallback(
                    Image.new("RGB", (10, 10), "white"),
                    field_name,
                    page_image=page_img,
                )

                qwen_val = qwen_res.get("value", "")
                qwen_conf = qwen_res.get("confidence", 0.5)

                if qwen_val and str(qwen_val).lower() != "null":
                    all_fields[field_name]["value"] = qwen_val
                    all_fields[field_name]["confidence"] = max(0.6, qwen_conf)
                    all_fields[field_name]["selected_model"] = "Qwen-VL-Fallback"
                    all_fields[field_name]["trocr_original"] = trocr_val

        # Map template field names to config field names
        mapped_fields = self._map_field_names(all_fields)

        # Calculate summary
        total_expected = len(FIELD_NAMES)
        extracted = sum(1 for f in mapped_fields.values() if f.get("value") and f.get("confidence", 0) > 0.3)
        missing = total_expected - extracted
        low_conf = sum(1 for f in mapped_fields.values() if 0 < f.get("confidence", 0) < CONFIDENCE_MEDIUM)

        confidences = [f.get("confidence", 0) for f in mapped_fields.values() if f.get("value")]
        overall_conf = float(np.mean(confidences)) if confidences else 0.0

        return {
            "fields": mapped_fields,
            "raw_template_fields": all_fields,
            "extraction_summary": {
                "total_fields_expected": total_expected,
                "fields_extracted": extracted,
                "fields_missing": missing,
                "fields_low_confidence": low_conf,
                "overall_confidence": round(overall_conf, 4),
            },
            "models_used": ["TrOCR-Primary", "Qwen-VL-Fallback"],
            "total_pages": len(images),
        }

    def _map_field_names(self, template_fields: Dict) -> Dict:
        """Map template-style field names to config-style field names."""
        mapped = {}

        for template_name, field_data in template_fields.items():
            config_name = REVERSE_FIELD_NAME_MAP.get(template_name)
            if config_name:
                if config_name not in mapped or field_data.get("confidence", 0) > mapped[config_name].get("confidence", 0):
                    mapped[config_name] = {**field_data, "template_name": template_name}
            else:
                mapped[template_name] = field_data

        return mapped

    def _full_qwen_fallback(self, images: List[Image.Image]) -> Dict:
        """Full Qwen extraction when template matcher fails."""
        logger.info("Running full Qwen VLM extraction as fallback...")
        all_fields = {}

        target_pages = [img for i, img in enumerate(images) if (i + 1) in {1, 2, 3, 5, 6, 7, 10, 14, 16, 28}]

        for img in target_pages[:5]:
            result = self.qwen.extract_from_image(img)
            if "fields" in result:
                for field_name, field_data in result["fields"].items():
                    if field_name not in all_fields or field_data.get("confidence", 0) > all_fields.get(field_name, {}).get("confidence", 0):
                        all_fields[field_name] = field_data

        return {
            "fields": all_fields,
            "extraction_summary": {
                "total_fields_expected": len(FIELD_NAMES),
                "fields_extracted": sum(1 for f in all_fields.values() if f.get("value")),
                "fields_missing": len(FIELD_NAMES) - sum(1 for f in all_fields.values() if f.get("value")),
                "overall_confidence": 0.3,
            },
            "models_used": [QWEN_MODEL_ID],
            "total_pages": len(images),
        }

    def cleanup(self):
        self.manager.unload_all()
