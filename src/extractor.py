"""
AI Extraction Engine
=====================
Dual-model extraction: Qwen2.5-VL (primary VLM) + TrOCR (field-level HTR).
Memory-managed for 24GB NVIDIA L4.
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
)

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
        """Free GPU memory."""
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
        """Load Qwen2.5-VL model (lazy, clears other models first)."""
        if self._qwen_model is not None:
            return self._qwen_model, self._qwen_processor

        if self._active_model == "trocr":
            self._clear_gpu()

        logger.info(f"Loading Qwen2.5-VL from {QWEN_MODEL_ID}...")
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        self._qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            QWEN_MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="flash_attention_2" if self._has_flash_attn() else "eager",
        )
        self._qwen_processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID)
        self._active_model = "qwen"

        vram_used = torch.cuda.memory_allocated() / 1e9
        logger.info(f"Qwen2.5-VL loaded. VRAM used: {vram_used:.1f} GB")
        return self._qwen_model, self._qwen_processor

    def get_trocr(self):
        """Load TrOCR model (lazy, clears other models first)."""
        if self._trocr_model is not None:
            return self._trocr_model, self._trocr_processor

        if self._active_model == "qwen":
            self._clear_gpu()

        # Try fine-tuned model first, fallback to base
        model_path = str(TROCR_FT_DIR) if TROCR_FT_DIR.exists() and any(TROCR_FT_DIR.iterdir()) else TROCR_MODEL_ID

        logger.info(f"Loading TrOCR from {model_path}...")
        from transformers import VisionEncoderDecoderModel, TrOCRProcessor

        self._trocr_model = VisionEncoderDecoderModel.from_pretrained(model_path)
        self._trocr_model.to(DEVICE)
        self._trocr_model.eval()

        self._trocr_processor = TrOCRProcessor.from_pretrained(TROCR_MODEL_ID)
        self._active_model = "trocr"

        vram_used = torch.cuda.memory_allocated() / 1e9
        logger.info(f"TrOCR loaded. VRAM used: {vram_used:.1f} GB")
        return self._trocr_model, self._trocr_processor

    def _has_flash_attn(self) -> bool:
        try:
            import flash_attn
            return True
        except ImportError:
            return False

    def unload_all(self):
        """Unload everything."""
        self._clear_gpu()


# Global model manager
_model_manager = ModelManager()


class QwenExtractor:
    """Primary extraction using Qwen2.5-VL-7B vision-language model."""

    def __init__(self, model_manager: ModelManager = None):
        self.manager = model_manager or _model_manager

    def extract_from_image(self, image: Image.Image, page_num: int = 1,
                           custom_prompt: str = None) -> Dict:
        """Extract all fields from a single form page image using Qwen2.5-VL."""
        model, processor = self.manager.get_qwen()

        prompt = custom_prompt or QWEN_EXTRACTION_PROMPT

        # Build Qwen2.5-VL message format
        messages = [
            {
                "role": "system",
                "content": QWEN_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        # Process with Qwen processor
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        # Generate
        start_time = time.time()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=QWEN_MAX_NEW_TOKENS,
                temperature=QWEN_TEMPERATURE,
                do_sample=False,
                num_beams=1,
            )

        # Decode
        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        output_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        elapsed = time.time() - start_time
        logger.info(f"Qwen extraction for page {page_num} took {elapsed:.1f}s")

        # Parse JSON from output
        result = self._parse_extraction_output(output_text)
        result["extraction_time_seconds"] = elapsed
        result["model_used"] = "Qwen2.5-VL-7B-Instruct"
        result["page_num"] = page_num

        return result

    def extract_multi_page(self, images: List[Image.Image]) -> Dict:
        """Extract from multiple pages of a single form."""
        all_fields = {}
        page_results = []

        for i, img in enumerate(images):
            page_result = self.extract_from_image(img, page_num=i + 1)
            page_results.append(page_result)

            if "fields" in page_result:
                for field_name, field_data in page_result["fields"].items():
                    # Keep highest confidence extraction
                    if field_name not in all_fields or \
                       field_data.get("confidence", 0) > all_fields[field_name].get("confidence", 0):
                        field_data["source_page"] = i + 1
                        all_fields[field_name] = field_data

        return {
            "fields": all_fields,
            "page_results": page_results,
            "total_pages": len(images),
            "model_used": "Qwen2.5-VL-7B-Instruct",
        }

    def _parse_extraction_output(self, output_text: str) -> Dict:
        """Parse Qwen's output into structured JSON."""
        # Try to extract JSON from the output
        try:
            # Find JSON block
            json_match = re.search(r'\{[\s\S]*\}', output_text)
            if json_match:
                parsed = json.loads(json_match.group())
                return parsed
        except json.JSONDecodeError:
            pass

        # Fallback: try to parse as key-value pairs
        logger.warning("Failed to parse JSON from Qwen output, attempting fallback parsing")
        return self._fallback_parse(output_text)

    def _fallback_parse(self, text: str) -> Dict:
        """Fallback parsing for non-JSON outputs."""
        fields = {}
        lines = text.strip().split("\n")

        for line in lines:
            # Try patterns like "Field: Value" or "Field = Value"
            for sep in [":", "=", "→", "->"]:
                if sep in line:
                    parts = line.split(sep, 1)
                    key = parts[0].strip().strip('"').strip("'")
                    value = parts[1].strip().strip('"').strip("'")
                    if key and value:
                        # Normalize field name
                        norm_key = key.replace(" ", "_").replace("-", "_")
                        fields[norm_key] = {
                            "value": value,
                            "confidence": 0.5,  # Low confidence for fallback
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


class TrOCRExtractor:
    """Field-level HTR using TrOCR for cropped field regions."""

    def __init__(self, model_manager: ModelManager = None):
        self.manager = model_manager or _model_manager

    def recognize_field(self, field_image: Image.Image) -> Dict:
        """Recognize text in a single cropped field image."""
        model, processor = self.manager.get_trocr()

        # Preprocess for TrOCR
        if field_image.mode != "RGB":
            field_image = field_image.convert("RGB")

        pixel_values = processor(images=field_image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(DEVICE)

        # Generate with output scores for confidence
        with torch.no_grad():
            outputs = model.generate(
                pixel_values,
                max_new_tokens=64,
                output_scores=True,
                return_dict_in_generate=True,
            )

        # Decode text
        generated_ids = outputs.sequences
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        # Compute confidence from logits
        confidence = self._compute_confidence(outputs)

        return {
            "value": text,
            "confidence": confidence,
            "model_used": "TrOCR-base-handwritten",
        }

    def recognize_batch(self, field_images: List[Image.Image],
                        field_names: List[str] = None) -> Dict[str, Dict]:
        """Batch recognize multiple field crops."""
        model, processor = self.manager.get_trocr()
        results = {}

        # Process in batches to manage memory
        batch_size = 8
        for i in range(0, len(field_images), batch_size):
            batch_imgs = field_images[i:i + batch_size]
            batch_names = field_names[i:i + batch_size] if field_names else [f"field_{i+j}" for j in range(len(batch_imgs))]

            # Preprocess batch
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

            for j, (name, text) in enumerate(zip(batch_names, texts)):
                results[name] = {
                    "value": text.strip(),
                    "confidence": self._compute_confidence_single(outputs, j),
                    "model_used": "TrOCR-base-handwritten",
                }

        return results

    def _compute_confidence(self, outputs) -> float:
        """Compute average confidence from generation logits."""
        if not hasattr(outputs, 'scores') or not outputs.scores:
            return 0.5

        confidences = []
        for score in outputs.scores:
            probs = torch.softmax(score[0], dim=-1)
            max_prob = probs.max().item()
            confidences.append(max_prob)

        return float(np.mean(confidences)) if confidences else 0.5

    def _compute_confidence_single(self, outputs, batch_idx: int) -> float:
        """Compute confidence for a single item in batch."""
        if not hasattr(outputs, 'scores') or not outputs.scores:
            return 0.5

        confidences = []
        for score in outputs.scores:
            if batch_idx < score.shape[0]:
                probs = torch.softmax(score[batch_idx], dim=-1)
                max_prob = probs.max().item()
                confidences.append(max_prob)

        return float(np.mean(confidences)) if confidences else 0.5


class DualModelExtractor:
    """
    Orchestrates TrOCR-First extraction with Qwen2.5-VL Agentic Fallback.
    
    Strategy:
    1. Run PaddleOCR Template Matcher to get field crops (instant)
    2. Run TrOCR on all crops (fast, primary OCR)
    3. Run Qwen2.5-VL ONLY on fields where TrOCR fails or has low confidence (heavy fallback)
    """
    
    def __init__(self):
        self.manager = ModelManager()
        self.qwen = QwenExtractor(self.manager)
        self.trocr = TrOCRExtractor(self.manager)
        
        # Lazy load template matcher
        self._template_matcher = None
        
    def _get_matcher(self):
        if self._template_matcher is None:
            from src.template_matcher import FormTemplateMatcher
            self._template_matcher = FormTemplateMatcher()
        return self._template_matcher

    def extract(self, images: List[Image.Image],
                field_crops: Optional[Dict[str, List[Image.Image]]] = None) -> Dict:
        """
        Full TrOCR-First extraction pipeline.
        """
        all_fields = {}
        matcher = self._get_matcher()
        
        # Phase 1: Layout Detection & Cropping
        logger.info("Phase 1: Running Layout Template Matching...")
        page_crops = []
        for i, img in enumerate(images):
            crops = matcher.match_and_crop(img)
            for field_name, crop_img in crops.items():
                if field_name not in all_fields:
                    page_crops.append((field_name, crop_img, i+1))
                    
        # Phase 2: TrOCR Primary Extraction
        logger.info(f"Phase 2: Running TrOCR on {len(page_crops)} detected fields...")
        low_conf_queue = []
        
        if page_crops:
            crop_imgs = [c[1] for c in page_crops]
            crop_names = [c[0] for c in page_crops]
            
            trocr_results = self.trocr.recognize_batch(crop_imgs, crop_names)
            
            for i, (field_name, crop_img, page_num) in enumerate(page_crops):
                result = trocr_results.get(field_name, {})
                conf = result.get("confidence", 0.0)
                
                field_data = {
                    "value": result.get("value", ""),
                    "confidence": conf,
                    "field_type": "handwritten",
                    "source_page": page_num,
                    "selected_model": "TrOCR",
                    "bounding_box": None # Could be added from template matcher
                }
                
                all_fields[field_name] = field_data
                
                # Check if it needs fallback (High threshold to force Qwen on messy Indian forms)
                if conf < 0.85 or not field_data["value"].strip() or len(field_data["value"]) < 2:
                    low_conf_queue.append((field_name, crop_img, page_num, field_data["value"]))
                    
        # Phase 3: Agentic Fallback to Qwen2.5-VL
        if low_conf_queue:
            logger.info(f"Phase 3: Agentic Fallback - Routing {len(low_conf_queue)} low-confidence fields to Qwen-3B...")
            for field_name, crop_img, page_num, trocr_val in low_conf_queue:
                # Ask Qwen to read just the crop and return JSON so the internal parser catches it
                prompt = f"What is the handwritten text in this image crop for the field '{field_name}'? Output ONLY a valid JSON dictionary in this exact format: {{\"fields\": {{\"{field_name}\": {{\"value\": \"extracted text\", \"confidence\": 0.95}}}}}}"
                
                qwen_res = self.qwen.extract_from_image(crop_img, page_num, custom_prompt=prompt)
                
                # Qwen returns a dict due to parsing, but custom prompt might just return raw text or fallback dict
                qwen_val = ""
                qwen_conf = 0.5
                
                if "fields" in qwen_res and field_name in qwen_res["fields"]:
                    qwen_val = qwen_res["fields"][field_name].get("value", "")
                    qwen_conf = qwen_res["fields"][field_name].get("confidence", 0.5)
                elif "fields" in qwen_res and qwen_res["fields"]:
                    # Grab first value
                    first_key = list(qwen_res["fields"].keys())[0]
                    qwen_val = qwen_res["fields"][first_key].get("value", "")
                    qwen_conf = qwen_res["fields"][first_key].get("confidence", 0.5)
                
                # If Qwen is NULL or empty, keep TrOCR. Otherwise use Qwen.
                if qwen_val and qwen_val.lower() != "null":
                    all_fields[field_name]["value"] = qwen_val
                    all_fields[field_name]["confidence"] = max(0.6, qwen_conf) # Boost confidence since VLM verified
                    all_fields[field_name]["selected_model"] = "Qwen2.5-VL-Fallback"
                    all_fields[field_name]["trocr_original"] = trocr_val
        
        # Calculate summary
        total_expected = len(FIELD_NAMES)
        extracted = sum(1 for f in all_fields.values() if f.get("value") and f.get("confidence", 0) > 0.3)
        missing = total_expected - extracted
        low_conf = sum(1 for f in all_fields.values() if 0 < f.get("confidence", 0) < CONFIDENCE_MEDIUM)
        
        confidences = [f.get("confidence", 0) for f in all_fields.values() if f.get("value")]
        overall_conf = float(np.mean(confidences)) if confidences else 0.0
        
        return {
            "fields": all_fields,
            "extraction_summary": {
                "total_fields_expected": total_expected,
                "fields_extracted": extracted,
                "fields_missing": missing,
                "fields_low_confidence": low_conf,
                "overall_confidence": round(overall_conf, 4),
            },
            "models_used": ["TrOCR-Primary", "Qwen2.5-VL-Fallback"],
            "total_pages": len(images),
        }

    def cleanup(self):
        """Release all GPU resources."""
        self.manager.unload_all()
