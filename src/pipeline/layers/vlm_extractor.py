"""
Layer 2: VLM-First Extraction Engine
======================================
Qwen2.5-VL directly understands the form, reads handwriting,
and outputs structured JSON. This is the PRIMARY extraction layer.

Features:
  - Smart tiling with per-tile VLM extraction
  - Multi-tile field merging with affinity scoring
  - Page-type-aware prompts
  - JSON repair for malformed VLM output
"""

import os
import sys
import json
import logging
import time
import re
from typing import Dict, List, Optional, Tuple
from PIL import Image

import torch

logger = logging.getLogger(__name__)

FIELDS = [
    "Proposer_Full_Name", "Proposer_First_Name", "Proposer_Middle_Name", "Proposer_Last_Name",
    "Proposer_Prefix", "Proposer_Father_Husband_Name", "Proposer_Mother_Name",
    "Proposer_Gender", "Proposer_Marital_Status", "Proposer_Spouse_Name",
    "Proposer_Date_of_Birth", "Proposer_Age", "Proposer_Birth_Place",
    "Proposer_Nationality", "Proposer_Citizenship",
    "Proposer_Customer_ID", "Proposer_CKYC",
    "Proposer_Address_Line1", "Proposer_Address_Line2",
    "Proposer_City", "Proposer_State", "Proposer_Pincode",
    "Proposer_Mobile_Number", "Proposer_Email",
    "Proposer_Aadhaar", "Proposer_PAN", "Proposer_Occupation", "Proposer_Annual_Income",
    "LA_Full_Name", "LA_Date_of_Birth", "LA_Age", "LA_Relationship",
    "Plan_Name", "Plan_Number", "Policy_Term", "Premium_Paying_Term",
    "Sum_Assured", "Premium_Amount", "Premium_Mode",
    "Nominee_Name", "Nominee_Relationship", "Nominee_Age", "Nominee_Address",
    "Bank_Account_Number", "Bank_Name", "Bank_IFSC", "Bank_Branch",
    "Agent_Code", "Agent_Name", "Branch_Code",
    "Date_of_Proposal", "Place_of_Signing",
]

FIELDS_JSON = json.dumps({f: None for f in FIELDS}, indent=2)

TILE_PROMPT = """You are an expert insurance document extraction AI. This is a section of an LIC (Life Insurance Corporation of India) Proposal Form.

The form is bilingual: English + Hindi/Marathi. Focus on FILLED-IN values (handwritten or typed), NOT printed labels.

CRITICAL RULES:
1. Read handwritten text very carefully — it may be in English or Devanagari script
2. Separate First Name, Middle Name, Last Name into individual fields
3. Father/Husband name: extract the full name as written next to the label
4. For gender/marital status: check which checkbox is marked (tick/cross)
5. For dates: use DD/MM/YYYY format
6. For PIN: extract exactly 6 digits
7. For phone: include STD code if present
8. Do NOT guess or fabricate values
9. Use null for fields NOT visible in this tile section

Return ONLY valid JSON with these keys:""" + "\n" + FIELDS_JSON

HALF_TOP_HINT = "\n\nNOTE: This is the TOP half of the page — personal details area (name, father, mother, gender, marital status, DOB, age, birth place, nationality, citizenship)."

HALF_BOTTOM_HINT = "\n\nNOTE: This is the BOTTOM half of the page — address area (house, street, city, state, PIN, phone, email, Aadhaar, PAN)."


class VLMExtractor:

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self._process_vision_info = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        logger.info(f"Loading VLM: {self.model_name}...")
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info
        self._process_vision_info = process_vision_info
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            dtype=torch.float16,
            device_map="auto",
        )
        self._loaded = True
        logger.info(f"VLM loaded on GPU")

    def extract_tile(self, pil_img: Image.Image, hint: str = "") -> Dict[str, Optional[str]]:
        self.load()
        prompt = TILE_PROMPT + hint
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": prompt},
            ]}
        ]

        text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self._process_vision_info(messages)
        inputs = self.processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=2048, do_sample=False)

        output = self.processor.batch_decode(
            output_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0].strip()

        del inputs, output_ids
        torch.cuda.empty_cache()

        return self._parse_json(output)

    def extract_page(self, tiles: List[Tuple[Image.Image, str]]) -> Tuple[Dict[str, Optional[str]], Dict[str, Dict]]:
        tile_results = {}
        raw_outputs = {}

        for tile_img, tile_name in tiles:
            hint = ""
            if tile_name == "half_top":
                hint = HALF_TOP_HINT
            elif tile_name == "half_bottom":
                hint = HALF_BOTTOM_HINT

            logger.info(f"  VLM extracting {tile_name}...")
            t0 = time.time()
            result = self.extract_tile(tile_img, hint)
            elapsed = time.time() - t0
            found = sum(1 for v in result.values() if v is not None)
            logger.info(f"  {tile_name}: {found} fields in {elapsed:.1f}s")

            tile_results[tile_name] = result
            raw_outputs[tile_name] = {"fields_found": found, "elapsed_s": round(elapsed, 1)}

        merged = FieldMerger().merge(tile_results)
        return merged, raw_outputs

    def _parse_json(self, raw: str) -> Dict[str, Optional[str]]:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:])
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()

        # Try direct parse
        try:
            parsed = json.loads(cleaned)
            return self._normalize(parsed)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in the text
        brace_start = cleaned.find("{")
        brace_end = cleaned.rfind("}")
        if brace_start >= 0 and brace_end > brace_start:
            try:
                parsed = json.loads(cleaned[brace_start:brace_end + 1])
                return self._normalize(parsed)
            except json.JSONDecodeError:
                pass

        # Try to fix common VLM JSON errors (trailing commas, unquoted keys)
        fixed = re.sub(r',\s*}', '}', cleaned)
        fixed = re.sub(r',\s*]', ']', fixed)
        try:
            parsed = json.loads(fixed)
            return self._normalize(parsed)
        except json.JSONDecodeError:
            logger.warning(f"VLM JSON parse failed: {raw[:200]}")
            return {}

    def _normalize(self, parsed: Dict) -> Dict[str, Optional[str]]:
        out = {}
        null_values = {"", "n/a", "na", "not found", "not visible", "none", "null", "nil"}
        for k, v in parsed.items():
            if isinstance(v, str) and v.strip().lower() in null_values:
                out[k] = None
            elif v is None:
                out[k] = None
            elif isinstance(v, str):
                out[k] = v.strip()
            else:
                out[k] = str(v).strip()
        return out


class FieldMerger:
    TILE_PRIORITY = {
        "half_top": 3, "half_bottom": 3,
        "tile_r0_c0": 2, "tile_r0_c1": 2,
        "tile_r1_c0": 2, "tile_r1_c1": 2,
        "tile_r2_c0": 2, "tile_r2_c1": 2,
    }

    FIELD_AFFINITY = {
        "Proposer_First_Name": ["half_top", "tile_r0_c1"],
        "Proposer_Middle_Name": ["half_top", "tile_r0_c1"],
        "Proposer_Last_Name": ["half_top", "tile_r0_c1"],
        "Proposer_Father_Husband_Name": ["half_top", "tile_r1_c0", "tile_r1_c1"],
        "Proposer_Mother_Name": ["half_top", "tile_r1_c0", "tile_r1_c1"],
        "Proposer_Gender": ["half_top", "tile_r1_c0", "tile_r1_c1"],
        "Proposer_Marital_Status": ["half_top", "tile_r1_c0", "tile_r1_c1"],
        "Proposer_Spouse_Name": ["half_top", "tile_r1_c0", "tile_r1_c1"],
        "Proposer_Date_of_Birth": ["half_top", "tile_r1_c0", "tile_r1_c1"],
        "Proposer_Age": ["half_top", "tile_r1_c0", "tile_r1_c1"],
        "Proposer_House_No": ["half_bottom", "tile_r2_c0"],
        "Proposer_Address_Line1": ["half_bottom", "tile_r2_c0"],
        "Proposer_Address_Line2": ["half_bottom", "tile_r2_c0"],
        "Proposer_City": ["half_bottom", "tile_r2_c0", "tile_r2_c1"],
        "Proposer_State": ["half_bottom", "tile_r2_c0", "tile_r2_c1"],
        "Proposer_Pincode": ["half_bottom", "tile_r2_c0", "tile_r2_c1"],
        "Proposer_Mobile_Number": ["half_bottom", "tile_r2_c0", "tile_r2_c1"],
    }

    def merge(self, tile_results: Dict[str, Dict[str, Optional[str]]]) -> Dict[str, Optional[str]]:
        merged = {}
        for field_name in FIELDS:
            candidates = {}
            for tile_name, tile_data in tile_results.items():
                val = tile_data.get(field_name)
                if val is not None and str(val).strip():
                    candidates[tile_name] = str(val).strip()

            if not candidates:
                merged[field_name] = None
                continue

            if len(candidates) == 1:
                merged[field_name] = list(candidates.values())[0]
                continue

            affinity = self.FIELD_AFFINITY.get(field_name, [])
            best_tile, best_val, best_priority = None, None, -1

            for tile_name, val in candidates.items():
                p = self.TILE_PRIORITY.get(tile_name, 1)
                if tile_name in affinity:
                    p += 5
                p += len(val) * 0.01
                if p > best_priority:
                    best_priority = p
                    best_tile = tile_name
                    best_val = val

            merged[field_name] = best_val

        return merged
