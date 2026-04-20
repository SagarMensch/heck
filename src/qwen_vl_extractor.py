"""
SOTA Qwen2.5-VL Extractor
The "End-to-End" Approach. No coordinates. No cropping. No guessing.
Just Image + Prompt -> JSON.
"""
import os
import torch
from typing import Dict, Any
import json

# Check for GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

class QwenVLExtractor:
    def __init__(self, model_path="Qwen/Qwen2.5-VL-7B-Instruct"):
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.loaded = False

    def load(self):
        if self.loaded:
            return
        print(f"Loading Qwen2.5-VL from {self.model_path}...")
        try:
            from transformers import AutoProcessor, AutoModelForVision2Seq
            
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
            )
            if device == "cpu":
                self.model = self.model.to(device)
            self.loaded = True
            print("Qwen2.5-VL loaded successfully.")
        except Exception as e:
            print(f"Failed to load Qwen2.5-VL: {e}")
            print("Ensure you have transformers, accelerate, and torch installed.")
            raise e

    def extract(self, image_path: str, fields: list) -> Dict[str, Any]:
        """
        Extract specific fields from an image using Qwen2.5-VL.
        """
        self.load()
        from PIL import Image
        image = Image.open(image_path)

        # Construct the prompt
        field_list = ", ".join(fields)
        prompt = (
            f"You are an expert data extraction AI. Analyze the provided image of an LIC Form 300. "
            f"Extract the following handwritten fields: {field_list}. "
            "Ignore all printed text, labels, and instructions. "
            "Only extract the handwritten values filled in by the user. "
            "If a field is empty or not found, return null for that field. "
            "Output the result as a valid JSON object. Do not include markdown formatting."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Prepare inputs
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False, # Greedy decoding for strict JSON
            )
        
        # Decode
        generated_ids_trimmed = [
            single_pred[prompt_len:] for single_pred, prompt_len in zip(generated_ids, inputs.input_shape[1:])
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # Parse JSON
        try:
            # Clean up response (remove markdown code blocks if any)
            response = response.replace("```json", "").replace("```", "").strip()
            return json.loads(response)
        except json.JSONDecodeError:
            return {"raw_response": response, "error": "Failed to parse JSON"}

def run_qwen_extraction(image_path: str):
    extractor = QwenVLExtractor()
    
    fields_to_extract = [
        "Proposer Full Name", "Father's Full Name", "Mother's Full Name",
        "Gender", "Marital Status", "Date of Birth", "Age", 
        "Place of Birth", "Nationality", "Citizenship", 
        "Permanent Address", "PIN Code", "Phone Number"
    ]
    
    print(f"Extracting from {image_path}...")
    result = extractor.extract(image_path, fields_to_extract)
    
    print("\n" + "="*50)
    print("EXTRACTED DATA (Qwen2.5-VL)")
    print("="*50)
    print(json.dumps(result, indent=2))
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python qwen_vl_extractor.py <image_path>")
        sys.exit(1)
    run_qwen_extraction(sys.argv[1])
