# Ultra-minimal training - just do one forward pass
import sys
sys.stdout = sys.stderr = open("step1.txt", "w", buffering=1)

print("START", flush=True)

import torch
print(f"CUDA: {torch.cuda.is_available()}", flush=True)

import os
model_dir = "models/tocr-trained"
print(f"Model dir exists: {os.path.exists(model_dir)}", flush=True)

from transformers import VisionEncoderDecoderModel
print("Loading model...", flush=True)
model = VisionEncoderDecoderModel.from_pretrained(model_dir)
print("Model loaded!", flush=True)

model = model.to("cuda")
print("Model to CUDA!", flush=True)

from transformers import TrOCRProcessor
print("Loading processor...", flush=True)
processor = TrOCRProcessor.from_pretrained(model_dir)
print("Processor loaded!", flush=True)

import json
with open("data/form300_factory/manifests/final_train_manifest.jsonl") as f:
    row = json.loads(f.readline())

from PIL import Image
img_path = row["image_path"]
if not os.path.exists(img_path):
    img_path = os.path.abspath(img_path)

print(f"Image: {img_path}", flush=True)

img = Image.open(img_path).convert("RGB")
text = row.get("label_text", "")

print(f"Text: {text}", flush=True)

pixel_values = processor(images=img, return_tensors="pt").pixel_values.to("cuda")
labels = processor.tokenizer(text, return_tensors="pt", padding=True).input_ids.to("cuda")

print(f"Pixel shape: {pixel_values.shape}", flush=True)
print(f"Labels shape: {labels.shape}", flush=True)

outputs = model(pixel_values=pixel_values, labels=labels)

print(f"Loss: {outputs.loss.item():.4f}", flush=True)
print("DONE!", flush=True)