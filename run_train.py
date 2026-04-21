# Simplified training script
import os
import json
from pathlib import Path
import traceback

log = open("train.log", "w")

def log_msg(msg):
    log.write(msg + "\n")
    log.flush()

log_msg("=== Starting Training ===")

try:
    # Use cached model path
    cache = r'C:\Users\aigcp_gpuadmin\.cache\huggingface\hub\models--microsoft--trocr-base-handwritten\snapshots\eaacaf452b06415df8f10bb6fad3a4c11e609406'
    log_msg(f"Cache: {cache}")
    log_msg(f"Cache exists: {os.path.exists(cache)}")

    # Try loading model first (smaller than processor)
    log_msg("Loading VisionEncoderDecoderModel...")
    from transformers import VisionEncoderDecoderModel
    model = VisionEncoderDecoderModel.from_pretrained(cache)
    log_msg("Model loaded!")

    log_msg("Loading processor...")
    from transformers import TrOCRProcessor
    processor = TrOCRProcessor.from_pretrained(cache)
    log_msg("Processor loaded!")

    # Load data
    log_msg("Loading data...")
    with open("data/form300_factory/manifests/final_train_manifest.jsonl") as f:
        row = json.loads(f.readline())
    
    img_path = row["image_path"]
    if not os.path.exists(img_path):
        img_path = os.path.abspath(img_path)
    log_msg(f"Image: {img_path}")
    log_msg(f"Image exists: {os.path.exists(img_path)}")

    from PIL import Image
    img = Image.open(img_path).convert("RGB")
    text = row.get("label_text", "")
    log_msg(f"Text: {text}")

    # Process
    log_msg("Processing...")
    from transformers import AutoProcessor
    import torch
    pixel_values = processor(images=img, return_tensors="pt").pixel_values
    labels = processor.tokenizer(text, return_tensors="pt").input_ids
    log_msg(f"Pixel shape: {pixel_values.shape}")

    # Forward
    log_msg("Running forward...")
    model.to("cuda")
    outputs = model(pixel_values=pixel_values, labels=labels)
    log_msg(f"Loss: {outputs.loss.item():.4f}")

    # Save
    log_msg("Saving...")
    output_dir = Path("models/tocr-trained")
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    
    log_msg("DONE!")
    
except Exception as e:
    log_msg(f"ERROR: {e}")
    log_msg(traceback.format_exc())

log.close()