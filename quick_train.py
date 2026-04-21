"""
Quick training test - minimal version.
"""

import json
import logging
from pathlib import Path
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load data
train_manifest = Path("data/form300_factory/manifests/final_train_manifest.jsonl")
val_manifest = Path("data/form300_factory/manifests/final_val_manifest.jsonl")

def load_manifest(path):
    rows = []
    with open(path, 'r') as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

train_rows = load_manifest(train_manifest)
val_rows = load_manifest(val_manifest)

logger.info(f"Loaded {len(train_rows)} train, {len(val_rows)} val samples")

# Dataset class
class TrOCRDataset(Dataset):
    def __init__(self, rows, processor, max_length=32):
        self.rows = rows
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img = Image.open(row['image_path']).convert("RGB")
        text = row.get('label_text', '')

        pixel_values = self.processor(images=img, return_tensors="pt").pixel_values.squeeze()
        labels = self.processor.tokenizer(
            text, padding="max_length", max_length=self.max_length,
            truncation=True, return_tensors="pt"
        ).input_ids.squeeze()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels}

# Load model and processor
logger.info("Loading TrOCR processor...")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")

logger.info("Loading TrOCR model...")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# Configure
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

# Apply LoRA
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Create datasets (small for quick test)
train_ds = TrOCRDataset(train_rows[:10], processor)  # Just 10 samples
val_ds = TrOCRDataset(val_rows[:5], processor)  # Just 5 samples

train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=2)

# Training loop - just 1 epoch, 5 steps
logger.info("Starting training...")
model.to("cuda" if torch.cuda.is_available() else "cpu")

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

for epoch in range(1):
    model.train()
    for i, batch in enumerate(train_loader):
        if i >= 5:  # Just 5 steps
            break
        
        pixel_values = batch['pixel_values'].to(model.device)
        labels = batch['labels'].to(model.device)
        
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        logger.info(f"Step {i+1}/5, Loss: {loss.item():.4f}")

logger.info("Training complete!")

# Save model
output_dir = Path("models/tocr-quick-test")
output_dir.mkdir(parents=True, exist_ok=True)
model.save_pretrained(output_dir)
processor.save_pretrained(output_dir)
logger.info(f"Model saved to {output_dir}")