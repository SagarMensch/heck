# Ultra simple training - single step
import json
import os
from pathlib import Path
import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, TrOCRProcessor

# Set paths
model_dir = "models/tocr-trained"
train_manifest = "data/form300_factory/manifests/final_train_manifest.jsonl"
output_dir = Path("models/tocr-finetuned")
output_dir.mkdir(parents=True, exist_ok=True)

print("Loading model...")
model = VisionEncoderDecoderModel.from_pretrained(model_dir, local_files_only=True)
model.config.decoder_start_token_id = 2
model.config.pad_token_id = 1
model = model.to("cuda")
print("Model loaded to CUDA")

print("Loading processor...")
processor = TrOCRProcessor.from_pretrained(model_dir, local_files_only=True)
print("Processor loaded")

print("Loading training data...")
with open(train_manifest, 'r') as f:
    train_rows = [json.loads(line) for line in f]

# Filter for quick training
train_rows = [r for r in train_rows if r.get('field_family') in ['date', 'short_id', 'proposal_number']][:5]
print(f"Training on {len(train_rows)} samples")

# Simple training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
model.train()

for epoch in range(1):
    total_loss = 0
    for i, row in enumerate(train_rows):
        img_path = row['image_path']
        if not os.path.exists(img_path):
            img_path = os.path.abspath(img_path)
        
        img = Image.open(img_path).convert("RGB")
        text = row.get('label_text', '')
        
        pixel_values = processor(images=img, return_tensors="pt").pixel_values.to("cuda")
        labels = processor.tokenizer(text, return_tensors="pt", padding=True).input_ids.to("cuda")
        
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        print(f"Step {i+1}/{len(train_rows)}, Loss: {loss.item():.4f}")
    
    print(f"Epoch {epoch+1} complete, Avg Loss: {total_loss/len(train_rows):.4f}")

print("Saving model...")
model.save_pretrained(output_dir)
processor.save_pretrained(output_dir)
print(f"Saved to {output_dir}")
print("DONE!")