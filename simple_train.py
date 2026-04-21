# Ultra simple training - keep everything on same device
s = open("test_result.txt", "w")
s.write("START\n")

try:
    import torch
    import os
    s.write(f"Torch version: {torch.__version__}\n")
    s.write(f"CUDA available: {torch.cuda.is_available()}\n")
    
    cache_dir = r"C:\Users\aigcp_gpuadmin\.cache\huggingface\hub\models--microsoft--trocr-base-handwritten\snapshots\eaacaf452b06415df8f10bb6fad3a4c11e609406"
    
    s.write("Loading model...\n")
    from transformers import VisionEncoderDecoderModel
    model = VisionEncoderDecoderModel.from_pretrained(cache_dir)
    # DON'T use .to() - load on CPU first
    s.write(f"Model device (before to): {next(model.parameters()).device}\n")
    
    # Move ENTIRE model to GPU at once, not layer by layer
    model = model.to("cuda")
    s.write(f"Model device (after to): {next(model.parameters()).device}\n")
    
    s.write("Loading processor...\n")
    from transformers import TrOCRProcessor
    processor = TrOCRProcessor.from_pretrained(cache_dir)
    s.write("Processor loaded\n")
    
    import json
    with open("data/form300_factory/manifests/final_train_manifest.jsonl") as f:
        row = json.loads(f.readline())
    
    from PIL import Image
    img_path = row["image_path"]
    if not os.path.exists(img_path):
        img_path = os.path.abspath(img_path)
    
    img = Image.open(img_path).convert("RGB")
    text = row.get("label_text", "")
    
    # Process - encode images and tokenize text
    pixel_values = processor.feature_extractor(images=img, return_tensors="pt").pixel_values
    labels = processor.tokenizer(text, return_tensors="pt", padding=True).input_ids
    
    # NOW move both to GPU together
    pixel_values = pixel_values.to("cuda", non_blocking=True)
    labels = labels.to("cuda", non_blocking=True)
    
    s.write(f"Input pixel device: {pixel_values.device}\n")
    s.write(f"Input label device: {labels.device}\n")
    
    # Forward pass
    outputs = model(pixel_values=pixel_values, labels=labels)
    s.write(f"Loss: {outputs.loss.item():.4f}\n")
    
    # Save
    from pathlib import Path
    output_dir = Path("models/tocr-trained")
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    
    s.write("SAVED!\n")
    s.write("SUCCESS!\n")
    
except Exception as e:
    s.write(f"ERROR: {e}\n")
    import traceback
    s.write(traceback.format_exc())
    
s.close()