"""
Download Nemotron Table Structure Model - Fixed Version
===========================================================
"""

import os
import sys

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def download_nemotron():
    model_name = "nvidia/nemotron-table-structure-v1"
    
    print("="*70)
    print("DOWNLOADING NEMOTRON TABLE STRUCTURE MODEL")
    print("="*70)
    print(f"Model: {model_name}")
    print("="*70)
    print()
    
    try:
        from transformers import AutoModelForObjectDetection, AutoProcessor
        import torch
        
        print("Step 1/3: Downloading processor...")
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        print("OK: Processor downloaded")
        print()
        
        print("Step 2/3: Downloading model weights...")
        print("  This will take 10-20 minutes")
        print("  Please wait...")
        print()
        
        model = AutoModelForObjectDetection.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print("OK: Model downloaded successfully!")
        print()
        
        print("Step 3/3: Testing model...")
        device = next(model.parameters()).device
        print(f"OK: Model loaded on device: {device}")
        print()
        
        print("="*70)
        print("DOWNLOAD COMPLETE - Ready for Techathon")
        print("="*70)
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        print()
        print("Troubleshooting:")
        print("  1. Check internet: ping google.com")
        print("  2. Check space: df -h")
        print("  3. Update: pip install --upgrade transformers")
        print()
        return False

if __name__ == "__main__":
    success = download_nemotron()
    sys.exit(0 if success else 1)
