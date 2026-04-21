"""
Download Nemotron Table Structure Model
==========================================
Pre-download before Techathon to avoid runtime delays.
"""

import os
import sys

# Disable symlink warnings
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def download_nemotron():
    """Download Nemotron table structure model."""
    
    model_name = "nvidia/nemotron-table-structure-v1"
    cache_dir = "./models/nemotron"
    
    print("="*70)
    print("DOWNLOADING NEMOTRON TABLE STRUCTURE MODEL")
    print("="*70)
    print(f"Model: {model_name}")
    print(f"Cache: {cache_dir}")
    print(f"Size: ~8-12 GB")
    print("="*70)
    print()
    
    try:
        from transformers import AutoModelForObjectDetection, AutoProcessor
        import torch
        
        print("Step 1/3: Downloading processor...")
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_dir
        )
        print("✓ Processor downloaded")
        print()
        
        print("Step 2/3: Downloading model weights...")
        print("  This will take 10-20 minutes depending on internet speed")
        print("  Please wait...")
        print()
        
        model = AutoModelForObjectDetection.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=cache_dir
        )
        
        print("✓ Model downloaded successfully!")
        print()
        
        # Test load
        print("Step 3/3: Testing model load...")
        device = next(model.parameters()).device
        print(f"✓ Model loaded on device: {device}")
        print()
        
        # Get model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,}")
        print()
        
        print("="*70)
        print("DOWNLOAD COMPLETE")
        print("="*70)
        print(f"Model cached at: {cache_dir}")
        print()
        print("You can now run:")
        print("  python test_pipeline_v2.py")
        print()
        
        return True
        
    except Exception as e:
        print(f"✗ Download failed: {e}")
        print()
        print("Troubleshooting:")
        print("  1. Check internet connection")
        print("  2. Check disk space: df -h")
        print("  3. Try: pip install --upgrade transformers")
        print()
        return False


if __name__ == "__main__":
    success = download_nemotron()
    sys.exit(0 if success else 1)
