# Build synthetic-only dataset (skip PDF extraction which is slow)
import json
from pathlib import Path

from src.dataset_factory import CPUForm300DatasetBuilder

def main():
    # Use synthetic-only mode - don't process PDFs
    # The build() will fall back to synthetic backgrounds
    
    # Clean output directory
    out_dir = Path("data/form300_factory_synthetic")
    if out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)
    
    builder = CPUForm300DatasetBuilder(out_dir, seed=7)
    
    # Build with NO pdf_dir - will use only synthetic data
    summary = builder.build(
        pdf_dir=None,  # Skip real PDF extraction
        num_records=500,  # More records = more training data
        num_projected_records=100,
    )
    
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()