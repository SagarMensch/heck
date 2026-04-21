"""
Quick check of training data availability and training setup.
"""

import json
from pathlib import Path

# Check what's in the manifests
train_manifest = Path("data/form300_factory/manifests/final_train_manifest.jsonl")
val_manifest = Path("data/form300_factory/manifests/final_val_manifest.jsonl")

print("=== Training Data ===")
train_count = 0
train_families = {}
with open(train_manifest, 'r') as f:
    for line in f:
        row = json.loads(line)
        train_count += 1
        family = row.get('field_family', 'unknown')
        train_families[family] = train_families.get(family, 0) + 1

print(f"Total train samples: {train_count}")
print(f"Field families: {train_families}")

print("\n=== Validation Data ===")
val_count = 0
val_families = {}
with open(val_manifest, 'r') as f:
    for line in f:
        row = json.loads(line)
        val_count += 1
        family = row.get('field_family', 'unknown')
        val_families[family] = val_families.get(family, 0) + 1

print(f"Total val samples: {val_count}")
print(f"Field families: {val_families}")

# Show one sample
print("\n=== Sample Record ===")
with open(train_manifest, 'r') as f:
    sample = json.loads(f.readline())
    print(json.dumps(sample, indent=2))