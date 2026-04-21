# Quick check which PDFs are valid (with timeout)
import os
from pathlib import Path
import fitz
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("PDF took too long")

def check_pdf(path):
    try:
        # Set a timeout
        doc = fitz.open(path)
        num_pages = len(doc)
        doc.close()
        return num_pages > 0
    except:
        return False

pdf_dir = Path("data/lic_samples")
out_dir = Path("data/form300_factory")
out_dir.mkdir(parents=True, exist_ok=True)

valid = []
invalid = []

for pdf_path in sorted(pdf_dir.glob("*.pdf")):
    print(f"Checking {pdf_path.name}...", end=" ", flush=True)
    try:
        doc = fitz.open(str(pdf_path))
        pages = len(doc)
        doc.close()
        print(f"OK ({pages} pages)")
        valid.append((pdf_path.name, pages))
    except Exception as e:
        print(f"FAIL - {str(e)[:40]}")
        invalid.append(pdf_path.name)

print(f"\n=== SUMMARY ===")
print(f"Valid: {len(valid)}")
print(f"Invalid: {len(invalid)}")

# Save list
with open(out_dir / "valid_pdfs.txt", "w") as f:
    for name, pages in valid:
        f.write(f"{name}\n")

print(f"\nSaved to {out_dir / 'valid_pdfs.txt'}")