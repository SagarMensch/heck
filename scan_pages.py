"""Dual-language PaddleOCR scan of all 28 pages of LIC Form 300 sample PDFs."""
import sys
import os
import json

sys.stdout.reconfigure(encoding="utf-8")
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import fitz
import numpy as np
import cv2

from paddleocr import PaddleOCR

print("Loading PaddleOCR engines...")
ocr_hi = PaddleOCR(lang="hi", use_textline_orientation=True)
ocr_en = PaddleOCR(lang="en", use_textline_orientation=True)
print("Engines loaded.")

sample_dir = r"C:\Users\aigcp_gpuadmin\Downloads\LICRFP\LICF\Techathon_Samples"
pdfs = sorted([f for f in os.listdir(sample_dir) if f.endswith(".pdf")])[:5]

catalog = {}

for pdf_name in pdfs:
    pdf_path = os.path.join(sample_dir, pdf_name)
    doc = fitz.open(pdf_path)
    print(f"\n{'='*60}")
    print(f"{pdf_name} ({len(doc)} pages)")
    print(f"{'='*60}")

    for pg in range(len(doc)):
        page = doc.load_page(pg)
        zoom = 200 / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        r_hi = ocr_hi.ocr(img)
        hi_texts = []
        if r_hi and r_hi[0]:
            t = r_hi[0].get("rec_texts") if hasattr(r_hi[0], "get") else None
            if t is not None:
                hi_texts = [str(x)[:100] for x in list(t)[:20]]
            elif isinstance(r_hi[0], list):
                hi_texts = [str(line[1][0])[:100] if len(line) >= 2 else "" for line in r_hi[0][:20]]

        r_en = ocr_en.ocr(img)
        en_texts = []
        if r_en and r_en[0]:
            t = r_en[0].get("rec_texts") if hasattr(r_en[0], "get") else None
            if t is not None:
                en_texts = [str(x)[:100] for x in list(t)[:20]]
            elif isinstance(r_en[0], list):
                en_texts = [str(line[1][0])[:100] if len(line) >= 2 else "" for line in r_en[0][:20]]

        key = f"{pdf_name}_p{pg+1}"
        catalog[key] = {"hi": hi_texts, "en": en_texts}

        hi_str = " | ".join(hi_texts) if hi_texts else "NONE"
        en_str = " | ".join(en_texts) if en_texts else "NONE"
        print(f"  Page {pg+1}:")
        print(f"    HI: {hi_str[:300]}")
        print(f"    EN: {en_str[:300]}")

    doc.close()

out = r"C:\Users\aigcp_gpuadmin\Downloads\LICRFP\LICF\page_type_catalog.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(catalog, f, indent=2, ensure_ascii=False)
print(f"\nSaved {out}")
