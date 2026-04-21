#!/usr/bin/env python
"""PaddleOCR v5 GPU Extraction - Single page test"""
import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import cv2
import numpy as np
import fitz

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "1"

PDF_PATH = r'Techathon_Samples\P02.pdf'
PAGE_NUM = 2
OUTPUT_DIR = r'data/simple_test'
os.makedirs(OUTPUT_DIR, exist_ok=True)

from paddleocr import PaddleOCR

print("Loading PaddleOCR v5 (English + Hindi)...")
ocr_en = PaddleOCR(use_textline_orientation=True, lang='en', text_det_thresh=0.3, text_det_box_thresh=0.5)
ocr_hi = PaddleOCR(use_textline_orientation=True, lang='hi', text_det_thresh=0.3, text_det_box_thresh=0.5)
print("OCR loaded!")

def parse_paddle_result(result, lang):
    regions = []
    if not result:
        return regions
    res0 = result[0] if isinstance(result, list) else result
    texts = res0.get("rec_texts") if hasattr(res0, "get") else None
    scores = res0.get("rec_scores") if hasattr(res0, "get") else None
    polys = res0.get("rec_polys") if hasattr(res0, "get") else None
    if texts is not None and scores is not None and polys is not None:
        for text, score, poly in zip(texts, scores, polys):
            pts = np.asarray(poly)
            if pts.ndim == 2 and pts.shape[0] >= 2 and pts.shape[1] == 2:
                xs, ys = pts[:, 0], pts[:, 1]
                regions.append({
                    'text': str(text).strip(),
                    'confidence': float(score),
                    'bbox': [int(np.min(xs)), int(np.min(ys)), int(np.max(xs)), int(np.max(ys))],
                    'lang': lang,
                    'y_pos': int(np.min(ys)),
                })
        return regions
    if isinstance(result, list) and result and isinstance(result[0], list):
        for line in result[0]:
            if isinstance(line, (list, tuple)) and len(line) >= 2:
                bbox_data = line[0]
                second = line[1]
                if isinstance(second, (list, tuple)) and len(second) == 2:
                    text, confidence = second
                else:
                    continue
                if isinstance(bbox_data, (list, tuple)) and len(bbox_data) == 4:
                    ys = [p[1] for p in bbox_data]
                    xs = [p[0] for p in bbox_data]
                    regions.append({
                        'text': str(text).strip(),
                        'confidence': float(confidence),
                        'bbox': [min(xs), min(ys), max(xs), max(ys)],
                        'lang': lang,
                        'y_pos': min(ys),
                    })
    return regions

def extract_all_text(img):
    all_results = []
    print("  Running English OCR...")
    en_result = ocr_en.predict(img)
    for r in en_result:
        all_results.extend(parse_paddle_result(r, 'en'))

    print("  Running Hindi OCR...")
    hi_result = ocr_hi.predict(img)
    for r in hi_result:
        all_results.extend(parse_paddle_result(r, 'hi'))

    all_results.sort(key=lambda x: x.get('y_pos', 0))
    return all_results

print(f"\nConverting {PDF_PATH} page {PAGE_NUM} to image...")
doc = fitz.open(PDF_PATH)
page = doc.load_page(PAGE_NUM - 1)
pix = page.get_pixmap(dpi=200)
img = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
if pix.n == 4:
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
elif pix.n == 3:
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
doc.close()

img_path = os.path.join(OUTPUT_DIR, f'page{PAGE_NUM}.png')
cv2.imwrite(img_path, img)
print(f"Saved: {img_path} ({pix.width}x{pix.height})")

print("\nRunning OCR...")
results = extract_all_text(img)

print("\n" + "="*100)
print("ALL EXTRACTED TEXT (sorted by position, conf>0.5):")
print("="*100)

meaningful = 0
for i, item in enumerate(results, 1):
    text = item['text']
    conf = item['confidence']
    lang = item['lang']
    y_pos = item.get('y_pos', 0)
    if conf > 0.5 and len(text.strip()) > 1:
        meaningful += 1
        print(f"{i:3}. [{lang}] {text[:100]:<100} (conf: {conf:.2f}, y: {y_pos:4.0f})")

print("\n" + "="*100)
print(f"Total regions: {len(results)}")
print(f"Meaningful (conf>0.5, len>1): {meaningful}")
