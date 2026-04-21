"""Raw PaddleOCR Extractor
========================
Extracts ALL text regions with bbox, text, confidence per page.
No field mapping, no template matching — just raw OCR output.
Saves per-page JSON with every detected region.

Output format per page:
{
  "page_num": 2,
  "page_width": 2538,
  "page_height": 3442,
  "regions": [
    {"bbox": [x1,y1,x2,y2], "text": "...", "confidence": 0.95, "lang": "hi"},
    ...
  ]
}
"""

import os
import json
import time
import logging
import argparse
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("raw_extract")


class RawPaddleExtractor:
    def __init__(self):
        self._hi = None
        self._en = None

    def _get_engine(self, lang):
        from paddleocr import PaddleOCR
        if lang == "hi" and self._hi is None:
            self._hi = PaddleOCR(lang="hi", use_textline_orientation=True)
            logger.info("PaddleOCR Hindi loaded")
        elif lang == "en" and self._en is None:
            self._en = PaddleOCR(lang="en", use_textline_orientation=True)
            logger.info("PaddleOCR English loaded")
        return self._hi if lang == "hi" else self._en

    def ocr_page(self, np_image: np.ndarray, lang: str = "hi") -> List[Dict]:
        engine = self._get_engine(lang)
        try:
            result = engine.ocr(np_image)
            return self._parse(result, lang)
        except Exception as e:
            logger.error(f"PaddleOCR {lang} failed: {e}")
            return []

    def _parse(self, result, lang: str) -> List[Dict]:
        regions = []
        if not result or not result[0]:
            return regions

        res0 = result[0]

        texts = res0.get("rec_texts") if hasattr(res0, "get") else None
        scores = res0.get("rec_scores") if hasattr(res0, "get") else None
        polys = res0.get("rec_polys") if hasattr(res0, "get") else None

        if texts is not None and scores is not None and polys is not None:
            if isinstance(texts, list):
                for text, score, poly in zip(texts, scores, polys):
                    if not isinstance(text, str):
                        text = str(text)
                    try:
                        confidence = float(score)
                    except (ValueError, TypeError):
                        confidence = 0.0
                    if isinstance(poly, (list, tuple, np.ndarray)):
                        pts = np.asarray(poly)
                        if pts.ndim == 2 and pts.shape[0] >= 2 and pts.shape[1] == 2:
                            xs, ys = pts[:, 0], pts[:, 1]
                            regions.append({
                                "bbox": [int(np.min(xs)), int(np.min(ys)),
                                         int(np.max(xs)), int(np.max(ys))],
                                "text": text.strip(),
                                "confidence": round(confidence, 4),
                                "lang": lang,
                            })
                return regions

        if isinstance(result, list) and result and isinstance(result[0], list):
            for line in result[0]:
                if isinstance(line, (list, tuple)) and len(line) >= 2:
                    bbox_data = line[0]
                    second = line[1]
                    if isinstance(second, (list, tuple)) and len(second) == 2:
                        text, confidence = second
                    elif isinstance(second, str):
                        text, confidence = second, 1.0
                    else:
                        continue
                    if isinstance(bbox_data, (list, tuple)) and len(bbox_data) == 4:
                        xs = [p[0] for p in bbox_data]
                        ys = [p[1] for p in bbox_data]
                        regions.append({
                            "bbox": [min(xs), min(ys), max(xs), max(ys)],
                            "text": str(text).strip(),
                            "confidence": round(float(confidence), 4),
                            "lang": lang,
                        })
        return regions

    def ocr_page_bilingual(self, np_image: np.ndarray) -> List[Dict]:
        hi = self.ocr_page(np_image, "hi")
        en = self.ocr_page(np_image, "en")

        seen = set()
        merged = []
        for r in hi + en:
            key = (r["text"].strip().lower()[:40],
                   tuple(r["bbox"]))
            if key not in seen:
                seen.add(key)
                merged.append(r)
        return merged


def pdf_to_pages(pdf_path: str, dpi: int = 300) -> List[Dict]:
    import fitz
    doc = fitz.open(pdf_path)
    pages = []
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    for page_idx in range(len(doc)):
        page = doc.load_page(page_idx)
        pix = page.get_pixmap(matrix=mat)
        img_data = np.frombuffer(pix.samples, dtype=np.uint8)
        img_data = img_data.reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img_data = cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:
            img_data = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
        pages.append({
            "page_num": page_idx + 1,
            "image": img_data,
            "width": pix.width,
            "height": pix.height,
        })
    doc.close()
    return pages


def extract_pdf_raw(pdf_path: str, output_dir: str, pages: List[int] = None,
                    lang: str = "bilingual") -> Dict:
    pdf_name = Path(pdf_path).stem
    logger.info(f"Processing: {pdf_path}")

    all_pages = pdf_to_pages(pdf_path)
    extractor = RawPaddleExtractor()

    if pages:
        all_pages = [p for p in all_pages if p["page_num"] in pages]

    results = []
    total_start = time.time()

    for page_info in all_pages:
        pnum = page_info["page_num"]
        img = page_info["image"]
        h, w = img.shape[:2]

        t0 = time.time()

        if lang == "bilingual":
            regions = extractor.ocr_page_bilingual(img)
        elif lang == "hi":
            regions = extractor.ocr_page(img, "hi")
        else:
            regions = extractor.ocr_page(img, "en")

        elapsed = time.time() - t0
        logger.info(f"  Page {pnum}: {len(regions)} regions in {elapsed:.1f}s ({w}x{h})")

        page_result = {
            "page_num": pnum,
            "page_width": w,
            "page_height": h,
            "ocr_time_seconds": round(elapsed, 2),
            "num_regions": len(regions),
            "regions": regions,
        }
        results.append(page_result)

    total_time = time.time() - total_start

    output = {
        "source_file": str(pdf_path),
        "source_name": pdf_name,
        "total_pages": len(results),
        "total_ocr_time_seconds": round(total_time, 2),
        "lang_mode": lang,
        "pages": results,
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{pdf_name}_raw.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved: {out_path} ({total_time:.1f}s)")

    return output


def main():
    parser = argparse.ArgumentParser(description="Raw PaddleOCR extraction")
    parser.add_argument("input", help="PDF file or folder of PDFs")
    parser.add_argument("--output-dir", default=r"C:\Users\aigcp_gpuadmin\Downloads\LICRFP\LICF\data\raw_ocr",
                        help="Output directory for raw JSON")
    parser.add_argument("--pages", type=int, nargs="+", default=None,
                        help="Specific page numbers to process (1-based)")
    parser.add_argument("--lang", choices=["bilingual", "hi", "en"],
                        default="bilingual", help="OCR language mode")
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.is_file():
        extract_pdf_raw(str(input_path), args.output_dir, args.pages, args.lang)
    elif input_path.is_dir():
        pdfs = sorted(input_path.glob("*.pdf"))
        logger.info(f"Found {len(pdfs)} PDFs in {input_path}")
        for i, pdf in enumerate(pdfs):
            logger.info(f"[{i+1}/{len(pdfs)}] {pdf.name}")
            try:
                extract_pdf_raw(str(pdf), args.output_dir, args.pages, args.lang)
            except Exception as e:
                logger.error(f"FAILED {pdf.name}: {e}")
    else:
        logger.error(f"Input not found: {input_path}")


if __name__ == "__main__":
    main()
