"""Test saved Qwen bounding boxes using the hot local Qwen service.

This renders one PDF page, aligns it, crops fields using the saved Qwen boxes,
and sends each crop to the hot Qwen server for extraction.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import urllib.request
from pathlib import Path

import fitz
import cv2
import numpy as np

from src.canonical_extractor import CanonicalExtractor
from src.form300_templates import PAGE_TYPE_TO_PAGE_NUM


HOT_SERVER_URL = "http://127.0.0.1:8765/extract_crop"


def call_hot_qwen(image_path: Path, field_name: str) -> dict:
    payload = json.dumps(
        {
            "image_path": str(image_path),
            "field_name": field_name,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        HOT_SERVER_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode("utf-8"))


def render_page(pdf_path: Path, page_num: int, dpi: int = 200) -> np.ndarray:
    doc = fitz.open(str(pdf_path))
    page = doc.load_page(page_num - 1)
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    doc.close()
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
    if pix.n == 4:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path")
    parser.add_argument("page_num", type=int)
    parser.add_argument("--output", default="")
    parser.add_argument(
        "--fields",
        default="",
        help="Comma-separated field names to test, e.g. first_name,last_name,date_of_birth",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    output_path = Path(args.output) if args.output else pdf_path.with_name(f"{pdf_path.stem}_page{args.page_num}_qwen_hot.json")

    img = render_page(pdf_path, args.page_num)
    extractor = CanonicalExtractor(audit_dir=str(Path("data") / "qwen_hot_test_audit" / pdf_path.stem))
    crops = extractor.extract_template_fields(img, args.page_num)
    selected_fields = {
        token.strip()
        for token in args.fields.split(",")
        if token.strip()
    }
    if selected_fields:
        crops = {name: crop for name, crop in crops.items() if name in selected_fields}

    results = {}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="qwen_hot_page_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        for field_name, crop in crops.items():
            crop_path = tmpdir_path / f"{field_name}.png"
            cv2.imwrite(str(crop_path), crop)
            try:
                result = call_hot_qwen(crop_path, field_name)
            except Exception as exc:
                result = {"value": "", "confidence": 0.0, "error": str(exc)}
            results[field_name] = result
            output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"{field_name}: {result.get('value', '')} (conf={result.get('confidence', 0.0)})")

    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
