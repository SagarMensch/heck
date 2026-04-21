"""Extract values from saved consensus coordinates using the hot Qwen server."""

from __future__ import annotations

import argparse
import json
import tempfile
import urllib.request
from pathlib import Path

import cv2
import fitz
import numpy as np

from src.canonical_extractor import CanonicalExtractor


HOT_EXTRACT_URL = "http://127.0.0.1:8765/extract_crop"


def post_json(url: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
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
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
    if pix.n == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def is_missing(value) -> bool:
    if value is None:
        return True
    text = str(value).strip().lower()
    return text in {"", "none", "null", "not found"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path")
    parser.add_argument("coords_json")
    parser.add_argument("--page", type=int, default=2)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    coords_path = Path(args.coords_json)
    output_path = Path(args.output) if args.output else coords_path.with_name(coords_path.stem + "_extracted.json")

    coord_data = json.loads(coords_path.read_text(encoding="utf-8"))
    extractor = CanonicalExtractor(audit_dir=str(Path("data") / "consensus_extract_audit" / pdf_path.stem))
    image = render_page(pdf_path, args.page)
    registration = extractor.align_page(image, args.page)
    aligned = registration.aligned_bgr
    cleaned = extractor.registrar.subtract_reference_form(aligned, args.page)

    results = {}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="consensus_extract_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        for field_name, item in coord_data.items():
            bbox = item.get("bbox")
            if not bbox or len(bbox) != 4:
                results[field_name] = {"value": None, "confidence": 0.0, "status": "missing_bbox"}
                continue

            x1, y1, x2, y2 = [int(v) for v in bbox]
            aligned_crop = aligned[y1:y2, x1:x2]
            cleaned_crop = cleaned[y1:y2, x1:x2]

            aligned_path = tmpdir_path / f"{field_name}_aligned.png"
            cleaned_path = tmpdir_path / f"{field_name}_cleaned.png"
            cv2.imwrite(str(aligned_path), aligned_crop)
            cv2.imwrite(str(cleaned_path), cleaned_crop)

            aligned_res = post_json(HOT_EXTRACT_URL, {"image_path": str(aligned_path), "field_name": field_name})
            cleaned_res = post_json(HOT_EXTRACT_URL, {"image_path": str(cleaned_path), "field_name": field_name})

            pick = aligned_res
            source = "aligned"
            if is_missing(aligned_res.get("value")) and not is_missing(cleaned_res.get("value")):
                pick = cleaned_res
                source = "cleaned"
            elif float(cleaned_res.get("confidence", 0.0) or 0.0) > float(aligned_res.get("confidence", 0.0) or 0.0):
                pick = cleaned_res
                source = "cleaned"

            results[field_name] = {
                "bbox": bbox,
                "value": pick.get("value"),
                "confidence": pick.get("confidence", 0.0),
                "source": source,
                "aligned_result": aligned_res,
                "cleaned_result": cleaned_res,
            }
            output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"{field_name}: {pick.get('value')} (conf={pick.get('confidence', 0.0)}, source={source})")

    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
