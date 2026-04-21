import json
import urllib.request
import cv2
import fitz
import numpy as np
from pathlib import Path
from PIL import Image

pdf_path = 'Techathon_Samples/P12.pdf'
dpi = 160
url = 'http://127.0.0.1:8765/extract_crop'
crop_dir = Path('data/p12_crops')
crop_dir.mkdir(parents=True, exist_ok=True)

def render_and_crop(pdf_path, page_num, box_norm, dpi=160):
    doc = fitz.open(str(pdf_path))
    page = doc.load_page(page_num - 1)
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    doc.close()
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR) if pix.n == 4 else cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    w, h = img.size
    x1, y1, x2, y2 = int(box_norm[0]*w), int(box_norm[1]*h), int(box_norm[2]*w), int(box_norm[3]*h)
    return img.crop((x1, y1, x2, y2))

def extract(crop_path, field_name):
    payload = {"image_path": str(crop_path), "field_name": field_name}
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))

# 22 field regions for P12
fields = [
    ("first_name", 2, (0.35, 0.18, 0.55, 0.26)),
    ("last_name", 2, (0.70, 0.18, 0.95, 0.26)),
    ("date_of_birth", 2, (0.35, 0.35, 0.60, 0.42)),
    ("age", 2, (0.70, 0.35, 0.85, 0.42)),
    ("gender", 2, (0.35, 0.27, 0.60, 0.32)),
    ("marital_status", 2, (0.35, 0.32, 0.60, 0.37)),
    ("address_line1", 2, (0.35, 0.54, 0.95, 0.62)),
    ("city", 2, (0.35, 0.66, 0.60, 0.72)),
    ("state", 2, (0.65, 0.66, 0.95, 0.72)),
    ("pincode", 2, (0.35, 0.72, 0.55, 0.78)),
    ("phone", 2, (0.65, 0.72, 0.95, 0.78)),
    ("email", 2, (0.35, 0.80, 0.95, 0.86)),
    ("place_of_birth", 2, (0.35, 0.42, 0.60, 0.48)),
    ("nationality", 2, (0.35, 0.48, 0.60, 0.54)),
    ("citizenship", 2, (0.65, 0.48, 0.95, 0.54)),
]

results = {}
for field_name, page_num, box in fields:
    print(f"Extracting {field_name}...")
    try:
        crop = render_and_crop(pdf_path, page_num, box, dpi)
        crop_path = crop_dir / f"{field_name}.png"
        crop.save(str(crop_path))
        result = extract(crop_path, field_name)
        results[field_name] = result
        val = result.get("value")
        conf = result.get("confidence")
        print(f"  -> {val} ({conf})")
    except Exception as e:
        print(f"  -> Error: {e}")

# Save all results
out_path = crop_dir / "22fields_results.json"
out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
print(f"\nSaved to {out_path}")