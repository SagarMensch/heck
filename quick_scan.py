"""Quick single-engine scan of P02 only."""
import sys, os, json
sys.stdout.reconfigure(encoding="utf-8")
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import fitz, numpy as np, cv2
from paddleocr import PaddleOCR

print("Loading PaddleOCR en...")
ocr = PaddleOCR(lang="en", use_textline_orientation=True)
print("Loaded. Scanning P02...")

doc = fitz.open(r"C:\Users\aigcp_gpuadmin\Downloads\LICRFP\LICF\Techathon_Samples\P02.pdf")
for pg in range(len(doc)):
    page = doc.load_page(pg)
    zoom = 120 / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4: img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3: img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    r = ocr.ocr(img)
    texts = []
    if r and r[0]:
        t = r[0].get("rec_texts") if hasattr(r[0], "get") else None
        if t is not None:
            texts = [str(x)[:80] for x in list(t)[:25]]
    label = "P{:2d}".format(pg+1)
    content = " | ".join(texts[:12])
    print(f"{label}: {content}")
doc.close()
print("DONE")
