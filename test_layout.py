from src.layout_detector import LayoutDetector
import cv2
import numpy as np
from PIL import Image
import fitz
import time

# Load first page of P02.pdf
doc = fitz.open('data/lic_samples/P02.pdf')
page = doc.load_page(0)
pix = page.get_pixmap(dpi=200)
img = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
np_img = np.array(img)
print('Image shape:', np_img.shape)

# Initialize LayoutDetector
detector = LayoutDetector()
start = time.time()
regions = detector.detect_text_regions(np_img)
end = time.time()
print('Detection took:', end-start, 'seconds')
print('Number of regions detected:', len(regions))
if regions:
    for i, r in enumerate(regions[:10]):
        text = r.get('text')
        text_preview = text[:50] if text and len(text) > 50 else text
        print('Region {}: text="{}", conf={}, bbox={}'.format(i, text_preview, r.get('confidence'), r.get('bbox')))
else:
    print('No regions detected')