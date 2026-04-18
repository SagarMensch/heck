from src.layout_detector import LayoutDetector
import fitz
from PIL import Image
import numpy as np

# Load first page of P02.pdf
doc = fitz.open('data/lic_samples/P02.pdf')
page = doc.load_page(0)
pix = page.get_pixmap(dpi=200)
img = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
np_img = np.array(img)
print('Image shape:', np_img.shape)

# Initialize LayoutDetector
detector = LayoutDetector()
regions = detector.detect_text_regions(np_img)
print('Number of regions detected:', len(regions))
# Print all regions
for i, r in enumerate(regions):
    text = r.get('text')
    if text and len(text.strip()) > 0:
        print('{:3d}: "{}" (conf: {:.3f})'.format(i, text[:80], r.get('confidence')))