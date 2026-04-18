from src.layout_detector import LayoutDetector
import cv2
import numpy as np
import fitz
from PIL import Image
import time
from paddleocr import PaddleOCR

# Load first page of P02.pdf
doc = fitz.open('data/lic_samples/P02.pdf')
page = doc.load_page(0)
pix = page.get_pixmap(dpi=200)
img = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
np_img = np.array(img)
print('Image shape:', np_img.shape)

# Initialize LayoutDetector with default-like settings
detector = LayoutDetector()
# Replace the engine with one having our desired parameters
engine = PaddleOCR(
    use_textline_orientation=False,
    lang='en',
    det_db_thresh=0.3,
    det_db_box_thresh=0.5,
    det_limit_type='max',
    det_limit_side_len=960,
    rec_batch_num=6
)
detector._engine = engine
start = time.time()
regions = detector.detect_text_regions(np_img)
end = time.time()
print('Detection took:', end-start, 'seconds')
print('Number of regions detected:', len(regions))
for i, r in enumerate(regions[:20]):
    text = r.get('text')
    if text and len(text.strip()) > 0:
        print('{:2d}: "{}" (conf: {:.3f})'.format(i, text[:80], r.get('confidence')))