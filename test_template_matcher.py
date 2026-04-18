from src.template_matcher import FormTemplateMatcher
from src.preprocessing import ImagePreprocessor
import time
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

# Initialize template matcher
matcher = FormTemplateMatcher()
start = time.time()
crops = matcher.match_and_crop(img)
end = time.time()
print('Template matching took:', end-start, 'seconds')
print('Number of crops:', len(crops))
for field_name, crop_img in crops.items():
    print(f'  {field_name}: {crop_img.size}')