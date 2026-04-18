import time
from src.preprocessing import ImagePreprocessor
from src.extractor import DualModelExtractor

preprocessor = ImagePreprocessor()
processed = preprocessor.process_file('data/lic_samples/P02.pdf')
print('Processed pages:', len(processed))
usable = [p for p in processed if p['is_usable']]
print('Usable pages:', len(usable))
if usable:
    # Take first usable page
    page = usable[0]
    pil_img = preprocessor.numpy_to_pil(page['preprocessed_color'])
    extractor = DualModelExtractor()
    start = time.time()
    result = extractor.extract([pil_img])
    end = time.time()
    print('Extraction took:', end-start, 'seconds')
    print('Fields extracted:', len(result.get('fields', {})))
    fields = result.get('fields', {})
    for k, v in list(fields.items())[:5]:
        print('  {}: {} (conf: {})'.format(k, v.get('value'), v.get('confidence')))