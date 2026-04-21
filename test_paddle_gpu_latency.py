import os
import time
from paddleocr import PaddleOCR
import fitz  # PyMuPDF
from PIL import Image
import io

def test_paddle_latency(pdf_path):
    print(f"Loading PaddleOCR with GPU...")
    start_load = time.time()
    # Initialize PaddleOCR with GPU enabled
    ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True, show_log=False)
    print(f"Model loaded in {time.time() - start_load:.2f} seconds.")

    print(f"\nProcessing {pdf_path}...")
    doc = fitz.open(pdf_path)
    page = doc.load_page(0)
    
    # Render page to an image
    pix = page.get_pixmap(dpi=150)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    img_path = "temp_page_0.png"
    img.save(img_path)

    print("Running GPU OCR on Page 1...")
    start_ocr = time.time()
    result = ocr.ocr(img_path, cls=True)
    end_ocr = time.time()
    
    latency = end_ocr - start_ocr
    print(f"\n✅ OCR Completed in {latency:.3f} seconds!")
    
    # Print some extracted text
    if result and result[0]:
        print("\n--- Extracted Text (First 10 lines) ---")
        lines = [line[1][0] for line in result[0]][:10]
        for line in lines:
            print(line)
        print("---------------------------------------")
        print(f"Total lines extracted: {len(result[0])}")
    else:
        print("No text found.")

    os.remove(img_path)

if __name__ == "__main__":
    test_pdf = "Techathon_Samples/P02.pdf"
    if os.path.exists(test_pdf):
        test_paddle_latency(test_pdf)
    else:
        print("Sample PDF not found.")
