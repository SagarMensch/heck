import fitz
import os
from pathlib import Path

PDFS_TO_ANALYZE = [
    "P02.pdf",
    "P05.pdf",
    "P07.pdf",
    "P17.pdf",
    "P23.pdf"
]

BASE_DIR = Path(r"C:\Users\aigcp_gpuadmin\Downloads\LICRFP\LICF\Techathon_Samples")

def analyze_pdf(pdf_name):
    path = BASE_DIR / pdf_name
    print(f"\n{'='*50}\nAnalyzing: {pdf_name}\n{'='*50}")
    
    if not path.exists():
        print("File not found!")
        return

    try:
        doc = fitz.open(path)
        print(f"Total Pages: {len(doc)}")
        
        # Check text vs image on first few pages
        for page_num in range(min(3, len(doc))): # Just check first 3 pages
            page = doc.load_page(page_num)
            text = page.get_text("text").strip()
            images = page.get_images()
            
            text_preview = text[:100].replace('\n', ' ') if text else "[NO TEXT LAYER]"
            print(f"  Page {page_num+1}:")
            print(f"    Text length: {len(text)}")
            print(f"    Text preview: {text_preview}")
            print(f"    Image count: {len(images)}")
            
            if images:
                for img in images[:2]:
                    print(f"      Img details: {img}")
                    
        doc.close()
    except Exception as e:
        print(f"Error reading PDF: {e}")

if __name__ == "__main__":
    for pdf in PDFS_TO_ANALYZE:
        analyze_pdf(pdf)
