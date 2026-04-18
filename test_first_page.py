import json
import logging
import time
from src.pipeline import LICExtractionPipeline

logging.basicConfig(level=logging.INFO)

def test_first_page():
    print("Starting pipeline test on first page of P02.pdf...")
    start = time.time()
    
    pipe = LICExtractionPipeline()
    # We'll process the whole PDF but then only take the first page result? 
    # Instead, let's use the pipeline's internal methods to process one page.
    # But for simplicity, we'll process the whole PDF and then break after first page in the pipeline? 
    # Let's just run the pipeline and see the output for the first page in the logs.
    # We can change the pipeline to only process one page by modifying the file path to a single page PDF.
    # Let's extract the first page to a temporary PDF.
    import fitz
    from pathlib import Path
    
    input_path = Path("data/lic_samples/P02.pdf")
    doc = fitz.open(input_path)
    # Create a new PDF with only the first page
    new_doc = fitz.open()
    new_doc.insert_pdf(doc, from_page=0, to_page=0)
    first_page_path = "data/lic_samples/P02_page1.pdf"
    new_doc.save(first_page_path)
    new_doc.close()
    doc.close()
    
    result = pipe.process_single_form(first_page_path)
    
    end = time.time()
    
    print(f"\n--- Extraction Complete in {end-start:.2f} seconds ---")
    print(f"Status: {result.get('form_status')}")
    print(f"Overall Confidence: {result.get('kpis', {}).get('overall_confidence', 0):.2%}")
    print(f"Fields Extracted: {result.get('kpis', {}).get('fields_extracted', 0)} / {result.get('kpis', {}).get('total_fields_expected', 0)}")
    
    print("\n--- Field Values ---")
    fields = result.get("fields", {})
    for field_name, data in fields.items():
        if data.get("value"):
            print(f"{field_name}: {data['value']} (Conf: {data.get('confidence', 0):.2%})")
            
    # Clean up
    import os
    os.remove(first_page_path)

if __name__ == "__main__":
    test_first_page()