import json
import logging
from src.pipeline import LICExtractionPipeline
import time

logging.basicConfig(level=logging.INFO)

def test_pipeline():
    print("Starting pipeline test on P02.pdf...")
    start = time.time()
    
    pipe = LICExtractionPipeline()
    result = pipe.process_single_form("C:/Users/aigcp_gpuadmin/Downloads/LICRFP/LICF/data/lic_samples/P02.pdf")
    
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
            
if __name__ == "__main__":
    test_pipeline()
