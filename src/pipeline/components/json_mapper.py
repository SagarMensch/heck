import json
import logging
import re
import math

logger = logging.getLogger(__name__)

# Real LIC Form 300 Fields based on Techathon PRD & Samples
FORM_300_FIELDS = {
    "Prefix_Name": {"type": "Printed", "length": "2-10", "mandatory": True, "data_type": "Alphabetic"},
    "First_Name": {"type": "Handwritten", "length": "1-50", "mandatory": True, "data_type": "Alphabetic"},
    "Last_Name": {"type": "Handwritten", "length": "1-50", "mandatory": True, "data_type": "Alphabetic"},
    "Father_Name": {"type": "Handwritten", "length": "1-50", "mandatory": False, "data_type": "Alphabetic"},
    "Mother_Name": {"type": "Handwritten", "length": "1-50", "mandatory": False, "data_type": "Alphabetic"},
    "Gender": {"type": "Handwritten/Tick", "length": "1-15", "mandatory": True, "data_type": "Alphabetic"},
    "Marital_Status": {"type": "Handwritten/Tick", "length": "1-15", "mandatory": True, "data_type": "Alphabetic"},
    "Spouse_Name": {"type": "Handwritten", "length": "1-50", "mandatory": False, "data_type": "Alphabetic"},
    "Date_of_Birth": {"type": "Handwritten", "length": "10", "mandatory": True, "data_type": "Date"},
    "Place_of_Birth": {"type": "Handwritten", "length": "2-50", "mandatory": False, "data_type": "Alphabetic"},
    "PAN": {"type": "Handwritten", "length": "10", "mandatory": True, "data_type": "AlphaNumeric"},
    "ID_Proof": {"type": "Handwritten", "length": "5-50", "mandatory": True, "data_type": "AlphaNumeric"},
    "ID_Number": {"type": "Handwritten", "length": "5-20", "mandatory": True, "data_type": "AlphaNumeric"},
    "Educational_Qualification": {"type": "Handwritten", "length": "2-50", "mandatory": True, "data_type": "AlphaNumeric"},
    "Present_Occupation": {"type": "Handwritten", "length": "2-50", "mandatory": True, "data_type": "AlphaNumeric"},
    "Source_of_Income": {"type": "Handwritten", "length": "2-50", "mandatory": True, "data_type": "AlphaNumeric"}
}

def map_nemotron_to_rfp_json(raw_json_path: str, output_json_path: str):
    with open(raw_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    extracted_fields = []
    all_cells = []
    
    for pnum_str, page_data in data.get("pages", {}).items():
        w_str, h_str = page_data["image_size"].split('x')
        page_w, page_h = int(w_str), int(h_str)
        
        for cell in page_data.get("stage3_cell_extractions", []):
            if cell.get("text") and len(cell["text"].strip()) > 1:
                cell["page_num"] = int(pnum_str)
                cell["page_w"] = page_w
                cell["page_h"] = page_h
                all_cells.append(cell)
                
    # Sort by Y, then X for logical reading order
    all_cells.sort(key=lambda c: (c["bbox_in_page"][1] // 20, c["bbox_in_page"][0]))
    
    mapped_keys = set()
    
    for i, cell in enumerate(all_cells):
        text = cell["text"].upper()
        conf = cell["confidence"]
        bbox = cell["bbox_in_page"]
        
        assigned_key = None
        value = text
        
        # Real LIC Form 300 Matching Rules based on P10
        if "PREFIX" in text and "NAME" in text:
            assigned_key = "Prefix_Name"
            if len(text.replace("PREFIX", "").replace("NAME", "").strip()) < 2 and i + 1 < len(all_cells):
                value = all_cells[i+1]["text"]
                bbox = all_cells[i+1]["bbox_in_page"]
                conf = all_cells[i+1]["confidence"]
                
        elif "FIRST NAME" in text:
            assigned_key = "First_Name"
            if i + 1 < len(all_cells) and not ("LAST NAME" in all_cells[i+1]["text"].upper()):
                value = all_cells[i+1]["text"]
                bbox = all_cells[i+1]["bbox_in_page"]
                conf = all_cells[i+1]["confidence"]
                
        elif "LAST NAME" in text and "FIRST" not in text:
            assigned_key = "Last_Name"
            if i + 1 < len(all_cells):
                value = all_cells[i+1]["text"]
                bbox = all_cells[i+1]["bbox_in_page"]
                conf = all_cells[i+1]["confidence"]
                
        elif "FATHER'S FULL NAME" in text or "FATHER" in text:
            assigned_key = "Father_Name"
            if i + 1 < len(all_cells) and len(text) < 30:
                value = all_cells[i+1]["text"]
                bbox = all_cells[i+1]["bbox_in_page"]
                conf = all_cells[i+1]["confidence"]
                
        elif "MOTHER'S FULL NAME" in text or "MOTHER" in text:
            assigned_key = "Mother_Name"
            if i + 1 < len(all_cells):
                value = all_cells[i+1]["text"]
                bbox = all_cells[i+1]["bbox_in_page"]
                conf = all_cells[i+1]["confidence"]
                
        elif "GENDER" in text:
            assigned_key = "Gender"
            if i + 1 < len(all_cells):
                value = all_cells[i+1]["text"]
                bbox = all_cells[i+1]["bbox_in_page"]
                conf = all_cells[i+1]["confidence"]
                
        elif "MARITAL STATUS" in text:
            assigned_key = "Marital_Status"
            if i + 1 < len(all_cells):
                value = all_cells[i+1]["text"]
                bbox = all_cells[i+1]["bbox_in_page"]
                conf = all_cells[i+1]["confidence"]
                
        elif "DATE OF BIRTH" in text:
            assigned_key = "Date_of_Birth"
            match = re.search(r'\d{2}[/-]\d{2}[/-]\d{2,4}', text)
            if match:
                value = match.group(0)
            elif i + 1 < len(all_cells):
                value = all_cells[i+1]["text"]
                bbox = all_cells[i+1]["bbox_in_page"]
                conf = all_cells[i+1]["confidence"]
                
        elif "CITY OF BIRTH" in text or "PLACE" in text:
            assigned_key = "Place_of_Birth"
            if i + 1 < len(all_cells):
                value = all_cells[i+1]["text"]
                bbox = all_cells[i+1]["bbox_in_page"]
                conf = all_cells[i+1]["confidence"]
                
        elif "PERMANENT ACCOUNT NUMBER" in text or "PAN" in text:
            assigned_key = "PAN"
            if i + 1 < len(all_cells):
                value = all_cells[i+1]["text"]
                bbox = all_cells[i+1]["bbox_in_page"]
                conf = all_cells[i+1]["confidence"]
                
        elif "PROOF OF IDENTITY" in text or "IDENTITY" in text:
            assigned_key = "ID_Proof"
            if i + 1 < len(all_cells):
                value = all_cells[i+1]["text"]
                bbox = all_cells[i+1]["bbox_in_page"]
                conf = all_cells[i+1]["confidence"]
                
        elif "NUMBER" in text and "ID" in text:
            assigned_key = "ID_Number"
            if i + 1 < len(all_cells):
                value = all_cells[i+1]["text"]
                bbox = all_cells[i+1]["bbox_in_page"]
                conf = all_cells[i+1]["confidence"]
                
        elif "EDUCATIONAL QUALIFICATION" in text:
            assigned_key = "Educational_Qualification"
            if i + 1 < len(all_cells):
                value = all_cells[i+1]["text"]
                bbox = all_cells[i+1]["bbox_in_page"]
                conf = all_cells[i+1]["confidence"]
                
        elif "PRESENT OCCUPATION" in text:
            assigned_key = "Present_Occupation"
            if i + 1 < len(all_cells):
                value = all_cells[i+1]["text"]
                bbox = all_cells[i+1]["bbox_in_page"]
                conf = all_cells[i+1]["confidence"]
                
        elif "SOURCE OF INCOME" in text:
            assigned_key = "Source_of_Income"
            if i + 1 < len(all_cells):
                value = all_cells[i+1]["text"]
                bbox = all_cells[i+1]["bbox_in_page"]
                conf = all_cells[i+1]["confidence"]

        if assigned_key and assigned_key not in mapped_keys:
            # Clean generic labels
            clean_value = re.sub(r'(?i)(first|last|name|prefix|pan|dob|policy|no|nominee|:|-)', '', value).strip()
            
            # Form 300 specific validation rules (Rule Engine)
            status = "Verified"
            if conf < 0.85:
                status = "Review Needed"
                
            # PAN Regex Validation
            if assigned_key == "PAN" and not re.match(r'[A-Z]{5}[0-9]{4}[A-Z]{1}', clean_value.replace(" ", "")):
                status = "Review Needed"
            
            # DOB Regex Validation
            if assigned_key == "Date_of_Birth" and not re.search(r'\d{2}[/-]\d{2}[/-]\d{2,4}', clean_value):
                status = "Review Needed"

            pw, ph = cell["page_w"], cell["page_h"]
            top_pct = round((bbox[1] / ph) * 100, 2)
            left_pct = round((bbox[0] / pw) * 100, 2)
            width_pct = round(((bbox[2] - bbox[0]) / pw) * 100, 2)
            
            field_data = {
                "field_name": assigned_key,
                "value": clean_value if clean_value else value, # fallback if clean erased it
                "confidence": math.floor(conf * 100),
                "status": status,
                "anchor": assigned_key.lower(),
                "bbox": bbox,
                "ui_coords": {
                    "top": f"{top_pct}%",
                    "left": f"{left_pct}%",
                    "width": f"{width_pct}%"
                },
                "page_num": cell["page_num"],
                "editable": True,
                "metadata": FORM_300_FIELDS.get(assigned_key, {})
            }
            extracted_fields.append(field_data)
            mapped_keys.add(assigned_key)
            
    final_output = {
        "document_id": data.get("document", "Unknown"),
        "total_fields": len(extracted_fields),
        "fields": extracted_fields
    }
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
        
    logger.info(f"Generated True LIC Form 300 UI JSON with {len(extracted_fields)} mapped fields")
    return final_output

if __name__ == "__main__":
    map_nemotron_to_rfp_json("output_nemotron_p10/NEMOTRON_PIPELINE_P10.json", "output_nemotron_p10/RFP_FINAL_P10.json")
