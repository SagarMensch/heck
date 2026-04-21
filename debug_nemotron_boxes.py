import json
import cv2
import os

def debug_boxes(json_path, img_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    for pnum_str, page_data in data.get("pages", {}).items():
        img_path = os.path.join(img_dir, f"page_{pnum_str}.png")
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        # Draw cells
        cells = page_data.get("stage3_cell_extractions", [])
        for cell in cells:
            bbox = cell["bbox_in_page"]
            text = cell.get("text", "")[:20] # first 20 chars
            
            # Draw rectangle (Green)
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            # Put text (Red)
            cv2.putText(img, text, (int(bbox[0]), int(bbox[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
        out_path = os.path.join(output_dir, f"debug_page_{pnum_str}.jpg")
        cv2.imwrite(out_path, img)
        print(f"Saved debug image to {out_path} with {len(cells)} boxes")

if __name__ == "__main__":
    debug_boxes(
        "output_nemotron_p10/NEMOTRON_PIPELINE_P10.json",
        "output_nemotron_p10",
        "output_nemotron_p10/debug_visuals"
    )
