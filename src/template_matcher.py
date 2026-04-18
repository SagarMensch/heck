import logging
import numpy as np
from typing import Dict, List, Tuple
from PIL import Image
import cv2
import re

from src.layout_detector import LayoutDetector
from src.config import FORM_300_FIELDS

logger = logging.getLogger(__name__)

class FormTemplateMatcher:
    """
    Uses PaddleOCR to find printed labels on a form and geometrically 
    calculates the bounding box for the adjacent handwritten fields.
    """
    
    def __init__(self):
        self.detector = LayoutDetector()
        
        # Heuristics: Where is the handwriting relative to the label?
        # Format: (Regex pattern for label, relative position, width_multiplier, height_multiplier)
        self.anchor_rules = {
            "Proposer_Full_Name": (r"(?i)name in full|full name", "right", 5.0, 1.5),
            "Proposer_Date_of_Birth": (r"(?i)date of birth|dob", "right", 3.0, 1.2),
            "Proposer_Age": (r"(?i)\bage\b", "right", 1.5, 1.2),
            "Proposer_PAN": (r"(?i)pan number|pan no", "right", 4.0, 1.5),
            "Proposer_Aadhaar": (r"(?i)aadhaar", "right", 4.0, 1.5),
            "Proposer_Mobile_Number": (r"(?i)mobile no|mobile number", "right", 4.0, 1.2),
            "Proposer_Email": (r"(?i)email|e-mail", "right", 5.0, 1.2),
            "Proposer_Address_Line1": (r"(?i)address", "below", 10.0, 2.5),
            "Proposer_City": (r"(?i)city|town", "right", 3.0, 1.2),
            "Proposer_State": (r"(?i)state", "right", 3.0, 1.2),
            "Proposer_Pincode": (r"(?i)pincode|pin code|pin", "right", 2.0, 1.2),
            "Bank_Account_Number": (r"(?i)account no|account number", "right", 4.0, 1.5),
            "Bank_IFSC": (r"(?i)ifsc", "right", 3.0, 1.5),
            "Bank_Name": (r"(?i)bank name", "right", 5.0, 1.5),
            "Nominee_Name": (r"(?i)name of nominee|nominee name", "right", 5.0, 1.5),
            "Sum_Assured": (r"(?i)sum assured", "right", 3.0, 1.5),
            "Premium_Amount": (r"(?i)premium|installment premium", "right", 3.0, 1.5),
        }

    def match_and_crop(self, pil_image: Image.Image) -> Dict[str, Image.Image]:
        """
        Detects text, matches against known anchors, and extracts crops.
        """
        # Convert PIL to CV2 format
        np_img = np.array(pil_image)
        if len(np_img.shape) == 3 and np_img.shape[2] == 3:
            np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
            
        h_img, w_img = np_img.shape[:2]
        
        # Get printed text regions from PaddleOCR
        regions = self.detector.detect_text_regions(np_img)
        
        crops = {}
        found_anchors = set()
        
        for region in regions:
            text = str(region.get("text", "")).strip()
            bbox = region.get("bbox") # [x_min, y_min, x_max, y_max]
            
            if not text or not bbox:
                continue
                
            x_min, y_min, x_max, y_max = bbox
            box_width = x_max - x_min
            box_height = y_max - y_min
            
            # Check against anchor rules
            for field_name, rule in self.anchor_rules.items():
                if field_name in found_anchors:
                    continue
                    
                pattern, position, w_mult, h_mult = rule
                
                if re.search(pattern, text):
                    # Calculate crop zone based on position
                    if position == "right":
                        crop_x1 = x_max + int(box_width * 0.1) # small padding
                        crop_y1 = y_min - int(box_height * 0.2)
                        crop_x2 = min(w_img, crop_x1 + int(box_width * w_mult))
                        crop_y2 = min(h_img, y_max + int(box_height * 0.2))
                    elif position == "below":
                        crop_x1 = x_min
                        crop_y1 = y_max + int(box_height * 0.1)
                        crop_x2 = min(w_img, x_min + int(box_width * w_mult))
                        crop_y2 = min(h_img, crop_y1 + int(box_height * h_mult))
                    else:
                        continue
                        
                    # Validate crop dimensions
                    if crop_x2 > crop_x1 and crop_y2 > crop_y1:
                        crop_img = np_img[crop_y1:crop_y2, crop_x1:crop_x2]
                        if crop_img.size > 0:
                            # Convert back to PIL
                            if len(crop_img.shape) == 3:
                                crop_img_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                            else:
                                crop_img_rgb = cv2.cvtColor(crop_img, cv2.COLOR_GRAY2RGB)
                            
                            crops[field_name] = Image.fromarray(crop_img_rgb)
                            found_anchors.add(field_name)
                            logger.debug(f"Anchored {field_name} via label '{text}'")
                            
        return crops
