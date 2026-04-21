"""
Quick test: Can preprocessing make Nemotron Table Structure detect cells?
Tries: raw vs binarized vs CLAHE-enhanced vs morphology-thickened lines
"""
import os, sys, time
import numpy as np
import torch
import cv2
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Load the table crop from the previous pipeline run
CROP_PATH = "output_nemotron_p10/page2_table0.png"

# Load Nemotron Table Structure
sys.path.insert(0, os.path.join(os.getcwd(), "nemotron_local"))
from nemotron_table_structure_v1.model import define_model
from nemotron_table_structure_v1.utils import postprocess_preds_table_structure

model = define_model("table_structure_v1")
print(f"Model on: {model.device}")

def run_nemotron(img_np, label=""):
    """Run Nemotron on an image and report results."""
    h, w = img_np.shape[:2]
    with torch.inference_mode():
        x = model.preprocess(img_np)
        preds = model(x, img_np.shape)[0]
    boxes, labels, scores = postprocess_preds_table_structure(preds, model.threshold, model.labels)
    cells = sum(1 for l in labels if l == "cell")
    rows = sum(1 for l in labels if l == "row")
    cols = sum(1 for l in labels if l == "column")
    print(f"  [{label:25s}] cells={cells:3d}  rows={rows:3d}  cols={cols:3d}  total={len(labels)}")
    return boxes, labels, scores

# Load raw crop
raw = cv2.imread(CROP_PATH)
raw_rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
print(f"Image: {raw.shape}")

# 1. RAW (baseline — should be 0 cells like before)
print("\n=== PREPROCESSING EXPERIMENTS ===")
run_nemotron(raw_rgb, "RAW (no processing)")

# 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
enhanced = clahe.apply(gray)
enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
cv2.imwrite("output_nemotron_p10/test_clahe.png", enhanced)
run_nemotron(enhanced_rgb, "CLAHE enhanced")

# 3. Adaptive Threshold (binarization)
binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)
binary_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
cv2.imwrite("output_nemotron_p10/test_binary.png", binary)
run_nemotron(binary_rgb, "Adaptive Threshold")

# 4. Otsu binarization
_, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
otsu_rgb = cv2.cvtColor(otsu, cv2.COLOR_GRAY2RGB)
cv2.imwrite("output_nemotron_p10/test_otsu.png", otsu)
run_nemotron(otsu_rgb, "Otsu Threshold")

# 5. CLAHE + Morphology (thicken lines)
kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
# Invert so lines are white
inv = cv2.bitwise_not(enhanced)
h_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel_h)
v_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel_v)
grid = cv2.add(h_lines, v_lines)
# Dilate to thicken
grid_thick = cv2.dilate(grid, np.ones((3,3), np.uint8), iterations=2)
# Overlay on original
combined = enhanced.copy()
combined[grid_thick > 50] = 0  # darken grid lines
combined_rgb = cv2.cvtColor(combined, cv2.COLOR_GRAY2RGB)
cv2.imwrite("output_nemotron_p10/test_grid_enhanced.png", combined)
run_nemotron(combined_rgb, "CLAHE + Grid thickened")

# 6. Sharp binarize with thick lines
sharp = cv2.GaussianBlur(gray, (0,0), 3)
sharp = cv2.addWeighted(gray, 1.5, sharp, -0.5, 0)
_, sharp_bin = cv2.threshold(sharp, 180, 255, cv2.THRESH_BINARY)
sharp_rgb = cv2.cvtColor(sharp_bin, cv2.COLOR_GRAY2RGB)
cv2.imwrite("output_nemotron_p10/test_sharp.png", sharp_bin)
run_nemotron(sharp_rgb, "Sharpened + Binary")

# 7. Lower threshold test — try with threshold=0.01 instead of default 0.05
print("\n=== LOWERED DETECTION THRESHOLD (0.01) ===")
with torch.inference_mode():
    x = model.preprocess(raw_rgb)
    preds = model(x, raw_rgb.shape)[0]
boxes_low, labels_low, scores_low = postprocess_preds_table_structure(preds, 0.01, model.labels)
cells_low = sum(1 for l in labels_low if l == "cell")
rows_low = sum(1 for l in labels_low if l == "row")
cols_low = sum(1 for l in labels_low if l == "column")
print(f"  [RAW @ threshold=0.01    ] cells={cells_low:3d}  rows={rows_low:3d}  cols={cols_low:3d}")

# Try CLAHE with low threshold
with torch.inference_mode():
    x = model.preprocess(enhanced_rgb)
    preds = model(x, enhanced_rgb.shape)[0]
boxes_low2, labels_low2, scores_low2 = postprocess_preds_table_structure(preds, 0.01, model.labels)
cells_low2 = sum(1 for l in labels_low2 if l == "cell")
rows_low2 = sum(1 for l in labels_low2 if l == "row")
cols_low2 = sum(1 for l in labels_low2 if l == "column")
print(f"  [CLAHE @ threshold=0.01  ] cells={cells_low2:3d}  rows={rows_low2:3d}  cols={cols_low2:3d}")

print("\nDone!")
