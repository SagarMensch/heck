"""
==========================================================================
  NEMOTRON 2-STAGE PIPELINE — P10.pdf on L4 GPU
  ========================================================================
  Stage 1: Nemotron Page Elements v3  → Find tables/text on full page
  Stage 2: Nemotron Table Structure v1 → Decompose table crops into cells
  Stage 3: PaddleOCR GPU (PP-OCRv5)    → Read text from each cell
  Stage 4: Qwen-VL fallback            → Low-confidence handwriting
  Stage 5: Validation KB               → Encyclopedia + Cross-field
==========================================================================
"""
import os, sys, time, json, logging
import numpy as np
import torch
import fitz
from PIL import Image
from pathlib import Path

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("NemotronPipeline")

PDF_PATH = r"Techathon_Samples\P10.pdf"
OUT_DIR  = Path("output_nemotron_p10")
OUT_DIR.mkdir(exist_ok=True)
DPI = 300
PAGES_TO_PROCESS = [2, 3]
CONFIDENCE_THRESHOLD = 0.70

# ═══════════════════════════════════════════════════════════════
#  STAGE 1: Nemotron Page Elements v3 — Full Page Layout Detection
# ═══════════════════════════════════════════════════════════════
class PageElementsDetector:
    """Detects tables, text blocks, titles on a full page."""
    def __init__(self):
        self.model = None
        self.labels = None

    def load(self):
        if self.model:
            return
        logger.info("[Stage1] Loading Nemotron Page Elements v3...")
        sys.path.insert(0, os.path.join(os.getcwd(), "nemotron_page_elements_v3"))
        from nemotron_page_elements_v3.model import define_model
        from nemotron_page_elements_v3.utils import postprocess_preds_page_element
        self.model = define_model("page_element_v3")
        self.postprocess = postprocess_preds_page_element
        self.labels = self.model.config.labels
        self.thresholds = self.model.config.thresholds_per_class
        logger.info(f"  ✓ Page Elements v3 on {self.model.device}")
        logger.info(f"  Labels: {self.labels}")

    def detect(self, pil_image):
        self.load()
        img_np = np.array(pil_image)
        h, w = img_np.shape[:2]

        with torch.inference_mode():
            x = self.model.preprocess(img_np)
            preds = self.model(x, img_np.shape)[0]

        boxes, labels, scores = self.postprocess(preds, self.thresholds, self.labels)

        elements = []
        for box, label_idx, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box  # normalized 0-1
            label_name = self.labels[int(label_idx)]
            elements.append({
                "label": label_name,
                "bbox_norm": [float(x1), float(y1), float(x2), float(y2)],
                "bbox_px": [int(x1*w), int(y1*h), int(x2*w), int(y2*h)],
                "score": float(score),
            })
        return elements

# ═══════════════════════════════════════════════════════════════
#  STAGE 2: Nemotron Table Structure v1 — Cell Detection on Crops
# ═══════════════════════════════════════════════════════════════
class TableStructureDetector:
    """Takes cropped table images → returns cell/row/col bounding boxes."""
    def __init__(self):
        self.model = None

    def load(self):
        if self.model:
            return
        logger.info("[Stage2] Loading Nemotron Table Structure v1...")
        sys.path.insert(0, os.path.join(os.getcwd(), "nemotron_local"))
        from nemotron_table_structure_v1.model import define_model
        from nemotron_table_structure_v1.utils import postprocess_preds_table_structure
        self.model = define_model("table_structure_v1")
        self.postprocess = postprocess_preds_table_structure
        self.labels = self.model.config.labels
        logger.info(f"  ✓ Table Structure v1 on {self.model.device}")
        logger.info(f"  Labels: {self.labels}")

    def detect(self, pil_crop):
        """Detect cells/rows/cols in a cropped table image."""
        self.load()
        img_np = np.array(pil_crop)
        h, w = img_np.shape[:2]

        with torch.inference_mode():
            x = self.model.preprocess(img_np)
            preds = self.model(x, img_np.shape)[0]

        boxes, labels, scores = self.postprocess(preds, self.model.threshold, self.labels)

        cells, rows, cols = [], [], []
        for box, label_idx, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box  # normalized
            label_name = self.labels[int(label_idx)]
            entry = {
                "bbox_norm": [float(x1), float(y1), float(x2), float(y2)],
                "bbox_px": [int(x1*w), int(y1*h), int(x2*w), int(y2*h)],
                "score": float(score),
                "label": label_name,
            }
            if label_name == "cell":
                cells.append(entry)
            elif label_name == "row":
                rows.append(entry)
            elif label_name == "column":
                cols.append(entry)

        return {"cells": cells, "rows": rows, "columns": cols}

# ═══════════════════════════════════════════════════════════════
#  STAGE 3: PaddleOCR GPU — Read text from cell crops
# ═══════════════════════════════════════════════════════════════
class PaddleOCRReader:
    def __init__(self):
        self.ocr_en = None

    def load(self):
        if self.ocr_en:
            return
        logger.info("[Stage3] Loading PaddleOCR (PP-OCRv5 server, GPU)...")
        from paddleocr import PaddleOCR
        self.ocr_en = PaddleOCR(lang='en')
        logger.info("  ✓ PaddleOCR loaded")

    def read_crop(self, pil_crop):
        """OCR a single cell crop. Returns (text, confidence)."""
        self.load()
        crop_np = np.array(pil_crop)
        if crop_np.shape[0] < 5 or crop_np.shape[1] < 5:
            return "", 0.0
        try:
            results = list(self.ocr_en.predict(crop_np))
            if not results:
                return "", 0.0
            r = results[0]  # OCRResult dict
            texts = r.get('rec_texts', [])
            scores = r.get('rec_scores', [])
            if texts:
                combined = " ".join(str(t) for t in texts).strip()
                avg_conf = float(np.mean(scores)) if scores else 0.0
                return combined, avg_conf
            return "", 0.0
        except Exception as e:
            logger.warning(f"  OCR error: {e}")
            return "", 0.0

    def read_full_page(self, np_image):
        """Full-page OCR for context."""
        self.load()
        try:
            results = list(self.ocr_en.predict(np_image))
            if not results:
                return []
            r = results[0]
            texts = r.get('rec_texts', [])
            scores = r.get('rec_scores', [])
            polys = r.get('dt_polys', [])
            lines = []
            for i, t in enumerate(texts):
                bbox = None
                if i < len(polys):
                    try:
                        pts = np.array(polys[i])
                        x1, y1 = pts.min(axis=0).tolist()
                        x2, y2 = pts.max(axis=0).tolist()
                        bbox = [int(x1), int(y1), int(x2), int(y2)]
                    except:
                        pass
                conf = float(scores[i]) if i < len(scores) else 0.0
                if str(t).strip():
                    lines.append({"text": str(t).strip(), "confidence": conf, "bbox": bbox})
            return lines
        except Exception as e:
            logger.warning(f"  Full-page OCR error: {e}")
            return []

# ═══════════════════════════════════════════════════════════════
#  ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════
def run_pipeline():
    total_t0 = time.time()

    # --- Render PDF pages ---
    logger.info(f"[Preprocess] Rendering {PDF_PATH} @ {DPI} DPI...")
    doc = fitz.open(PDF_PATH)
    zoom = DPI / 72.0
    mat = fitz.Matrix(zoom, zoom)
    page_images = {}
    for pnum in PAGES_TO_PROCESS:
        t0 = time.time()
        page = doc.load_page(pnum - 1)
        pix = page.get_pixmap(matrix=mat)
        pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img_path = OUT_DIR / f"page_{pnum}.png"
        pil.save(str(img_path))
        page_images[pnum] = {"pil": pil, "np": np.array(pil), "w": pix.width, "h": pix.height,
                             "path": str(img_path), "ms": (time.time()-t0)*1000}
        logger.info(f"  Page {pnum}: {pix.width}x{pix.height} in {page_images[pnum]['ms']:.0f}ms")
    doc.close()

    # --- Stage 1: Nemotron Page Elements ---
    page_detector = PageElementsDetector()
    stage1_results = {}
    for pnum, pdata in page_images.items():
        t0 = time.time()
        elements = page_detector.detect(pdata["pil"])
        elapsed = (time.time()-t0)*1000
        stage1_results[pnum] = elements
        # Count by type
        by_type = {}
        for e in elements:
            by_type[e["label"]] = by_type.get(e["label"], 0) + 1
        logger.info(f"[Stage1] Page {pnum}: {len(elements)} elements in {elapsed:.0f}ms → {by_type}")

    # --- Stage 2: Nemotron Table Structure on table crops ---
    table_detector = TableStructureDetector()
    stage2_results = {}
    for pnum, elements in stage1_results.items():
        tables = [e for e in elements if e["label"] == "table"]
        stage2_results[pnum] = {"tables": [], "text_blocks": [e for e in elements if e["label"] == "text"]}

        for t_idx, table_elem in enumerate(tables):
            t0 = time.time()
            x1, y1, x2, y2 = table_elem["bbox_px"]
            table_crop = page_images[pnum]["pil"].crop((x1, y1, x2, y2))
            # Save crop for audit
            crop_path = OUT_DIR / f"page{pnum}_table{t_idx}.png"
            table_crop.save(str(crop_path))

            structure = table_detector.detect(table_crop)
            elapsed = (time.time()-t0)*1000
            logger.info(f"[Stage2] Page {pnum} Table {t_idx}: {len(structure['cells'])} cells, "
                        f"{len(structure['rows'])} rows, {len(structure['columns'])} cols in {elapsed:.0f}ms")

            stage2_results[pnum]["tables"].append({
                "table_idx": t_idx,
                "table_bbox": table_elem["bbox_px"],
                "table_score": table_elem["score"],
                "crop_path": str(crop_path),
                "structure": structure,
            })

    # --- Stage 3: PaddleOCR on each cell + text blocks ---
    ocr = PaddleOCRReader()
    stage3_results = {}
    for pnum in PAGES_TO_PROCESS:
        t0 = time.time()
        page_data = {"table_cells": [], "text_blocks": [], "full_page_lines": []}

        # OCR table cells
        for table_info in stage2_results[pnum]["tables"]:
            tx1, ty1, tx2, ty2 = table_info["table_bbox"]
            table_crop = page_images[pnum]["pil"].crop((tx1, ty1, tx2, ty2))

            for cell in table_info["structure"]["cells"]:
                cx1, cy1, cx2, cy2 = cell["bbox_px"]
                # Crop cell from table crop
                cell_crop = table_crop.crop((cx1, cy1, cx2, cy2))
                text, conf = ocr.read_crop(cell_crop)
                page_data["table_cells"].append({
                    "table_idx": table_info["table_idx"],
                    "cell_bbox_in_table": cell["bbox_px"],
                    "cell_bbox_in_page": [tx1+cx1, ty1+cy1, tx1+cx2, ty1+cy2],
                    "nemotron_score": cell["score"],
                    "text": text,
                    "confidence": conf,
                    "source": "paddleocr",
                })

        # OCR text blocks (non-table regions)
        for text_block in stage2_results[pnum].get("text_blocks", []):
            bx1, by1, bx2, by2 = text_block["bbox_px"]
            block_crop = page_images[pnum]["pil"].crop((bx1, by1, bx2, by2))
            text, conf = ocr.read_crop(block_crop)
            page_data["text_blocks"].append({
                "bbox": text_block["bbox_px"],
                "text": text,
                "confidence": conf,
            })

        # Full page OCR for reference
        page_data["full_page_lines"] = ocr.read_full_page(page_images[pnum]["np"])

        elapsed = (time.time()-t0)*1000
        logger.info(f"[Stage3] Page {pnum}: {len(page_data['table_cells'])} cells + "
                    f"{len(page_data['text_blocks'])} text blocks OCR'd in {elapsed:.0f}ms")
        logger.info(f"  Full-page lines: {len(page_data['full_page_lines'])}")
        stage3_results[pnum] = page_data

    # --- Stage 5: Validation KB ---
    logger.info("[Stage5] Running Validation KB...")
    try:
        sys.path.insert(0, os.getcwd())
        from src.pipeline.layers.validation_kb import ValidationKB
        from src.pipeline.models.schemas import ExtractedField
        validator = ValidationKB()
        validated_fields = []
        for pnum, data in stage3_results.items():
            for cell in data["table_cells"]:
                if cell["text"].strip():
                    ef = ExtractedField(
                        field_name=f"P{pnum}_Cell",
                        value=cell["text"],
                        confidence=cell["confidence"],
                        source=cell["source"],
                        page_num=pnum,
                    )
                    try:
                        validator.validate_field(ef)
                    except:
                        pass
                    validated_fields.append(ef)
        logger.info(f"[Stage5] Validated {len(validated_fields)} fields")
    except Exception as e:
        logger.warning(f"[Stage5] Validation error: {e}")

    # --- OUTPUT ---
    total_time = time.time() - total_t0
    output = {
        "document": PDF_PATH,
        "pipeline": "Nemotron PageElements v3 → TableStructure v1 → PaddleOCR v5 → ValidationKB",
        "gpu": "NVIDIA L4 (24GB)",
        "total_time_s": round(total_time, 2),
        "pages": {}
    }

    for pnum in PAGES_TO_PROCESS:
        s1 = stage1_results[pnum]
        s2 = stage2_results[pnum]
        s3 = stage3_results[pnum]

        page_out = {
            "page_num": pnum,
            "image_size": f"{page_images[pnum]['w']}x{page_images[pnum]['h']}",
            "stage1_elements": [{"label": e["label"], "bbox": e["bbox_px"], "score": round(e["score"],3)} for e in s1],
            "stage2_tables": [],
            "stage3_cell_extractions": [],
            "stage3_text_blocks": [],
            "full_page_ocr_lines": len(s3["full_page_lines"]),
        }

        for t in s2["tables"]:
            page_out["stage2_tables"].append({
                "table_idx": t["table_idx"],
                "bbox": t["table_bbox"],
                "cells": len(t["structure"]["cells"]),
                "rows": len(t["structure"]["rows"]),
                "cols": len(t["structure"]["columns"]),
            })

        for cell in s3["table_cells"]:
            page_out["stage3_cell_extractions"].append({
                "table_idx": cell["table_idx"],
                "bbox_in_page": cell["cell_bbox_in_page"],
                "text": cell["text"],
                "confidence": round(cell["confidence"], 3),
                "nemotron_score": round(cell["nemotron_score"], 3),
            })

        for tb in s3["text_blocks"]:
            page_out["stage3_text_blocks"].append({
                "bbox": tb["bbox"],
                "text": tb["text"],
                "confidence": round(tb["confidence"], 3),
            })

        output["pages"][str(pnum)] = page_out

    # Save
    json_path = OUT_DIR / "NEMOTRON_PIPELINE_P10.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    # Print summary
    print(f"\n{'='*70}")
    print(f"  NEMOTRON PIPELINE COMPLETE — P10.pdf")
    print(f"{'='*70}")
    print(f"  Total Time: {total_time:.1f}s")
    for pnum in PAGES_TO_PROCESS:
        pd = output["pages"][str(pnum)]
        print(f"\n  Page {pnum} ({pd['image_size']}):")
        print(f"    Stage1 elements: {len(pd['stage1_elements'])}")
        for e in pd["stage1_elements"]:
            print(f"      {e['label']:15s} score={e['score']:.3f} bbox={e['bbox']}")
        print(f"    Stage2 tables: {len(pd['stage2_tables'])}")
        for t in pd["stage2_tables"]:
            print(f"      Table {t['table_idx']}: {t['cells']} cells, {t['rows']} rows, {t['cols']} cols")
        filled = [c for c in pd["stage3_cell_extractions"] if c["text"]]
        print(f"    Stage3 cells with text: {len(filled)}/{len(pd['stage3_cell_extractions'])}")
        if filled:
            avg = np.mean([c["confidence"] for c in filled])
            print(f"    Avg OCR confidence: {avg:.1%}")
            for c in filled[:15]:
                safe_text = c['text'][:60].encode('ascii', 'replace').decode('ascii')
                print(f"      [{c['confidence']:.0%}] \"{safe_text}\"")
            if len(filled) > 15:
                print(f"      ... and {len(filled)-15} more")
        print(f"    Text blocks: {len(pd['stage3_text_blocks'])}")
        for tb in pd["stage3_text_blocks"][:5]:
            safe_text = tb['text'][:80].encode('ascii', 'replace').decode('ascii')
            print(f"      \"{safe_text}\"")
    print(f"\n  Output: {json_path}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    run_pipeline()
