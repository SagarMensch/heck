"""
PaddleX Layout Extraction Module
Extract structured regions from LIC Form 300
"""
import os
import json
import logging
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class TextRegion:
    """A detected text region with metadata"""
    text: str
    label: str
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    region_type: str
    page_num: int
    
    def to_dict(self) -> Dict:
        return asdict(self)


class PaddleXLayoutExtractor:
    """Extract layout regions using PaddleX layout_parsing pipeline"""
    
    def __init__(self):
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "1"
        self.pipeline = None
        self._loaded = False
    
    def load(self):
        """Lazy load PaddleX pipeline"""
        if self._loaded:
            return
        
        from paddlex import create_pipeline
        logger.info("Loading PaddleX layout_parsing pipeline...")
        self.pipeline = create_pipeline(pipeline="layout_parsing")
        self._loaded = True
        logger.info("PaddleX loaded successfully")
    
    def extract_from_image(self, img_path: str, page_num: int = 1) -> List[TextRegion]:
        """
        Extract text regions from an image
        
        Args:
            img_path: Path to image file
            page_num: Page number for tracking
        
        Returns:
            List of TextRegion objects
        """
        self.load()
        
        result = self.pipeline.predict(img_path)
        r = list(result)[0]
        prl = r["parsing_res_list"]
        
        regions = []
        for item in prl:
            txt = item.get("block_content", "").strip()
            if not txt:
                continue
            
            label = item.get("block_label", "text")
            bbox_raw = item.get("block_bbox", [])
            
            # Convert bbox
            if isinstance(bbox_raw, np.ndarray):
                bbox = bbox_raw.tolist()
            elif isinstance(bbox_raw, (list, tuple)):
                bbox = [float(x) for x in bbox_raw]
            else:
                bbox = [0, 0, 0, 0]
            
            regions.append(TextRegion(
                text=txt,
                label=label,
                bbox=bbox,
                confidence=1.0,  # PaddleX doesn't provide per-block confidence
                region_type=label,
                page_num=page_num
            ))
        
        return regions
    
    def extract_from_pdf(self, pdf_path: str, pages: List[int] = None) -> Dict[int, List[TextRegion]]:
        """
        Extract from PDF (converts pages to images internally)
        
        Args:
            pdf_path: Path to PDF
            pages: List of page numbers (1-indexed), or None for all
        
        Returns:
            Dict mapping page_num -> List[TextRegion]
        """
        import fitz  # PyMuPDF
        import tempfile
        
        self.load()
        
        doc = fitz.open(pdf_path)
        all_regions = {}
        
        page_nums = pages if pages else range(1, len(doc) + 1)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            for pnum in page_nums:
                page = doc.load_page(pnum - 1)
                pix = page.get_pixmap(dpi=150)
                img_path = os.path.join(tmpdir, f"page_{pnum}.png")
                pix.save(img_path)
                
                regions = self.extract_from_image(img_path, pnum)
                all_regions[pnum] = regions
        
        doc.close()
        return all_regions
    
    def save_regions(self, regions: Dict[int, List[TextRegion]], output_dir: str):
        """Save regions to JSON files"""
        os.makedirs(output_dir, exist_ok=True)
        
        for pnum, region_list in regions.items():
            data = {
                "page_num": pnum,
                "num_regions": len(region_list),
                "regions": [r.to_dict() for r in region_list]
            }
            
            with open(os.path.join(output_dir, f"page_{pnum}_regions.json"), "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Also save markdown
            md_content = f"# Page {pnum}\n\n"
            for r in region_list:
                md_content += f"[{r.label}] {r.text}\n\n"
            
            with open(os.path.join(output_dir, f"page_{pnum}.md"), "w") as f:
                f.write(md_content)
        
        logger.info(f"Saved {len(regions)} pages to {output_dir}")


def extract_layout(pdf_path: str, output_dir: str = None, pages: List[int] = None) -> Dict[int, List[TextRegion]]:
    """
    Convenience function to extract layout from PDF
    
    Args:
        pdf_path: Path to PDF
        output_dir: Optional directory to save results
        pages: Optional list of page numbers to process
    
    Returns:
        Dict mapping page_num -> List[TextRegion]
    """
    extractor = PaddleXLayoutExtractor()
    regions = extractor.extract_from_pdf(pdf_path, pages)
    
    if output_dir:
        extractor.save_regions(regions, output_dir)
    
    return regions
