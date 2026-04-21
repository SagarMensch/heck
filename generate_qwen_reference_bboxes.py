"""Generate Qwen-VL reference bounding boxes for LIC Form 300 fields.

Usage examples:
  python generate_qwen_reference_bboxes.py --pages 2,3
  python generate_qwen_reference_bboxes.py --all --force
"""

from __future__ import annotations

import argparse
import logging

from src.form300_templates import PAGE_TYPE_TO_PAGE_NUM
from src.qwen_bbox_grounder import (
    QWEN_REFERENCE_AUDIT_DIR,
    QWEN_REFERENCE_BBOX_PATH,
    QwenBBoxGrounder,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("QwenBBoxBootstrap")


def parse_pages(raw: str):
    pages = set()
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            for page in range(start, end + 1):
                pages.add(page)
        else:
            pages.add(int(token))
    return sorted(pages)


def main():
    parser = argparse.ArgumentParser(description="Bootstrap Qwen reference bounding boxes for LIC Form 300")
    parser.add_argument("--pages", default="", help="Comma-separated pages or ranges, e.g. 2,3,6-7")
    parser.add_argument("--all", action="store_true", help="Generate boxes for all currently templated pages")
    parser.add_argument("--force", action="store_true", help="Regenerate boxes even if they already exist")
    parser.add_argument(
        "--pdf",
        default="",
        help="Optional filled sample PDF to use for grounding before alignment, e.g. Techathon_Samples\\P10.pdf",
    )
    args = parser.parse_args()

    if args.all:
        pages = sorted(PAGE_TYPE_TO_PAGE_NUM.values())
    elif args.pages.strip():
        pages = parse_pages(args.pages)
    else:
        pages = [2, 3, 6, 7]

    logger.info("Generating Qwen reference boxes for pages: %s", pages)
    grounder = QwenBBoxGrounder(bootstrap_pdf_path=args.pdf or None)
    grounder.bootstrap_pages(pages, force=args.force)

    logger.info("Saved reference boxes to: %s", QWEN_REFERENCE_BBOX_PATH)
    logger.info("Saved audit overlays to: %s", QWEN_REFERENCE_AUDIT_DIR)


if __name__ == "__main__":
    main()
