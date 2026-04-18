"""
CLI for polished Form 300 labeling stack.
"""

import argparse
import json
from pathlib import Path

from src.polished_labeling import PolishedLabelingPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Build polished labeling assets for Form 300.")
    parser.add_argument(
        "--factory-dir",
        type=str,
        default="data/form300_factory",
        help="Dataset factory output directory.",
    )
    parser.add_argument(
        "--teacher-predictions",
        type=str,
        default="",
        help="Optional JSONL of OCR-VLM teacher predictions.",
    )
    parser.add_argument(
        "--gold-csv",
        type=str,
        default="",
        help="Optional reviewed gold-label CSV. Defaults to manifests/gold_label_sheet.csv",
    )
    args = parser.parse_args()

    factory_dir = Path(args.factory_dir)
    teacher_predictions = Path(args.teacher_predictions) if args.teacher_predictions else None
    gold_csv = Path(args.gold_csv) if args.gold_csv else None

    summary = PolishedLabelingPipeline().run(
        factory_dir=factory_dir,
        teacher_predictions=teacher_predictions,
        gold_csv=gold_csv,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
