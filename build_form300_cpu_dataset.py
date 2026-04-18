"""
CLI wrapper for the CPU-only Form 300 dataset factory.
"""

import argparse
import json
from pathlib import Path

from src.dataset_factory import CPUForm300DatasetBuilder


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CPU-side Form 300 synthetic and real crop datasets.")
    parser.add_argument(
        "--pdf-dir",
        type=str,
        default="data/training_samples",
        help="Directory containing sample training PDFs. Optional but recommended.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/form300_factory",
        help="Output directory for crops, pages, backgrounds, and manifests.",
    )
    parser.add_argument(
        "--num-records",
        type=int,
        default=2000,
        help="Number of coherent synthetic records to generate. Each record produces one crop per template field.",
    )
    parser.add_argument(
        "--num-projected-records",
        type=int,
        default=250,
        help="Number of synthetic records to project into full-page samples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for deterministic dataset generation.",
    )
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.exists():
        pdf_dir = None

    builder = CPUForm300DatasetBuilder(Path(args.out_dir), seed=args.seed)
    summary = builder.build(
        pdf_dir=pdf_dir,
        num_records=args.num_records,
        num_projected_records=args.num_projected_records,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
