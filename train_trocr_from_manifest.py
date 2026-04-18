"""
GPU-side TrOCR fine-tuning entrypoint from final manifests.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from src.trocr_finetuning import TrOCRFineTuner


def _read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _extract_label(row: Dict[str, object], use_canonical: bool = False) -> Optional[str]:
    if use_canonical and row.get("canonical_text"):
        return str(row["canonical_text"])
    for key in ("label_text", "target_text", "text"):
        if row.get(key) is not None and str(row[key]).strip():
            return str(row[key]).strip()
    return None


def _filter_rows(rows: Sequence[Dict[str, object]], families: Optional[Sequence[str]]) -> List[Dict[str, object]]:
    if not families:
        return list(rows)
    family_set = {item.strip() for item in families if item.strip()}
    return [row for row in rows if str(row.get("field_family", "")).strip() in family_set]


def load_training_samples(
    train_manifest: Path,
    val_manifest: Path,
    families: Optional[Sequence[str]] = None,
    use_canonical: bool = False,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    train_rows = _filter_rows(_read_jsonl(train_manifest), families)
    val_rows = _filter_rows(_read_jsonl(val_manifest), families)

    train_paths: List[str] = []
    train_labels: List[str] = []
    val_paths: List[str] = []
    val_labels: List[str] = []

    for row in train_rows:
        label = _extract_label(row, use_canonical=use_canonical)
        image_path = str(row.get("image_path", "")).strip()
        if label and image_path:
            train_paths.append(image_path)
            train_labels.append(label)

    for row in val_rows:
        label = _extract_label(row, use_canonical=use_canonical)
        image_path = str(row.get("image_path", "")).strip()
        if label and image_path:
            val_paths.append(image_path)
            val_labels.append(label)

    return train_paths, train_labels, val_paths, val_labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TrOCR from final manifest JSONL files.")
    parser.add_argument(
        "--train-manifest",
        type=str,
        default="data/form300_factory/manifests/final_train_manifest.jsonl",
        help="Path to final train manifest JSONL.",
    )
    parser.add_argument(
        "--val-manifest",
        type=str,
        default="data/form300_factory/manifests/final_val_manifest.jsonl",
        help="Path to final val manifest JSONL.",
    )
    parser.add_argument(
        "--field-family",
        type=str,
        default="",
        help="Comma-separated field families to train on. Example: short_id,date,numeric",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="microsoft/trocr-base-handwritten",
        help="Base TrOCR checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/trocr-finetuned",
        help="Output directory for fine-tuned model.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Per-device train batch size.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--use-canonical",
        action="store_true",
        help="Prefer canonical_text over label_text when available.",
    )
    args = parser.parse_args()

    families = [part.strip() for part in args.field_family.split(",") if part.strip()] or None
    train_manifest = Path(args.train_manifest)
    val_manifest = Path(args.val_manifest)

    train_paths, train_labels, val_paths, val_labels = load_training_samples(
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        families=families,
        use_canonical=args.use_canonical,
    )

    if not train_paths:
        raise ValueError("No training samples selected from manifests.")
    if not val_paths:
        raise ValueError("No validation samples selected from manifests.")

    finetuner = TrOCRFineTuner(base_model=args.base_model, output_dir=args.output_dir)
    train_ds, _, processor = finetuner.prepare_dataset(train_paths, train_labels, val_split=0.0)
    # Build validation dataset separately to preserve manifest split.
    _, val_ds, _ = finetuner.prepare_dataset(val_paths, val_labels, val_split=1.0)

    finetuner.fine_tune(
        train_ds=train_ds,
        val_ds=val_ds,
        processor=processor,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()
