"""
Generate starter lexicon files from local LIC manifests.

Outputs category-wise CSV lexicons under ``data/lexicons`` with columns:
``value,count``.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

from src.name_resolver import (
    ALL_CATEGORIES,
    EMPLOYER_CATEGORY,
    PERSON_FULL_CATEGORY,
    PERSON_STOP_TOKENS,
    PERSON_TOKEN_CATEGORY,
    PLAN_CATEGORY,
    _field_to_categories,
    _normalize,
    _smart_title,
    _tokenize,
)


DEFAULT_MANIFESTS = [
    Path("data/form300_factory/manifests/synthetic_crops.jsonl"),
    Path("data/form300_factory/manifests/final_train_manifest.jsonl"),
    Path("data/form300_factory/manifests/final_val_manifest.jsonl"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate starter lexicon CSV files from local manifests.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/lexicons"),
        help="Directory to write lexicon CSV files into.",
    )
    parser.add_argument(
        "--manifest",
        action="append",
        type=Path,
        dest="manifests",
        help="Optional additional manifest path. Can be repeated.",
    )
    return parser.parse_args()


def _ingest_text(counters: Dict[str, Counter[str]], field_name: str, text: str) -> None:
    categories = _field_to_categories(field_name)
    if not categories or not text:
        return

    canonical = _smart_title(text)
    if not _normalize(canonical, preserve_digits=True):
        return

    if PERSON_FULL_CATEGORY in categories:
        counters[PERSON_FULL_CATEGORY][canonical] += 1
        for token in _tokenize(_normalize(canonical)):
            if token in PERSON_STOP_TOKENS or len(token) < 2:
                continue
            counters[PERSON_TOKEN_CATEGORY][_smart_title(token)] += 1
        return

    for category in categories:
        counters[category][canonical] += 1


def _iter_records(manifest_paths: Iterable[Path]) -> Iterable[dict]:
    for path in manifest_paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _write_csv(path: Path, counter: Counter[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["value", "count"])
        for value, count in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
            writer.writerow([value, count])


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_paths: List[Path] = [repo_root / rel for rel in DEFAULT_MANIFESTS]
    if args.manifests:
        manifest_paths.extend((repo_root / path).resolve() if not path.is_absolute() else path for path in args.manifests)

    counters: Dict[str, Counter[str]] = {category: Counter() for category in ALL_CATEGORIES}

    for record in _iter_records(manifest_paths):
        field_name = str(record.get("field_name", "")).strip().lower()
        text = (
            record.get("target_text")
            or record.get("label_text")
            or record.get("rendered_text")
            or ""
        )
        _ingest_text(counters, field_name, str(text))

    for category in ALL_CATEGORIES:
        _write_csv(output_dir / f"{category}.csv", counters[category])

    summary_path = output_dir / "summary.json"
    summary = {
        "person_token": {"distinct_values": len(counters[PERSON_TOKEN_CATEGORY]), "total_frequency": int(sum(counters[PERSON_TOKEN_CATEGORY].values()))},
        "person_full": {"distinct_values": len(counters[PERSON_FULL_CATEGORY]), "total_frequency": int(sum(counters[PERSON_FULL_CATEGORY].values()))},
        "employer_name": {"distinct_values": len(counters[EMPLOYER_CATEGORY]), "total_frequency": int(sum(counters[EMPLOYER_CATEGORY].values()))},
        "plan_name": {"distinct_values": len(counters[PLAN_CATEGORY]), "total_frequency": int(sum(counters[PLAN_CATEGORY].values()))},
        "manifest_paths": [str(path) for path in manifest_paths if path.exists()],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Lexicons written to: {output_dir}")


if __name__ == "__main__":
    main()
