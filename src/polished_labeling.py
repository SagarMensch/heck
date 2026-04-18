"""
Polished labeling stack for Form 300.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.labeling_schema import ValidationResult, compare_normalized_labels, normalize_field_value


def _read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


FAMILY_PRIORITY = {
    "long_text": 3.2,
    "name_text": 3.0,
    "short_id": 2.8,
    "date": 2.6,
    "amount": 2.5,
    "short_text": 2.3,
    "numeric": 2.1,
    "binary_mark": 1.5,
}


class CropQCAuditor:
    """
    Scores harvested real crops for likely template or crop issues.
    """

    def _audit_one(self, row: Dict[str, object]) -> Dict[str, object]:
        image_path = Path(str(row["image_path"]))
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return {
                **row,
                "qc_status": "missing_image",
                "qc_score": 0.0,
                "qc_flags": ["missing_image"],
            }

        h, w = img.shape[:2]
        dark = (img < 210).astype(np.uint8)
        dark_ratio = float(dark.mean())
        edge_var = float(cv2.Laplacian(img, cv2.CV_64F).var())
        top = float(dark[: max(1, h // 8), :].mean())
        bottom = float(dark[h - max(1, h // 8) :, :].mean())
        left = float(dark[:, : max(1, w // 8)].mean())
        right = float(dark[:, w - max(1, w // 8) :].mean())
        flags: List[str] = []

        if dark_ratio < 0.01:
            flags.append("very_blank")
        if edge_var < 20:
            flags.append("low_edge_energy")
        if top > 0.20:
            flags.append("ink_top_edge")
        if bottom > 0.20:
            flags.append("ink_bottom_edge")
        if left > 0.20:
            flags.append("ink_left_edge")
        if right > 0.20:
            flags.append("ink_right_edge")

        score = 1.0
        score -= 0.35 if "very_blank" in flags and not bool(row.get("is_blank")) else 0.0
        score -= 0.20 if "low_edge_energy" in flags else 0.0
        score -= 0.10 * sum(flag.startswith("ink_") for flag in flags)
        score = max(0.0, min(1.0, score))

        status = "good"
        if score < 0.45:
            status = "poor"
        elif score < 0.70:
            status = "review"

        return {
            **row,
            "qc_status": status,
            "qc_score": round(score, 4),
            "qc_flags": flags,
            "qc_metrics": {
                "dark_ratio": round(dark_ratio, 4),
                "edge_var": round(edge_var, 2),
                "top_edge_dark": round(top, 4),
                "bottom_edge_dark": round(bottom, 4),
                "left_edge_dark": round(left, 4),
                "right_edge_dark": round(right, 4),
            },
        }

    def audit(self, rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
        return [self._audit_one(row) for row in rows]


class GoldLabelPrioritizer:
    """
    Ranks real crops for manual correction.
    """

    def prioritize(self, qc_rows: Sequence[Dict[str, object]], per_family_cap: int = 40) -> List[Dict[str, object]]:
        buckets: Dict[str, List[Dict[str, object]]] = defaultdict(list)
        for row in qc_rows:
            if bool(row.get("is_blank")):
                continue
            family = str(row["field_family"])
            base = FAMILY_PRIORITY.get(family, 1.0)
            qc_score = float(row.get("qc_score", 0.0))
            priority = base * (0.6 + 0.4 * qc_score)
            if family in {"long_text", "name_text"}:
                priority += 0.20
            if row.get("qc_status") == "review":
                priority += 0.10
            buckets[family].append(
                {
                    "crop_id": row["id"],
                    "image_path": row["image_path"],
                    "field_name": row["field_name"],
                    "field_family": family,
                    "pdf_path": row.get("pdf_path", ""),
                    "qc_status": row.get("qc_status", ""),
                    "qc_score": qc_score,
                    "priority_score": round(priority, 4),
                    "suggested_text": "",
                    "gold_text": "",
                    "review_status": "pending",
                    "notes": ",".join(row.get("qc_flags", [])),
                }
            )

        final_rows: List[Dict[str, object]] = []
        for family, rows in buckets.items():
            ranked = sorted(rows, key=lambda item: item["priority_score"], reverse=True)
            final_rows.extend(ranked[:per_family_cap])
        final_rows.sort(key=lambda item: item["priority_score"], reverse=True)
        return final_rows


class WeakLabelAcceptancePolicy:
    """
    Accepts or rejects OCR-VLM teacher outputs.

    Expected input JSONL row:
    {
      "id": "...",
      "field_name": "...",
      "field_family": "...",
      "text": "...",
      "confidence": 0.93,
      "model": "DeepSeek-OCR-2"
    }
    """

    def _score_candidate(self, row: Dict[str, object]) -> Tuple[ValidationResult, float]:
        result = normalize_field_value(str(row["field_name"]), str(row.get("text", "")))
        model_conf = float(row.get("confidence", 0.0) or 0.0)
        combined = 0.65 * result.score + 0.35 * model_conf
        return result, combined

    def accept(self, rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
        grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
        for row in rows:
            grouped[str(row["id"])].append(row)

        accepted: List[Dict[str, object]] = []
        for crop_id, candidates in grouped.items():
            scored: List[Tuple[Dict[str, object], ValidationResult, float]] = []
            for row in candidates:
                result, combined = self._score_candidate(row)
                scored.append((row, result, combined))
            scored.sort(key=lambda item: item[2], reverse=True)
            best_row, best_result, best_score = scored[0]

            consensus = False
            if len(scored) > 1:
                for _, other_result, _ in scored[1:]:
                    if compare_normalized_labels(best_result.canonical_text, other_result.canonical_text):
                        consensus = True
                        best_score += 0.08
                        break

            threshold = 0.78
            if best_result.field_family == "binary_mark":
                threshold = 0.70
            elif best_result.field_family in {"name_text", "long_text"}:
                threshold = 0.80

            if best_result.is_valid and best_score >= threshold:
                accepted.append(
                    {
                        "id": crop_id,
                        "image_path": best_row.get("image_path", ""),
                        "field_name": best_result.field_name,
                        "field_family": best_result.field_family,
                        "source_tier": "weak",
                        "teacher_model": best_row.get("model", ""),
                        "teacher_confidence": float(best_row.get("confidence", 0.0) or 0.0),
                        "label_text": str(best_row.get("text", "")).strip(),
                        "canonical_text": best_result.canonical_text,
                        "validation_score": round(best_result.score, 4),
                        "acceptance_score": round(best_score, 4),
                        "consensus": consensus,
                        "issues": best_result.issues,
                    }
                )
        return accepted


class ManifestMerger:
    """
    Merge gold + weak + synthetic manifests into final training manifests.
    """

    def _read_gold(self, gold_csv: Path) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        if not gold_csv.exists():
            return rows
        with open(gold_csv, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                gold_text = str(row.get("gold_text", "")).strip()
                if gold_text:
                    rows.append(
                        {
                            "id": row["crop_id"],
                            "image_path": row["image_path"],
                            "field_name": row["field_name"],
                            "field_family": row["field_family"],
                            "source_tier": "gold",
                            "label_text": gold_text,
                            "canonical_text": normalize_field_value(row["field_name"], gold_text).canonical_text,
                        }
                    )
        return rows

    def merge(
        self,
        synthetic_train: Sequence[Dict[str, object]],
        synthetic_val: Sequence[Dict[str, object]],
        gold_rows: Sequence[Dict[str, object]],
        weak_rows: Sequence[Dict[str, object]],
    ) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
        train_rows: List[Dict[str, object]] = []
        val_rows: List[Dict[str, object]] = []

        for row in synthetic_train:
            train_rows.append(
                {
                    "id": row["id"],
                    "image_path": row["image_path"],
                    "field_name": row["field_name"],
                    "field_family": row["field_family"],
                    "source_tier": "synthetic",
                    "label_text": row["target_text"],
                }
            )
        for row in synthetic_val:
            val_rows.append(
                {
                    "id": row["id"],
                    "image_path": row["image_path"],
                    "field_name": row["field_name"],
                    "field_family": row["field_family"],
                    "source_tier": "synthetic",
                    "label_text": row["target_text"],
                }
            )

        train_rows.extend(gold_rows)
        train_rows.extend(weak_rows)
        return train_rows, val_rows

    def write_final_manifests(
        self,
        manifest_dir: Path,
        accepted_weak_path: Optional[Path] = None,
        gold_csv: Optional[Path] = None,
    ) -> Dict[str, int]:
        synthetic_train = _read_jsonl(manifest_dir / "train_crops.jsonl")
        synthetic_val = _read_jsonl(manifest_dir / "val_crops.jsonl")
        gold_rows = self._read_gold(gold_csv or (manifest_dir / "gold_label_sheet.csv"))
        weak_rows = _read_jsonl(accepted_weak_path or (manifest_dir / "accepted_weak_labels.jsonl"))
        train_rows, val_rows = self.merge(synthetic_train, synthetic_val, gold_rows, weak_rows)
        _write_jsonl(manifest_dir / "final_train_manifest.jsonl", train_rows)
        _write_jsonl(manifest_dir / "final_val_manifest.jsonl", val_rows)
        summary = {
            "synthetic_train": len(synthetic_train),
            "synthetic_val": len(synthetic_val),
            "gold_train": len(gold_rows),
            "weak_train": len(weak_rows),
            "final_train": len(train_rows),
            "final_val": len(val_rows),
        }
        with open(manifest_dir / "final_manifest_summary.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        return summary


class PolishedLabelingPipeline:
    """
    Orchestrates QC, gold prioritization, weak label acceptance, and final manifest merge.
    """

    def run(
        self,
        factory_dir: Path,
        teacher_predictions: Optional[Path] = None,
        gold_csv: Optional[Path] = None,
    ) -> Dict[str, object]:
        manifest_dir = factory_dir / "manifests"
        real_rows = _read_jsonl(manifest_dir / "real_crops_unlabeled.jsonl")

        qc_rows = CropQCAuditor().audit(real_rows)
        _write_jsonl(manifest_dir / "crop_qc_report.jsonl", qc_rows)

        priority_rows = GoldLabelPrioritizer().prioritize(qc_rows)
        if priority_rows:
            with open(manifest_dir / "gold_label_priority.csv", "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(priority_rows[0].keys()))
                writer.writeheader()
                writer.writerows(priority_rows)

        accepted_weak_rows: List[Dict[str, object]] = []
        if teacher_predictions and teacher_predictions.exists():
            teacher_rows = _read_jsonl(teacher_predictions)
            accepted_weak_rows = WeakLabelAcceptancePolicy().accept(teacher_rows)
            _write_jsonl(manifest_dir / "accepted_weak_labels.jsonl", accepted_weak_rows)

        merge_summary = ManifestMerger().write_final_manifests(
            manifest_dir=manifest_dir,
            accepted_weak_path=(manifest_dir / "accepted_weak_labels.jsonl"),
            gold_csv=(gold_csv or manifest_dir / "gold_label_sheet.csv"),
        )

        summary = {
            "real_rows": len(real_rows),
            "qc_rows": len(qc_rows),
            "gold_priority_rows": len(priority_rows),
            "accepted_weak_rows": len(accepted_weak_rows),
            "merge_summary": merge_summary,
        }
        with open(manifest_dir / "polished_labeling_summary.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        return summary
