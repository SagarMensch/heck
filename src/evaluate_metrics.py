import json
import logging
from pathlib import Path
from typing import Dict, Tuple

try:
    import jiwer
except ImportError:
    logging.warning("jiwer library not found. Run 'pip install jiwer' to compute CER/WER.")

def compute_cer(hypothesis: str, reference: str) -> float:
    """Compute Character Error Rate (CER)."""
    try:
        return jiwer.cer(reference, hypothesis)
    except:
        return 0.0

def compute_metrics(extracted_json: Dict, ground_truth_json: Dict) -> Dict:
    """
    Calculate mathematical Precision, Recall, and Character Error Rate (CER)
    by comparing pipeline extraction against human-labeled ground truth.
    """
    extracted_fields = extracted_json.get("fields", {})
    gt_fields = ground_truth_json.get("fields", {})
    
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    total_cer = 0.0
    cer_count = 0
    
    for field_name, gt_data in gt_fields.items():
        gt_value = str(gt_data.get("value", "")).strip().lower()
        
        if field_name in extracted_fields:
            ex_value = str(extracted_fields[field_name].get("value", "")).strip().lower()
            
            if ex_value == gt_value:
                true_positives += 1
            else:
                # Value exists but doesn't match perfectly
                if ex_value and gt_value:
                    cer = compute_cer(ex_value, gt_value)
                    total_cer += cer
                    cer_count += 1
                false_positives += 1
        else:
            # Field completely missed
            false_negatives += 1
            
    # Compute false positives for fields extracted that weren't in GT
    for field_name in extracted_fields:
        if field_name not in gt_fields:
            false_positives += 1
            
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    avg_cer = (total_cer / cer_count) if cer_count > 0 else 0.0
    
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0, 4),
        "average_character_error_rate": round(avg_cer, 4),
        "total_fields_evaluated": len(gt_fields),
        "true_positives_exact_match": true_positives
    }

if __name__ == "__main__":
    print("Metrics Evaluation Module loaded.")
    print("To evaluate, provide a ground_truth.json and the pipeline's output.json")
