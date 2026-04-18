# Polished Labeling Stack

## Goal

Turn the CPU factory output into training-ready manifests that are materially better than a bootstrap synthetic-only setup.

This stack does four things:

1. audits harvested real crops
2. prioritizes the best manual gold-label work
3. accepts or rejects OCR-VLM weak labels using field-aware rules
4. merges `gold + weak + synthetic` into final train/val manifests

## Run Order

### 1. Build CPU dataset bundle

```powershell
python .\build_form300_cpu_dataset.py `
  --pdf-dir "C:\Users\sagar\Downloads\heck-main\_tmp_techathon_forms" `
  --out-dir ".\data\form300_factory" `
  --num-records 1500 `
  --num-projected-records 250 `
  --seed 7
```

### 2. Build polished labeling assets

```powershell
python .\build_polished_labeling_stack.py `
  --factory-dir ".\data\form300_factory"
```

This generates:

- `manifests/crop_qc_report.jsonl`
- `manifests/gold_label_priority.csv`
- `manifests/final_train_manifest.jsonl`
- `manifests/final_val_manifest.jsonl`

## Teacher Predictions Format

When you run OCR-VLM teachers on the GPU box, write one JSONL row per prediction in this format:

```json
{
  "id": "P17__pan_number",
  "image_path": "/abs/path/to/crop.png",
  "field_name": "pan_number",
  "field_family": "short_id",
  "text": "CBVPV12345",
  "confidence": 0.93,
  "model": "DeepSeek-OCR-2"
}
```

You can have multiple rows with the same `id` from different teacher models.

Then run:

```powershell
python .\build_polished_labeling_stack.py `
  --factory-dir ".\data\form300_factory" `
  --teacher-predictions ".\teacher_predictions.jsonl"
```

Accepted weak labels are written to:

- `manifests/accepted_weak_labels.jsonl`

## Gold Labels

Fill:

- `manifests/gold_label_sheet.csv`

Populate `gold_text` for reviewed rows. Then rerun:

```powershell
python .\build_polished_labeling_stack.py `
  --factory-dir ".\data\form300_factory" `
  --gold-csv ".\data\form300_factory\manifests\gold_label_sheet.csv"
```

## Final Training Manifests

The merger writes:

- `manifests/final_train_manifest.jsonl`
- `manifests/final_val_manifest.jsonl`

The intended composition is:

- `synthetic` as the bulk base
- `weak` as validator-filtered real supervision
- `gold` as highest-value correction signal

## Current Acceptance Logic

Weak labels are validated by field family:

- `short_id`: PAN-like / CKYC-like / numeric id shape checks
- `date`: date-shape checks
- `amount`: numeric-with-comma checks
- `numeric`: digit checks
- `binary_mark`: yes/no normalization
- `name_text`, `short_text`, `long_text`: character and length sanity checks

Consensus across multiple teacher models increases acceptance score.

## What This Fixes

Compared with the bootstrap-only setup, this stack adds:

- crop-level QC before training
- smarter manual labeling prioritization
- automatic rejection of bad weak labels
- one merged manifest instead of ad hoc datasets

This does not guarantee perfect blind accuracy, but it is the correct path to a much stronger real-plus-synthetic training corpus.
