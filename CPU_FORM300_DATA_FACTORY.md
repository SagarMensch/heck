# CPU Form300 Data Factory

## Purpose

This build step is designed for the CPU machine.

It generates:

- canonical cleaned page backgrounds
- real unlabeled field crops from the LIC sample PDFs
- synthetic handwritten field crops
- synthetic projected full pages
- manifests for:
  - `gold` manual correction
  - `weak` OCR-VLM labeling on the GPU box
  - `synthetic` TrOCR training

## Run

From the project root:

```powershell
python .\build_form300_cpu_dataset.py `
  --pdf-dir "C:\Users\sagar\Downloads\heck-main\_tmp_techathon_forms" `
  --out-dir ".\data\form300_factory" `
  --num-records 2000 `
  --num-projected-records 250 `
  --seed 7
```

## Scaling

Approximate crop count is:

`num_records * number_of_template_fields`

Current template coverage is `56` fields.

Examples:

- `2000` records -> about `112,000` synthetic crops
- `1000` records -> about `56,000` synthetic crops
- `500` records -> about `28,000` synthetic crops

If you want the initial target of `50k-100k` synthetic crops, use `1000-1800` records.

## Output Layout

```text
data/form300_factory/
  backgrounds/
  real_crops/
    images/
  synthetic_crops/
    <family>/<field_name>/*.png
  synthetic_pages/
    <page_type>/*.png
  manifests/
    summary.json
    real_crops_unlabeled.jsonl
    weak_label_requests.jsonl
    gold_label_sheet.csv
    synthetic_crops.jsonl
    train_crops.jsonl
    val_crops.jsonl
    synthetic_pages.jsonl
```

## What Goes To The GPU Box

Move these directories/files:

- `synthetic_crops/`
- `synthetic_pages/`
- `manifests/train_crops.jsonl`
- `manifests/val_crops.jsonl`
- `manifests/weak_label_requests.jsonl`
- `manifests/gold_label_sheet.csv`

## Training Tiers

- `gold`
  - manually corrected real crops
  - highest precision
- `weak`
  - OCR-VLM labeled real crops accepted by validators
  - generated later on GPU
- `synthetic`
  - generated on CPU
  - perfect labels by construction

## Current Coverage

The current template set focuses on the highest-value handwritten fields for blind evaluation:

- office proposal / deposit fields
- customer id / CKYC
- name fields
- DOB / age
- nationality / citizenship
- address lines / pincode
- PAN / Aadhaar / KYC marks
- occupation / income
- health yes-no marks
- suitability table values

This is enough to start the TrOCR family training pipeline and later refine with real error analysis.
