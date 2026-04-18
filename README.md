# LIC Form 300 Extraction Stack

CPU-first dataset factory and labeling stack for LIC Form 300 handwritten proposal extraction, with GPU-side TrOCR fine-tuning from final manifests.

## What Is In This Repo

- CPU dataset factory
  - canonical Form 300 page templates
  - real crop harvesting from scanned LIC PDFs
  - synthetic handwritten crop generation
  - synthetic full-page projection
- polished labeling stack
  - crop QC
  - gold-label prioritization
  - weak-label acceptance for OCR-VLM teachers
  - final train/val manifest merge
- GPU training entrypoint
  - TrOCR fine-tuning directly from final manifests

Core files:

- [build_form300_cpu_dataset.py](./build_form300_cpu_dataset.py)
- [build_polished_labeling_stack.py](./build_polished_labeling_stack.py)
- [train_trocr_from_manifest.py](./train_trocr_from_manifest.py)
- [src/dataset_factory.py](./src/dataset_factory.py)
- [src/polished_labeling.py](./src/polished_labeling.py)
- [src/form300_templates.py](./src/form300_templates.py)
- [src/trocr_finetuning.py](./src/trocr_finetuning.py)
- [CPU_FORM300_DATA_FACTORY.md](./CPU_FORM300_DATA_FACTORY.md)
- [POLISHED_LABELING_STACK.md](./POLISHED_LABELING_STACK.md)
- [SOTA_LIC_FORM300_STRATEGY.md](./SOTA_LIC_FORM300_STRATEGY.md)

## Recommended Architecture

For blind LIC Techathon-style evaluation, the intended architecture is:

1. template-first page registration and field routing
2. family-specific field OCR
3. validator-aware post-processing
4. OCR-specific VLM fallback for hard crops
5. exact-match field scoring with audit trail

This repo implements the data and labeling side of that plan.

## Environment

### CPU box

Use the CPU box for:

- template definition
- real crop harvesting
- synthetic data generation
- QC and labeling manifests

### GPU box

Use the GPU box for:

- OCR-VLM teacher inference for weak labels
- TrOCR family fine-tuning
- evaluation and threshold calibration

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Optional GPU-side packages:

- `torch` with the correct CUDA build
- `flash-attn` if supported

## Input Assumption

The CPU factory expects scanned Form 300 PDF packs similar to the LIC sample zip.

By default, runtime data goes under `data/`, but `data/` is gitignored and should not be committed.

## Step 1: Build CPU Dataset Bundle

```powershell
python .\build_form300_cpu_dataset.py `
  --pdf-dir "C:\path\to\training_pdfs" `
  --out-dir ".\data\form300_factory" `
  --num-records 1500 `
  --num-projected-records 250 `
  --seed 7
```

What it produces:

- cleaned background pages
- real unlabeled field crops
- synthetic field crops
- synthetic full pages
- manifests for gold, weak, and synthetic tiers

Approximate synthetic crop count:

`num_records * template_field_count`

Current template field coverage is `56` fields, so:

- `1000` records -> about `56,000` synthetic crops
- `1500` records -> about `84,000` synthetic crops
- `1800` records -> about `100,800` synthetic crops

## Step 2: Build Polished Labeling Assets

```powershell
python .\build_polished_labeling_stack.py `
  --factory-dir ".\data\form300_factory"
```

This writes:

- `manifests/crop_qc_report.jsonl`
- `manifests/gold_label_priority.csv`
- `manifests/final_train_manifest.jsonl`
- `manifests/final_val_manifest.jsonl`

## Step 3: Gold Labels

Review and fill:

- `data/form300_factory/manifests/gold_label_sheet.csv`

Populate `gold_text` for the selected high-priority crops, then rerun:

```powershell
python .\build_polished_labeling_stack.py `
  --factory-dir ".\data\form300_factory" `
  --gold-csv ".\data\form300_factory\manifests\gold_label_sheet.csv"
```

## Step 4: Weak Labels From OCR-VLM Teachers

Run OCR-VLM teacher inference on the GPU box over:

- `data/form300_factory/manifests/weak_label_requests.jsonl`

Expected teacher prediction JSONL format:

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

You can emit multiple rows per crop id from different teacher models.

Then rerun:

```powershell
python .\build_polished_labeling_stack.py `
  --factory-dir ".\data\form300_factory" `
  --teacher-predictions ".\teacher_predictions.jsonl"
```

Accepted weak labels are written to:

- `data/form300_factory/manifests/accepted_weak_labels.jsonl`

## Step 5: Train TrOCR On GPU

### Family-specific training

Recommended families:

- `short_id,date,numeric`
- `name_text`
- `long_text`
- `short_text`

Example:

```powershell
python .\train_trocr_from_manifest.py `
  --train-manifest ".\data\form300_factory\manifests\final_train_manifest.jsonl" `
  --val-manifest ".\data\form300_factory\manifests\final_val_manifest.jsonl" `
  --field-family "short_id,date,numeric" `
  --base-model "microsoft/trocr-base-handwritten" `
  --output-dir ".\models\trocr-shortid" `
  --epochs 10 `
  --batch-size 8 `
  --learning-rate 5e-5 `
  --use-canonical
```

Names:

```powershell
python .\train_trocr_from_manifest.py `
  --train-manifest ".\data\form300_factory\manifests\final_train_manifest.jsonl" `
  --val-manifest ".\data\form300_factory\manifests\final_val_manifest.jsonl" `
  --field-family "name_text" `
  --output-dir ".\models\trocr-names" `
  --epochs 10 `
  --batch-size 8 `
  --learning-rate 5e-5
```

Addresses:

```powershell
python .\train_trocr_from_manifest.py `
  --train-manifest ".\data\form300_factory\manifests\final_train_manifest.jsonl" `
  --val-manifest ".\data\form300_factory\manifests\final_val_manifest.jsonl" `
  --field-family "long_text" `
  --output-dir ".\models\trocr-address" `
  --epochs 10 `
  --batch-size 6 `
  --learning-rate 3e-5
```

## Labeling Policy

The polished stack validates labels by field family:

- `short_id`
  - PAN-like and CKYC-like shape checks
- `date`
  - date-shape checks
- `amount`
  - numeric/amount checks
- `numeric`
  - digits-only canonicalization
- `binary_mark`
  - yes/no normalization
- `name_text`, `short_text`, `long_text`
  - length and character sanity checks

Weak labels are accepted only if:

- the normalized label passes the family rules
- the combined confidence clears threshold
- consensus across teacher models strengthens the score when available

## Hindi / Multilingual Note

This repo is currently optimized for the Techathon-first case:

- Indian entities
- mostly Roman-script handwritten values
- English structured output

Hindi or multilingual handwriting can be added as a separate track later, but should not be mixed blindly into the first TrOCR dataset unless the target evaluation truly includes it.

## Git Notes

The repo intentionally does not commit:

- `data/`
- `models/`
- large PDFs / zips
- virtual environments

Commit the code and docs only. Build datasets and models locally on the target machines.

## Current Scope

This repo now contains:

- CPU synthetic generation
- CPU real-crop harvesting
- polished labeling stack
- manifest-driven GPU TrOCR training entrypoint

What still remains outside this repo:

- OCR-VLM teacher runner implementation for the GPU box
- final production inference router that chooses among family-specific TrOCR models and fallback OCR-VLMs
- exact LIC scoring dashboard against ground truth

That is enough to start building the real training corpus and fine-tune TrOCR cleanly.
