# SOTA Strategy for LIC Form 300 Handwritten Proposal Extraction

## Executive Decision

This is not a generic document OCR problem. It is a fixed-template, multi-page, handwritten field extraction problem under blind evaluation, sovereign deployment, and exact-match scoring constraints.

The winning architecture is:

1. Template-first page registration and routing.
2. Specialized field-family extractors.
3. Constrained validation and normalization.
4. OCR-specific VLM fallback only for hard cases.
5. Synthetic data generated mainly at crop level, not page level.

Do not use a general VLM as the primary extractor for all fields. That is slower, less auditable, and more likely to hallucinate under LIC's exact-match and non-billable-failure rules.

## What The RFP Actually Demands

From the LIC documents, the hard constraints are:

- The benchmark target is `>=95%` field-level accuracy and `>=97%` character-level accuracy, with `<=5%` rejection and `<=10%` manual correction.
- Corrigendum softens shortlisting in practice: if too few bidders hit `95%`, LIC can consider top bidders down to `85%`.
- Techathon evaluation is blind: `100` sample proposal forms are shared only on the day of the Techathon.
- During the challenge, bidders must not use public cloud, external AI APIs, or internet-exposed processing unless LIC explicitly approves it.
- The solution must be based on open-weight or sovereign VLMs. API-only GPT/Gemini/Claude-style vision solutions are not eligible.
- Fine-tuning capability is required, but LIC treats future fine-tuning as a separate commercial activity and does not want those costs embedded into per-proposal pricing.
- Structured output must end in English, even if the input contains Hindi or other Indian languages.

Implication:

- The Techathon is fundamentally an "offline blind robustness" challenge.
- The system must already generalize on unseen handwritten samples without depending on last-minute tuning.
- Evaluation, audit trail, confidence, and rejection logic matter almost as much as raw OCR quality.

## What The Sample Zip Shows

I inspected the `30` PDFs in `Techathon Sample form for Training.zip`.

Observed properties:

- These are scanned, image-heavy PDF packs, not text PDFs.
- Most samples are `28` pages. Shorter variants exist at `15`, `20`, `26`, and `27` pages.
- The printed template geometry is highly stable across files.
- Page 1 is a standard cover/inward-photo page.
- Page 2 and later pages contain bounded handwritten fields, repetitive labels, and bilingual printed scaffolding.
- The hard content is concentrated in:
  - names
  - parent/spouse names
  - DOB and age
  - PAN / CKYC / Aadhaar / proposal numbers
  - multi-line addresses
  - occupation / income
  - yes-no medical declarations
  - handwritten amounts
  - signatures and photo-related presence checks

Implication:

- This is template registration plus field routing.
- Whole-page OCR is useful for pseudo-labeling and failure recovery, but not as the primary exact extractor.

## Current Open-Weight SOTA Landscape

The current model landscape separates into three roles:

### 1. Page-Level Document Parsing

Best current open-weight option:

- `PaddleOCR-VL-1.5`

Why:

- Officially positioned as a SOTA lightweight document VLM for document parsing.
- Explicitly robust to scanning, skew, warping, illumination, and photographed documents.
- Strong fit for page understanding, structure extraction, and pseudo-label generation.

### 2. Localization And Production OCR

Best practical production option:

- `PP-OCRv5`

Why:

- Official paper argues it is competitive with many billion-parameter VLMs on OCR while offering better localization precision and reduced hallucinations.
- For fixed-form extraction, localization precision matters more than benchmark glamour.
- Better suited to printed anchors, label detection, and lightweight deployment.

### 3. OCR-Specific VLM Fallback / Teacher

Strong open-weight options:

- `HunyuanOCR`
- `DeepSeek-OCR-2`

Why:

- Both are OCR-specialized VLMs and more appropriate as OCR teachers or hard-case fallback than older general VLM baselines.
- HunyuanOCR shows strong document parsing and information extraction results in its official repo/model card.
- DeepSeek-OCR-2 is a dedicated OCR model with document-to-markdown grounding and GPU-serving support.

### 4. General VLM Fallback

Acceptable but not the first choice for OCR-specific fallback:

- `Qwen2.5-VL-7B`

Why:

- Good structured output and visual reasoning.
- Still useful for cross-field consistency checks and JSON repair.
- But it is no longer the best OCR-specialized model to anchor the OCR stack around.

## Recommended Final Architecture

## Layer 0: PDF Ingest And Rasterization

- Render every PDF page to images at fixed DPI.
- Treat PDF structure as unreliable; some sample PDFs already show malformed internals while rendering correctly.
- Standardize to one canonical image pipeline before anything else.

## Layer 1: Page Registration And Page-Type Routing

Use:

- geometric registration
- template matching
- printed-anchor detection
- page-type classification

Recommended approach:

- Use PP-OCRv5 for label anchors and printed text regions.
- Register each page to a canonical template using printed lines, boxes, and label anchors.
- Assign a `page_type_id` such as:
  - cover_page
  - proposer_details_page
  - kyc_occupation_page
  - health_declaration_page
  - signature_declaration_page
  - suitability_page

This is the most important engineering decision in the whole system.

If registration is stable, field extraction becomes a bounded crop problem.

## Layer 2: Field Router

For each `page_type_id`, define:

- canonical field boxes
- field family
- expected datatype
- validator
- optional/mandatory status
- fallback policy

Field families should be separated, not mixed:

- `short_id`: PAN, CKYC, proposal no, IFSC, PIN
- `date`: DOB, proposal date, expiry
- `numeric`: age, term, counts, phone, Aadhaar digits
- `amount`: premium, sum assured, deposit
- `short_text`: city, state, occupation, relationship
- `name_text`: first/middle/last/father/mother/spouse/nominee
- `long_text`: address lines
- `binary_mark`: yes/no, selected option, male/female/married
- `signature_presence`
- `photo_presence`

## Layer 3: Family-Specific Extractors

### Printed Labels / Anchors

- `PP-OCRv5`

### Checkbox / Mark / Binary Option

- small classifier or rule-based mark detector
- do not run a generative VLM for these by default

### Handwritten Short IDs And Constrained Fields

- `TrOCR-large-handwritten + LoRA`
- decode with field constraints

Examples:

- PAN must be `AAAAA9999A`
- PIN must be 6 digits
- date must parse
- IFSC shape must match bank code pattern

For these fields, constrained decoding plus validator-backed correction will outperform general free-text OCR.

### Handwritten Names And Short Free Text

- `TrOCR-large-handwritten + LoRA`
- separate adapters or heads per family:
  - names
  - addresses
  - occupation / relation / city-state

Do not use one universal adapter for everything. Name handwriting and address handwriting have different length, rhythm, and failure modes.

### Long Address Fields

- segment line-by-line first
- then run TrOCR per line
- merge with address normalizer

### Hard Fallback

Use one OCR-specialized VLM:

- `HunyuanOCR` or `DeepSeek-OCR-2`

Use one general reasoning VLM only if needed:

- `Qwen2.5-VL-7B`

Fallback should be field-scoped, not full-page by default.

Example:

- primary crop OCR fails PAN regex
- rerun only the PAN crop with OCR VLM prompt
- compare candidate outputs
- pass through validator
- log both attempts

## Layer 4: Validator And Decision Engine

This layer is mandatory for LIC.

Per-field validators:

- regex
- type parser
- semantic range
- cross-field consistency
- null/blank legality

Cross-field rules:

- DOB <-> age coherence
- city/state/pincode plausibility
- gender/marital/spouse coherence
- amount / income / suitability consistency flags
- impossible date detection

Decision policy:

- accept
- auto-correct
- fallback
- reject / needs review

All of this must be auditable.

## Layer 5: Audit Output

For every field, save:

- page id
- bbox
- extracted text
- normalized text
- model used
- confidence
- validator status
- fallback history
- final decision

This is required both for LIC verification and for your own error analysis.

## Synthetic Data Program

The synthetic program should be built around coverage, not volume.

The paper in `6171_Reasoning_Driven_Syntheti.md` is useful for one core idea:

- define taxonomy and coverage explicitly

It is not proof that one synthetic recipe is universally optimal, and it is not an OCR-specific result. Use it as a data-design framework, not as a claim that synthetic data alone guarantees `100%`.

## How To Use The 30 Real Packs

The 30 real packs are enough to bootstrap a serious synthetic program if used correctly.

Use them for:

- page template geometry
- printed anchor vocabulary
- writer-style bank
- scan/background/noise bank
- real blank-field frequency
- overwrite / correction patterns
- field co-occurrence statistics
- semantic realism

Do not spend them only as supervised OCR labels.

## Synthetic Generation Levels

### Level A: Crop-Level Synthetic

Main training signal for TrOCR.

Generate for each field family:

- PAN / CKYC / Aadhaar / IFSC / PIN / phone / proposal no
- dates
- names
- addresses
- occupation / relation / city / state
- amounts and numeric values
- yes/no and selection marks
- blanks / strike-through / overwritten entries

This is the highest ROI dataset.

### Level B: Template-Projected Synthetic Pages

Take synthetic crop text and paste into real template boxes.

Purpose:

- train page registration tolerance
- simulate neighboring handwriting interference
- create realistic multi-field pages
- support pseudo-labeling and end-to-end dry runs

### Level C: Hard-Negative Synthetic

Required for robust rejection logic.

Examples:

- near-valid PAN
- malformed date
- over-inked field
- blank but noisy field
- multiple strokes in checkbox
- clipped or partially visible writing

### Level D: Teacher-Labeled Real Crops

Use OCR-VLM teachers to propose text for unlabeled or weakly labeled crops.

Then:

- validate with regex and cross-field logic
- keep only high-precision pseudo-labels
- do not blindly absorb teacher outputs

## Coverage Taxonomy

Track synthetic coverage on these axes:

- page type
- field family
- character set
- string length
- writing density
- line count
- uppercase/lowercase/mixed case
- ink thickness
- slant
- pressure / broken strokes
- blur
- skew
- border clipping
- background shading
- overwrite / correction
- blank / NA / same-as-above
- semantic coherence with neighboring fields

If you cannot measure coverage, you cannot know whether your synthetic data is doing useful work.

## Practical Counts

For this project, the first serious synthetic corpus should look like:

- `10k-20k` constrained ID/date/numeric crops
- `20k-30k` name crops
- `20k-30k` address-line crops
- `10k-15k` short-text crops
- `10k-20k` checkbox/mark samples
- `5k-10k` hard negatives
- `2k-5k` full-page projected pages

This is more useful than generating hundreds of thousands of weak full-page images.

## Model Recommendation By Role

| Role | Recommended choice | Why |
| --- | --- | --- |
| Page parsing / teacher | PaddleOCR-VL-1.5 | Best current lightweight open document parser for scanned/warped docs |
| Printed anchors / localization | PP-OCRv5 | Better localization precision, lower hallucination risk |
| Handwritten field OCR | TrOCR-large-handwritten + LoRA | Fine-tunable, controllable, well-suited to crop-level HTR |
| OCR-specific fallback | HunyuanOCR or DeepSeek-OCR-2 | Better OCR specialization than older general VLM baselines |
| Reasoning / JSON repair / cross-field judge | Qwen2.5-VL-7B | Strong structured outputs and reasoning, useful as secondary verifier |

## What Should Not Be The Primary Strategy

Avoid these traps:

- one-shot full-page VLM extraction as the main engine
- one generic TrOCR adapter for every field type
- synthetic generation that only varies rendered fonts
- measuring "field accuracy" using extracted-field count without ground truth
- English-only anchor assumptions on a bilingual form
- treating checkbox / marks / signatures as OCR text tasks

## Gap Versus The Current Repo

The current repo is a useful prototype, but it is not yet aligned to the challenge.

Main gaps:

1. Model choice mismatch:
   - `src/config.py` sets `Qwen/Qwen2.5-VL-3B-Instruct`
   - `src/extractor.py` comments and outputs still refer to `Qwen2.5-VL-7B`
   - This needs one explicit model policy.

2. Metrics are currently not true field-accuracy metrics:
   - `src/pipeline.py` increments `correct_fields` using `fields_extracted`, then derives `field_level_accuracy`.
   - That is not LIC-compliant accuracy measurement.

3. Field inventory is still generic:
   - `src/config.py` field list is not yet grounded to exact Form 300 page templates and field boxes.

4. Template matching is too narrow:
   - `src/template_matcher.py` uses a small English anchor rule set.
   - The actual form is bilingual and layout-stable; this should be replaced with page-type-specific canonical templates and printed-anchor registration.

5. Synthetic generation is too weak:
   - `src/trocr_finetuning.py` is still mostly string rendering plus augmentation.
   - It does not yet model field semantics, real scan artifacts, checkbox marks, blanks, overwrites, or page-level context.

## Recommended Immediate Work Sequence

### Phase 1: Lock The Page Taxonomy

- enumerate all page archetypes in the 30 packs
- choose one canonical template per page type
- define registration anchors
- define required output fields per page type

### Phase 2: Build The Field Ontology

For every field:

- page type
- bbox
- family
- datatype
- validator
- mandatory flag
- fallback policy

### Phase 3: Build The Crop Extractor Stack

- PP-OCRv5 for printed anchors
- checkbox classifier
- TrOCR family adapters
- OCR-VLM fallback
- Qwen verifier

### Phase 4: Build The Synthetic Factory

- crop generators by family
- page-projected generation
- hard negatives
- pseudo-label acceptance pipeline
- coverage dashboard

### Phase 5: Build LIC-Style Evaluation

- exact-match field scorer
- character scorer
- rejection-rate scorer
- manual-correction-rate estimator
- full audit log

## Bottom Line

If the goal is a serious bid, the SOTA approach is not "bigger VLM first."

It is:

- specialized OCR where precision matters
- template registration because the form is stable
- field-family modeling because the handwriting tasks are heterogeneous
- OCR-VLM fallback only where needed
- synthetic data designed around coverage and validators, not just volume

That architecture is much more likely to produce Techathon-grade results than a generic page-to-JSON VLM pipeline.

## Source Notes

Local LIC docs used:

- `LIC RFP-Proposal form data extraction_290126 (4).md`
- `Corrigendum 1 (1).md`
- `Pre-Bid Responses - Proposal Form Data Extraction RFP (1).md`
- `Executive Summary (2).md`
- `6171_Reasoning_Driven_Syntheti.md`

Primary web sources used:

- PaddleOCR official GitHub: https://github.com/PaddlePaddle/PaddleOCR
- PaddleOCR-VL-1.5 paper: https://arxiv.org/abs/2601.21957
- PP-OCRv5 paper: https://arxiv.org/abs/2603.24373
- Tencent HunyuanOCR official GitHub: https://github.com/Tencent-Hunyuan/HunyuanOCR
- Tencent HunyuanOCR official model card: https://huggingface.co/tencent/HunyuanOCR
- DeepSeek-OCR official model card: https://huggingface.co/deepseek-ai/DeepSeek-OCR
- DeepSeek-OCR-2 official model card: https://huggingface.co/deepseek-ai/DeepSeek-OCR-2
- Qwen2.5-VL-7B official model card: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
