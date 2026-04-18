Executive Summary
We propose a hybrid pipeline centered on fine-tuning an open-weight TrOCR-based OCR engine for field￾level HTR, augmented with lightweight layout analysis and fallback vision-language verification. Synthetic
data and careful augmentation will supplement the limited real samples, and PEFT/LoRA will enable fine￾tuning on a single 24 GB NVIDIA L4 GPU. Key models include TrOCR (primary recognizer) , PaddleOCR
(for layout/text detection) , and an open VLM (Qwen2.5-VL-7B with AWQ quantization) for logical field￾checks and low-confidence fallback . We will apply data-centric methods (field cropping, GAN/
diffusion synth., heavy augmentation ), and PEFT (LoRA) to train within our resource envelope .
The 5-day plan includes data prep and synthetic generation (Days 1–2), TrOCR fine-tuning with LoRA (Day 3),
system integration and rule-based validation (Day 4), and final testing/deployment prep (Day 5). With
aggressive early prototyping and contingency for accuracy (e.g. fallback to VLM reasoning or template￾based correction), we target ≥95% field-level accuracy. 
Model Selection and Comparison
We compare candidate OCR/VDU models on accuracy, efficiency, and LIC compliance (open-license, self￾ownership). Key contenders: 
Model/Tool Type Size Open
Weights
Fine￾tunable
Memory
(24GB
L4)
Pros/Cons
TrOCR (Base/
Handwritten)
Vision￾Encoder–
Decoder
(Transformer)
240M/
470M
Yes (MIT) Yes (HF)
~6–
10GB
(w/FP16)
SOTA HTR,
supports LoRA;
may need
many samples.
Donut (Base)
Encoder–
Decoder VDU
(OCR-free)
~102M Yes (MIT) Yes (HF)
~4–6GB
(FP16)
End-to-end
VDU; excels at
structured
docs. Synthetic
pretraining
. Might
overkill for
small forms.
1
2
3 4
5 6 7
1
8
8
1

Model/Tool Type Size Open
Weights
Fine￾tunable
Memory
(24GB
L4)
Pros/Cons
PaddleOCR
(PP-OCRv5,
det+rec)
Pipeline
(Detectors +
CRNN/Trans
OCR)
(Det:~10M,
Rec:~100M)
Yes
(Apache-2)
Yes
(with
code)
~2–4GB
per
module
Fast,
production￾ready toolkit;
smaller models
match large
VLMs . Ideal
for detection
and printed
text.
Tesseract
OCR
Classical OCR n/a
Yes
(Apache)
No
(engine
only)
~2GB
Open-source
but poor on
messy
handwriting;
only candidate
fallback for
printed text.
Qwen2.5-
VL-7B
(Instruct)
Vision￾Language
(Large LLM)
7B (with
AWQ 4-bit)
Yes
(Apache) Yes (HF)
2GB
(AWQ 4-
bit
~0.6GB)
Excellent OCR
and reasoning
, JSON/
table outputs.
AWQ
quantization
enables fit in
24GB .
Higher latency
but strong
fallback.
LLaVA-7B
(NeXT)
Vision￾Language
7B
Mostly
(LGPL)
Yes (HF)
~3–4GB
(FP16)
Good local
agent model;
decent OCR
but
outperformed
by Qwen2.5
. Smaller
context than
Qwen.
2 2
3
4
3
9
10
2

Model/Tool Type Size Open
Weights
Fine￾tunable
Memory
(24GB
L4)
Pros/Cons
SmolVLM2
(500M–2B)
Very Small
VLM
0.5–2.2B
Yes
(Apache) Yes
<1GB–
2GB
Extremely
lightweight
, ideal edge
use. Good for
simple OCR/
VQA tasks but
limited
capacity for
complex forms.
InternVL
(3.5B)
Vision￾Language
(MoE)
3.5B
(activates
1–4B)
Yes
(Apache) Yes
~3–4GB
(FP16)
Competitive
with GPT-4V on
DocVQA .
Large 15B MoE
variant
available but
out of scope.
Idefics2 (8B) Vision￾Language
8B
Yes
(Apache) Yes ~6–7GB
Good balance
(SigLIP+Mistral)
for local;
strong text/
scene
understanding
. Slightly
heavier.
Compliance: TrOCR, Donut, PaddleOCR, Qwen2.5, LLaVA-NeXT, SmolVLM, InternVL, Idefics2 are
open-source or have permissive licenses (most Apache/MIT) allowing self-hosting and fine-tuning.
Tesseract is open but not trainable beyond lexicons. Qwen2.5-VL and SmolVLM have official AWQ/
GGUF quantized checkpoints to meet limited GPU constraints .
Primary Recognizer: We choose TrOCR (handwritten variant) . It achieved SOTA on handwritten
text when pretrained on synthetic data and its HF implementation supports LoRA adapters
(PEFT) for memory-efficient tuning . 
Detection/Layout: Use PaddleOCR’s PP-Structure or PP-OCRv5 for text detection and table/line
segmentation. It’s industrial-strength and lightweight . Alternatively, a simple contour/Canny
approach can crop fields if forms are clean. 
VLM Verifier: Qwen2.5-VL-7B (AWQ quantized) is a strong open VLM with structured-output support
and trained OCR capabilities . We’ll use it to check multi-field consistency (e.g. cross-field
logic) and low-confidence cases. 
Alternate Options: Donut could replace TrOCR for end-to-end form parsing, but we prefer modular
approach for greater control. LLaVA/InternVL are candidates but slightly weaker at raw OCR than
11
11
12
12
13
14
• 
9 15
• 1
1
7
• 
2
• 
3 4
• 
3

Qwen and lack the ready structured-output support. Tesseract may be used for checks or very clear
print, but will not drive main accuracy.
Data Preparation
Field Cropping: Assume “Form 300” with fixed layout. Manually define bounding boxes (x,y,w,h) for
each field (Name, PAN, DOB, etc.) using sample forms. This yields (image, label) pairs per field. Tools
like LabelImg or simple coordinate lists suffice. 
Synthetic Data Generation: Use a text generator (e.g. lists of first/last names, random PAN-like
strings, dates) and multiple cursive fonts to render synthetic field images. We can adapt open tools
like SynthOCR-Gen to generate diverse handwriting: draw random text with fonts and apply
augmentations . Example script (Python/PIL):
from PIL import Image, ImageDraw, ImageFont
import random, os
fonts = [ImageFont.truetype(f, size) for f in
["BrushScript.ttf","DancingScript.ttf",...]]
texts = ["John Doe","Amit Sharma",...]
for text in texts:
for font in fonts:
img = Image.new("L", (800,100), color=255)
draw = ImageDraw.Draw(img)
w,h = draw.textsize(text, font=font)
draw.text(((800-w)/2,10), text, font=font, fill=0)
# Save and later augment (blur, rotate, etc.)
We then stack/augment these synthetic crops. Augmentations include rotation (±5°), Gaussian blur/noise,
elastic warp, brightness/contrast jitters, background textures, to mimic scanned handwriting . In
practice, a library like Albumentations or custom PIL script can apply: 
import albumentations as A
aug = A.Compose([A.Rotate(limit=5), A.GaussNoise(var_limit=(10,50)),
A.ElasticTransform(alpha=1, sigma=50)])
augmented = aug(image=img_np)["image"]
- Data Mix: Combine ~1000 real field images (if available) with ~10× that many synthetic examples. Shuffle
and split ~90% train/10% val. Ensure class balance (e.g. PAN fields vs Name fields) and that each field type
appears sufficiently. 
Fine-Tuning (PEFT/LoRA)
Model & Tokenizer: Load Microsoft’s TrOCR processor and model (e.g. microsoft/trocr-base￾handwritten ) via Transformers. Wrap in a VisionEncoderDecoderModel if needed. Use PEFT (LoRA)
to reduce GPU load . Example LoRA setup:
• 
• 
6
5 6
• 
7
4

from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from peft import LoraConfig
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base￾handwritten")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base￾handwritten")
# Attach LoRA adapter
lora_config = LoraConfig(task_type="SEQ_2_SEQ_LM", r=8, lora_alpha=16,
target_modules=["q_proj","v_proj"])
model.add_adapter("htrocr", config=lora_config)
model.train_adapter("htrocr")
Training Data: Use the (image, text) pairs for each field. Convert images to pixel values and texts to
token IDs via the processor. We treat HTR as a sequence-to-sequence task (image→text). 
Training Arguments: Given 24GB GPU, use mixed precision (fp16 or bf16) to save memory. Batch
size ≈4–8 (to fit). Use gradient_accumulation_steps to simulate larger batches if needed.
Example:
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
args = Seq2SeqTrainingArguments(
output_dir="./trocr-ft",
per_device_train_batch_size=4,
gradient_accumulation_steps=2,
fp16=True,
learning_rate=3e-5,
num_train_epochs=5,
evaluation_strategy="epoch",
logging_steps=100,
save_total_limit=2,
predict_with_generate=True
)
trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=train_ds,
eval_dataset=val_ds, data_collator=proc.data_collator)
trainer.train()
Checkpointing: Save checkpoints each epoch. We expect field-level accuracy to improve to ~85–90%
on held-out data. If lagging, try TrOCR-large (if time permits) or augment more data. Target CER ~5–
7%, WER ~10–15% as success criteria . 
LoRA Benefits: By freezing 99% of parameters and only training adapters, memory is drastically
reduced . This lets 24GB L4 handle TrOCR-base or even -large with LoRA. Gradient checkpointing
(HuggingFace deepspeed or torch.utils.checkpoint ) can further cut VRAM use if needed.
• 
• 
• 
16
• 
7
5

Integration & Inference Pipeline
graph TB
 subgraph Input/Preprocess
 A[Raw Form Images] --> B[Preprocess: Binarize/Denoise]
 B --> C[Layout Analysis (e.g. PaddleOCR)]
 end
 subgraph Extraction
 C --> D[Crop Field Regions]
 D --> E[TrOCR Recognition (loRA)]
 E --> F[Validate/Normalize (regex rules)]
 E --> G[VLM (Qwen2.5-VL) Fallback]
 end
 subgraph Output
 F --> H[Structured Data (JSON/CSV)]
 G --> H
 end
Pipeline Steps: 1) Preprocessing: Standardize image size/color (grayscale), deskew if necessary. 2) 
Layout/Text Detection: Use PaddleOCR/PP-Structure to segment lines and detect text blocks, or use
heuristics (fixed form layout). 3) Field Cropping: Based on known coordinates, crop individual fields
(name, PAN, etc.). 4) Primary OCR: Run fine-tuned TrOCR on each crop. 5) Post-Validation: Apply
field-specific rules (see below). 6) VLM Check: For any field with low-confidence or rule-violation,
feed image+prompt to Qwen2.5-VL to double-check/extract text . 7) Output Assembly: Combine
results into structured output (CSV, JSON). 
Inference Wrapper Example: Using the 24 GB L4, load models with auto device mapping and mixed
precision:
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
"Qwen/Qwen2.5-VL-7B-Instruct-AWQ", torch_dtype="auto",
device_map="auto")
qwen_proc = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct￾AWQ")
# For TrOCR:
trocr_model = VisionEncoderDecoderModel.from_pretrained("./trocr-ft") # 
fine-tuned
trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base￾handwritten")
Throughput Tuning: Since forms are limited (50 eval forms in 1hr), sequential processing is fine. For
production, one could batch field images to increase GPU utilization. Ensure to pin batch sizes to
avoid OOM. 
• 
3
• 
• 
6

Evaluation Metrics & Scripts
Field-Level Accuracy: % of fields (across all forms) that match ground truth exactly. We also track
per-field CER/WER for non-exact:
def cer(pred, gold): # char error rate
# compute Levenshtein distance normalized by len(gold)
F1-Score (entity-level): If structured fields (like key–value), evaluate F1 on correctly extracted
entities. 
Confidence Thresholds: Record model confidence/logits. We may set a threshold (e.g. 0.8) below
which outputs trigger VLM fallback. 
Validation Script: After inference, run a script that checks:
Field regexes: e.g. PAN: ^[A-Z]{5}\d{4}[A-Z]$ , DOB: valid date format (DD/MM/YYYY) with
logical date checks.
Checksums: e.g. verify PAN checksum (Indian PAN uses weighted checksum) or credit card LUHN if
needed.
Fields failing validation are flagged or re-processed. All decisions and overrides are logged for audit. 
Post-Processing & Rules
Regex Validation:
PAN: Must be 10 chars (5 letters + 4 digits + 1 letter) . If OCR gives invalid PAN (e.g. wrong
pattern), correct common OCR confusions (0/O, 1/I) or mark for review. 
DOB: Parse with datetime ; enforce reasonable age (e.g. 18–100 years). 
Other Numeric Fields: Use numeric ranges or known check algorithms (e.g. Luhn for card numbers) to
detect errors. 
Normalization: Standardize formats (uppercase, fixed date format, remove extraneous characters). 
Confidence-Based Decision: Log all confidences. Optionally only accept fields if confidence >
threshold (empirically chosen, e.g. 0.5). Otherwise, defer to human or VLM check. 
Auditing: Keep a JSON log of each field’s image, OCR text, confidence, rule status, and any
corrections. This supports traceability. 
Deployment
Containerization: Package as a Docker container with dependencies (PyTorch, PaddleOCR, HF
Transformers). For example, FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime . 
Inference Endpoint: Expose a REST/CLI that takes scanned forms and returns JSON. Tools: FastAPI
or HuggingFace Inference Endpoints for scalability. 
Throughput Tuning: On L4, one inference pass (crop+TrOCR) per field in ~0.1–0.5s. Batch VLM
queries if possible. Use torch.compile() or transformers optimized kernels if needed. 
Fallback Mechanism: If TrOCR and validators fail, automatically call the VLM (Qwen2.5-VL-AWQ) to
parse the full form via a prompt like: 
• 
• 
• 
• 
• 
• 
• 
• 3
• 
• 
• 
• 
• 
• 
• 
• 
• 
7

“Extract all fields [Name, PAN, DOB, ...] from the attached form image as JSON.”
If even that fails, flag form for human review. 
5-Day Step-by-Step Plan
gantt
 dateFormat YYYY-MM-DD
 title 5-Day LIC Techathon Plan
 section Day 1 (Apr 16)
 Define pipeline architecture :done, 2026-04-16, 0.5d
 Label sample forms (field boxes) :done, 2026-04-16, 0.5d
 Set up env (install libs, HF login) :done, 2026-04-16, 0.5d
 section Day 2 (Apr 17)
 Generate synthetic field images :done, 2026-04-17, 1d
 Prepare training dataset (split) :done, 2026-04-17, 0.5d
 section Day 3 (Apr 18)
 Fine-tune TrOCR with LoRA :active, 2026-04-18, 1d
 Monitor metrics (CER/WER) :active, 2026-04-18, 1d
 section Day 4 (Apr 19)
 Integrate detection + TrOCR pipeline :active, 2026-04-19, 0.5d
 Implement regex rules & VLM calls :active, 2026-04-19, 0.5d
 Dry-run on eval forms :active, 2026-04-19, 1d
 section Day 5 (Apr 20)
 Full evaluation & error analysis :active, 2026-04-20, 0.5d
 Throughput tuning & optimizations :active, 2026-04-20, 0.5d
 Prepare final presentation/demo :active, 2026-04-20, 1d
Day 1: Prototype the pipeline on one form. Label field coordinates. Install Transformers, PEFT,
PaddleOCR, etc. 
Day 2: Write scripts to render synthetic handwriting for each field (using diverse fonts/
augmentations ). Assemble the (image, text) dataset. 
Day 3: Fine-tune TrOCR-base (with LoRA) on field crops for ~5 epochs. Validate on held-out crops.
Adjust LR or LoRA rank if needed. 
Day 4: Build end-to-end pipeline: detect fields on full form (hardcode coords or use detector), run
TrOCR on each. Apply post-rules. Integrate Qwen2.5-VL for mismatches (using AWQ weights for
memory). 
Day 5: Evaluate on hidden test set (50 forms). Compute field-level accuracy, CER/WER . If below
targets, identify failure modes (handwriting style, new tokens) and augment or tweak accordingly.
Finalize throughput and containerize solution.
Evaluation and Contingency Plans
Checkpoints: We aim for ≥95% of fields correct. If a field falls below ~90%, investigate: add more
synthetic examples for that field type, adjust post-rules, or increase model size (e.g. TrOCR-large with
LoRA). 
• 
• 
6 5
• 
• 
• 16
• 
8

Fallback Accuracy: Qwen2.5-VL can salvage tough fields (it handles OCR on charts/forms ).
Ensure it is 4-bit AWQ to run on the L4 (model card provides an AWQ variant ). 
Manual Review: For critical fields (PAN, DOB), low-confidence OCR can be flagged for human check.
Track such cases for continuous improvement. 
Explainability: Log attention maps (TrOCR encoder–decoder attention) or use SHAP on the VLM
outputs to understand errors. But due to time, emphasis is on accuracy over explainability. 
References
Li et al. (2022) – TrOCR: Transformer-based OCR with pre-trained models. Introduced TrOCR (vision+text
transformers), pretrained on synthetic data for OCR, achieving SOTA in handwritten and scene text
. 
Kim et al. (ECCV 2022) – OCR-free Document Understanding Transformer (Donut). Presented an end-to￾end VDU model (Donut) with synthetic pretraining and a data generator, enabling OCR-free parsing
of documents . 
Cui et al. (arXiv 2025) – PaddleOCR 3.0. Describes PP-OCRv5 (<100M params) and PP-StructureV3 for
OCR and layout parsing; despite its small size, PP-OCRv5 rivals billion-parameter VLMs in accuracy
. 
Garrido-Muñoz et al. (arXiv 2026) – Zero-Shot Synthetic-to-Real HTR. Proposes transferring synthetic￾trained HTR models to real data via “task analogies” without target labels, highlighting synthetic data
augmentation challenges . 
Rassul et al. (arXiv 2025) – HTR Augmentation Survey. Systematic review of offline HTR data
augmentation: GANs, diffusion, and transformer-based synthetic methods are key to generating
realistic handwriting for robust OCR . 
Malik et al. (arXiv 2026) – SynthOCR-Gen. An open tool to generate synthetic OCR datasets (multi-font
rendering + 25+ augmentations) for low-resource scripts. Demonstrates automated word-image
generation and extensive augmentations (rotation, blur, noise) . 
Hugging Face Documentation (2024–25) – PEFT/LoRA Guides. Describes adapter-based fine-tuning
(LoRA) that updates few parameters to save memory . Example HF cookbook recipes show fine￾tuning vision-LMs (e.g. Qwen2-VL-7B) with TRL/TRL . 
Roboflow Blog (2025) – Best Local VLMs. Highlights that Qwen2.5-VL-7B (6GB, 125K context)
outperforms larger 11B Llama-3.2-Vision on OCR/Doc benchmarks . Discusses small VLMs (LLaVA￾NeXT, SmolVLM, Idefics2) for local deployment. 
Hugging Face Model Cards (2024–25) – Qwen2.5-VL. Qwen’s release notes confirm its structured￾output and JSON form parsing abilities . An AWQ-quantized 7B checkpoint is available, enabling
efficient inference . 
[2109.10282] TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models
https://arxiv.org/abs/2109.10282
PaddleOCR 3.0 Technical Report
https://arxiv.org/pdf/2507.05595
Qwen/Qwen2.5-VL-7B-Instruct · Hugging Face
https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
• 3
15
• 
• 
• 
1
• 
8
• 
2
• 
17 18
• 
5
• 
6
• 
7
19
• 
4
• 
3
15
1
2
3
9

Best Local Vision-Language Models for Offline AI
https://blog.roboflow.com/local-vision-language-models/
Advancing Offline Handwritten Text Recognition: A Systematic Review of Data Augmentation and
Generation Techniques
https://arxiv.org/html/2507.06275v1
SYNTHOCR-GEN: A SYNTHETIC OCR DATASET GENERATOR FOR LOW-RESOURCE LANGUAGES- BREAKING
THE DATA BARRIER
https://arxiv.org/html/2601.16113v1
Parameter-efficient fine-tuning · Hugging Face
https://huggingface.co/docs/transformers/en/peft
[2111.15664] OCR-free Document Understanding Transformer
https://arxiv.org/abs/2111.15664
Quantization · Hugging Face
https://huggingface.co/docs/transformers/en/main_classes/quantization
Qwen/Qwen2.5-VL-7B-Instruct-AWQ · Hugging Face
https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct-AWQ
Fine tune trocr model - Models - Hugging Face Forums
https://discuss.huggingface.co/t/fine-tune-trocr-model/151014
Zero-Shot Synthetic-to-Real Handwritten Text Recognition via Task Analogies
https://arxiv.org/html/2604.09713v1
Fine-Tuning a Vision Language Model (Qwen2-VL-7B) with the Hugging Face Ecosystem (TRL) · Hugging
Face
https://huggingface.co/learn/cookbook/en/fine_tuning_vlm_trl
4 10 11 12 13 14
5
6
7
8
9
15
16
17 18
19
10

