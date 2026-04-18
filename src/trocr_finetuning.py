"""
TrOCR Fine-Tuning with LoRA + Synthetic Data Generation
=========================================================
Implements reasoning-driven synthetic data (inspired by Simula framework)
with taxonomy-based coverage for insurance form field types.
"""

import os
import json
import random
import logging
import string
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


# ──────────────────────── SYNTHETIC DATA TAXONOMY ────────────────────────

# Inspired by the Simula paper's taxonomy-driven approach for coverage
FIELD_TAXONOMIES = {
    "Name": {
        "subtypes": ["first_name", "full_name", "father_name", "nominee_name"],
        "patterns": {
            "indian_male": ["Rajesh Kumar", "Amit Sharma", "Suresh Patel", "Vinod Gupta", "Manoj Singh",
                           "Ravi Shankar", "Deepak Verma", "Arun Joshi", "Sanjay Mishra", "Prakash Rao",
                           "Ramesh Chandra", "Vijay Nair", "Sunil Reddy", "Pradeep Tiwari", "Ashok Mehta",
                           "Gopal Krishna", "Mohan Lal", "Dinesh Choudhary", "Rakesh Agarwal", "Umesh Yadav"],
            "indian_female": ["Sunita Devi", "Priya Sharma", "Anita Kumari", "Kavita Singh", "Meena Patel",
                             "Rekha Gupta", "Pooja Verma", "Asha Rani", "Lakshmi Narayanan", "Savita Jain",
                             "Geeta Desai", "Kamla Devi", "Sushila Bai", "Padma Reddy", "Usha Kapoor"],
            "south_indian": ["Venkatesh Murthy", "Lakshmi Prasad", "Ramachandran Iyer", "Subramaniam Pillai",
                            "Thirumurugan Selvam", "Nagalakshmi Raman", "Krishnamurthy Swamy"],
        },
        "augmentations": ["uppercase", "lowercase", "title_case", "all_caps", "initials_expanded"],
    },
    "PAN": {
        "subtypes": ["individual_pan"],
        "pattern_regex": r"[A-Z]{5}\d{4}[A-Z]",
        "generator": "generate_pan",
    },
    "Date": {
        "subtypes": ["dob", "proposal_date", "general_date"],
        "formats": ["DD/MM/YYYY", "DD-MM-YYYY", "DD.MM.YYYY"],
        "ranges": {"dob_min_year": 1940, "dob_max_year": 2008, "recent_min_year": 2024},
    },
    "Mobile": {
        "subtypes": ["mobile_number"],
        "pattern": r"[6-9]\d{9}",
    },
    "Address": {
        "subtypes": ["full_address", "city", "state", "pincode"],
        "components": {
            "house_numbers": ["12", "45/A", "B-302", "Flat 7", "H.No. 234", "Plot 56"],
            "streets": ["MG Road", "Station Road", "Gandhi Nagar", "Nehru Street", "Park Avenue",
                       "Civil Lines", "Rajaji Nagar", "Vikas Puri", "Sector 21"],
            "areas": ["Andheri East", "Bandra West", "Koramangala", "Adyar", "Salt Lake",
                     "Jubilee Hills", "Connaught Place", "Lajpat Nagar", "Malviya Nagar"],
            "cities": ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Hyderabad",
                      "Pune", "Ahmedabad", "Jaipur", "Lucknow", "Patna", "Bhopal",
                      "Kochi", "Chandigarh", "Nagpur", "Indore", "Vadodara"],
            "states": ["Maharashtra", "Delhi", "Karnataka", "Tamil Nadu", "West Bengal",
                      "Telangana", "Rajasthan", "Gujarat", "Uttar Pradesh", "Bihar",
                      "Madhya Pradesh", "Kerala", "Punjab", "Andhra Pradesh"],
            "pincodes": [f"{random.randint(100000, 999999)}" for _ in range(50)],
        },
    },
    "Amount": {
        "subtypes": ["sum_assured", "premium", "annual_income"],
        "ranges": {
            "sum_assured": (100000, 50000000),
            "premium": (1000, 500000),
            "annual_income": (100000, 100000000),
        },
        "formats": ["plain", "with_commas", "in_words"],
    },
    "Numeric": {
        "subtypes": ["age", "term", "account_number", "aadhaar"],
        "ranges": {
            "age": (18, 75),
            "term": (5, 40),
        },
    },
    "IFSC": {
        "subtypes": ["bank_ifsc"],
        "bank_codes": ["SBIN", "HDFC", "ICIC", "UTIB", "PUNB", "BARB", "CNRB",
                       "BKID", "IDIB", "UBIN", "MAHB", "CORP", "IOBA"],
    },
    "Occupation": {
        "subtypes": ["profession"],
        "values": ["Service", "Business", "Self Employed", "Government Service",
                  "Teacher", "Doctor", "Engineer", "Lawyer", "Farmer", "Retired",
                  "Housewife", "Student", "Accountant", "Shopkeeper", "Contractor"],
    },
    "Relationship": {
        "subtypes": ["nominee_relation"],
        "values": ["Wife", "Husband", "Son", "Daughter", "Father", "Mother",
                  "Brother", "Sister", "Spouse", "Self"],
    },
}


class SyntheticDataGenerator:
    """
    Generates synthetic handwriting data for TrOCR fine-tuning.
    Uses taxonomy-driven approach for comprehensive field coverage.
    """

    def __init__(self, fonts_dir: str = None, output_dir: str = None):
        self.fonts_dir = Path(fonts_dir) if fonts_dir else Path(__file__).parent.parent / "fonts"
        self.output_dir = Path(output_dir) if output_dir else Path(__file__).parent.parent / "data" / "synthetic"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fonts = self._load_fonts()

    def _load_fonts(self) -> List:
        """Load handwriting-style fonts."""
        fonts = []
        self.fonts_dir.mkdir(parents=True, exist_ok=True)

        # Search for fonts
        for ext in ["*.ttf", "*.otf", "*.TTF", "*.OTF"]:
            fonts.extend(self.fonts_dir.glob(ext))

        # Also check system fonts
        system_font_dirs = [
            Path("C:/Windows/Fonts"),
            Path(os.path.expanduser("~/.fonts")),
        ]
        handwriting_keywords = ["script", "hand", "cursive", "comic", "brush", "casual", "indie"]
        for font_dir in system_font_dirs:
            if font_dir.exists():
                for ext in ["*.ttf", "*.otf"]:
                    for f in font_dir.glob(ext):
                        if any(kw in f.stem.lower() for kw in handwriting_keywords):
                            fonts.append(f)

        if not fonts:
            # Fallback: use default font
            logger.warning("No handwriting fonts found. Using default font.")
            fonts = [None]  # Will use PIL default

        logger.info(f"Loaded {len(fonts)} fonts for synthetic data generation")
        return fonts

    def generate_dataset(self, num_samples: int = 5000) -> Tuple[List[str], List[str]]:
        """Generate a full synthetic dataset covering all field types."""
        images_dir = self.output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        image_paths = []
        labels = []
        metadata = []

        samples_per_type = num_samples // len(FIELD_TAXONOMIES)

        for field_type, taxonomy in FIELD_TAXONOMIES.items():
            logger.info(f"Generating {samples_per_type} samples for {field_type}...")

            for i in range(samples_per_type):
                text = self._generate_text(field_type, taxonomy)
                img = self._render_text(text)

                if img is not None:
                    # Apply augmentations
                    img = self._augment(img)

                    fname = f"{field_type}_{i:05d}.png"
                    fpath = images_dir / fname
                    img.save(fpath)

                    image_paths.append(str(fpath))
                    labels.append(text)
                    metadata.append({
                        "field_type": field_type,
                        "text": text,
                        "image_path": str(fpath),
                    })

        # Save metadata
        meta_file = self.output_dir / "metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"Generated {len(image_paths)} synthetic samples")
        return image_paths, labels

    def _generate_text(self, field_type: str, taxonomy: Dict) -> str:
        """Generate realistic text for a field type."""
        if field_type == "Name":
            category = random.choice(list(taxonomy["patterns"].keys()))
            name = random.choice(taxonomy["patterns"][category])
            aug = random.choice(taxonomy["augmentations"])
            if aug == "uppercase":
                return name.upper()
            elif aug == "lowercase":
                return name.lower()
            elif aug == "all_caps":
                return name.upper()
            return name

        elif field_type == "PAN":
            return self._generate_pan()

        elif field_type == "Date":
            fmt = random.choice(taxonomy["formats"])
            if random.random() < 0.7:  # DOB
                year = random.randint(taxonomy["ranges"]["dob_min_year"],
                                     taxonomy["ranges"]["dob_max_year"])
            else:  # Recent date
                year = random.randint(2024, 2026)
            month = random.randint(1, 12)
            day = random.randint(1, 28)
            if fmt == "DD/MM/YYYY":
                return f"{day:02d}/{month:02d}/{year}"
            elif fmt == "DD-MM-YYYY":
                return f"{day:02d}-{month:02d}-{year}"
            else:
                return f"{day:02d}.{month:02d}.{year}"

        elif field_type == "Mobile":
            first = random.choice([6, 7, 8, 9])
            rest = "".join([str(random.randint(0, 9)) for _ in range(9)])
            return f"{first}{rest}"

        elif field_type == "Address":
            comp = taxonomy["components"]
            parts = [
                random.choice(comp["house_numbers"]),
                random.choice(comp["streets"]),
                random.choice(comp["areas"]),
            ]
            return ", ".join(parts)

        elif field_type == "Amount":
            range_key = random.choice(list(taxonomy["ranges"].keys()))
            low, high = taxonomy["ranges"][range_key]
            amount = random.randint(low, high)
            if random.random() < 0.3:
                return f"{amount:,}"
            return str(amount)

        elif field_type == "Numeric":
            range_key = random.choice(list(taxonomy["ranges"].keys()))
            low, high = taxonomy["ranges"][range_key]
            return str(random.randint(low, high))

        elif field_type == "IFSC":
            code = random.choice(taxonomy["bank_codes"])
            branch = "".join([str(random.randint(0, 9)) for _ in range(6)])
            return f"{code}0{branch}"

        elif field_type in ("Occupation", "Relationship"):
            return random.choice(taxonomy["values"])

        return "UNKNOWN"

    def _generate_pan(self) -> str:
        """Generate valid PAN format."""
        letters1 = "".join(random.choices(string.ascii_uppercase, k=5))
        digits = "".join(random.choices(string.digits, k=4))
        letter2 = random.choice(string.ascii_uppercase)
        return f"{letters1}{digits}{letter2}"

    def _render_text(self, text: str, img_width: int = 800, img_height: int = 100) -> Optional[Image.Image]:
        """Render text as handwriting-style image with form-like complexification."""
        try:
            img = Image.new("L", (img_width, img_height), color=255)
            draw = ImageDraw.Draw(img)

            # --- COMPLEXIFICATION: Add Form Artifacts ---
            # Random horizontal line (simulating the form box baseline)
            if random.random() < 0.8:
                line_y = random.randint(img_height - 30, img_height - 10)
                draw.line([(0, line_y), (img_width, line_y)], fill=random.randint(0, 100), width=random.randint(1, 3))
            
            # Random printed text noise on the left (simulating the field label like 'Name:')
            if random.random() < 0.6:
                try:
                    printed_font = ImageFont.truetype("arial.ttf", random.randint(16, 24))
                except:
                    printed_font = ImageFont.load_default()
                labels = ["Name:", "PAN No:", "Mobile:", "Address", "City", "Date of Birth:"]
                draw.text((5, random.randint(10, img_height-40)), random.choice(labels), font=printed_font, fill=random.randint(0, 50))

            # Pick random handwriting font and size
            font_path = random.choice(self.fonts)
            font_size = random.randint(32, 52)

            if font_path:
                try:
                    font = ImageFont.truetype(str(font_path), font_size)
                except (OSError, IOError):
                    font = ImageFont.load_default()
            else:
                font = ImageFont.load_default()

            # Random position offset (place it after the potential label)
            x_offset = random.randint(100, 200) if random.random() < 0.6 else random.randint(5, 50)
            y_offset = random.randint(5, 25)

            # Random ink color (dark variations)
            ink_color = random.randint(0, 60)

            draw.text((x_offset, y_offset), text, font=font, fill=ink_color)

            # Tight crop
            bbox = img.getbbox()
            if bbox:
                # Add small padding
                pad = 10
                bbox = (max(0, bbox[0] - pad), max(0, bbox[1] - pad),
                       min(img_width, bbox[2] + pad), min(img_height, bbox[3] + pad))
                img = img.crop(bbox)

            return img
        except Exception as e:
            logger.warning(f"Failed to render text '{text}': {e}")
            return None

    def _augment(self, img: Image.Image) -> Image.Image:
        """Apply augmentations to simulate real scan conditions."""
        img_np = np.array(img)

        # Random rotation (slight)
        if random.random() < 0.5:
            import cv2
            angle = random.uniform(-3, 3)
            h, w = img_np.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            img_np = cv2.warpAffine(img_np, M, (w, h), borderValue=255)

        # Gaussian noise
        if random.random() < 0.4:
            noise = np.random.normal(0, random.uniform(5, 20), img_np.shape)
            img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)

        # Gaussian blur
        if random.random() < 0.3:
            import cv2
            ksize = random.choice([3, 5])
            img_np = cv2.GaussianBlur(img_np, (ksize, ksize), 0)

        # Brightness/contrast jitter
        if random.random() < 0.4:
            alpha = random.uniform(0.8, 1.2)  # contrast
            beta = random.uniform(-20, 20)  # brightness
            img_np = np.clip(alpha * img_np + beta, 0, 255).astype(np.uint8)

        # Elastic transform (simulates paper warping)
        if random.random() < 0.2:
            img_np = self._elastic_transform(img_np)

        return Image.fromarray(img_np)

    def _elastic_transform(self, image: np.ndarray, alpha: float = 15,
                           sigma: float = 3) -> np.ndarray:
        """Apply elastic deformation to simulate paper warping."""
        import cv2
        h, w = image.shape[:2]
        dx = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1).astype(np.float32),
                              (0, 0), sigma) * alpha
        dy = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1).astype(np.float32),
                              (0, 0), sigma) * alpha

        x, y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)

        return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT)


class TrOCRFineTuner:
    """Fine-tune TrOCR with LoRA on synthetic + real field data."""

    def __init__(self, base_model: str = "microsoft/trocr-base-handwritten",
                 output_dir: str = None):
        self.base_model = base_model
        self.output_dir = Path(output_dir) if output_dir else Path(__file__).parent.parent / "models" / "trocr-finetuned"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def prepare_dataset(self, image_paths: List[str], labels: List[str],
                        val_split: float = 0.1):
        """Prepare HuggingFace dataset from image paths and labels."""
        from torch.utils.data import Dataset
        from transformers import TrOCRProcessor

        processor = TrOCRProcessor.from_pretrained(self.base_model)

        class TrOCRDataset(Dataset):
            def __init__(self, image_paths, labels, processor, max_length=64):
                self.image_paths = image_paths
                self.labels = labels
                self.processor = processor
                self.max_length = max_length

            def __len__(self):
                return len(self.image_paths)

            def __getitem__(self, idx):
                img = Image.open(self.image_paths[idx]).convert("RGB")
                text = self.labels[idx]

                pixel_values = self.processor(images=img, return_tensors="pt").pixel_values.squeeze()
                labels = self.processor.tokenizer(
                    text, padding="max_length", max_length=self.max_length,
                    truncation=True, return_tensors="pt"
                ).input_ids.squeeze()
                # Replace padding with -100 for loss computation
                labels[labels == self.processor.tokenizer.pad_token_id] = -100

                return {"pixel_values": pixel_values, "labels": labels}

        # Split
        n = len(image_paths)
        indices = list(range(n))
        random.shuffle(indices)
        val_size = int(n * val_split)

        train_paths = [image_paths[i] for i in indices[val_size:]]
        train_labels = [labels[i] for i in indices[val_size:]]
        val_paths = [image_paths[i] for i in indices[:val_size]]
        val_labels = [labels[i] for i in indices[:val_size]]

        train_ds = TrOCRDataset(train_paths, train_labels, processor)
        val_ds = TrOCRDataset(val_paths, val_labels, processor)

        return train_ds, val_ds, processor

    def fine_tune(self, train_ds, val_ds, processor,
                  num_epochs: int = 10, batch_size: int = 8,
                  learning_rate: float = 5e-5):
        """Fine-tune TrOCR with LoRA adapters."""
        from transformers import (
            VisionEncoderDecoderModel, Seq2SeqTrainer,
            Seq2SeqTrainingArguments, default_data_collator,
        )
        from peft import LoraConfig, get_peft_model, TaskType

        logger.info(f"Loading base model: {self.base_model}")
        model = VisionEncoderDecoderModel.from_pretrained(self.base_model)

        # Configure model for generation
        model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
        model.config.pad_token_id = processor.tokenizer.pad_token_id
        model.config.vocab_size = model.config.decoder.vocab_size

        # Apply LoRA
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
            modules_to_save=["lm_head"],  # Also train the head
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=str(self.output_dir),
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=2,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            warmup_ratio=0.1,
            weight_decay=0.01,
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=3,
            predict_with_generate=True,
            generation_max_length=64,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            report_to="none",
            dataloader_num_workers=0,
        )

        # Trainer
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=processor,
        )

        logger.info("Starting TrOCR fine-tuning with LoRA...")
        trainer.train()

        # Save
        model.save_pretrained(str(self.output_dir))
        processor.save_pretrained(str(self.output_dir))
        logger.info(f"Fine-tuned model saved to {self.output_dir}")

        return model, processor


def run_synthetic_generation_and_finetuning(
    num_synthetic_samples: int = 5000,
    num_epochs: int = 10,
    batch_size: int = 8,
):
    """Convenience function to generate data and fine-tune in one go."""
    logger.info("=" * 60)
    logger.info("STEP 1: Generating synthetic handwriting data")
    logger.info("=" * 60)

    generator = SyntheticDataGenerator()
    image_paths, labels = generator.generate_dataset(num_synthetic_samples)

    # --- Load Pseudo Labeled Data ---
    pseudo_meta = Path("data/pseudo_labeled/metadata.json")
    if pseudo_meta.exists():
        logger.info("Found pseudo-labeled real data! Combining with synthetic data...")
        with open(pseudo_meta, "r", encoding="utf-8") as f:
            real_data = json.load(f)
        for item in real_data:
            image_paths.append(item["image_path"])
            labels.append(item["text"])
        logger.info(f"Added {len(real_data)} real crops. Total training size: {len(image_paths)}")
    else:
        logger.warning("No pseudo-labeled data found. Training on purely synthetic data.")

    logger.info("=" * 60)
    logger.info("STEP 2: Preparing dataset for TrOCR")
    logger.info("=" * 60)

    finetuner = TrOCRFineTuner()
    train_ds, val_ds, processor = finetuner.prepare_dataset(image_paths, labels)

    logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    logger.info("=" * 60)
    logger.info("STEP 3: Fine-tuning TrOCR with LoRA")
    logger.info("=" * 60)

    model, processor = finetuner.fine_tune(
        train_ds, val_ds, processor,
        num_epochs=num_epochs,
        batch_size=batch_size,
    )

    logger.info("=" * 60)
    logger.info("DONE! Fine-tuned TrOCR model ready.")
    logger.info("=" * 60)

    return model, processor


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_synthetic_generation_and_finetuning(
        num_synthetic_samples=10000,
        num_epochs=5,
        batch_size=8,
    )
