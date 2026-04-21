import argparse
import importlib.util
import json
import logging
from pathlib import Path


def load_pipeline_class():
    module_path = Path(__file__).resolve().parent / "src" / "pipeline.py"
    spec = importlib.util.spec_from_file_location("lic_full_pipeline_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load pipeline module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LICExtractionPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the current LIC full pipeline on one PDF.")
    parser.add_argument("pdf_path", help="Path to input PDF")
    parser.add_argument(
        "--output-json",
        default="data/current_full_pipeline_result.json",
        help="Path to write output JSON",
    )
    parser.add_argument(
        "--qwen-votes",
        type=int,
        default=3,
        help="Self-consistency vote count for Qwen fallback",
    )
    parser.add_argument(
        "--disable-qwen",
        action="store_true",
        help="Disable Qwen fallback",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    LICExtractionPipeline = load_pipeline_class()
    pipeline = LICExtractionPipeline(
        use_qwen_fallback=not args.disable_qwen,
        qwen_votes=args.qwen_votes,
    )
    result = pipeline.process_single_form(args.pdf_path, form_id=Path(args.pdf_path).stem)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"OUTPUT_JSON={output_path.resolve()}")
    print(f"FORM_STATUS={result.get('form_status', '')}")
    print(f"PROCESSING_TIME_SECONDS={result.get('processing_time_seconds', '')}")
    print(f"PAGES_PROCESSED={result.get('pages_processed', '')}")
    print(f"PAGES_SKIPPED={result.get('pages_skipped', '')}")


if __name__ == "__main__":
    main()
