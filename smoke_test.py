"""
Quick end-to-end smoke test for the LIC pipeline.
Tests: preprocessing -> template matching -> extraction -> validation
Uses a single sample PDF (P02).
"""
import sys
import logging
import json
import traceback

logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger("smoke_test")

def test_step(step_name, fn):
    try:
        result = fn()
        logger.info(f"[PASS] {step_name}")
        return True, result
    except Exception as e:
        logger.error(f"[FAIL] {step_name}: {e}")
        traceback.print_exc()
        return False, None

def main():
    results = {}

    # Step 1: Preprocess P02.pdf (just first 3 pages to save time)
    def step_preprocess():
        from src.preprocessing import ImagePreprocessor
        preprocessor = ImagePreprocessor()
        pdf_path = r"C:\Users\aigcp_gpuadmin\Downloads\LICRFP\LICF\Techathon_Samples\P02.pdf"
        pages = preprocessor.process_file(pdf_path)
        logger.info(f"  -> {len(pages)} pages preprocessed")
        for p in pages[:5]:
            logger.info(f"     Page {p['page_num']}: usable={p['is_usable']}, target={p.get('is_target_page', '?')}, quality={p['quality_score']:.1f}")
        return pages

    ok, pages = test_step("Preprocessing P02.pdf", step_preprocess)
    results["preprocess"] = ok
    if not ok:
        logger.error("Preprocessing failed, cannot continue")
        return results

    # Step 2: Template matching on page 2 (proposer details)
    def step_template_match():
        from src.template_matcher import FormTemplateMatcher
        from PIL import Image
        import numpy as np

        matcher = FormTemplateMatcher()

        # Get page 2 image (proposer_details)
        target_pages = [p for p in pages if p.get("page_num") == 2]
        if not target_pages:
            raise ValueError("Page 2 not found in preprocessed output")

        page2 = target_pages[0]
        color_img = page2.get("preprocessed_color", page2.get("original"))
        pil_img = Image.fromarray(color_img) if isinstance(color_img, np.ndarray) else color_img

        crops = matcher.match_and_crop(pil_img, page_num=2)
        logger.info(f"  -> {len(crops)} fields cropped from page 2")
        for name, crop_img in list(crops.items())[:5]:
            logger.info(f"     {name}: size={crop_img.size}")
        return crops

    ok, crops = test_step("Template matching on page 2", step_template_match)
    results["template_match"] = ok

    # Step 3: Test HallucinationDetector
    def step_hallucination():
        from src.validators import HallucinationDetector
        detector = HallucinationDetector()

        tests = [
            ("avelandian", "name_text", 0.5, True),
            ("Rajesh Kumar", "name_text", 0.8, False),
            ("AAAAAAA", "name_text", 0.3, True),
            ("1234567890", "numeric", 0.9, False),
            ("abc", "numeric", 0.5, True),
            ("15/06/1985", "date", 0.9, False),
        ]

        all_pass = True
        for val, family, conf, expected_halluc in tests:
            is_h, reason = detector.is_hallucination(val, f"test_{family}", family, conf)
            status = "PASS" if is_h == expected_halluc else "FAIL"
            if is_h != expected_halluc:
                all_pass = False
            logger.info(f"  -> [{status}] val='{val}' family={family} halluc={is_h} reason='{reason}'")

        if not all_pass:
            raise ValueError("Some hallucination tests failed")
        return all_pass

    ok, _ = test_step("Hallucination detection", step_hallucination)
    results["hallucination"] = ok

    # Step 4: Test SemanticValidator
    def step_semantic():
        from src.validators import SemanticValidator
        sv = SemanticValidator()

        tests = [
            ("Proposer_State", "Maharashtra", True),
            ("Proposer_State", "xy", False),
            ("Premium_Mode", "yearly", True),
            ("Premium_Mode", "xyz", False),
            ("Proposer_Email", "test@gmail.com", True),
            ("Proposer_Email", "no-at-sign", False),
            ("Proposer_Mobile_Number", "9876543210", True),
            ("Proposer_Mobile_Number", "12345", False),
        ]

        all_pass = True
        for fname, val, expected_ok in tests:
            is_ok, msg = sv.validate(fname, val, {})
            status = "PASS" if is_ok == expected_ok else "FAIL"
            if is_ok != expected_ok:
                all_pass = False
            logger.info(f"  -> [{status}] {fname}='{val}' valid={is_ok} msg='{msg}'")

        if not all_pass:
            raise ValueError("Some semantic validation tests failed")
        return all_pass

    ok, _ = test_step("Semantic validation", step_semantic)
    results["semantic"] = ok

    # Step 5: Test FieldValidator.validate_all with mock data
    def step_validate_all():
        from src.validators import FieldValidator
        fv = FieldValidator()

        mock_fields = {
            "Proposer_PAN": {"value": "ABCDE1234F", "confidence": 0.9, "field_family": "short_id"},
            "Proposer_Mobile_Number": {"value": "9876543210", "confidence": 0.85, "field_family": "numeric"},
            "Proposer_Pincode": {"value": "400001", "confidence": 0.8, "field_family": "numeric"},
            "Proposer_Date_of_Birth": {"value": "15/06/1985", "confidence": 0.75, "field_family": "date"},
            "Proposer_Full_Name": {"value": "", "confidence": 0.0, "field_family": "name_text"},
            "test_gibberish": {"value": "avelandian", "confidence": 0.4, "field_family": "name_text"},
        }

        validated = fv.validate_all(mock_fields)
        for fname, fdata in validated.items():
            logger.info(f"  -> {fname}: status={fdata.get('validation_status')} cat={fdata.get('category')} val='{fdata.get('value', '')[:30]}'")

        # Check hallucination was caught
        gib = validated.get("test_gibberish", {})
        if gib.get("validation_status") != "hallucination":
            raise ValueError(f"Gibberish not caught: {gib.get('validation_status')}")

        return validated

    ok, _ = test_step("Full field validation", step_validate_all)
    results["validate_all"] = ok

    # Summary
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    logger.info(f"\n{'='*60}")
    logger.info(f"SMOKE TEST: {passed}/{total} steps passed")
    for k, v in results.items():
        logger.info(f"  {k}: {'PASS' if v else 'FAIL'}")
    logger.info(f"{'='*60}")

    return results

if __name__ == "__main__":
    main()
