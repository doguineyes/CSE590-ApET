"""Scoring ambiguity and evidence preservation checks; standard library only."""

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from check_original_env import REPO, verify_native_reference
from mmstar_scoring import score_answer
from rescore_original_mmstar import rescore, validate_saved

QUESTION = "Which color? Options: A: red, B: blue, C: green, D: black"


class ScoringTests(unittest.TestCase):
    def test_matching_option_text_is_correct_but_not_format_compliant(self):
        result = score_answer("A: red", QUESTION, "A")
        self.assertEqual(result["prediction"], "A")
        self.assertTrue(result["correct"])
        self.assertFalse(result["strict_correct"])
        self.assertFalse(result["format_compliant"])
        self.assertEqual(result["parse_method"], "matched_option_text")
        self.assertTrue(score_answer("A", QUESTION, "A")["format_compliant"])

    def test_choice_parsing_never_uses_the_ground_truth(self):
        for gold in "ABCD":
            result = score_answer("B: blue", QUESTION, gold)
            self.assertEqual(result["prediction"], "B")
            self.assertEqual(result["correct"], gold == "B")

    def test_ambiguity_conflicting_text_and_free_prose_are_not_guessed(self):
        for response in ["A or B", "A: blue", "A: red, B: blue", "A because it is red",
                         "red", "", "E", "A: red or maybe blue"]:
            with self.subTest(response=response):
                self.assertIsNone(score_answer(response, QUESTION, "A")["prediction"])
        self.assertIsNone(score_answer("A: red", "No labeled options", "A")["prediction"])

    def test_labels_inside_option_text_are_not_treated_as_conflicting_answers(self):
        question = "Which diagram? Options: A: B, B: C, C: A, D: D"
        self.assertEqual(score_answer("B: C", question, "B")["prediction"], "B")

    def test_whitespace_normalization_and_parenthesized_options(self):
        result = score_answer("Answer: A: A RED\n coat.",
                              "Which?\n(A) A red coat\n(B) A blue coat\n(C) A green coat\n(D) A black coat", "A")
        self.assertTrue(result["correct"])
        self.assertIsNone(score_answer("A: red", QUESTION + ", A: another", "A")["prediction"])


class RescoringTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.raw = b"image bytes are copied without reencoding"
        manifest = json.loads((REPO / "research/provenance/original_llava_reference.json").read_text())
        self.report = {
            "status": "passed", "commit": "generation-commit", "source_sha256": manifest["native_source_sha256"],
            "protocol": {"configurations": {name: {} for name in ["baseline", "input-only"]}, "scoring": "old strict"},
            "dataset": {"sample_ids": [6], "image_directory": "images"},
            "summaries": {"baseline": {"correct": 1}, "input-only": {"correct": 0}}, "comparisons": {},
        }
        self.rows = []
        for name, answer in [("baseline", "A"), ("input-only", "A: red")]:
            self.rows.append({"configuration": name, "index": 6, "gold": "A", "original_question": QUESTION,
                              "question": QUESTION, "prompt": "same native prompt", "seed": 596,
                              "image": "images/6.jpg", "image_sha256": hashlib.sha256(self.raw).hexdigest(),
                              "category": "coarse perception", "response": answer,
                              "prediction": "A" if name == "baseline" else None,
                              "correct": name == "baseline", "generation_seconds": 0.5, "generated_tokens": 2})
        self.source = self.root / "source.zip"
        with zipfile.ZipFile(self.source, "w") as archive:
            archive.writestr("run.json", json.dumps(self.report))
            archive.writestr("run.jsonl", "\n".join(json.dumps(row) for row in self.rows))
            archive.writestr("images/6.jpg", self.raw)

    def test_rescore_preserves_generation_and_source_zip_and_keeps_old_metrics(self):
        digest = hashlib.sha256(self.source.read_bytes()).hexdigest()
        output = self.root / "new.json"
        result = rescore(self.source, output)
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), digest)
        self.assertEqual(result["commit"], "generation-commit")
        self.assertFalse(result["rescoring"]["gpu_inference_rerun"])
        self.assertEqual(result["summaries"]["input-only"]["correct"], 1)
        self.assertEqual(result["summaries"]["input-only"]["strict_correct"], 0)
        self.assertEqual(result["comparisons"]["input-only"]["apr"], 1)
        self.assertEqual(result["strict_comparisons"]["input-only"]["apr"], 0)
        rows = [json.loads(line) for line in output.with_suffix(".jsonl").read_text().splitlines()]
        for old, new in zip(self.rows, rows):
            for key in ["response", "prompt", "seed", "generation_seconds", "generated_tokens", "image_sha256"]:
                self.assertEqual(old[key], new[key])
            self.assertEqual((self.root / new["image"]).read_bytes(), self.raw)
        with self.assertRaisesRegex(ValueError, "Output exists"):
            rescore(self.source, output)

    def test_duplicate_or_mismatched_pairs_cannot_be_rescored_as_complete(self):
        with self.assertRaisesRegex(ValueError, "IDs/configurations"):
            validate_saved(copy.deepcopy(self.report), self.rows + [self.rows[0]])
        changed = copy.deepcopy(self.rows)
        changed[1]["prompt"] = "different"
        with self.assertRaisesRegex(ValueError, "inputs differ"):
            validate_saved(copy.deepcopy(self.report), changed)

    def test_native_reference_detects_compression_source_changes(self):
        report = copy.deepcopy(self.report)
        report["source_sha256"]["llava/model/modeling_llama_x.py"] = "changed"
        with self.assertRaisesRegex(RuntimeError, "Native sources differ"):
            verify_native_reference(report)


if __name__ == "__main__":
    unittest.main()
