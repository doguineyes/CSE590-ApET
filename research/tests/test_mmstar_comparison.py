"""Check comparison guards and paired scoring without ML packages."""

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("compare_mmstar", REPO / "scripts/compare_mmstar.py")
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


class ComparisonTests(unittest.TestCase):
    def fixture(self):
        report = {"status": "passed", "lock_sha256": "lock", "packages": {"torch": "2.9"},
                  "dataset": {"id": "MMStar", "revision": "sha", "split": "val",
                              "parquet_sha256": "parquet", "sample_ids": [6, 12]},
                  "checkpoint": {"source": "huggingface", "model_id": "llava", "revision": "sha"},
                  "protocol": {"instruction": "letter", "max_new_tokens": 16, "do_sample": False,
                               "precision": "float16", "batch_size": 1, "scoring": "strict"},
                  "summary": {"evaluated": 2, "accuracy": 0.5, "median_generation_seconds": 1}}
        rows = {6: {"index": 6, "question": "question6", "gold": "A", "response": "A",
                    "prediction": "A", "correct": True},
                12: {"index": 12, "question": "question12", "gold": "B", "response": "unsure",
                     "prediction": None, "correct": False}}
        return report, rows

    def test_changed_samples_checkpoint_or_protocol_are_rejected(self):
        base, _ = self.fixture()
        for section, field, value in [("dataset", "sample_ids", [6, 13]),
                                      ("checkpoint", "revision", "other"),
                                      ("protocol", "max_new_tokens", 32)]:
            current = copy.deepcopy(base)
            current[section][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "differs"):
                comparison.validate_compatible(base, current)

    def test_partial_progress_reports_cannot_pass_as_completed_comparisons(self):
        base, rows = self.fixture()
        current = copy.deepcopy(base)
        for status in ["running", "interrupted", "failed"]:
            current["status"] = status
            with self.subTest(status=status), self.assertRaisesRegex(ValueError, "did not complete"):
                comparison.compare_runs(base, current, rows, rows)

    def test_gain_loss_and_unparsed_response_are_preserved(self):
        base, rows = self.fixture()
        current, changed = copy.deepcopy(base), copy.deepcopy(rows)
        changed[6].update(response="B", prediction="B", correct=False)
        changed[12].update(response="B", prediction="B", correct=True)
        result = comparison.compare_runs(base, current, rows, changed)
        self.assertEqual(result["gained_correct_ids"], [12])
        self.assertEqual(result["lost_correct_ids"], [6])
        self.assertEqual(result["baseline_unparsed"], [{"index": 12, "response": "unsure"}])

    def test_identity_rejects_changed_raw_responses_even_if_scores_match(self):
        base, rows = self.fixture()
        current, changed = copy.deepcopy(base), copy.deepcopy(rows)
        current["compression"] = {"enabled": True, "original_tokens": 576, "keep_tokens": 576}
        self.assertEqual(comparison.compare_runs(base, current, rows, changed)["identity_check"], "passed")
        changed[6]["response"] = "Answer: A"
        with self.assertRaisesRegex(ValueError, "Identity adapter changed"):
            comparison.compare_runs(base, current, rows, changed)

    def test_timing_ratio_keeps_output_length_and_device_differences_visible(self):
        base, rows = self.fixture()
        current, changed = copy.deepcopy(base), copy.deepcopy(rows)
        base["summary"]["total_generation_seconds"] = 10
        current["summary"]["total_generation_seconds"] = 5
        base["device_map"], current["device_map"] = {"layer": "0"}, {"layer": "1"}
        rows[6]["generated_tokens"] = changed[6]["generated_tokens"] = 2
        rows[12]["generated_tokens"], changed[12]["generated_tokens"] = 16, 2
        timing = comparison.compare_runs(base, current, rows, changed)["performance"]
        self.assertEqual(timing["generation_time_ratio_baseline_over_current"], 2)
        self.assertEqual(timing["same_generated_token_count_samples"], 1)
        self.assertFalse(timing["same_device_map"])
        self.assertIsNone(timing["same_hardware"])

    def test_restored_predictions_are_found_and_duplicate_ids_are_rejected(self):
        report, rows = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report["predictions_file"] = "/missing-original-session/predictions.jsonl"
            predictions = root / "predictions.jsonl"
            predictions.write_text("\n".join(json.dumps(row) for row in rows.values()) + "\n")
            self.assertEqual(comparison.read_predictions(report, root / "baseline.json"), rows)
            predictions.write_text((json.dumps(rows[6]) + "\n") * 2)
            with self.assertRaisesRegex(ValueError, "Prediction IDs"):
                comparison.read_predictions(report, root / "baseline.json")


if __name__ == "__main__":
    unittest.main()
