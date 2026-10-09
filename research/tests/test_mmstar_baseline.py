"""Check sampling and scoring without installing ML libraries on the Mac."""

import importlib.util
import io
import json
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("eval_mmstar", REPO / "scripts/eval_mmstar.py")
baseline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(baseline)


class BaselineProtocolTests(unittest.TestCase):
    def test_selection_is_repeatable_balanced_and_has_no_duplicates(self):
        categories = ["perception"] * 250 + ["math"] * 250 + ["reasoning"] * 250
        selected = baseline.select_rows(categories, 4, 590)
        self.assertEqual(selected, baseline.select_rows(categories, 4, 590))
        self.assertEqual(len(selected), len(set(selected)))
        self.assertEqual(Counter(categories[row] for row in selected),
                         {"perception": 4, "math": 4, "reasoning": 4})
        self.assertNotEqual(selected, baseline.select_rows(categories, 4, 591))
        self.assertEqual(baseline.select_rows(categories, 250, 590), list(range(750)))

    def test_insufficient_category_is_not_silently_undersampled(self):
        with self.assertRaisesRegex(ValueError, "Not enough"):
            baseline.select_rows(["math", "perception", "perception"], 2, 590)

    def test_single_answers_are_parsed_and_ambiguous_answers_are_rejected(self):
        for response, expected in [("A", "A"), (" b. ", "B"), ("(C)", "C"),
                                   ("Answer: D", "D"), ("The answer is A.", "A")]:
            with self.subTest(response=response):
                self.assertEqual(baseline.parse_answer(response), expected)
        for response in ["A or B", "A. B.", "A because it is red", "", "Red", "E"]:
            with self.subTest(response=response):
                self.assertIsNone(baseline.parse_answer(response))

    def test_unparsed_responses_stay_in_the_accuracy_denominator(self):
        records = [
            {"correct": True, "prediction": "A", "category": "math", "generation_seconds": 1},
            {"correct": False, "prediction": None, "category": "math", "generation_seconds": 2},
            {"correct": False, "prediction": "B", "category": "perception", "generation_seconds": 3},
        ]
        summary = baseline.summarize(records)
        self.assertEqual(summary["evaluated"], 3)
        self.assertEqual(summary["accuracy"], 1 / 3)
        self.assertEqual(summary["unparsed"], 1)
        self.assertEqual(summary["per_category"]["math"]["accuracy"], 0.5)

    def test_question_retains_options_and_removes_duplicate_image_markers(self):
        question = "<image 1> Choose a color. Options: A: Red, B: Blue."
        prepared = baseline.prepare_question(question)
        self.assertNotIn("<image", prepared)
        self.assertIn("Options: A: Red, B: Blue.", prepared)
        self.assertTrue(prepared.endswith(baseline.ANSWER_INSTRUCTION))

    def test_latency_token_and_memory_summaries_include_every_answer(self):
        records = [{"correct": True, "prediction": "A", "category": "math",
                    "generation_seconds": seconds, "generated_tokens": tokens,
                    "peak_allocated_gib": {"0": memory, "1": 7}}
                   for seconds, tokens, memory in [(1, 2, 6), (2, 4, 8), (9, 6, 7)]]
        summary = baseline.summarize(records)
        self.assertEqual(summary["total_generation_seconds"], 12)
        self.assertEqual(summary["mean_generation_seconds"], 4)
        self.assertEqual(summary["median_generation_seconds"], 2)
        self.assertEqual(summary["p95_generation_seconds"], 9)
        self.assertEqual(summary["total_generated_tokens"], 12)
        self.assertEqual(summary["mean_generated_tokens"], 4)
        self.assertEqual(summary["peak_allocated_gib"], {"0": 8, "1": 7})
        empty = baseline.summarize([])
        self.assertEqual(empty["evaluated"], 0)
        self.assertIsNone(empty["p95_generation_seconds"])

    def test_interrupted_run_saves_partial_summary_and_returns_failure(self):
        def interrupt(args, report):
            report["summary"] = {"evaluated": 7}
            raise KeyboardInterrupt

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "interrupted.json"
            with patch.dict("sys.modules", {"smoke_llava": SimpleNamespace(
                    MODEL_ID="model", MODEL_REVISION="revision")}), \
                    patch.object(baseline.sys, "argv", ["eval_mmstar.py", "--output", str(output)]), \
                    patch.object(baseline.sys, "prefix", str(REPO / ".venv")), \
                    patch.object(baseline.importlib.metadata, "version", return_value="test"), \
                    patch.object(baseline.subprocess, "check_output", return_value="commit"), \
                    patch.object(baseline, "run", side_effect=interrupt), redirect_stdout(io.StringIO()):
                self.assertEqual(baseline.main(), 1)
            result = json.loads(output.read_text())
            self.assertEqual(result["status"], "interrupted")
            self.assertEqual(result["summary"]["evaluated"], 7)
            self.assertIn("total_wall_seconds", result["timing"])


if __name__ == "__main__":
    unittest.main()
