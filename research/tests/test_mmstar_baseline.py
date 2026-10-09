"""Check sampling and scoring without installing ML libraries on the Mac."""

import importlib.util
import unittest
from collections import Counter
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
