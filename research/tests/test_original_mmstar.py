"""Data integrity, paired scoring, and single-load orchestration (no model downloads)."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import eval_original_mmstar as pilot


def prediction(index, correct, category="reasoning"):
    return {"index": index, "gold": "A", "category": category, "correct": correct}


class PairedScoringTests(unittest.TestCase):
    def test_equal_accuracy_can_hide_lost_baseline_answers(self):
        before = [prediction(1, True), prediction(2, True), prediction(3, False)]
        after = [prediction(1, True), prediction(2, False), prediction(3, True)]
        result = pilot.paired_summary(before, after)
        self.assertEqual(result["apr"], 0.5)
        self.assertEqual(result["lost_correct_ids"], [2])
        self.assertEqual(result["gained_correct_ids"], [3])
        self.assertEqual(result["per_category"]["reasoning"]["apr"], 0.5)

    def test_partial_pairs_and_zero_denominator(self):
        result = pilot.paired_summary([prediction(1, False), prediction(2, True)],
                                      [prediction(1, True), prediction(3, False)])
        self.assertEqual(result["paired_samples"], 1)
        self.assertEqual(result["baseline_correct_in_pairs"], 0)
        self.assertIsNone(result["apr"])
        self.assertEqual(result["gained_correct_ids"], [1])

    def test_duplicate_ids_and_mismatched_labels_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            pilot.paired_summary([prediction(1, True)] * 2, [])
        with self.assertRaisesRegex(ValueError, "differ"):
            pilot.paired_summary([prediction(1, True)], [prediction(1, True, "other")])


HAS_DATA_LIBS = all(importlib.util.find_spec(name) for name in ("PIL", "pyarrow", "huggingface_hub"))


@unittest.skipUnless(HAS_DATA_LIBS, "Use the existing editor or Kaggle environment for data-only checks")
class DataAndWorkflowTests(unittest.TestCase):
    def setUp(self):
        from PIL import Image

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        buffer = io.BytesIO()
        Image.new("RGB", (12, 8), "red").save(buffer, format="PNG")
        self.raw = buffer.getvalue()
        self.args = SimpleNamespace(output=self.root / "pilot.json", dataset_cache=self.root / "cache",
                                    per_category=1, seed=590, max_new_tokens=16,
                                    configurations=list(pilot.CONFIGURATIONS), model=pilot.MODEL_ID,
                                    vision_model=pilot.VISION_ID, cache_dir=self.root / "models")

    def prepare(self, image_bytes=None, gold="A"):
        import pyarrow as pa
        import pyarrow.parquet as pq

        parquet = self.root / "mmstar.parquet"
        pq.write_table(pa.Table.from_pylist([
            {"index": 11, "image": {"bytes": self.raw if image_bytes is None else image_bytes, "path": None},
             "question": "<image 1> Which color? A: red B: blue C: green D: black",
             "answer": gold, "category": "perception", "l2_category": "color"},
        ]), parquet)
        digest = hashlib.sha256(parquet.read_bytes()).hexdigest()
        self.report = {}
        with patch("huggingface_hub.hf_hub_download", return_value=str(parquet)), \
                patch.object(pilot, "PARQUET_SHA256", digest):
            return pilot.prepare_samples(self.args, self.report)

    def test_embedded_image_bytes_and_ids_preserved_without_reencoding(self):
        samples = self.prepare()
        sample = samples[0]
        self.assertEqual(self.report["dataset"]["sample_ids"], [11])
        self.assertEqual((self.root / sample["image"]).read_bytes(), self.raw)
        self.assertEqual(sample["image_size"], [12, 8])
        self.assertNotIn("<image", sample["question"])
        self.assertEqual(sample["image_sha256"], hashlib.sha256(self.raw).hexdigest())

    def test_corrupt_image_fails_before_model_loading(self):
        from PIL import UnidentifiedImageError

        with self.assertRaises(UnidentifiedImageError):
            self.prepare(image_bytes=b"broken")

    def test_invalid_label_cannot_be_silently_scored(self):
        with self.assertRaisesRegex(ValueError, "Invalid answer"):
            self.prepare(gold="E")

    def execute_mock_model(self, fail_at=None):
        samples = self.prepare()
        calls = []
        objects = (object(), object(), object())

        def generate(tokenizer, model, processor, image, question, name, **options):
            self.assertIs(model, objects[1])
            self.assertNotIn("min_new_tokens", options)
            self.assertFalse(options["require_nonempty"])
            calls.append((name, options["max_new_tokens"]))
            if len(calls) == fail_at:
                raise RuntimeError("simulated interruption")
            # An empty decoded answer must remain in the scored records.
            return {"configuration": name, "response": "" if name == "combined" else "A",
                    "generation_seconds": 0.1, "generated_tokens": 1}

        with patch.object(pilot, "prepare_samples", return_value=samples), \
                patch.object(pilot, "checkpoint_plan", return_value={"missing_bytes": 0}), \
                patch.object(pilot, "download", return_value="cached"), \
                patch.object(pilot, "load_native_model", return_value=objects) as loader, \
                patch.object(pilot, "generate_native_answer", side_effect=generate):
            if fail_at:
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    pilot.run(self.args, self.report)
            else:
                pilot.run(self.args, self.report)
            loader.assert_called_once()
        return calls, [json.loads(line) for line in self.args.output.with_suffix(".jsonl").read_text().splitlines()]

    def test_all_stages_share_model_and_empty_response_counts_in_denominator(self):
        calls, rows = self.execute_mock_model()
        self.assertEqual(len(calls), 9)  # Four warmups, four predictions, baseline repeat.
        self.assertEqual(len(rows), 4)
        self.assertEqual(self.report["status"], "passed")
        self.assertEqual(self.report["summaries"]["combined"]["unparsed"], 1)
        self.assertEqual(self.report["comparisons"]["combined"]["apr"], 0)
        self.assertTrue(self.report["baseline_state_check"]["matched"])

    def test_partial_predictions_survive_failure_in_next_configuration(self):
        _, rows = self.execute_mock_model(fail_at=6)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["configuration"], "baseline")
        self.assertEqual(self.report["progress"]["completed_predictions"], 1)
        self.assertEqual(self.report["comparisons"]["combined"]["paired_samples"], 0)


if __name__ == "__main__":
    unittest.main()
