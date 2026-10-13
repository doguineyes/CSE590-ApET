"""CPU data, scoring, observation and single-model orchestration checks."""

from contextlib import contextmanager
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import eval_original_gqa as gqa
from gqa_scoring import score_answer
from original_diagnostics import capture_merge, compare_diagnostics, trace_returns
from prepare_gqa_pilot import DATASET_ID, DATASET_REVISION, FILES, SCHEMA_VERSION, image_seed, select_groups


def question(qid, image_id, structural, semantic):
    return {"id": qid, "imageId": image_id, "question": f"What is shown in {qid}?", "answer": "blue",
            "isBalanced": True, "types": {"structural": structural, "semantic": semantic, "detailed": qid}}


class ProtocolTests(unittest.TestCase):
    def test_same_image_groups_are_repeatable_and_type_diverse(self):
        rows = [question(f"{image}-{i}", image, structural, semantic) for image in ["a", "b", "c"]
                for i, (structural, semantic) in enumerate([("query", "attr"), ("verify", "rel"),
                                                          ("choose", "obj"), ("logical", "global"), ("query", "rel")])]
        selected, _ = select_groups(rows, {"a", "b", "c"}, 2, 4, 590)
        self.assertEqual(selected, select_groups(rows, {"a", "b", "c"}, 2, 4, 590)[0])
        self.assertEqual(len(selected), 8)
        for image in {row["imageId"] for row in selected}:
            group = [row for row in selected if row["imageId"] == image]
            self.assertTrue({"query", "verify"} <= {row["types"]["structural"] for row in group})
            self.assertGreaterEqual(len({row["types"]["semantic"] for row in group}), 2)
            self.assertEqual(len({image_seed(590, row["imageId"]) for row in group}), 1)
        self.assertNotEqual(image_seed(590, "a"), image_seed(590, "b"))

    def test_missing_images_and_duplicate_questions_cannot_be_silently_selected(self):
        row = question("q", "a", "query", "attr")
        with self.assertRaisesRegex(ValueError, "image missing"):
            select_groups([row], set(), 1, 2, 590)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            select_groups([row, row], {"a"}, 1, 2, 590)

    def test_short_answer_scoring_does_not_extract_answers_from_prose(self):
        self.assertTrue(score_answer(" Blue. ", "blue")["correct"])
        self.assertTrue(score_answer("YES", "yes")["correct"])
        for response in ["The shirt is blue.", "blue or red", "no", ""]:
            self.assertFalse(score_answer(response, "blue")["correct"])
        self.assertIsNone(score_answer("", "blue")["prediction"])

    def test_return_observation_calls_original_once_and_restores_trace_on_failure(self):
        calls, observed = [], []

        def original(value):
            calls.append(value)
            computed = value * 2
            return computed + 1

        with self.assertRaisesRegex(ValueError, "after generation"):
            with trace_returns({original.__code__: lambda local: observed.append(local["computed"])}):
                self.assertEqual(original(4), 9)
                raise ValueError("after generation")
        self.assertEqual(calls, [4])
        self.assertEqual(observed, [8])
        self.assertIsNone(sys.gettrace())

    def test_merge_targets_and_basis_slots_have_distinct_meanings(self):
        class Tensor:
            def __init__(self, values): self.data = values
            def detach(self): return self
            def cpu(self): return self
            def tolist(self): return copy.deepcopy(self.data)

        merged = capture_merge({"errors_indices": Tensor([4, 2, 1]), "index_mask": Tensor([False, True, True, False, True, False]),
                                "merge_count": Tensor([1, 2]), "nearest_token_indices": Tensor([1, 0, 1]),
                                "k": 1, "scaling": 1}, False)
        self.assertEqual(merged["basis_slots"], [1])
        self.assertEqual(merged["dropped_indices"], [0, 3, 5])
        self.assertEqual(merged["assigned_target_indices"], [2, 4, 2])
        tensors = {"errors_indices": Tensor([[4, 2, 5, 1]]), "index_mask": Tensor([[False, True, True, False, True, True]]),
                   "merge_count": Tensor([[1, 1]]), "nearest_token_indices": Tensor([[1, 0]]), "k": 2, "scaling": 1}
        input_merge = capture_merge(tensors, True)
        self.assertEqual(input_merge["basis_slots"], [5, 1])
        self.assertEqual(input_merge["basis_assignment_slots"], [1, 5])  # Native boolean-index assignment order.

    def test_same_image_check_does_not_hide_changed_or_missing_stages(self):
        before = {"input": {"enabled": False}, "decoder": [], "decoder_merge": []}
        after = copy.deepcopy(before)
        after["decoder"].append({"enabled": True})
        with self.assertRaisesRegex(ValueError, "counts differ"):
            compare_diagnostics(before, after)


@unittest.skipUnless(importlib.util.find_spec("PIL"), "Use the existing editor or original Kaggle environment")
class DataAndRunnerTests(unittest.TestCase):
    def setUp(self):
        from PIL import Image

        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        source = self.root / "source"
        (source / "images").mkdir(parents=True)
        path = source / "images/a.png"
        Image.new("RGB", (12, 8), "blue").save(path)
        with Image.open(path) as image:
            pixels = hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
        rows = [question(f"q{i}", "a", structural, semantic) for i, (structural, semantic) in
                enumerate([("query", "attr"), ("verify", "rel"), ("choose", "obj"), ("logical", "global")])]
        self.manifest = {"schema": SCHEMA_VERSION, "source": {"dataset": DATASET_ID, "revision": DATASET_REVISION, "files": FILES},
                         "selection": {"seed": 590, "image_count": 1, "questions_per_image": 4,
                                       "image_ids": ["a"], "question_ids": [row["id"] for row in rows]},
                         "images": {"a": {"file": "images/a.png", "image_size": [12, 8], "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                          "rgb_pixels_sha256": pixels, "compression_seed": image_seed(590, "a")}}, "questions": rows}
        self.path = source / "manifest.json"
        self.path.write_text(json.dumps(self.manifest))
        self.args = SimpleNamespace(data_manifest=self.path, output=self.root / "result.json", configurations=list(gqa.CONFIGURATIONS),
                                    max_new_tokens=16, model=gqa.MODEL_ID, vision_model=gqa.VISION_ID,
                                    cache_dir=self.root / "models", download_budget_gib=0)

    def test_manifest_validates_pixels_and_image_based_seed_before_model_loading(self):
        self.assertEqual(gqa.load_manifest(self.path)["selection"]["question_ids"], ["q0", "q1", "q2", "q3"])
        self.manifest["images"]["a"]["compression_seed"] += 1
        self.path.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, "depend on image"):
            gqa.load_manifest(self.path)

    def test_changed_image_bytes_are_rejected(self):
        (self.path.parent / "images/a.png").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Image bytes changed"):
            gqa.load_manifest(self.path)

    def execute(self, fail_after=None):
        objects = (object(), object(), object())
        calls = []

        def native(tokenizer, model, processor, image, text, name, **kwargs):
            self.assertIs(model, objects[1])
            calls.append((text, name, kwargs["seed"]))
            if fail_after and len(calls) == fail_after:
                raise RuntimeError("interrupted generation")
            return {"configuration": name, "response": "blue", "generated_tokens": 2, "generation_seconds": 0.1,
                    "trace": {"configuration": name}, "seed": kwargs["seed"]}

        def observed(tokenizer, model, processor, image, sample, name, current=None, max_new_tokens=16):
            generated = native(tokenizer, model, processor, image, sample["question"], name,
                               seed=sample["compression_seed"])
            return {**generated, "diagnostics": {"input": {"enabled": False}, "decoder": [], "decoder_merge": []}}

        report = {}
        with patch.object(gqa, "checkpoint_plan", return_value={"missing_bytes": 0}), \
                patch.object(gqa, "download", return_value="cached"), \
                patch.object(gqa, "load_native_model", return_value=objects) as loader, \
                patch.object(gqa, "generate_native_answer", side_effect=native), \
                patch.object(gqa, "generate_observed", side_effect=observed):
            if fail_after:
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    gqa.run(self.args, report)
            else:
                gqa.run(self.args, report)
            loader.assert_called_once()
        rows = [json.loads(line) for line in self.args.output.with_suffix(".jsonl").read_text().splitlines()]
        return report, rows, calls

    def test_four_conditions_and_questions_share_model_and_seed(self):
        report, rows, calls = self.execute()
        self.assertEqual(len(rows), 16)
        self.assertEqual(len({call[2] for call in calls}), 1)
        self.assertEqual(report["status"], "passed")
        self.assertTrue(all(value["matched"] for value in report["observer_parity"].values()))
        self.assertEqual(report["comparisons"]["combined"]["apr"], 1)
        self.assertEqual(len(report["image_groups"]["a"]["same_image_checks"]["combined"]["other_questions"]), 3)

    def test_partial_predictions_survive_later_generation_failure(self):
        report, rows, _ = self.execute(fail_after=14)  # 4 warmups, 8 parity generations, then one prediction.
        self.assertEqual(len(rows), 1)
        self.assertEqual(report["progress"]["completed_predictions"], 1)


if __name__ == "__main__":
    unittest.main()
