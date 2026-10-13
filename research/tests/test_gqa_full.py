"""CPU checks for full-data coverage, paired scoring and safe interrupted-run recovery."""

import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import eval_original_gqa_full as runner
from gqa_scoring import score_answer
from prepare_gqa_full import SCHEMA_VERSION, full_questions
from prepare_gqa_pilot import DATASET_ID, DATASET_REVISION, FILES, image_seed
from smoke_original_apet import CONFIGURATIONS


def question(qid, image_id, text="What color is it?", kind="attr"):
    return {"id": qid, "imageId": image_id, "question": text, "answer": "red", "isBalanced": True,
            "types": {"semantic": kind, "structural": "query", "detailed": "color"}}


def fixture(root):
    from PIL import Image
    data = root / "prepared"
    (data / "images").mkdir(parents=True)
    images = {}
    for image_id in ["a", "b"]:
        file = data / "images" / (image_id + ".png")
        Image.new("RGB", (8, 8), "red").save(file)
        with Image.open(file) as image:
            pixels = hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
        images[image_id] = {"file": "images/" + file.name, "image_size": [8, 8],
            "sha256": runner.digest(file), "rgb_pixels_sha256": pixels, "compression_seed": image_seed(590, image_id)}
    rows = [question("q1", "a"), question("q2", "a"), question("q3", "b", kind="obj")]
    manifest = {"schema": SCHEMA_VERSION, "source": {"dataset": DATASET_ID, "revision": DATASET_REVISION,
        "split": "testdev", "tier": "testdev_balanced", "files": FILES, "total_questions": 3, "total_images": 2},
        "selection": {"seed": 590, "question_count": 3, "image_count": 2,
            "question_ids": [row["id"] for row in rows], "image_ids": ["a", "b"], "questions_by_image": {"a": 2, "b": 1}},
        "images": images, "questions": rows}
    path = data / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def args_for(root):
    return SimpleNamespace(data_manifest=fixture(root), output=root / "run" / "result.json", resume=False,
        max_new_tokens=16, configurations=list(CONFIGURATIONS), model=runner.MODEL_ID,
        vision_model=runner.VISION_ID, cache_dir=root / "cache", download_budget_gib=0, report_every=2)


def generation(sample, name):
    setting = CONFIGURATIONS[name]
    trace = {"input": {"before": 576, "after": setting["input_tokens"]}}
    if setting["layers"]:
        # See validate_trace: a decoder shrink must match the sequence shrink.
        trace["decoder"] = [{"layer": 16, "visual_before": setting["input_tokens"], "visual_after": 96,
            "sequence_before": setting["input_tokens"] + 20, "sequence_after": 116}]
    return {"configuration": name, "settings": {**setting, "basis_tokens": 10},
        "seed": sample["compression_seed"], "status": "passed", "response": "Red.",
        "question": sample["question"], "prompt": "native prompt", "trace": trace,
        "generation_seconds": 0.5, "generated_tokens": 2, "peak_allocated_gib": {"0": 7.0}}


class FullGQATests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.count_patches = [patch.object(runner, "QUESTION_COUNT", 3), patch.object(runner, "IMAGE_COUNT", 2)]
        for item in self.count_patches:
            item.start()
        self.args = args_for(self.root)
        self.args.output.parent.mkdir()
        self.current = {"lock_sha256": runner.digest(REPO / "environments/original/uv.lock"),
                        "source_sha256": {}, "implementation": {"native_matches_reference": True}}
        self.args.session_provenance = self.current

    def tearDown(self):
        for item in reversed(self.count_patches):
            item.stop()
        self.temp.cleanup()

    def test_full_selection_retains_repeated_text_and_variable_group_sizes(self):
        rows = [question("q3", "b"), question("q2", "a"), question("q1", "a")]
        self.assertEqual([row["id"] for row in full_questions(rows, {"a", "b"})], ["q1", "q2", "q3"])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            full_questions(rows + [rows[0]], {"a", "b"})
        with self.assertRaisesRegex(ValueError, "cover exactly"):
            full_questions(rows, {"a", "b", "c"})

    def test_manifest_rejects_missing_questions_changed_images_and_wrong_seed(self):
        good = json.loads(self.args.data_manifest.read_text())
        self.assertEqual(len(runner.load_manifest(self.args.data_manifest)["questions"]), 3)
        for mutate in [lambda m: m["questions"].pop(),
                       lambda m: m["images"]["a"].update(compression_seed=1),
                       lambda m: m["images"]["a"].update(sha256="bad")]:
            changed = copy.deepcopy(good)
            mutate(changed)
            self.args.data_manifest.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                runner.load_manifest(self.args.data_manifest)
        self.args.data_manifest.write_text(json.dumps(good))

    def saved_rows(self, samples):
        by_id = {row["index"]: row for row in samples}
        return [{**by_id[qid], **generation(by_id[qid], name), **score_answer("Red.", "red"), "session_id": 1}
                for qid, name in runner.expected_order(samples, self.args.configurations)]

    def test_resume_guards_settings_and_preserves_original_results(self):
        report, samples, _ = runner.open_run(self.args, self.current)
        path = self.args.output.with_suffix(".jsonl")
        path.write_text(json.dumps(self.saved_rows(samples)[0]) + "\n")
        before = self.args.output.read_bytes(), path.read_bytes()
        self.args.resume = True
        self.args.max_new_tokens = 8
        with self.assertRaisesRegex(ValueError, "Resume refused"):
            runner.open_run(self.args, self.current)
        self.assertEqual(before, (self.args.output.read_bytes(), path.read_bytes()))
        self.args.max_new_tokens = 16
        _, _, saved = runner.open_run(self.args, self.current)
        self.assertEqual(len(saved), 1)

    def test_torn_final_line_preserved_but_complete_corruption_and_duplicates_rejected(self):
        _, samples, _ = runner.open_run(self.args, self.current)
        rows = self.saved_rows(samples)
        path = self.args.output.with_suffix(".jsonl")
        first = (json.dumps(rows[0]) + "\n").encode()
        path.write_bytes(first + b'{"index":')
        self.assertEqual(len(runner.read_completed(path, samples, self.args)), 1)
        self.assertEqual(path.read_bytes(), first)
        backups = list(path.parent.glob("*.partial-*"))
        self.assertEqual(backups[0].read_bytes(), b'{"index":')
        for corrupt in [first + b"invalid\n", first + first]:
            path.write_bytes(corrupt)
            with self.assertRaises(ValueError):
                runner.read_completed(path, samples, self.args)
            self.assertEqual(path.read_bytes(), corrupt)

    def test_valid_unterminated_line_gets_one_newline_and_scoring_mismatch_is_rejected(self):
        _, samples, _ = runner.open_run(self.args, self.current)
        row = self.saved_rows(samples)[0]
        path = self.args.output.with_suffix(".jsonl")
        path.write_text(json.dumps(row))
        runner.read_completed(path, samples, self.args)
        correct = path.read_bytes()
        runner.read_completed(path, samples, self.args)
        self.assertEqual(path.read_bytes(), correct)
        self.assertTrue(correct.endswith(b"\n"))
        self.assertFalse(correct.endswith(b"\n\n"))
        row["correct"] = False
        path.write_text(json.dumps(row) + "\n")
        with self.assertRaisesRegex(ValueError, "scoring differs"):
            runner.read_completed(path, samples, self.args)

    def test_equal_accuracy_still_records_lost_gained_and_type_apr(self):
        report, samples, _ = runner.open_run(self.args, self.current)
        rows = self.saved_rows(samples)
        for row in rows:
            row["correct"] = (row["index"] == "q1") if row["configuration"] == "baseline" else row["index"] == "q2"
        runner.update_summary(report, [runner.compact(row) for row in rows], self.args.configurations)
        comparison = report["comparisons"]["combined"]
        self.assertEqual(comparison["apr"], 0)
        self.assertEqual(comparison["lost_correct_ids"], ["q1"])
        self.assertEqual(comparison["gained_correct_ids"], ["q2"])
        self.assertEqual(comparison["by_question_type"]["semantic"]["attr"]["apr"], 0)
        self.assertIsNone(comparison["by_question_type"]["semantic"]["obj"]["apr"])

    def test_mid_question_resume_runs_only_missing_predictions_and_loads_once_per_invocation(self):
        report, samples, predictions = runner.open_run(self.args, self.current)
        # q1 and q2 have identical text: generation must still preserve both IDs in output.
        scored_calls, seeds = [], []
        fail = [True]
        def fake_generate(tokenizer, model, processor, image, text, name, seed, max_new_tokens, **kwargs):
            if max_new_tokens == 2:
                return {}
            if fail[0] and len(scored_calls) == 3:
                raise RuntimeError("simulated interruption mid-question")
            sample = next(row for row in samples if row["question"] == text and row["compression_seed"] == seed)
            scored_calls.append(name)
            seeds.append(seed)
            return generation(sample, name)
        with patch.object(runner, "checkpoint_plan", return_value={"missing_bytes": 0}), \
             patch.object(runner, "download", return_value="cached"), \
             patch.object(runner, "load_native_model", return_value=(None, None, None)) as loader, \
             patch.object(runner, "generate_native_answer", side_effect=fake_generate):
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                runner.run(self.args, report, samples, predictions)
            self.assertEqual(len(predictions), 3)
            runner.atomic_report(self.args.output, report)
            self.args.resume = True
            fail[0] = False
            report, samples, predictions = runner.open_run(self.args, self.current)
            runner.run(self.args, report, samples, predictions)
            runner.atomic_report(self.args.output, report)
            self.assertEqual(loader.call_count, 2)
            self.assertEqual(report["progress"]["completed_predictions"], 12)
            self.assertEqual(len(scored_calls), 12)
            saved = [json.loads(line) for line in self.args.output.with_suffix(".jsonl").read_text().splitlines()]
            self.assertEqual(Counter(row["index"] for row in saved), {"q1": 4, "q2": 4, "q3": 4})
            self.assertEqual(seeds[:8], [image_seed(590, "a")] * 8)
            report, samples, predictions = runner.open_run(self.args, self.current)
            runner.run(self.args, report, samples, predictions)
            self.assertEqual(loader.call_count, 2)  # Completed runs need no model download/load.


if __name__ == "__main__":
    unittest.main()
