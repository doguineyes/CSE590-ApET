"""CPU checks for scientific controls, fixed case identity and partial evidence.

Generation is mocked here. Actual native/GPU repeatability is measured by the
Kaggle notebook, not established by these orchestration tests.
"""

from contextlib import contextmanager, redirect_stdout
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import repeat_original_gqa as repeat

try:
    from PIL import Image
except ImportError:
    Image = None


class ProtocolTests(unittest.TestCase):
    def test_each_case_has_identical_repeats_logging_sandwiches_and_return_controls(self):
        plan = repeat.trial_plan()
        self.assertEqual(len(plan), 72)
        self.assertEqual([row["trial_id"] for row in plan], list(range(1, 73)))
        for qid in repeat.CASE_SPECS:
            case = [row for row in plan if row["question_id"] == qid]
            self.assertEqual(len(case), 36)
            for name in repeat.CONFIGURATIONS:
                same = [row for row in case if row["configuration"] == name and row["phase"] == "identical-repeats"]
                self.assertEqual(len(same), 3)
                self.assertEqual({row["logging"] for row in same}, {"counts-only"})
                sandwich = [row for row in case if row["configuration"] == name and row["phase"] == "logging-sandwich"]
                self.assertEqual([row["logging"] for row in sandwich], ["counts-only", "native-values", "counts-only"])
            switches = [row for row in case if row["phase"] == "configuration-switch"]
            for start in range(0, len(switches), 3):
                a, b, again = switches[start:start + 3]
                self.assertEqual((a["configuration"], a["logging"]), (again["configuration"], again["logging"]))
                self.assertNotEqual(a["configuration"], b["configuration"])

    def test_reserved_manifest_is_rejected_before_reading_images(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest = Path(folder) / "manifest.json"
            manifest.write_text(json.dumps({"source": {"dataset": repeat.DATASET_ID,
                "revision": repeat.DATASET_REVISION, "files": repeat.FILES}, "study": {"role": "evaluation"}}))
            with self.assertRaisesRegex(ValueError, "not reserved evaluation"):
                repeat.case_inputs(SimpleNamespace(data_manifest=manifest))


@unittest.skipIf(Image is None, "Use the Mac editor environment for Pillow")
class CaseAndExecutionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        specs = copy.deepcopy(repeat.CASE_SPECS)
        self.questions, self.raw_images = {}, {}
        for qid, spec in specs.items():
            image = Image.new("RGB", (8, 6), (50, 100, 150))
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            raw = buffer.getvalue()
            spec.update(image_sha256=hashlib.sha256(raw).hexdigest(),
                        rgb_pixels_sha256=hashlib.sha256(image.tobytes()).hexdigest(), size=[8, 6])
            self.raw_images[spec["image_id"]] = raw
            self.questions[qid] = {"id": qid, "imageId": spec["image_id"], "question": spec["question"],
                                   "answer": spec["gold"], "isBalanced": True, "types": {}}
        self.specs = specs
        self.real_case_inputs = repeat.case_inputs
        patcher = patch.object(repeat, "CASE_SPECS", specs)
        patcher.start()
        self.addCleanup(patcher.stop)
        inputs = patch.object(repeat, "case_inputs", side_effect=lambda args:
                              (self.questions, self.raw_images, {"synthetic_test_fixture": True}))
        inputs.start()
        self.addCleanup(inputs.stop)
        self.args = SimpleNamespace(output=self.root / "repeat.json", model=repeat.MODEL_ID,
                                    vision_model=repeat.VISION_ID, cache_dir=self.root / "cache",
                                    download_budget_gib=0, repeats=3)

    def test_preparation_preserves_question_instruction_seed_and_image_bytes(self):
        report = {}
        samples = repeat.prepare_cases(self.args, report)
        for sample in samples:
            spec = self.specs[sample["question_id"]]
            self.assertEqual(sample["question"], spec["question"] + "\n" + repeat.INSTRUCTION)
            self.assertEqual(sample["seed"], spec["seed"])
            self.assertEqual((self.root / sample["image"]).read_bytes(), self.raw_images[spec["image_id"]])
        self.assertTrue((self.root / report["dataset"]["manifest_file"]).is_file())

    def test_parquet_path_selects_only_the_two_inspected_cases(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("Use the Mac editor environment for PyArrow")
        question_path, image_path = self.root / "questions.parquet", self.root / "images.parquet"
        questions = list(self.questions.values())
        questions = [{**row, "types": {"semantic": "obj"}} for row in questions]
        questions.append({**questions[0], "id": "uninspected-question"})
        pq.write_table(pa.Table.from_pylist(questions), question_path)
        pq.write_table(pa.Table.from_pylist([
            {"id": image_id, "image": {"bytes": raw}} for image_id, raw in self.raw_images.items()
        ]), image_path)
        paths = dict(zip(repeat.FILES, [question_path, image_path]))
        args = SimpleNamespace(data_manifest=None, dataset_cache=self.root / "dataset-cache",
                               source_dir=None, download_budget_mib=100)
        with patch.object(repeat, "fetch_sources", return_value=paths):
            selected, images, origin = self.real_case_inputs(args)
        self.assertEqual(set(selected), set(self.specs))
        self.assertEqual(images, self.raw_images)
        self.assertEqual(origin["pinned_source_files"], repeat.FILES)

    def test_changed_gold_and_changed_image_are_rejected(self):
        self.questions["2058428"]["answer"] = "no"
        with self.assertRaisesRegex(ValueError, "Question/label/image identity"):
            repeat.prepare_cases(self.args, {})
        self.questions["2058428"]["answer"] = "yes"
        self.args.output = self.root / "other.json"
        self.raw_images["n497658"] += b"changed"
        with self.assertRaisesRegex(ValueError, "Historical image bytes differ"):
            repeat.prepare_cases(self.args, {})

    def execute(self, vary_answer=False, fail_after=None):
        report = {"status": "running", "plan": repeat.trial_plan()}
        samples = repeat.prepare_cases(self.args, report)
        active, calls = [], []

        @contextmanager
        def observe():
            data = {"input": {"enabled": False}, "decoder": [], "decoder_merge": []}
            active.append(data)
            try:
                yield data
            finally:
                active.pop()

        def native(tokenizer, model, processor, image, text, name, **options):
            calls.append({"text": text, "name": name, "seed": options["seed"], "max_tokens": options["max_new_tokens"]})
            record = options.get("record", {})
            record["status"] = "running"
            # First four calls are warmups; then fail after N completed trials.
            if fail_after is not None and len(calls) > 4 + fail_after:
                raise RuntimeError("synthetic generation failure")
            qid = next(qid for qid, spec in self.specs.items() if text.startswith(spec["question"]))
            answer = self.specs[qid]["development_answers"][name]
            if vary_answer and qid == "2058428" and name == "combined" and active:
                answer = "No"
            record.update(configuration=name, settings={**repeat.CONFIGURATIONS[name], "basis_tokens": 10},
                          seed=options["seed"], response=answer, prompt="<image>\n" + text,
                          generated_tokens=2, generation_seconds=0.01, trace={"configuration": name}, status="passed")
            return record

        cuda = SimpleNamespace(device_count=lambda: 2, get_device_name=lambda i: "Tesla T4",
                               get_device_capability=lambda i: (7, 5))
        torch = SimpleNamespace(cuda=cuda, version=SimpleNamespace(cuda="12.1"),
            are_deterministic_algorithms_enabled=lambda: False,
            backends=SimpleNamespace(cudnn=SimpleNamespace(deterministic=False, benchmark=False, allow_tf32=True),
                                     cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False))))
        with patch.dict(sys.modules, {"torch": torch}), \
                patch.object(repeat, "checkpoint_plan", return_value={"missing_bytes": 0}), \
                patch.object(repeat, "download", return_value="fixture"), \
                patch.object(repeat, "load_native_model", return_value=(object(), object(), object())) as loader, \
                patch.object(repeat, "generate_native_answer", side_effect=native), \
                patch.object(repeat, "observe_native_values", side_effect=observe), \
                patch.object(repeat, "validate_diagnostics", side_effect=lambda settings, data: data.update(fingerprint="fixture")), \
                redirect_stdout(io.StringIO()):
            if fail_after is None:
                repeat.run(self.args, report, samples)
            else:
                with self.assertRaisesRegex(RuntimeError, "synthetic generation failure"):
                    repeat.run(self.args, report, samples)
            self.assertEqual(loader.call_count, 1)
        return report, calls

    def test_one_model_load_and_identical_historical_seeds_and_prompts(self):
        report, calls = self.execute()
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["progress"], {"completed_trials": 72, "expected_trials": 72})
        self.assertEqual(len(calls), 76)
        for call in calls:
            spec = next(s for s in self.specs.values() if call["text"].startswith(s["question"]))
            self.assertEqual(call["seed"], spec["seed"])
            self.assertEqual(call["text"], spec["question"] + "\n" + repeat.INSTRUCTION)
        self.assertFalse(report["findings"]["answer_variation_detected"])
        self.assertFalse(report["findings"]["control_difference_detected"])
        self.assertFalse(report["findings"]["native_value_difference_detected"])

    def test_answer_variation_is_a_saved_finding_and_does_not_abort(self):
        report, _ = self.execute(vary_answer=True)
        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["findings"]["answer_variation_detected"])
        self.assertTrue(report["findings"]["control_difference_detected"])
        records = [json.loads(line) for line in self.args.output.with_suffix(".jsonl").read_text().splitlines()]
        self.assertEqual(len(records), 72)
        self.assertTrue(any(r["diagnostics"] is not None for r in records))
        self.assertEqual(set(report["summaries"]["2058428"]["combined"]["raw_answer_counts"]), {"Yes", "No"})

    def test_partial_trials_survive_later_failure(self):
        report, _ = self.execute(fail_after=5)
        records = [json.loads(line) for line in self.args.output.with_suffix(".jsonl").read_text().splitlines()]
        self.assertEqual([row["trial_id"] for row in records], [1, 2, 3, 4, 5])
        self.assertEqual(report["progress"]["completed_trials"], 5)
        self.assertEqual(report["current_trial"]["trial_id"], 6)
        self.assertIn("generation", report["current_trial"])


if __name__ == "__main__":
    unittest.main()
