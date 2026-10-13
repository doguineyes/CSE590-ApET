"""CPU tests for deterministic image splits, frozen data, cached baselines and resume."""

from contextlib import ExitStack
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_original_gqa_subset as runner
import prepare_gqa_subsets as prepare
from prepare_gqa_pilot import FILES
from test_gqa_full import generation


class SubsetTests(unittest.TestCase):
    def setUp(self):
        from PIL import Image
        import pyarrow as pa
        import pyarrow.parquet as pq

        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.stack = ExitStack()
        self.questions, self.images = [], []
        for number in range(24):
            image_id = f"i{number:02}"
            output = io.BytesIO()
            Image.new("RGB", (8, 8), (number * 10, 20, 30)).save(output, format="PNG")
            self.images.append({"id": image_id, "image": {"bytes": output.getvalue()}})
            for q in range(8):
                self.questions.append({"id": f"{image_id}-q{q}", "imageId": image_id,
                    "question": f"Question {q} about image {image_id}?", "answer": "red", "isBalanced": True,
                    "types": {"semantic": "attr" if q % 2 else "rel",
                              "structural": ["query", "verify", "logical", "choose"][q % 4], "detailed": f"type{q}"}})
        paths = {}
        for name, rows in zip(FILES, [self.questions, self.images]):
            path = self.root / name
            path.parent.mkdir(parents=True)
            pq.write_table(pa.Table.from_pylist(rows), path)
            paths[name] = path
        for module in [prepare, runner]:
            self.stack.enter_context(patch.object(module, "IMAGE_COUNT", 24))
            self.stack.enter_context(patch.object(module, "QUESTION_COUNT", 192))
        self.stack.enter_context(patch.object(prepare, "fetch_sources", return_value=paths))
        options = SimpleNamespace(output_dir=self.root / "sets", dataset_cache=self.root / "cache",
            source_dir=None, download_budget_mib=0, seed=590, development_images=12,
            evaluation_images=10, questions_per_image=8, diagnostic_images=10,
            diagnostic_questions=4, no_zip=True)
        self.study = prepare.prepare(options)
        self.args = SimpleNamespace(data_manifest=options.output_dir / "development/manifest.json",
            output=self.root / "run" / "result.json", resume=False, reference_report=None,
            max_new_tokens=16, configurations=list(runner.CONFIGURATIONS), model=runner.MODEL_ID,
            vision_model=runner.VISION_ID, cache_dir=self.root / "model-cache", download_budget_gib=0,
            report_every=1000)
        self.args.output.parent.mkdir()
        self.current = {"lock_sha256": runner.shared.digest(REPO / "environments/original/uv.lock"),
            "source_sha256": {name: runner.shared.digest(REPO / name) for name in
                              ["scripts/smoke_original_apet.py", "llava/model/llava_arch.py"]},
            "implementation": {"native_matches_reference": True}}
        self.args.session_provenance = self.current

    def tearDown(self):
        self.stack.close()
        self.temp.cleanup()

    def fake_models(self, samples, fail_after=None, wrong_combined=False):
        calls = []
        stack = ExitStack()
        def generate(tokenizer, model, processor, image, text, name, seed, max_new_tokens, **kwargs):
            if max_new_tokens == 2:
                return {}
            if fail_after is not None and len(calls) == fail_after:
                raise RuntimeError("simulated mid-question interruption")
            sample = next(row for row in samples if row["question"] == text)
            self.assertEqual(seed, sample["compression_seed"])
            calls.append((sample["index"], name))
            result = generation(sample, name)
            if wrong_combined and name == "combined":
                result["response"] = "Blue"
            return result
        stack.enter_context(patch.object(runner, "checkpoint_plan", return_value={"missing_bytes": 0}))
        stack.enter_context(patch.object(runner, "download", return_value="cached"))
        loader = stack.enter_context(patch.object(runner, "load_native_model", return_value=(None, None, None)))
        stack.enter_context(patch.object(runner, "generate_native_answer", side_effect=generate))
        return stack, calls, loader

    def finalize(self, report):
        report["predictions_sha256"] = runner.shared.digest(self.args.output.with_suffix(".jsonl"))
        runner.atomic_report(self.args.output, report)

    def make_reference(self):
        self.args.configurations = ["baseline"]
        report, samples, predictions, reference = runner.open_run(self.args, self.current)
        with self.fake_models(samples)[0]:
            runner.run(self.args, report, samples, predictions, reference)
        self.finalize(report)
        reference_path = self.args.output
        self.args.output = self.root / "new" / "variant.json"
        self.args.output.parent.mkdir()
        self.args.reference_report = reference_path
        self.args.configurations = ["combined"]
        return reference_path

    def test_sampling_repeatable_disjoint_and_diagnostics_nested(self):
        images = {row["id"]: {"sha256": runner.shared.hashlib.sha256(row["image"]["bytes"]).hexdigest()} for row in self.images}
        a = prepare.select_sets(self.questions, images, 590, 12, 10, 8, 10, 4)
        b = prepare.select_sets(list(reversed(self.questions)), images, 590, 12, 10, 8, 10, 4)
        self.assertEqual(a, b)
        sets, assignment, pilot_ids, _ = a
        self.assertEqual([len(sets[name]) for name in ["development", "evaluation", "diagnostic"]], [96, 80, 40])
        self.assertFalse(set(assignment["development"]) & set(assignment["evaluation"]))
        self.assertFalse(set(pilot_ids) & set(assignment["evaluation"]))
        self.assertTrue({row["id"] for row in sets["diagnostic"]} <= {row["id"] for row in sets["development"]})
        changed = prepare.select_sets(self.questions, images, 591, 12, 10, 8, 10, 4)
        self.assertNotEqual(a[1], changed[1])

    def test_identical_photo_bytes_cannot_cross_splits(self):
        images = {row["id"]: {"sha256": "same-photo"} for row in self.images}
        with self.assertRaisesRegex(ValueError, "Duplicate raw photos"):
            prepare.select_sets(self.questions, images, 590, 12, 10, 8, 10, 4)

    def test_both_manifests_validate_and_metadata_tampering_is_rejected(self):
        manifest, _ = runner.load_manifest(self.args.data_manifest)
        self.assertEqual(manifest["study"]["role"], "development")
        evaluation = self.args.data_manifest.parent.parent / "evaluation/manifest.json"
        self.assertEqual(runner.load_manifest(evaluation)[0]["selection"]["question_count"], 80)
        changed = copy.deepcopy(manifest)
        changed["questions"][0]["answer"] = "blue"
        self.args.data_manifest.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "metadata"):
            runner.load_manifest(self.args.data_manifest)

    def test_cached_baseline_skips_baseline_inference_and_keeps_reference_immutable(self):
        reference_path = self.make_reference()
        before = reference_path.read_bytes(), reference_path.with_suffix(".jsonl").read_bytes()
        report, samples, predictions, reference = runner.open_run(self.args, self.current)
        models, calls, loader = self.fake_models(samples, wrong_combined=True)
        with models:
            runner.run(self.args, report, samples, predictions, reference)
        self.assertEqual(loader.call_count, 1)
        self.assertEqual(len(calls), 96)
        self.assertEqual({name for _, name in calls}, {"combined"})
        self.assertEqual(report["progress"], {"completed_predictions": 96, "expected_predictions": 96, "saved_reference_predictions": 96})
        self.assertEqual(report["summaries"]["baseline"]["correct"], 96)
        self.assertEqual(report["comparisons"]["combined"]["apr"], 0)
        self.assertEqual(report["baseline_origin"], "saved_reference")
        self.assertEqual(before, (reference_path.read_bytes(), reference_path.with_suffix(".jsonl").read_bytes()))

    def test_reference_rejects_other_subset_changed_settings_or_tampered_predictions(self):
        reference_path = self.make_reference()
        self.args.max_new_tokens = 8
        with self.assertRaisesRegex(ValueError, "matching-subset"):
            runner.open_run(self.args, self.current)
        self.args.max_new_tokens = 16
        old_manifest = self.args.data_manifest
        self.args.data_manifest = old_manifest.parent.parent / "evaluation/manifest.json"
        with self.assertRaisesRegex(ValueError, "matching-subset"):
            runner.open_run(self.args, self.current)
        self.args.data_manifest = old_manifest
        with reference_path.with_suffix(".jsonl").open("a") as stream:
            stream.write("{}\n")
        with self.assertRaisesRegex(ValueError, "Reference predictions changed"):
            runner.open_run(self.args, self.current)

    def test_mid_question_resume_completes_once_and_completed_run_does_not_load_model(self):
        report, samples, predictions, reference = runner.open_run(self.args, self.current)
        with self.fake_models(samples, fail_after=3)[0]:
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                runner.run(self.args, report, samples, predictions, reference)
        self.assertEqual(len(predictions), 3)
        self.finalize(report)
        self.args.resume = True
        report, samples, predictions, reference = runner.open_run(self.args, self.current)
        models, calls, loader = self.fake_models(samples)
        with models:
            runner.run(self.args, report, samples, predictions, reference)
            self.assertEqual(len(calls), 381)
            self.assertEqual(loader.call_count, 1)
            self.finalize(report)
            report, samples, predictions, reference = runner.open_run(self.args, self.current)
            runner.run(self.args, report, samples, predictions, reference)
            self.assertEqual(loader.call_count, 1)
        rows = [json.loads(line) for line in self.args.output.with_suffix(".jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 384)
        self.assertEqual(len({(row["index"], row["configuration"]) for row in rows}), 384)

    def test_saved_reference_resume_uses_its_portable_snapshot(self):
        original = self.make_reference()
        report, samples, predictions, reference = runner.open_run(self.args, self.current)
        with self.fake_models(samples, fail_after=3)[0]:
            with self.assertRaises(RuntimeError):
                runner.run(self.args, report, samples, predictions, reference)
        self.finalize(report)
        original.rename(original.with_suffix(".preserved"))
        self.args.reference_report = None
        self.args.resume = True
        report, samples, predictions, reference = runner.open_run(self.args, self.current)
        with self.fake_models(samples)[0]:
            runner.run(self.args, report, samples, predictions, reference)
        self.assertEqual(report["progress"]["completed_predictions"], 96)
        self.assertEqual(len(reference), 96)


if __name__ == "__main__":
    unittest.main()
