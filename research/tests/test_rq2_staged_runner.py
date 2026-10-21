"""CPU export, prerequisite and native-cache schedule validation tests."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import run_rq2_staged_calibration as runner
from rq2_staged_protocol import specification
from rq2_staged_native import validate_cache


class StagedRunnerTests(unittest.TestCase):
    def test_same_image_reuses_input_bank_but_each_question_generates_its_own_stages(self):
        import numpy as np
        from PIL import Image
        from rq2_staged_protocol import load_profile
        cases = load_profile()["cases"][:2]
        records = [{"case": c, "rq2": c, "reference": {"seed": c["seed"]}} for c in cases]
        def capture(model, pixels, seed, condition, key):
            return {"synthetic": condition}, {"condition": condition}
        def generate(tokenizer, model, processor, image, case, pixels, condition, snapshot, seed, **kwargs):
            spec = specification(condition)
            row = {"case_id": case["case_id"], "image_sha256": case["image_sha256"], "condition": condition,
                **spec, "seed": seed, "response": "wrong", "scoring": {"correct": False},
                "generation_tokens": {"generated_token_ids": [2]}, "generation": {"prompt": case["question"]}}
            return row, {"independent_question": case["case_id"]} if spec["layer"] else None
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = SimpleNamespace(inputs=root, output=root)
            report = {}
            model = SimpleNamespace(model=SimpleNamespace(layers=[None] * 32))
            with patch.object(runner, "REPEAT_IDS", ()), \
                 patch.object(runner, "save_image", return_value=Image.new("RGB", (2, 2))), \
                 patch.object(runner, "prepare_pixels", return_value=(object(), np.zeros((2, 2, 3), dtype=np.uint8))) as prepared, \
                 patch.object(runner, "save_input_bank", return_value={"file": "synthetic"}) as saved, \
                 patch.object(runner, "check_bank", return_value={"status": "synthetic"}), \
                 patch("rq2_staged_native.capture_input", side_effect=capture) as captured, \
                 patch("rq2_staged_native.generate_condition", side_effect=generate) as generated, \
                 contextlib.redirect_stdout(io.StringIO()):
                runner.run_panel(args, report, {}, {"gqa": {}}, records, (None, model, None))
            prepared.assert_called_once()
            saved.assert_called_once()
            self.assertEqual(captured.call_count, 10)
            self.assertEqual(generated.call_count, 20)
            self.assertEqual(report["progress"]["completed_questions"], 2)
            self.assertEqual(report["summaries"]["gqa"]["baseline_correct"], 0)
            self.assertEqual(len((root / "predictions.jsonl").read_text().splitlines()), 18)

    def test_input_bank_gate_rejects_source_drift_and_accidental_random_compression(self):
        from research.tests.test_rq2_random_runner import bank_fixture
        original, item = bank_fixture()
        observed = {}
        for name in (*runner.CONDITIONS, runner.IDENTITY_CONDITION):
            spec = specification(name)
            if spec["method"] == "baseline":
                row = copy.deepcopy(original["baseline576"])
            elif spec["method"] == "native":
                row = copy.deepcopy(original["native-288-merge"])
                row["encoded"]["kept_tokens"] = spec["input_tokens"]
            else:
                row = {"source_clip": original["baseline576"]["source_clip"],
                    "source_patches": original["baseline576"]["source_patches"],
                    "full_projected": original["baseline576"]["encoded"]["features"],
                    "pure_input": {"fps": False, "E": False, "basis_replacement": False, "merging": False},
                    "encoded": {"kept_tokens": spec["input_tokens"]}}
                if name == runner.IDENTITY_CONDITION:
                    row["encoded"] = copy.deepcopy(original["baseline576"]["encoded"])
            observed[name] = row
        self.assertEqual(runner.check_bank(observed, item)["status"], "passed")
        bad = copy.deepcopy(observed)
        bad["pure-random-96-to-32-s101"]["pure_input"]["E"] = True
        with self.assertRaisesRegex(ValueError, "Pure input"):
            runner.check_bank(bad, item)
        bad = copy.deepcopy(observed)
        bad["native-96-to-32"]["source_clip"]["sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "Source/budget"):
            runner.check_bank(bad, item)

    def test_cache_schedule_rejects_uncompressed_late_cache_or_bad_positions(self):
        spec = specification("native-96-to-32")
        events = []
        for block, length in ((1, 160), (17, 96)):
            events.append({"block": block, "sequence_tokens": length, "position_first": 0,
                           "position_last": length - 1, "cache_tokens_before": 0})
            for position in (length, length + 1, length + 2):
                events.append({"block": block, "sequence_tokens": 1, "position_first": position,
                               "position_last": position, "cache_tokens_before": position})
        self.assertEqual(validate_cache(events, spec, 160, 4)["cached_decode_steps"], 3)
        for field in ("cache_tokens_before", "position_first", "sequence_tokens"):
            bad = copy.deepcopy(events)
            bad[-1][field] += 64
            with self.assertRaises(ValueError):
                validate_cache(bad, spec, 160, 4)

    def test_noop_parity_allows_only_the_declared_identity_condition(self):
        row = {"case_id": "gqa:1", "image_sha256": "a", "condition": "baseline576", "seed": 590,
               "response": "red", "scoring": {"correct": True}, "generation_tokens": {"generated_token_ids": [1]},
               "generation": {"prompt": "p"}}
        noop = {**row, "condition": runner.IDENTITY_CONDITION}
        self.assertEqual(runner.compare_replay(row, noop)["status"], "passed")
        changed = {**noop, "generation_tokens": {"generated_token_ids": [2]}}
        with self.assertRaises(ValueError):
            runner.compare_replay(row, changed)

    def test_failed_runtime_preserves_archive_without_loading_model(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            out = root / "failed"
            with patch.object(runner, "provenance", return_value={}), \
                 patch.object(runner, "validate_runtime", side_effect=RuntimeError("synthetic failure")), \
                 patch.object(runner, "load_model") as load, patch.object(runner.signal, "signal"), \
                 patch.dict(os.environ), contextlib.redirect_stdout(io.StringIO()):
                code = runner.main(["--inputs", str(root / "inputs"), "--output", str(out), "--calibration",
                    "--setup-log", str(root / "setup"), "--runtime-report", str(root / "runtime"),
                    "--prerequisite-report", str(root / "proof")])
            load.assert_not_called()
            self.assertEqual(code, 1)
            with zipfile.ZipFile(out.with_suffix(".zip")) as z:
                self.assertIsNone(z.testzip())
                report = json.loads(z.read("report.json"))
                self.assertEqual(report["status"], "failed")
                self.assertFalse(report["calibration_passed"])
                self.assertIn("artifact-sha256.json", z.namelist())

    def test_stale_prerequisite_sources_or_runtime_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = SimpleNamespace(prerequisite_report=root / "proof", setup_log=root / "setup", runtime_report=root / "runtime")
            args.setup_log.write_text("setup")
            args.runtime_report.write_text("runtime")
            proof = {"version": runner.VERSION, "status": "passed", "test_modules": list(runner.TESTS),
                "tests_run": 30, "skipped": 0, "source_sha256": {"source": "same"},
                "setup_log_sha256": runner.sha256(args.setup_log), "runtime_report_sha256": runner.sha256(args.runtime_report)}
            args.prerequisite_report.write_text(json.dumps(proof))
            with patch.object(runner, "prerequisite_signature", return_value={"source": "same"}):
                self.assertEqual(runner.verify_prerequisites(args)["status"], "passed")
                args.setup_log.write_text("changed")
                with self.assertRaisesRegex(ValueError, "changed since prerequisites"):
                    runner.verify_prerequisites(args)

    def test_notebook_import_idle_and_pretrained_guard_requires_tests(self):
        notebook = json.loads((REPO / "research/notebooks/kaggle_rq2_staged_calibration.ipynb").read_text())
        code = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
        for i, cell in enumerate(code):
            compile(cell, f"cell-{i}", "exec")
        with patch("subprocess.run") as run:
            with self.assertRaisesRegex(RuntimeError, "RUN_STAGED_CALIBRATION"):
                exec(code[0], {})
        run.assert_not_called()
        helper = next(s for s in code if s.startswith("def verify_rq2_staged_prerequisites("))
        namespace = {"rq2_staged_prerequisites_passed": True}
        exec(helper, namespace)
        self.assertFalse(namespace["rq2_staged_prerequisites_passed"])
        with self.assertRaisesRegex(RuntimeError, "prerequisites"):
            namespace["run_rq2_staged_calibration"]()


if __name__ == "__main__":
    unittest.main()
