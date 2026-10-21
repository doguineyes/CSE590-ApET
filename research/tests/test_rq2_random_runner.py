"""CPU export and source/selection gates. Synthetic fixtures are not GPU evidence."""
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
import run_rq2_random_calibration as runner
from rq2_random_protocol import CASE_IDS, CONDITIONS, PARITY_CONDITION, random_selection
from research.tests.test_rq2_runner import synthetic_encoding


def bank_fixture():
    observations = {}
    reference = synthetic_encoding("native-input288")
    archive = {"clip_sha256": reference["source_clip"]["sha256"], "projected_sha256": "full"}
    for name, spec in CONDITIONS.items():
        if spec["method"] == "baseline":
            obs = synthetic_encoding("baseline576")
            obs["encoded"]["features"]["sha256"] = "full"
        elif spec["method"] == "native":
            obs = synthetic_encoding("native-input288" if spec["merge"] else "native-input288-no-merge")
        else:
            obs = {"source_clip": reference["source_clip"], "source_patches": reference["source_patches"],
                "uncompressed_projected": {"sha256": "full"},
                "controls": {"no_fps_E_basis_or_merge": True},
                "encoded": {"selected_indices_sorted": random_selection("a" * 64, spec["random_seed"], spec["budget"])[0]}}
        observations[name] = obs
    observations[PARITY_CONDITION] = {**copy.deepcopy(observations["pure-random-288-s101"]),
                                     "encoded": copy.deepcopy(observations["baseline576"]["encoded"])}
    for obs in observations.values():
        if "uncompressed_projected" in obs:
            obs["uncompressed_projected"] = copy.deepcopy(observations["baseline576"]["encoded"]["features"])
    return observations, {"reference": archive, "feature": {"diagnostics": {"input": reference["input"]}}}


class RandomRunnerTests(unittest.TestCase):
    def test_same_image_questions_share_one_bank_including_full576_parity(self):
        import numpy as np
        from PIL import Image
        observations, item = bank_fixture()
        records = []
        for cid in CASE_IDS[:2]:
            case = {"case_id": cid, "image_sha256": "a" * 64, "question": "Synthetic color?",
                    "generation_question": "Synthetic color?", "canonical_target": "red"}
            records.append({"case": case, "rq2": case, "feature": item["feature"],
                "reference": {**item["reference"], "seed": 590, "measurement_target": "Red",
                    "prompt": "synthetic prompt", "target_token_ids": [7],
                    "controls": {"reference": {"token_logprobs": [-1.]}}}})

        def capture(model, pixels, seed, condition, key):
            observation = copy.deepcopy(observations[condition])
            return observation, {"observation": observation}

        def generate(tokenizer, model, processor, image, case, pixels, condition, snapshot, seed, target):
            return {"case_id": case["case_id"], "condition": condition, "response": "Red",
                "scoring": {"correct": True}, "encoding": snapshot["observation"],
                "generation_tokens": {"generated_token_ids": [7]},
                "reference_score": {"prompt": "synthetic prompt", "target_token_ids": [7],
                                    "token_logprobs": [-1.], "mean_logprob": -1.}}

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = SimpleNamespace(output=root, inputs=root / "inputs")
            report = {}
            with patch.dict(sys.modules, {"torch": SimpleNamespace()}), \
                 patch.object(runner, "REPEAT_IDS", ()), \
                 patch.object(runner, "save_image", return_value=Image.new("RGB", (2, 2))), \
                 patch.object(runner, "prepare_pixels", return_value=(object(), np.zeros((2, 2, 3), dtype=np.uint8))) as pixels, \
                 patch.object(runner, "save_bank", return_value={"file": "synthetic"}) as saved, \
                 patch("rq2_random_native.capture_condition", side_effect=capture) as captured, \
                 patch("rq2_random_native.generate_condition", side_effect=generate) as generated, \
                 contextlib.redirect_stdout(io.StringIO()):
                runner.run_panel(args, report, {"gqa": {}}, records, (None, None, None))
            pixels.assert_called_once()
            saved.assert_called_once()
            self.assertEqual(captured.call_count, 12)
            self.assertEqual(generated.call_count, 24)
            self.assertEqual(report["progress"]["completed_questions"], 2)
            self.assertEqual(report["summaries"]["gqa"]["unique_questions"], 2)
            q = [json.loads((root / "questions" / (cid.replace(":", "-") + ".json")).read_text())
                 for cid in CASE_IDS[:2]]
            self.assertEqual(q[0]["feature_bank"], q[1]["feature_bank"])
            self.assertEqual(q[1]["controls"]["pure576_parity"]["status"], "passed")

    def test_bank_gate_rejects_non_nested_selections_and_changed_sources(self):
        observations, item = bank_fixture()
        self.assertEqual(runner.validate_bank(observations, item)["status"], "passed")
        changed = copy.deepcopy(observations)
        changed["pure-random-64-s101"]["source_clip"]["sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "Source CLIP"):
            runner.validate_bank(changed, item)
        changed = copy.deepcopy(observations)
        outside = next(i for i in range(576) if i not in changed["pure-random-288-s101"]["encoded"]["selected_indices_sorted"])
        changed["pure-random-64-s101"]["encoded"]["selected_indices_sorted"][0] = outside
        with self.assertRaisesRegex(ValueError, "not nested"):
            runner.validate_bank(changed, item)

    def test_runtime_failure_preserves_report_and_verified_archive_without_loading_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "failed-run"
            with patch.object(runner, "provenance", return_value={}), \
                 patch.object(runner, "validate_runtime", side_effect=RuntimeError("synthetic failure")), \
                 patch.object(runner, "load_model") as load, patch.object(runner.signal, "signal"), \
                 patch.dict(os.environ), contextlib.redirect_stdout(io.StringIO()):
                code = runner.main(["--inputs", str(Path(folder) / "inputs"), "--output", str(output), "--calibration"])
            self.assertEqual(code, 1)
            load.assert_not_called()
            with zipfile.ZipFile(output.with_suffix(".zip")) as archive:
                self.assertIsNone(archive.testzip())
                report = json.loads(archive.read("report.json"))
                self.assertEqual(report["status"], "failed")
                self.assertFalse(report["calibration_passed"])
                self.assertFalse(report["scientific_benchmark_passed"])
                self.assertIn("artifact-sha256.json", archive.namelist())

    def test_existing_outputs_and_missing_calibration_flag_are_rejected_before_runtime(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root / "existing"
            output.mkdir()
            marker = output / "keep.txt"
            marker.write_text("preserve")
            with patch.object(runner, "provenance") as get, contextlib.redirect_stderr(io.StringIO()):
                for options in ([], ["--calibration"]):
                    with self.assertRaises(SystemExit):
                        runner.main(["--inputs", str(root / "inputs"), "--output", str(output), *options])
            get.assert_not_called()
            self.assertEqual(marker.read_text(), "preserve")

    def test_notebook_cells_compile_and_import_cannot_start_a_gpu_run(self):
        notebook = json.loads((REPO / "research/notebooks/kaggle_rq2_random_calibration.ipynb").read_text())
        code = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
        for index, cell in enumerate(code):
            compile(cell, f"cell-{index}", "exec")
        self.assertIn("RUN_RANDOM_CALIBRATION = False", code[0])
        with patch("subprocess.run") as run:
            with self.assertRaisesRegex(RuntimeError, "RUN_RANDOM_CALIBRATION"):
                exec(compile(code[0], "first-cell", "exec"), {})
        run.assert_not_called()

    def test_notebook_refuses_pretrained_run_without_passed_prerequisites(self):
        notebook = json.loads((REPO / "research/notebooks/kaggle_rq2_random_calibration.ipynb").read_text())
        helper = next("".join(c["source"]) for c in notebook["cells"]
                      if "".join(c["source"]).startswith("def verify_rq2_random_prerequisites("))
        namespace = {"rq2_random_prerequisites_passed": True}
        exec(compile(helper, "helpers", "exec"), namespace)
        self.assertFalse(namespace["rq2_random_prerequisites_passed"])
        with self.assertRaisesRegex(RuntimeError, "prerequisites"):
            namespace["run_rq2_random_calibration"]()


if __name__ == "__main__":
    unittest.main()
