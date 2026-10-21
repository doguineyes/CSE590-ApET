"""CPU controls/export tests. These do not claim pretrained correctness."""
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
import run_rq2_textvqa100 as runner
from rq2_textvqa_protocol import load_profile, load_reference
from rq2_random_protocol import CONDITIONS
from research.tests.test_rq2_random_runner import bank_fixture


class TextVQARunnerTests(unittest.TestCase):
    def test_all_arms_run_for_a_baseline_wrong_case_and_only_declared_tensors_are_saved(self):
        import numpy as np
        from PIL import Image
        case = copy.deepcopy(load_reference()["cases"][0])
        case["compression_seed"] = None
        data = {"cases": [case]}
        observations, _ = bank_fixture()
        # A bank fixture is intentionally synthetic; the separate native tests
        # exercise actual masks, source gates and 64-token generation.
        def capture(model, pixels, seed, condition, key):
            return copy.deepcopy(observations[condition]), {"observation": observations[condition]}

        def generate(tokenizer, model, processor, image, c, pixels, condition, snapshot, seed):
            from rq2_protocol import score_response
            from rq2_random_protocol import specification
            return {"case_id": c["case_id"], "condition": condition, **specification(condition),
                "image_sha256": c["image_sha256"], "seed": seed, "response": "deliberately wrong",
                "scoring": score_response(c, "deliberately wrong"), "generated_tokens": 2,
                "generation_tokens": {"generated_token_ids": [8, 9]},
                "encoded_fingerprint": {"sha256": "synthetic"},
                "generation": {"prompt": "same synthetic prompt", "expanded_prompt_tokens": 100}}

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = SimpleNamespace(output=root, inputs=root)
            report = {}
            with patch.object(runner, "REPEAT_IDS", ()), patch.object(runner, "DIAGNOSTIC_IDS", ()), \
                 patch.object(runner, "save_image", return_value=Image.new("RGB", (2, 2))), \
                 patch.object(runner, "prepare_pixels", return_value=(object(), np.zeros((2, 2, 3), dtype=np.uint8))), \
                 patch.object(runner, "validate_bank", return_value={"status": "synthetic"}), \
                 patch.object(runner, "save_bank") as saved, \
                 patch.object(runner, "generate_answer", side_effect=generate) as generated, \
                 patch("rq2_random_native.capture_condition", side_effect=capture) as captured, \
                 contextlib.redirect_stdout(io.StringIO()):
                runner.run_panel(args, report, data, load_reference(), (None, None, None))
            saved.assert_not_called()
            self.assertEqual(captured.call_count, 11)
            self.assertEqual(generated.call_count, 11)
            self.assertEqual(report["summaries"]["all100"]["baseline_full_credit"], 0)
            self.assertEqual(report["progress"]["completed_questions"], 1)
            self.assertEqual(len((root / "predictions.jsonl").read_text().splitlines()), 11)
            encodings = list((root / "encodings").glob("*.json"))
            self.assertEqual(len(encodings), 1)

    def test_frozen_replay_rejects_changed_generated_ids_or_processed_scores(self):
        row = {"case_id": "textvqa:1", "image_sha256": "a", "seed": 590, "response": "red",
            "scoring": {"correct": True}, "generation_tokens": {"generated_token_ids": [1], "scores": [-.1]},
            "encoded_fingerprint": {"sha256": "full"},
            "generation": {"prompt": "p", "expanded_prompt_tokens": 600}}
        self.assertEqual(runner.compare_replay(row, copy.deepcopy(row))["status"], "passed")
        for field, value in (("generated_token_ids", [2]), ("scores", [-.2])):
            changed = copy.deepcopy(row)
            changed["generation_tokens"][field] = value
            with self.assertRaisesRegex(ValueError, "Frozen replay"):
                runner.compare_replay(row, changed)

    def test_changed_sources_or_unmatched_selection_fail_bank_gate(self):
        observations, _ = bank_fixture()
        # Supply the real per-condition budget metadata absent from old fixture.
        for name, observation in observations.items():
            from rq2_random_protocol import specification
            observation["experiment"] = specification(name)
            observation["encoded"]["kept_tokens"] = specification(name)["budget"]
        self.assertEqual(runner.validate_bank(observations, "a" * 64)["status"], "passed")
        changed = copy.deepcopy(observations)
        changed["native-64-merge"]["input"]["basis_indices"][0] = -1
        with self.assertRaises((ValueError, RuntimeError)):
            runner.validate_bank(changed, "a" * 64)
        changed = copy.deepcopy(observations)
        changed["pure-random-64-s101"]["encoded"]["kept_tokens"] = 288
        with self.assertRaisesRegex(ValueError, "budget"):
            runner.validate_bank(changed, "a" * 64)

    def test_prerequisite_proof_is_bound_to_current_sources_and_runtime(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = SimpleNamespace(prerequisite_report=root / "prerequisites.json",
                                   runtime_report=root / "runtime.json", setup_log=root / "setup.log")
            args.runtime_report.write_text("runtime")
            args.setup_log.write_text("tests")
            proof = {"version": runner.VERSION, "status": "passed", "test_modules": list(runner.TESTS),
                     "tests_run": 40, "skipped": 0, "source_sha256": {"source": "good"},
                     "runtime_report_sha256": runner.sha256(args.runtime_report),
                     "setup_log_sha256": runner.sha256(args.setup_log)}
            args.prerequisite_report.write_text(json.dumps(proof))
            with patch.object(runner, "prerequisite_signature", return_value={"source": "good"}):
                self.assertEqual(runner.verify_prerequisite_report(args)["status"], "passed")
                args.runtime_report.write_text("changed")
                with self.assertRaisesRegex(ValueError, "runtime/setup"):
                    runner.verify_prerequisite_report(args)
            with patch.object(runner, "prerequisite_signature", return_value={"source": "different"}):
                with self.assertRaisesRegex(ValueError, "changed since prerequisites"):
                    runner.verify_prerequisite_report(args)

    def test_runtime_failure_preserves_report_archive_and_does_not_load_model(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            out = root / "failed"
            with patch.object(runner, "provenance", return_value={}), \
                 patch.object(runner, "validate_runtime", side_effect=RuntimeError("synthetic runtime failure")), \
                 patch.object(runner, "load_model") as load, patch.object(runner.signal, "signal"), \
                 patch.dict(os.environ), contextlib.redirect_stdout(io.StringIO()):
                code = runner.main(["--inputs", str(root / "inputs"), "--output", str(out), "--run-textvqa100",
                    "--prerequisite-report", str(root / "proof"), "--runtime-report", str(root / "runtime"),
                    "--setup-log", str(root / "setup")])
            load.assert_not_called()
            self.assertEqual(code, 1)
            with zipfile.ZipFile(out.with_suffix(".zip")) as archive:
                self.assertIsNone(archive.testzip())
                report = json.loads(archive.read("report.json"))
                self.assertEqual(report["status"], "failed")
                self.assertFalse(report["execution_passed"])
                self.assertIn("artifact-sha256.json", archive.namelist())

    def test_notebook_import_is_idle_and_helper_invalidates_old_prerequisites(self):
        notebook = json.loads((REPO / "research/notebooks/kaggle_rq2_textvqa100.ipynb").read_text())
        code = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
        for i, cell in enumerate(code):
            compile(cell, f"cell-{i}", "exec")
        with patch("subprocess.run") as run:
            with self.assertRaisesRegex(RuntimeError, "RUN_TEXTVQA100"):
                exec(code[0], {})
        run.assert_not_called()
        helper = next(cell for cell in code if cell.startswith("def verify_rq2_textvqa_prerequisites("))
        namespace = {"rq2_textvqa_prerequisites_passed": True}
        exec(helper, namespace)
        self.assertFalse(namespace["rq2_textvqa_prerequisites_passed"])
        with self.assertRaisesRegex(RuntimeError, "prerequisites"):
            namespace["run_rq2_textvqa100"]()


if __name__ == "__main__":
    unittest.main()
