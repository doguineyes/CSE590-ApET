"""CPU checks of TextVQA audit gating and retained failure evidence; no model load."""

from contextlib import nullcontext
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import run_rq1_input as runner
from rq1_protocol import TEXTVQA_SELECTION


def synthetic_selection_data():
    profile = json.loads(TEXTVQA_SELECTION.read_text())
    return {"dataset": "textvqa", "cases": [
        {**{k: e[k] for k in ("case_id", "question_id", "image_id", "image_sha256", "canonical_target", "question")},
         "gold_answers": [e["canonical_target"]] * 10, "compression_seed": None,
         "generation_question": e["question"], "predictions": {}, "outcomes": {}}
        for e in profile["cases"]]}


class TextVQARunnerTests(unittest.TestCase):
    def test_prerequisite_report_and_all_six_details_are_preserved_with_hashes(self):
        profile = json.loads(TEXTVQA_SELECTION.read_text())
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            audit = temp / "audit";audit.mkdir();(audit / "target-audit").mkdir()
            (audit / "report.json").write_text('{"status":"passed"}')
            for entry in profile["cases"]:
                (audit / "target-audit" / f"{entry['question_id']}.json").write_text(json.dumps(entry))
            output = temp / "run";output.mkdir()
            manifest = temp / "manifest.json";manifest.write_text('{}')
            args = SimpleNamespace(manifest=manifest, output=output, mode="interventions",
                                   target_audit_report=audit / "report.json")
            report = {"implementation": json.loads((REPO / "research/provenance/original_llava_reference.json").read_text()),
                      "target_audit_prerequisite": {"case_ids": [e["case_id"] for e in profile["cases"]]}}
            runner.copy_evidence(args, report, {"dataset": "textvqa"})
            self.assertIn("prerequisite-target-audit.json", report["evidence_sha256"])
            self.assertEqual(len(list((output / "evidence/prerequisite-target-audit").glob('*.json'))), 6)
            self.assertEqual((output / "evidence/prerequisite-target-audit.json").read_bytes(),
                             (audit / "report.json").read_bytes())

    def test_partial_credit_is_not_full_credit_alignment_authorization(self):
        case = {"canonical_target": "two", "gold_answers": ["two"] * 3 + ["other"] * 7}
        score = runner.generation_score("textvqa", case, "2")
        self.assertAlmostEqual(score["consensus_score"], .9)
        self.assertFalse(score["correct"])
        self.assertFalse(score["full_credit"])

    def test_missing_audit_stops_before_loading_weights_and_still_exports_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "run"
            (Path(temp) / "manifest.json").write_text('{}')
            argv = ["run_rq1_input.py", "--manifest", str(Path(temp) / "manifest.json"),
                    "--output", str(output), "--questions", "2", "--tokens", "32"]
            with patch.object(sys, "argv", argv), \
                 patch.object(runner, "provenance", return_value={}), \
                 patch.object(runner, "validate_runtime"), patch.object(runner, "verify_native_reference"), \
                 patch.object(runner, "load_manifest", return_value=synthetic_selection_data()), \
                 patch.object(runner, "load_model") as load, \
                 patch.object(runner.signal, "signal"), patch("builtins.print"):
                self.assertEqual(runner.main(), 1)
            load.assert_not_called()
            with zipfile.ZipFile(output.with_suffix(".zip")) as archive:
                report = json.loads(archive.read("report.json"))
            self.assertEqual(report["status"], "failed")
            self.assertIn("--target-audit-report", report["error"])

    def test_failed_textvqa_audit_saves_finite_details_and_never_retargets(self):
        data = synthetic_selection_data()
        case = data["cases"][0]
        fixed = case["canonical_target"][:1].upper() + case["canonical_target"][1:]
        variant = {"name": "initial_capital", "text": fixed, "post_inference": False,
                   "target_ids_match_generated_prefix": False,
                   "generated_prefix_score_max_delta": None,
                   "reference_repeat_max_delta": 0.0, "score": {"sum_logprob": -5.0}}
        with tempfile.TemporaryDirectory() as temp:
            args = SimpleNamespace(output=Path(temp), questions=2, seed=590, control_tolerance=.001)
            report = {}
            def generate(*args, **kwargs):
                kwargs["record"].update(response="Wrong")
            with patch.object(runner, "pilot_cases", return_value=[case]), \
                 patch.object(runner, "save_image"), \
                 patch.object(runner, "generate_native_answer", side_effect=generate), \
                 patch("rq1_native.encode_image", return_value={"seed": runner.case_seed(case, 590), "canvas": None}), \
                 patch("PIL.Image.fromarray"), \
                 patch("rq1_native.observe_generation_tokens", return_value=nullcontext()), \
                 patch("rq1_native.audit_target_forms", return_value={"variants": [variant]}), \
                 patch("builtins.print"):
                with self.assertRaisesRegex(RuntimeError, "reference audit failed"):
                    runner.run_target_audit(args, report, data, (None, SimpleNamespace(), None))
            raw = (args.output / "target-audit" / f"{case['question_id']}.json").read_text()
            self.assertNotIn("Infinity", raw)
            record = json.loads(raw)
            self.assertFalse(record["fixed_reference"]["passed"])
            self.assertEqual(record["fixed_reference"]["text"], fixed)
            self.assertIsNone(record["fixed_reference"]["max_score_delta"])
            self.assertFalse(report["target_audit"][0]["fixed_reference"]["baseline_full_credit"])


if __name__ == "__main__":
    unittest.main()
