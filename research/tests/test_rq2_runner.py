"""CPU orchestration/export tests; synthetic execution never claims model parity.

Actual frozen questions and unchanged dataset scorers are exercised. CUDA/model
execution is explicitly stubbed, and real pretrained gates still belong to Kaggle.
"""

import contextlib
import copy
import csv
import hashlib
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
import run_rq2_input_merge as runner
from rq1_protocol import sha256, write_json
from rq2_protocol import ARMS, PROFILE, load_profile, paired_summary, score_response


def fixture_inputs(directory):
    """Minimal byte-preservation fixture, not an accepted inference bundle."""
    root = Path(directory) / "inputs"
    write_json(root / "references/index.json", {"synthetic_cpu_test": True})
    (root / "references/raw-evidence.bin").write_bytes(b"\x00original bytes\xff\r\n")
    write_json(root / "protocol.json", {"synthetic_cpu_test": True})
    for dataset in ("gqa", "mmstar", "textvqa"):
        write_json(root / dataset / "manifest.json", {"dataset": dataset, "synthetic_cpu_test": True})
    return root


def synthetic_panel(args, report, profile, manifests, records, loaded):
    """Generate CPU export fixtures from real archived strings and real scorers."""
    report["synthetic_cpu_test"] = True
    entries = profile["cases"][:args.questions]
    report.update(expected_case_ids=[c["case_id"] for c in entries], questions=[], question_sha256={})
    rows = []
    for case in entries:
        arms = {}
        for arm in ARMS:
            previous = "input-only" if arm == "native-input288" else "baseline"
            answer = case["historical_predictions"][previous]["response"]
            # Illustrative no-merge responses are archived baseline strings,
            # never predictions or an asserted outcome of actual no-merge inference.
            row = {"case_id": case["case_id"], "arm": arm, "response": answer,
                   "scoring": score_response(case, answer),
                   "reference_score": {"mean_logprob": -1., "synthetic_cpu_test": True},
                   "synthetic_cpu_test": True}
            arms[arm] = row
            rows.append(row)
        path = args.output / "questions" / (case["case_id"].replace(":", "-") + ".json")
        write_json(path, {"case_id": case["case_id"], "status": "passed", "case": case,
                          "arms": arms, "controls": {"synthetic_cpu_test": True}})
        relative = str(path.relative_to(args.output))
        report["question_sha256"][relative] = sha256(path)
        report["questions"].append({"case_id": case["case_id"], "status": "passed"})
    with (args.output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    report["summaries"] = paired_summary(rows)
    report["progress"] = {"completed_questions": len(entries), "expected_questions": len(entries)}


def synthetic_encoding(arm):
    """JSON fingerprints for testing gate wiring, never real tensor evidence."""
    selected = list(range(576 if arm == "baseline576" else 288))
    mask = {"sha256": "synthetic-mask-" + str(len(selected))}
    post = "synthetic-pre" if arm == "native-input288-no-merge" else "synthetic-post"
    return {"source_clip": {"sha256": "synthetic-clip"},
            "source_patches": {"sha256": "synthetic-patches"},
            "pre_merge": {"sha256": "synthetic-pre"}, "post_merge": {"sha256": post},
            "input": {"enabled": arm != "baseline576", "errors": [0.] * 576,
                      "basis_indices": list(range(10)), "selected_indices": selected,
                      "selected_indices_sorted": selected},
            "input_merge": {"synthetic_destinations": [1, 2, 3]},
            "basis_slot_to_source": [{"source_index": 0, "assignment_slot": 287}],
            "encoded": {"features": {"sha256": post}, "mask": mask,
                        "selected_indices_sorted": selected, "kept_tokens": len(selected)}}


def gate_records(profile):
    records = []
    for case in profile["cases"]:
        records.append({"case": case, "reference": {
            "seed": case["seed"], "clip_sha256": "synthetic-clip", "projected_sha256": "synthetic-projected",
            "target_token_ids": [1], "measurement_target": case["canonical_target"],
            "prompt": "synthetic fixed prompt", "controls": {"reference": {
                "token_logprobs": [-1.], "sum_logprob": -1., "mean_logprob": -1.}}},
            "feature": {"diagnostics": {"input": {
                "errors": [0.] * 576, "basis_indices": list(range(10))}}}})
    return records


class RQ2RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = load_profile()

    def verify_zip(self, output):
        archive = output.with_suffix(".zip")
        self.assertTrue(archive.is_file())
        with zipfile.ZipFile(archive) as z:
            self.assertIsNone(z.testzip())
            self.assertEqual(len(z.namelist()), len(set(z.namelist())))
            inventory = json.loads(z.read("artifact-sha256.json"))
            self.assertEqual(set(z.namelist()), set(inventory) | {"artifact-sha256.json"})
            for relative, digest in inventory.items():
                self.assertEqual(hashlib.sha256(z.read(relative)).hexdigest(), digest)
                self.assertEqual(z.read(relative), (output / relative).read_bytes())
            return json.loads(z.read("report.json")), inventory

    def test_runtime_failure_is_archived_and_model_is_never_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "runtime-failed"
            with patch.object(runner, "provenance", return_value={"synthetic_cpu_test": True}), \
                    patch.object(runner, "validate_runtime", side_effect=RuntimeError("CPU test runtime gate failure")), \
                    patch.object(runner, "verify_native_reference") as native, \
                    patch.object(runner, "load_model") as load, \
                    patch.object(runner.signal, "signal"), patch.dict(os.environ), \
                    contextlib.redirect_stdout(io.StringIO()):
                code = runner.main(["--inputs", str(root / "missing-inputs"), "--output", str(output)])
            self.assertEqual(code, 1)
            native.assert_not_called()
            load.assert_not_called()
            report, inventory = self.verify_zip(output)
            self.assertEqual(report["status"], "failed")
            self.assertIn("CPU test runtime gate failure", report["error"])
            self.assertIn("traceback", report)
            self.assertGreaterEqual(report["wall_seconds"], 0)
            self.assertNotIn("predictions.jsonl", inventory)
            self.assertFalse(report.get("questions"))
            self.assertIn("index.html", inventory)

    def test_overwrite_invalid_prefix_and_output_inside_inputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs"
            inputs.mkdir()
            existing = root / "existing"
            existing.mkdir()
            sentinel = existing / "unchanged.bin"
            sentinel.write_bytes(b"preserve existing result")
            zipped = root / "zip-exists"
            zipped.with_suffix(".zip").write_bytes(b"preserve existing zip")
            cases = [(existing, []), (zipped, []), (inputs / "nested", [])]
            cases += [(root / ("bad-prefix-" + value), ["--questions", value]) for value in ("0", "31", "-1")]
            with patch.object(runner, "provenance") as provenance, contextlib.redirect_stderr(io.StringIO()):
                for output, extra in cases:
                    with self.subTest(output=output), self.assertRaises(SystemExit) as stopped:
                        runner.main(["--inputs", str(inputs), "--output", str(output), *extra])
                    self.assertEqual(stopped.exception.code, 2)
            provenance.assert_not_called()
            self.assertEqual(sentinel.read_bytes(), b"preserve existing result")
            self.assertEqual(zipped.with_suffix(".zip").read_bytes(), b"preserve existing zip")
            self.assertFalse((inputs / "nested").exists())
            self.assertFalse(any(root.glob("bad-prefix-*")))

    def test_setup_runtime_and_source_evidence_are_copied_byte_for_byte(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = fixture_inputs(root)
            output = root / "copy"
            output.mkdir()
            setup = root / "setup.log"
            setup.write_bytes("uv setup → unchanged\r\n".encode("utf-8"))
            runtime = root / "runtime.json"
            runtime.write_bytes(b'{ "status": "passed", "synthetic_cpu_test": true }\n')
            source = "scripts/rq2_protocol.py"
            report = {"measurement_sources_sha256": {source: sha256(REPO / source)}}
            args = SimpleNamespace(output=output, inputs=inputs, setup_log=setup, runtime_report=runtime)
            before = {str(path): path.read_bytes() for path in [setup, runtime, REPO / source]}
            runner.copy_evidence(args, report)
            mapping = [(setup, "notebook/setup.log"), (runtime, "notebook/runtime.json"),
                       (REPO / source, "sources/" + source),
                       (inputs / "references/raw-evidence.bin", "references/raw-evidence.bin"),
                       (inputs / "protocol.json", "accepted-input-protocol.json")]
            for original, relative in mapping:
                self.assertEqual((output / "evidence" / relative).read_bytes(), original.read_bytes())
                self.assertEqual(report["evidence_sha256"][relative], sha256(original))
            for original, content in before.items():
                self.assertEqual(Path(original).read_bytes(), content)
            self.assertEqual(len(report["evidence_sha256"]), 9)

    def test_supplied_but_missing_notebook_evidence_is_not_silently_omitted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = fixture_inputs(root)
            output = root / "missing-evidence"
            output.mkdir()
            args = SimpleNamespace(output=output, inputs=inputs, setup_log=root / "absent.log", runtime_report=None)
            with self.assertRaisesRegex(ValueError, "notebook evidence missing"):
                runner.copy_evidence(args, {"measurement_sources_sha256": {}})

    def test_complete_three_arm_main_export_uses_frozen_questions_and_real_scorers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = fixture_inputs(root)
            output = root / "synthetic-complete"
            setup = root / "setup.log"
            setup.write_bytes(b"synthetic CPU setup control\n")
            runtime = root / "runtime.json"
            runtime.write_bytes(b'{"synthetic_cpu_test":true}\n')
            fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 2))
            source_hashes = {"scripts/rq2_protocol.py": sha256(REPO / "scripts/rq2_protocol.py")}
            with patch.dict(sys.modules, {"torch": fake_torch}), \
                    patch.object(runner, "provenance", return_value={"synthetic_cpu_test": True}), \
                    patch.object(runner, "validate_runtime"), patch.object(runner, "verify_native_reference"), \
                    patch.object(runner, "validate_bundle", return_value=(copy.deepcopy(self.profile), {}, {}, [])), \
                    patch.object(runner, "source_hashes", return_value=source_hashes), \
                    patch.object(runner, "load_model", return_value=(None, None, None)) as load, \
                    patch.object(runner, "run_panel", side_effect=synthetic_panel), \
                    patch.object(runner.signal, "signal"), patch.dict(os.environ), \
                    contextlib.redirect_stdout(io.StringIO()):
                code = runner.main(["--inputs", str(inputs), "--output", str(output),
                                    "--setup-log", str(setup), "--runtime-report", str(runtime)])
            self.assertEqual(code, 0)
            load.assert_called_once()
            report, inventory = self.verify_zip(output)
            self.assertEqual(report["status"], "passed")
            self.assertTrue(report["synthetic_cpu_test"])
            self.assertEqual(report["expected_case_ids"], self.profile["case_ids"])
            self.assertEqual(report["progress"]["completed_questions"], 30)
            self.assertEqual(len(report["question_sha256"]), 30)
            self.assertEqual([report["summaries"][d]["questions"] for d in ("gqa", "mmstar", "textvqa")], [12, 12, 6])
            rows = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 90)
            self.assertEqual({row["arm"] for row in rows}, set(ARMS))
            by_key = {(row["case_id"], row["arm"]): row for row in rows}
            self.assertFalse(by_key[("textvqa:37533", "native-input288")]["scoring"]["correct"])
            self.assertEqual(by_key[("textvqa:37533", "native-input288")]["response"], "Bottoli")
            self.assertEqual(by_key[("textvqa:37533", "native-input288")]["scoring"]["consensus_score"], 0)
            self.assertTrue(by_key[("textvqa:37533", "baseline576")]["scoring"]["correct"])
            with (output / "answers.csv").open(newline="") as stream:
                table = list(csv.DictReader(stream))
            self.assertEqual(len(table), 90)
            self.assertEqual(sum(row["mmstar_grounding_flag"] == "True" for row in table), 15)
            self.assertEqual(sum(row["mmstar_prior_quality_flag"] == "True" for row in table), 6)
            for relative, digest in report["question_sha256"].items():
                self.assertEqual(inventory[relative], digest)
            self.assertEqual((output / "evidence/notebook/setup.log").read_bytes(), setup.read_bytes())
            self.assertEqual((output / "evidence/notebook/runtime.json").read_bytes(), runtime.read_bytes())

    def run_two_question_gates(self, root, *, change_same_image_mapping=False):
        """Exercise actual run_panel bookkeeping/parity using synthetic fingerprints."""
        output = root / "run-panel"
        output.mkdir()
        (output / "images").mkdir()
        args = SimpleNamespace(output=output, inputs=root / "inputs", questions=2)
        report = {"status": "running"}
        records = gate_records(self.profile)
        encoded = {"seed": records[0]["reference"]["seed"], "clip_sha256": "synthetic-clip",
                   "projected_sha256": "synthetic-projected", "canvas": "synthetic canvas", "pixels": None}
        generated_ids = {"generated_token_ids": [1, 2]}

        def arm(tokenizer, model, processor, image, case, pixels, name, target, seed):
            encoding = synthetic_encoding(name)
            if change_same_image_mapping and case["case_id"] == self.profile["case_ids"][1]:
                encoding["input_merge"]["synthetic_destinations"] = [3, 2, 1]
            prior = "input-only" if name == "native-input288" else "baseline"
            answer = case["historical_predictions"][prior]["response"]
            return {"case_id": case["case_id"], "arm": name, "response": answer,
                    "scoring": score_response(case, answer), "encoding": encoding,
                    "generation_tokens": generated_ids, "reference_score": {
                        "target_token_ids": [1], "prompt": "synthetic fixed prompt",
                        "token_logprobs": [-1.], "sum_logprob": -1., "mean_logprob": -1.}}, {}

        def native(*args):
            return {"encoded": synthetic_encoding("native-input288")["encoded"],
                    "generation_tokens": generated_ids}

        fake_pil = SimpleNamespace(Image=SimpleNamespace(fromarray=lambda canvas: SimpleNamespace(
            save=lambda path: path.write_bytes(b"synthetic canvas; not a rendered image"))))
        with patch.dict(sys.modules, {"torch": SimpleNamespace(), "PIL": fake_pil}), \
                patch.object(runner, "save_image", return_value=None), \
                patch("rq1_native.encode_image", return_value=encoded) as encode, \
                patch.object(runner, "raw_native_control", side_effect=native), \
                patch.object(runner, "run_arm", side_effect=arm), \
                contextlib.redirect_stdout(io.StringIO()):
            if change_same_image_mapping:
                with self.assertRaisesRegex(ValueError, "Same-image compression changed"):
                    runner.run_panel(args, report, self.profile, {"gqa": {}}, records, (None, None, None))
            else:
                runner.run_panel(args, report, self.profile, {"gqa": {}}, records, (None, None, None))
        encode.assert_called_once()
        return output, report

    def test_actual_panel_pairs_same_image_once_and_validates_three_arms(self):
        with tempfile.TemporaryDirectory() as directory:
            output, report = self.run_two_question_gates(Path(directory))
            self.assertEqual(report["progress"]["completed_questions"], 2)
            self.assertEqual(len(report["question_sha256"]), 2)
            self.assertEqual(len((output / "predictions.jsonl").read_text().splitlines()), 6)
            for cid in self.profile["case_ids"][:2]:
                detail = json.loads((output / "questions" / (cid.replace(":", "-") + ".json")).read_text())
                self.assertEqual(detail["status"], "passed")
                self.assertTrue(detail["controls"]["same_image_exact"])
                self.assertEqual(detail["controls"]["native_parity"]["status"], "passed")
                self.assertEqual(detail["controls"]["selection_and_basis"]["status"], "passed")
                self.assertEqual(detail["controls"]["baseline_reference_max_delta"], 0)

    def test_diagnostic_mode_cannot_be_exported_as_a_passed_scientific_pilot(self):
        """CLI dispatch/export only; a synthetic probe is not model evidence."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = fixture_inputs(root)
            output = root / "synthetic-diagnostic"
            fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 2))

            def probe(args, report, profile, manifests, records, loaded):
                report.update(diagnostic_only=True, scientific_pilot_passed=False,
                    expected_case_ids=[profile['case_ids'][0]],
                    progress={'completed_passes': 9, 'expected_passes': 9},
                    diagnostic={'status': 'passed', 'synthetic_cpu_test': True,
                        'completion_meaning': 'Probe executed; not passed scientific parity'})
                write_json(args.output / 'parity-diagnostic.json', report['diagnostic'])

            fake_module = SimpleNamespace(run_diagnostic=probe)
            with patch.dict(sys.modules, {'torch': fake_torch, 'rq2_parity_diagnostic': fake_module}), \
                    patch.object(runner, 'provenance', return_value={'synthetic_cpu_test': True}), \
                    patch.object(runner, 'validate_runtime'), patch.object(runner, 'verify_native_reference'), \
                    patch.object(runner, 'validate_bundle', return_value=(copy.deepcopy(self.profile), {}, {}, [])), \
                    patch.object(runner, 'source_hashes', return_value={}), \
                    patch.object(runner, 'load_model', return_value=(None, None, None)), \
                    patch.object(runner, 'run_panel') as panel, patch.object(runner.signal, 'signal'), \
                    patch.dict(os.environ), contextlib.redirect_stdout(io.StringIO()):
                status = runner.main(['--inputs', str(inputs), '--output', str(output), '--diagnose-native-parity'])
            self.assertEqual(status, 0)
            panel.assert_not_called()
            report, _ = self.verify_zip(output)
            self.assertEqual(report['version'], 'rq2-native-parity-diagnostic-v1')
            self.assertTrue(report['diagnostic_only'])
            self.assertFalse(report['scientific_pilot_passed'])
            self.assertEqual(report['expected_case_ids'], [self.profile['case_ids'][0]])
            self.assertNotIn('summaries', report)
            self.assertIn('does not establish a passed RQ2 pilot', (output/'index.html').read_text())

    def test_actual_panel_preserves_partial_records_when_same_image_gate_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            output, report = self.run_two_question_gates(Path(directory), change_same_image_mapping=True)
            self.assertEqual(report["progress"]["completed_questions"], 1)
            self.assertEqual(len((output / "predictions.jsonl").read_text().splitlines()), 3)
            cid = self.profile["case_ids"][1]
            detail = json.loads((output / "questions" / (cid.replace(":", "-") + ".json")).read_text())
            self.assertEqual(detail["status"], "failed")
            self.assertEqual(len(detail["arms"]), 3)
            self.assertIn("Same-image compression changed", detail["error"])


if __name__ == "__main__":
    unittest.main()
