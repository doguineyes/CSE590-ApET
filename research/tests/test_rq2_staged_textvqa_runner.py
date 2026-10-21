"""CPU execution/export checks with synthetic adapters, not pretrained results."""
import contextlib
import copy
import csv
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import run_rq2_staged_textvqa100 as runner
from rq2_staged_protocol import CONDITIONS, IDENTITY_CONDITION, specification, random_plan
from rq2_staged_textvqa_protocol import load_profile, load_reference
from rq2_protocol import score_response


def bank_fixture(key="a" * 64):
    source = dict(shape=[1, 577, 1024], sha256="synthetic-source")
    full = dict(shape=[1, 576, 4096], sha256="synthetic-full")
    rows = {}
    for condition in (*CONDITIONS, IDENTITY_CONDITION):
        spec = specification(condition); first = spec["input_tokens"]
        selected = list(range(first))
        obs = dict(source_clip=source, source_patches=source,
            encoded=dict(kept_tokens=first, selected_indices_sorted=selected, features=full))
        if spec["method"] == "native":
            obs.update(native_merge_executed=True, merge_calls=1,
                input=dict(errors=list(range(576)), basis_indices=list(range(10))))
        elif spec["method"] == "pure-random":
            plan = random_plan(key, spec["random_seed"], first, spec["decoder_tokens"])
            obs.update(full_projected=full, random_plan=plan,
                pure_input=dict(fps=False, E=False, basis_replacement=False, merging=False))
            obs["encoded"]["selected_indices_sorted"] = plan["input_original_positions"]
        rows[condition] = obs
    return rows


class StagedTextVQARunnerTests(unittest.TestCase):
    def test_bank_rejects_source_budget_random_compression_and_mask_drift(self):
        bank = bank_fixture()
        self.assertEqual(runner.validate_bank(bank, "a" * 64)["status"], "passed")
        for field in ("source", "budget", "compression", "mask"):
            bad = copy.deepcopy(bank)
            name = "pure-random-96-to-32-s101"
            if field == "source":
                bad[name]["source_clip"] = {"sha256": "changed"}
            elif field == "budget":
                bad[name]["encoded"]["kept_tokens"] = 288
            elif field == "compression":
                bad[name]["pure_input"]["E"] = True
            else:
                bad[name]["encoded"]["selected_indices_sorted"][0] = -1
            with self.assertRaises(ValueError):
                runner.validate_bank(bad, "a" * 64)

    def test_all100_baseline_wrong_cases_run_900_arms_32_controls_and_compact_export(self):
        import numpy as np
        from PIL import Image
        cases = copy.deepcopy(load_reference()["cases"])
        for case in cases:
            case["compression_seed"] = None  # Present in the accepted input manifest.
        data = {"cases": cases}
        profile = load_profile()
        def capture(model, pixels, seed, condition, key):
            obs = copy.deepcopy(bank_fixture(key)[condition])
            return obs, {"obs": obs}
        def generate(tokenizer, model, processor, image, case, pixels, condition, snapshot, seed,
                     *, decoder_replay=None, probe=False):
            count = 4 if probe else 2
            row = dict(case_id=case["case_id"], image_sha256=case["image_sha256"], condition=condition,
                **specification(condition), seed=seed, response="synthetic wrong",
                scoring=score_response(case, "synthetic wrong"), input_encoding=snapshot["obs"],
                generation=dict(prompt="same synthetic prompt", generated_tokens=count),
                generation_tokens=dict(generated_token_ids=list(range(count))),
                control_only=bool(probe or decoder_replay is not None or condition==IDENTITY_CONDITION))
            return row, None if condition == "baseline576" else ("synthetic-layer", condition)
        def raw(tokenizer, model, processor, image, case, condition, seed):
            return dict(case_id=case["case_id"], condition=condition, response="synthetic wrong",
                generation_tokens=dict(generated_token_ids=[0, 1]), scoring=score_response(case, "synthetic wrong"))
        def save(args, report, key, *unused):
            name = "features/" + key + "-synthetic.pt"
            # Bytes are explicitly synthetic; this test never deserializes tensors.
            path = args.output / name
            path.write_text("synthetic fixture only")
            report["feature_sha256"][name] = runner.sha256(path)
            return dict(file=name, sha256=runner.sha256(path))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = SimpleNamespace(output=root, inputs=root)
            report = dict(version=runner.VERSION, status="passed")
            model = SimpleNamespace(model=SimpleNamespace(layers=[None]*32))
            with patch.object(runner, "save_image", return_value=Image.new("RGB", (2, 2))), \
                 patch.object(runner, "prepare_pixels", return_value=(object(), np.zeros((2, 2, 3), dtype=np.uint8))), \
                 patch("rq2_staged_native.capture_input", side_effect=capture) as cap, \
                 patch("rq2_staged_native.generate_condition", side_effect=generate) as gen, \
                 patch("rq2_staged_native.generate_raw_native", side_effect=raw) as fresh, \
                 patch.object(runner, "save_input_bank", side_effect=save) as save_input, \
                 patch.object(runner, "save_decoder", side_effect=lambda a,r,c,condition,s: save(a,r,c+"-"+condition)) as save_layer, \
                 contextlib.redirect_stdout(io.StringIO()):
                runner.run_panel(args, report, profile, data, load_reference(), (None, model, None))
                runner.finish_run(root, report, time.perf_counter())
            self.assertEqual(cap.call_count, 906)
            self.assertEqual(gen.call_count, 928)
            self.assertEqual(fresh.call_count, 4)
            self.assertEqual(save_input.call_count, 2)
            self.assertEqual(save_layer.call_count, 16)
            self.assertEqual(report["progress"]["completed_questions"], 100)
            self.assertEqual(report["progress"]["completed_primary_predictions"], 900)
            self.assertEqual(report["summaries"]["all100"]["baseline_full_credit"], 0)
            self.assertEqual(report["summaries"]["outside_previous_six"]["questions"], 94)
            self.assertEqual(len(report["encoding_sha256"]), 100)
            self.assertEqual(len(report["feature_sha256"]), 18)
            predictions = [json.loads(line) for line in (root / "predictions.jsonl").read_text().splitlines()]
            self.assertEqual(len(predictions), 900)
            self.assertTrue(all("input_encoding" not in row and "encoding_reference" in row for row in predictions))
            with zipfile.ZipFile(root.with_suffix(".zip")) as z:
                self.assertIsNone(z.testzip())
                hashes = json.loads(z.read("artifact-sha256.json"))
                import hashlib
                self.assertTrue(all(hashlib.sha256(z.read(name)).hexdigest() == digest for name,digest in hashes.items()))
                self.assertEqual(len(list(csv.DictReader(io.StringIO(z.read("answers.csv").decode())))), 900)
                self.assertIn("94 questions", z.read("index.html").decode())
                self.assertEqual(len(json.loads(z.read("historical-comparison.json"))), 100)
            # The exported ZIP sits next to the TemporaryDirectory.
            root.with_suffix(".zip").unlink()

    def test_partial_export_keeps_failed_arms_but_only_complete_blocks_in_summary(self):
        from research.tests.test_rq2_staged_textvqa_protocol import primary_rows
        reference = load_reference()["cases"]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "partial"; (root / "questions").mkdir(parents=True)
            for index, status in ((0,"passed"), (1,"failed")):
                case = reference[index]
                rows = primary_rows([case["case_id"]])[:9 if status == "passed" else 2]
                arms = {r["condition"]: {**r, **specification(r["condition"]), "seed":590} for r in rows}
                (root/"questions"/(case["case_id"].replace(":","-")+".json")).write_text(json.dumps(dict(case_id=case["case_id"], case=case, status=status, arms=arms)))
            report = dict(status="interrupted", execution_passed=False)
            with contextlib.redirect_stdout(io.StringIO()):
                runner.finish_run(root, report, time.perf_counter())
            self.assertEqual(report["summaries"]["all100"]["questions"], 1)
            self.assertFalse(report["execution_passed"])
            with (root/"answers.csv").open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 11)

    def test_stale_proof_or_skipped_tests_fail_before_pretrained_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = SimpleNamespace(prerequisite_report=root/"proof", setup_log=root/"setup", runtime_report=root/"runtime")
            args.setup_log.write_text("tests"); args.runtime_report.write_text("runtime")
            proof = dict(version=runner.VERSION, status="passed", test_modules=list(runner.TESTS), tests_run=40,
                skipped=0, source_sha256={"code":"same"}, setup_log_sha256=runner.sha256(args.setup_log),
                runtime_report_sha256=runner.sha256(args.runtime_report))
            args.prerequisite_report.write_text(json.dumps(proof))
            with patch.object(runner, "prerequisite_signature", return_value={"code":"same"}):
                self.assertEqual(runner.verify_prerequisites(args)["status"], "passed")
                args.runtime_report.write_text("changed")
                with self.assertRaisesRegex(ValueError, "changed since prerequisites"):
                    runner.verify_prerequisites(args)
                proof["skipped"] = 1; args.prerequisite_report.write_text(json.dumps(proof))
                with self.assertRaisesRegex(ValueError, "zero skips"):
                    runner.verify_prerequisites(args)

    def test_failed_runtime_preserves_archive_and_does_not_load_model(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); out=root/"failed"
            with patch.object(runner, "provenance", return_value={}), \
                 patch.object(runner, "validate_runtime", side_effect=RuntimeError("synthetic runtime failure")), \
                 patch.object(runner, "load_model") as load, patch.object(runner.signal, "signal"), \
                 patch.dict(os.environ), contextlib.redirect_stdout(io.StringIO()):
                status=runner.main(["--inputs",str(root/"inputs"),"--output",str(out),"--run-textvqa100",
                    "--setup-log",str(root/"setup"),"--runtime-report",str(root/"runtime"),"--prerequisite-report",str(root/"proof")])
            load.assert_not_called(); self.assertEqual(status,1)
            with zipfile.ZipFile(out.with_suffix(".zip")) as z:
                self.assertIsNone(z.testzip())
                self.assertEqual(json.loads(z.read("report.json"))["status"],"failed")

    def test_notebook_is_idle_until_enabled_and_helper_invalidates_old_proof(self):
        nb=json.loads((REPO/"research/notebooks/kaggle_rq2_staged_textvqa100.ipynb").read_text())
        code=["".join(c["source"]) for c in nb["cells"] if c["cell_type"]=="code"]
        for i,cell in enumerate(code):compile(cell,f"cell-{i}","exec")
        with patch("subprocess.run") as run:
            with self.assertRaisesRegex(RuntimeError,"RUN_STAGED_TEXTVQA100"):
                exec(code[0],{})
        run.assert_not_called()
        helper=next(c for c in code if c.startswith("def verify_rq2_staged_textvqa_prerequisites("))
        namespace={"rq2_staged_textvqa_prerequisites_passed":True}
        exec(helper,namespace)
        self.assertFalse(namespace["rq2_staged_textvqa_prerequisites_passed"])
        with self.assertRaisesRegex(RuntimeError,"prerequisites"):
            namespace["run_rq2_staged_textvqa100"]()


if __name__ == "__main__":
    unittest.main()
