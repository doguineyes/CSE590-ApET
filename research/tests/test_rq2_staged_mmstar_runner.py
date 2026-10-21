"""Synthetic full execution, interruption, notebook and saved-ZIP validation."""
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
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import run_rq2_staged_mmstar1500 as runner
import summarize_rq2_staged_mmstar1500 as aggregate
from rq2_staged_mmstar_protocol import load_profile, load_reference, manifest_data, selected_ids, summarize
from rq2_staged_protocol import CONDITIONS, IDENTITY_CONDITION, specification
from rq2_staged_native import validate_cache
from rq2_protocol import score_response
from research.tests.test_rq2_staged_textvqa_runner import bank_fixture
from research.tests.test_rq2_staged_mmstar_protocol import primary_rows


def generation_fixture(case, condition, snapshot, seed, *, probe=False, replay=False):
    spec = specification(condition)
    count, expanded = (4 if probe else 2), spec["input_tokens"] + 10
    last = spec["decoder_tokens"] if spec["layer"] else spec["input_tokens"]
    events = []
    for block, length in ((1, expanded), (17, expanded - spec["input_tokens"] + last)):
        events.append(dict(block=block, sequence_tokens=length, position_first=0,
            position_last=length-1, cache_tokens_before=0))
        events += [dict(block=block, sequence_tokens=1, position_first=length+i,
            position_last=length+i, cache_tokens_before=length+i) for i in range(count-1)]
    return dict(case_id=case["case_id"], image_sha256=case["image_sha256"], condition=condition,
        **spec, seed=seed, response="synthetic unparsed answer", scoring=score_response(case, "synthetic unparsed answer"),
        input_encoding=snapshot["obs"], generation=dict(prompt=case["generation_question"],
            generated_tokens=count, expanded_prompt_tokens=expanded),
        generation_tokens=dict(generated_token_ids=list(range(count))),
        cache=validate_cache(events, spec, expanded, count),
        control_only=bool(probe or replay or condition == IDENTITY_CONDITION))


class MMStarStagedRunnerTests(unittest.TestCase):
    def test_full1500_keeps13500_primary_32_controls_and_distinct_reused_image_encodings(self):
        import numpy as np
        from PIL import Image
        data, profile = manifest_data(), load_profile()
        latest = [None, None]
        def capture(model, pixels, seed, condition, key):
            if key != latest[0]:
                latest[:] = [key, bank_fixture(key)]
            obs = copy.deepcopy(latest[1][condition])
            return obs, dict(obs=obs)
        def generate(tokenizer, model, processor, image, case, pixels, condition, snapshot, seed,
                     *, decoder_replay=None, probe=False):
            r = generation_fixture(case, condition, snapshot, seed, probe=probe, replay=decoder_replay is not None)
            return r, None if condition == "baseline576" else ("synthetic-layer", condition)
        def raw(tokenizer, model, processor, image, case, condition, seed):
            return dict(case_id=case["case_id"], condition=condition, response="synthetic unparsed answer",
                scoring=score_response(case, "synthetic unparsed answer"), generation_tokens=dict(generated_token_ids=[0, 1]))
        def save(args, report, key, *unused):
            name = "features/" + key + "-synthetic.pt"
            path = args.output / name
            path.write_text("synthetic fixture only; not real tensor data")
            report["feature_sha256"][name] = runner.sha256(path)
            return dict(file=name, sha256=runner.sha256(path))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "run"
            root.mkdir()
            args = SimpleNamespace(output=root, inputs=root)
            report = dict(status="passed", version=runner.VERSION)
            model = SimpleNamespace(model=SimpleNamespace(layers=[None]*32))
            original_summary = runner.summarize
            with patch.object(runner, "save_image", return_value=Image.new("RGB", (2, 2))), \
                 patch.object(runner, "prepare_pixels", return_value=(object(), np.zeros((2, 2, 3), dtype=np.uint8))), \
                 patch("rq2_staged_native.capture_input", side_effect=capture) as cap, \
                 patch("rq2_staged_native.generate_condition", side_effect=generate) as gen, \
                 patch("rq2_staged_native.generate_raw_native", side_effect=raw) as fresh, \
                 patch.object(runner, "save_input_bank", side_effect=save) as save_input, \
                 patch.object(runner, "save_decoder", side_effect=lambda a,r,c,n,s: save(a,r,c+"-"+n)) as save_layer, \
                 patch.object(runner, "summarize", side_effect=lambda rows, **kw: original_summary(rows, expected_ids=kw.get("expected_ids"))), \
                 contextlib.redirect_stdout(io.StringIO()):
                runner.run_panel(args, report, profile, data, load_reference(), (None, model, None))
                runner.finish_run(root, report, time.perf_counter())
            self.assertEqual(cap.call_count, 13506)
            self.assertEqual(gen.call_count, 13528)
            self.assertEqual(fresh.call_count, 4)
            self.assertEqual(save_input.call_count, 2)
            self.assertEqual(save_layer.call_count, 16)
            self.assertEqual(report["progress"]["completed_primary_predictions"], 13500)
            self.assertTrue(report["summaries"]["full_mmstar1500"])
            self.assertEqual(report["summaries"]["baseline_correct"], 0)
            self.assertEqual(len(report["encoding_sha256"]), 1500)  # Not 1430 overwritten files.
            self.assertEqual(len(report["feature_sha256"]), 18)
            with zipfile.ZipFile(root.with_suffix(".zip")) as z:
                self.assertIsNone(z.testzip())
                inventory = json.loads(z.read("artifact-sha256.json"))
                self.assertTrue(all(hashlib.sha256(z.read(n)).hexdigest() == h for n,h in inventory.items()))
                self.assertEqual(len(z.read("predictions.jsonl").splitlines()), 13500)
                self.assertEqual(len(list(csv.DictReader(io.StringIO(z.read("answers.csv").decode())))), 13500)

    def test_interrupted_export_preserves_partial_arms_and_only_summarizes_complete_questions(self):
        cases = manifest_data()["cases"]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "partial"
            (root / "questions").mkdir(parents=True)
            for i in range(2):
                rows = primary_rows([cases[i]["case_id"]])[:9 if i == 0 else 2]
                arms = {r["condition"]: {**r, **specification(r["condition"]), "seed":590+i} for r in rows}
                (root / "questions" / f"mmstar-{i}.json").write_text(json.dumps(dict(case_id=cases[i]["case_id"],
                    case=cases[i], arms=arms, status="passed" if i == 0 else "failed")))
            report = dict(status="interrupted", execution_passed=False)
            with contextlib.redirect_stdout(io.StringIO()):
                runner.finish_run(root, report, time.perf_counter())
            self.assertEqual(report["summaries"]["questions"], 1)
            self.assertFalse(report["summary_coverage"]["full_mmstar1500"])
            with (root / "answers.csv").open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 11)

    def test_runtime_failure_preserved_before_model_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(runner, "provenance", return_value={}), \
                 patch.object(runner, "validate_runtime", side_effect=RuntimeError("synthetic runtime failure")), \
                 patch.object(runner, "load_model") as load, patch.object(runner.signal, "signal"), \
                 patch.dict(os.environ), contextlib.redirect_stdout(io.StringIO()):
                status = runner.main(["--inputs", str(root/"inputs"), "--output", str(root/"failed"),
                    "--run-mmstar", "--setup-log", str(root/"setup"), "--runtime-report", str(root/"runtime"),
                    "--prerequisite-report", str(root/"proof")])
            load.assert_not_called()
            self.assertEqual(status, 1)
            with zipfile.ZipFile(root / "failed.zip") as z:
                self.assertEqual(json.loads(z.read("report.json"))["status"], "failed")

    def test_proof_bound_to_current_sources_runtime_and_zero_skips(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = SimpleNamespace(setup_log=root/"setup", runtime_report=root/"runtime", prerequisite_report=root/"proof")
            args.setup_log.write_text("tests")
            args.runtime_report.write_text("runtime")
            proof = dict(version=runner.VERSION, status="passed", tests_run=40, skipped=0,
                test_modules=list(runner.TESTS), source_sha256={"code":"same"},
                setup_log_sha256=runner.sha256(args.setup_log), runtime_report_sha256=runner.sha256(args.runtime_report))
            args.prerequisite_report.write_text(json.dumps(proof))
            with patch.object(runner, "prerequisite_signature", return_value={"code":"same"}):
                self.assertEqual(runner.verify_prerequisites(args)["status"], "passed")
                args.runtime_report.write_text("changed")
                with self.assertRaisesRegex(ValueError, "changed since prerequisites"):
                    runner.verify_prerequisites(args)
                proof["skipped"] = 1
                args.prerequisite_report.write_text(json.dumps(proof))
                with self.assertRaisesRegex(ValueError, "zero skips"):
                    runner.verify_prerequisites(args)

    def test_notebook_idle_compiles_and_requires_prerequisites(self):
        nb = json.loads((REPO/"research/notebooks/kaggle_rq2_staged_mmstar1500.ipynb").read_text())
        code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
        for i,c in enumerate(code):
            compile(c, f"cell-{i}", "exec")
            self.assertNotIn("mmstar800", c)
        self.assertIn("MMSTAR_SHARDS = [0, 1, 2, 3, 4, 5]", code[0])
        with patch("subprocess.run") as run:
            with self.assertRaisesRegex(RuntimeError, "RUN_STAGED_MMSTAR"):
                exec(code[0], {})
        run.assert_not_called()
        helper = next(c for c in code if c.startswith("def verify_rq2_staged_mmstar_prerequisites("))
        namespace = {"rq2_staged_mmstar_prerequisites_passed": True}
        exec(helper, namespace)
        self.assertFalse(namespace["rq2_staged_mmstar_prerequisites_passed"])
        with self.assertRaisesRegex(RuntimeError, "prerequisites"):
            namespace["run_rq2_staged_mmstar1500"]()

    def test_combiner_disjoint_parts_keep1500_denominator_and_reject_overlap_or_runtime_drift(self):
        profile = load_profile()
        def read(path):
            rows = primary_rows(selected_ids(profile, [int(str(path))]))
            return ([{**r, **specification(r["condition"]), "seed":590} for r in rows], {"code":"same"}, {"part":str(path)})
        with tempfile.TemporaryDirectory() as folder, patch.object(aggregate, "read_archive", side_effect=read):
            root = Path(folder)
            with patch.object(aggregate, "summarize", side_effect=lambda r, **kw: summarize(r)):
                result = aggregate.combine(list(range(6)), root/"all")
                self.assertEqual(result["status"], "complete")
                self.assertEqual(result["summaries"]["questions"], 1500)
                self.assertEqual(result["summaries"]["images"], 1430)
                self.assertEqual(aggregate.combine([0], root/"part")["status"], "partial-coverage")
                with self.assertRaisesRegex(ValueError, "Overlapping"):
                    aggregate.combine([0, 0], root/"duplicate")
                with patch.object(aggregate, "read_archive", side_effect=[read(0), (read(1)[0], {"code":"changed"}, {})]):
                    with self.assertRaisesRegex(ValueError, "different measurement"):
                        aggregate.combine([0,1], root/"changed")

    def test_saved_zip_reader_verifies_sources_controls_cache_and_raw_scores(self):
        # Small self-contained archive fixture: no local result ZIP/model needed.
        profile, data = load_profile(), manifest_data()
        ids = ["mmstar:0", "mmstar:136", "mmstar:617"]
        cases = {c["case_id"]: c for c in data["cases"] if c["case_id"] in ids}
        encode = lambda v: json.dumps(v).encode()
        digest = lambda b: hashlib.sha256(b).hexdigest()
        files = {"evidence/notebook/setup.log": b"synthetic setup", "evidence/notebook/runtime.json": b"{}"}
        measurement, signature = runner.measurement_hashes(), runner.prerequisite_signature()
        for name in signature:
            files["evidence/sources/" + name] = (REPO/name).read_bytes()
        proof = dict(version=runner.VERSION, status="passed", skipped=0, tests_run=40,
            test_modules=list(runner.TESTS), source_sha256=signature,
            setup_log_sha256=digest(files["evidence/notebook/setup.log"]),
            runtime_report_sha256=digest(files["evidence/notebook/runtime.json"]))
        files["evidence/notebook/prerequisites.json"] = encode(proof)
        rows = []
        for cid, c in cases.items():
            blob = ("synthetic image " + cid).encode()
            key = digest(blob)
            c["image_sha256"] = key
            files["images/"+key+".jpg"] = blob
            bank = bank_fixture(key)
            names = (*CONDITIONS, IDENTITY_CONDITION) if cid in runner.DIAGNOSTIC_IDS else tuple(CONDITIONS)
            bank = {n: bank[n] for n in names}
            checks = runner.validate_bank(bank, key)
            encoding_path = "encodings/" + cid.replace(":", "-") + ".json"
            files[encoding_path] = encode(dict(encodings=bank, controls=checks))
            ref = dict(file=encoding_path, sha256=digest(files[encoding_path]))
            arms = {}
            def generate(name, *, probe=False, replay=False):
                r = generation_fixture(c, name, dict(obs=bank[name]), c["compression_seed"], probe=probe, replay=replay)
                return runner.compact_row(r, ref)
            for name in CONDITIONS:
                r = generate(name)
                r.update(index=c["index"], image_id=c["image_id"], category=c["category"], l2_category=c["l2_category"])
                arms[name] = r
                rows.append(r)
            controls = dict(input_bank=checks)
            if cid in runner.DIAGNOSTIC_IDS:
                controls["identity576"] = runner.compare_replay(arms["baseline576"], generate(IDENTITY_CONDITION))
            if cid in runner.REPEAT_IDS:
                controls["fixed_replays"] = {n: runner.compare_replay(arms[n], generate(n, replay=True)) for n in CONDITIONS}
            if cid == runner.PROBE_CASE:
                controls["cached_decoding_probes"] = [generate(n, probe=True) for n in runner.PROBE_CONDITIONS]
            if cid == runner.FRESH_CASE:
                controls["fresh_raw_native"] = [dict(row=generate(n), repeat=i+1) for n in ("native-96-to-32", "native-288-to-96") for i in range(2)]
                for v in controls["fresh_raw_native"]:
                    v["row"]["control_only"] = True
            files["questions/"+cid.replace(":", "-")+".json"] = encode(dict(case=c, status="passed", arms=arms,
                encoding_reference=ref, controls=controls))
        # Patch only synthetic input identities/coverage; exercise real ZIP/source/control reader.
        fixture = {**data, "cases": list(cases.values())}
        files["evidence/inputs/mmstar/manifest.json"] = encode(fixture)
        files["predictions.jsonl"] = b"".join(encode(r)+b"\n" for r in rows)
        native = json.loads((REPO/"research/provenance/original_llava_reference.json").read_text())
        report = dict(version=runner.VERSION, status="passed", execution_passed=True, profile=profile,
            profile_sha256=runner.sha256(runner.PROFILE), conditions=CONDITIONS, shards=[0], expected_case_ids=ids,
            progress=dict(completed_primary_predictions=len(ids)*9), questions=[dict(case_id=c,status="passed") for c in ids],
            feature_sha256={}, encoding_sha256={n:digest(b) for n,b in files.items() if n.startswith("encodings/")},
            question_sha256={n:digest(b) for n,b in files.items() if n.startswith("questions/")},
            evidence_sha256={n[len("evidence/"):]:digest(b) for n,b in files.items() if n.startswith("evidence/")},
            measurement_sources_sha256=measurement, prerequisite=proof, implementation=dict(native_matches_reference=True),
            source_sha256=native["native_source_sha256"], lock_sha256=measurement["environments/original/uv.lock"],
            checkpoints={}, packages={"fixture":"no inference"}, commit="synthetic")
        files["report.json"] = encode(report)
        def write(path, content):
            content = {**content, "artifact-sha256.json":encode({n:digest(b) for n,b in content.items()})}
            with zipfile.ZipFile(path,"x",zipfile.ZIP_DEFLATED) as z:
                for n,b in content.items(): z.writestr(n,b)
        with tempfile.TemporaryDirectory() as folder, patch.object(aggregate,"manifest_data",return_value=fixture), \
             patch.object(aggregate,"selected_ids",return_value=ids):
            root = Path(folder)
            write(root/"valid.zip",files)
            admitted,_,_ = aggregate.read_archive(root/"valid.zip")
            self.assertEqual(len(admitted),27)
            bad = copy.deepcopy(rows)
            bad[0]["scoring"]["correct"] = True
            write(root/"bad-score.zip",{**files,"predictions.jsonl":b"".join(encode(r)+b"\n" for r in bad)})
            with self.assertRaisesRegex(ValueError,"Prediction identity/scoring"):
                aggregate.read_archive(root/"bad-score.zip")
            bad = copy.deepcopy(rows)
            bad[0]["cache"]["events"][0]["sequence_tokens"] = -1
            qpath = "questions/mmstar-0.json"
            q = json.loads(files[qpath]); q["arms"]["baseline576"] = bad[0]
            changed = {**files,qpath:encode(q),"predictions.jsonl":b"".join(encode(r)+b"\n" for r in bad)}
            r = copy.deepcopy(report)
            r["question_sha256"][qpath] = digest(changed[qpath])
            changed["report.json"] = encode(r)
            write(root/"bad-cache.zip",changed)
            with self.assertRaisesRegex(ValueError,"Prefill/cache"):
                aggregate.read_archive(root/"bad-cache.zip")


if __name__ == "__main__":
    unittest.main()
