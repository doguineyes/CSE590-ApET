"""CPU tests for the fixed RQ2 cohort and paired generated-answer evaluation."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from rq2_protocol import (ARMS, PROFILE, case_panel, load_profile, paired_summary,
                          score_response, source_hashes, validate_bundle)

BUNDLE = REPO / "research/results/rq1-ih-inputs-v1.zip"


class RQ2ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = load_profile()

    def changed_profile(self, mutate):
        value = copy.deepcopy(self.profile)
        mutate(value)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(value))
            return load_profile(path)

    def test_profile_freezes_only_three_arms_and_all_previous_cases_and_flags(self):
        self.assertEqual(list(ARMS), ["baseline576", "native-input288", "native-input288-no-merge"])
        self.assertEqual(len(self.profile["case_ids"]), 30)
        self.assertEqual(len({r["image_sha256"] for r in self.profile["cases"]}), 24)
        self.assertEqual(sum(r["flags"]["mmstar_grounding"] for r in self.profile["cases"]), 5)
        self.assertEqual(sum(r["flags"]["mmstar_prior_quality"] for r in self.profile["cases"]), 2)
        self.assertEqual([r["case_id"] for r in self.profile["cases"][-6:]],
                         ["textvqa:37533", "textvqa:35670", "textvqa:35962",
                          "textvqa:39323", "textvqa:36001", "textvqa:38025"])

    def test_protocol_rejects_decoder_pruning_budget_or_generation_change(self):
        changes = [
            lambda p: p["arms"]["native-input288-no-merge"].update(decoder_layers=[16]),
            lambda p: p["arms"]["native-input288-no-merge"].update(input_tokens=96),
            lambda p: p["arms"]["native-input288-no-merge"].update(basis_tokens=0),
            lambda p: p["arms"]["native-input288-no-merge"].update(merge_enabled=True),
            lambda p: p["arms"].update(random_merge={}),
            lambda p: p["generation"].update(do_sample=True),
            lambda p: p["cases"][13]["flags"].update(mmstar_prior_quality=False),
            lambda p: p["cases"][1].update(seed=1),
            lambda p: p["case_ids"].reverse(),
        ]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.changed_profile(change)

    def test_existing_scorers_do_not_credit_similar_but_wrong_literal_text(self):
        cases = {c["case_id"]: c for c in self.profile["cases"]}
        for cid, right, wrong in [("textvqa:37533", "Brolio", "Bottoli"),
                                  ("textvqa:35962", "Cookery school", "Culinary school"),
                                  ("textvqa:36001", "Give up", "Gone")]:
            with self.subTest(case=cid):
                self.assertEqual(score_response(cases[cid], right)["consensus_score"], 1)
                self.assertEqual(score_response(cases[cid], wrong)["consensus_score"], 0)
                self.assertFalse(score_response(cases[cid], wrong)["correct"])
        self.assertTrue(score_response(cases["gqa:201636988"], "NO.")["correct"])
        ambiguous = score_response(cases["mmstar:313"], "B or D")
        self.assertIsNone(ambiguous["prediction"])
        self.assertFalse(ambiguous["correct"])

    def test_paired_metrics_track_rescues_harms_and_partial_credit_separately(self):
        ids = self.profile["case_ids"][:2]
        rows = []
        for arm, outcomes in zip(ARMS, [(True, False), (False, True), (True, False)]):
            rows += [{"case_id": cid, "arm": arm, "scoring": {"correct": correct}}
                     for cid, correct in zip(ids, outcomes)]
        summary = paired_summary(rows)["gqa"]
        self.assertEqual(summary["baseline_correct"], 1)
        self.assertEqual(summary["arms"]["native-input288"]["APR"], 0)
        self.assertEqual(summary["arms"]["native-input288-no-merge"]["APR"], 1)
        self.assertEqual(summary["arms"]["native-input288"]["gains"], [ids[1]])
        self.assertEqual(summary["no_merge_vs_native"]["rescues"], [ids[0]])
        self.assertEqual(summary["no_merge_vs_native"]["new_losses"], [ids[1]])
        self.assertEqual(summary["no_merge_vs_native"]["mean_score_delta"], 0)
        ids = self.profile["case_ids"][-6:-4]
        rows = []
        for arm, scores in zip(ARMS, [(1., .6), (.6, 1.), (1., .3)]):
            rows += [{"case_id": cid, "arm": arm,
                      "scoring": {"correct": score == 1, "full_credit": score == 1,
                                  "consensus_score": score}}
                     for cid, score in zip(ids, scores)]
        summary = paired_summary(rows)["textvqa"]
        self.assertEqual(summary["arms"]["native-input288"]["accuracy"], .5)
        self.assertEqual(summary["arms"]["native-input288-no-merge"]["accuracy"], .5)
        self.assertAlmostEqual(summary["no_merge_vs_native"]["mean_score_delta"], -.15)

    def test_mismatched_pairs_repeats_and_bad_consensus_are_rejected(self):
        cid = self.profile["case_ids"][-1]
        rows = [{"case_id": cid, "arm": arm,
                 "scoring": {"correct": True, "full_credit": True, "consensus_score": 1}}
                for arm in ARMS]
        for bad in [rows[:-1], rows + [rows[0]],
                    [*rows[:-1], {**rows[-1], "case_id": self.profile["case_ids"][-2]}]]:
            with self.assertRaises(ValueError):
                paired_summary(bad)
        for score in [float("nan"), 1.1, -.1, .6]:
            bad = copy.deepcopy(rows)
            bad[-1]["scoring"]["consensus_score"] = score
            with self.assertRaises(ValueError):
                paired_summary(bad)

    def test_zero_baseline_correct_has_no_defined_preservation_rate(self):
        rows = [{"case_id": self.profile["case_ids"][0], "arm": arm,
                 "scoring": {"correct": arm != "baseline576"}} for arm in ARMS]
        out = paired_summary(rows)["gqa"]
        self.assertIsNone(out["arms"]["native-input288"]["APR"])
        self.assertEqual(out["arms"]["native-input288"]["gains"], [self.profile["case_ids"][0]])

    def test_import_and_protocol_validation_need_no_torch(self):
        result = subprocess.run([sys.executable, "-I", "-X", "utf8", "-c",
            "import sys;sys.path.insert(0,sys.argv[1]);import rq2_protocol;"
            "rq2_protocol.load_profile();assert 'torch' not in sys.modules;print('CPU protocol passed')",
            str(REPO / "scripts")], check=True, text=True, capture_output=True)
        self.assertIn("CPU protocol passed", result.stdout)
        hashes = source_hashes()
        self.assertIn("llava/model/llava_arch.py", hashes)
        self.assertIn("research/protocols/rq2_input_merge_v1.json", hashes)
        with self.assertRaises(ValueError):
            source_hashes(["../not-a-project-source.py"])


@unittest.skipUnless(BUNDLE.is_file(), "Private unchanged RQ1 IH input bundle unavailable")
class RQ2PrivateBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.inputs = Path(cls.temp.name) / "inputs"
        with zipfile.ZipFile(BUNDLE) as archive:
            archive.extractall(cls.inputs)
        cls.profile, cls.index, cls.manifests, cls.records = validate_bundle(cls.inputs)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_same_images_questions_gold_and_seeds_are_preserved(self):
        panel = case_panel(self.records)
        self.assertEqual(len(panel), 30)
        self.assertEqual(sum(len(r["reference"]["tokens"]) for r in panel), 17280)
        self.assertEqual(panel[24]["rq2"]["historical_predictions"]["input-only"]["response"], "Bottoli")
        self.assertEqual(panel[0]["rq2"]["seed"], panel[1]["rq2"]["seed"])
        self.assertEqual(case_panel(self.records, 2), panel[:2])
        for records in [self.records[:-1], list(reversed(self.records))]:
            with self.assertRaises(ValueError):
                case_panel(records)

    def test_mismatched_frozen_question_or_reference_seed_stops_preflight(self):
        for key, value in [("question", "New unrelated question"), ("gold_answers", ["other"])]:
            changed = copy.deepcopy(self.profile)
            changed["cases"][0][key] = value
            with patch("rq2_protocol.load_profile", return_value=changed), self.assertRaisesRegex(ValueError, "identity"):
                validate_bundle(self.inputs)
        changed = copy.deepcopy(self.profile)
        changed["cases"][0]["seed"] += 1
        with patch("rq2_protocol.load_profile", return_value=changed), self.assertRaisesRegex(ValueError, "seed"):
            validate_bundle(self.inputs)


if __name__ == "__main__":
    unittest.main()
