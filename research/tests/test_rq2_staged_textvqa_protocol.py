"""Cohort/scoring/seed-denominator checks; no Torch or model downloads."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import rq2_staged_textvqa_protocol as p
from rq2_staged_protocol import CONDITIONS, random_plan
from rq2_protocol import score_response


def primary_rows(ids=None):
    ids = p.load_profile()["case_ids"] if ids is None else ids
    return [dict(case_id=cid, condition=condition, response="red", control_only=False, generated_tokens=2,
        scoring=dict(prediction="red", correct=True, full_credit=True, consensus_score=1.))
        for cid in ids for condition in CONDITIONS]


class StagedTextVQAProtocolTests(unittest.TestCase):
    def test_cohort_has100_unique_images_including56_historically_not_full_credit(self):
        profile = p.load_profile()
        old = p.load_reference()["cases"]
        self.assertEqual(len(profile["case_ids"]), 100)
        self.assertEqual(len({c["image_sha256"] for c in old}), 100)
        self.assertEqual(sum(c["historical_predictions"]["baseline"]["scoring"]["correct"] for c in old), 44)
        self.assertEqual(profile["schedules"], [[96, 32], [288, 96]])
        self.assertEqual(profile["layer"], 16)
        self.assertEqual(len(CONDITIONS), 9)

    def test_profile_rejects_filtering_or_changed_stage_counts(self):
        profile = p.load_profile()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "profile.json"
            for key, value in (("case_ids", profile["case_ids"][:44]), ("schedules", [[64, 32], [288, 96]])):
                bad = copy.deepcopy(profile); bad[key] = value
                path.write_text(json.dumps(bad))
                with self.assertRaisesRegex(ValueError, "Frozen staged"):
                    p.load_profile(path)

    def test_reviewed_calibration_rejects_operator_source_drift(self):
        self.assertEqual(p.load_calibration_reference()["progress"]["completed_primary_predictions"], 54)
        with patch("run_rq2_staged_calibration.measurement_hashes", return_value={"changed": "source"}):
            with self.assertRaisesRegex(ValueError, "calibrated operator sources changed"):
                p.load_calibration_reference()

    def test_partial_credit_and_seed_mean_keep_question_denominator(self):
        ids = p.load_profile()["case_ids"][:2]
        rows = primary_rows(ids)
        for row in rows:
            if row["case_id"] == ids[0] and row["condition"] == "pure-random-96-to-32-s101":
                row["scoring"].update(correct=False, full_credit=False, consensus_score=.4)
        s = p.summarize(rows)["all100"]
        arm = s["conditions"]["pure-random-96-to-32-s101"]
        self.assertEqual(arm["mean_consensus_score"], .7)
        self.assertEqual(arm["APR"], .5)
        delta = s["paired_comparisons"]["pure-random-96-to-32-seed-mean minus native-96-to-32"]
        self.assertEqual(delta["questions"], 2)
        self.assertAlmostEqual(delta["mean_score_delta"], -.1)

    def test_complete_groups_94_case_complement_and_control_exclusion(self):
        rows = primary_rows()
        s = p.summarize(rows, expected_ids=p.load_profile()["case_ids"])
        self.assertEqual([s[k]["questions"] for k in ("all100", "outside_previous_six", "previous_six")], [100, 94, 6])
        for bad in (rows[:-1], rows + [rows[0]], [{**rows[0], "control_only": True}],
                    [{**rows[0], "case_id": "textvqa:unknown"}]):
            with self.assertRaises(ValueError):
                p.summarize(bad)
        with self.assertRaisesRegex(ValueError, "incomplete or filtered"):
            p.summarize(primary_rows(p.load_profile()["case_ids"][:1]), expected_ids=p.load_profile()["case_ids"])

    def test_zero_correct_baseline_has_undefined_APR_but_counts_gains(self):
        rows = primary_rows(p.load_profile()["case_ids"][:1])
        rows[0]["scoring"].update(correct=False, full_credit=False, consensus_score=0.)
        s = p.summarize(rows)["all100"]
        self.assertIsNone(s["conditions"]["native-96-to-32"]["APR"])
        self.assertIsNone(s["conditions"]["native-96-to-32"]["paired_mean_score_ratio"])
        self.assertEqual(len(s["conditions"]["native-96-to-32"]["full_credit_gains"]), 1)
        self.assertIsNone(s["random_seed_summary"]["96-to-32"]["APR"]["mean"])

    def test_only_matched_baseline_compared_to_history_without_answer_gate(self):
        reference = p.load_reference(); case = reference["cases"][0]
        arms = {"baseline576": dict(response="unrelated", scoring=score_response(case, "unrelated"))}
        h = p.historical_comparison(case, arms, reference)
        self.assertEqual(set(h), {"baseline576"})
        self.assertIn("never filter", h["baseline576"]["policy"])

    def test_paired_intervals_and_random_streams_preserve_calibration(self):
        ids = p.load_profile()["case_ids"][:2]
        s = p.summarize(primary_rows(ids), intervals=True)["all100"]
        delta = s["paired_comparisons"]["pure-random-96-to-32-seed-mean minus native-96-to-32"]
        self.assertEqual(delta["paired_bootstrap_95_ci"], [0., 0.])
        plan = random_plan("a" * 64, 101, 96, 32)
        self.assertEqual(p.RANDOM_VERSION, plan["version"])
        self.assertEqual(len(plan["decoder_original_positions"]), 32)


if __name__ == "__main__":
    unittest.main()
