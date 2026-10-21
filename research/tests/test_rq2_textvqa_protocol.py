"""CPU checks of cohort identity, scoring denominators and paired random controls."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import rq2_textvqa_protocol as protocol
from rq2_random_protocol import CONDITIONS, random_selection
from rq2_protocol import score_response


def primary_rows(ids=None):
    ids = protocol.load_profile()["case_ids"] if ids is None else ids
    return [{"case_id": cid, "condition": condition, "response": "red",
             "scoring": {"correct": True, "full_credit": True, "consensus_score": 1., "prediction": "red"},
             "generated_tokens": 1}
            for cid in ids for condition in CONDITIONS]


class TextVQAProtocolTests(unittest.TestCase):
    def test_frozen100_includes_baseline_wrong_questions_and_94_case_complement(self):
        profile = protocol.load_profile()
        reference = protocol.load_reference()
        self.assertEqual(len(profile["case_ids"]), 100)
        full = [r for r in reference["cases"] if r["historical_predictions"]["baseline"]["scoring"]["correct"]]
        self.assertEqual(len(full), 44)
        self.assertEqual(len(set(profile["case_ids"]) - set(protocol.DIAGNOSTIC_IDS)), 94)
        self.assertEqual(list(profile["conditions"]), list(CONDITIONS))
        self.assertEqual(profile["decoder_layers"], [])

    def test_profile_rejects_a_filtered_cohort_or_changed_budget(self):
        profile = protocol.load_profile()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "profile.json"
            for key, value in (("case_ids", profile["case_ids"][:44]), ("budgets", [288, 96])):
                changed = copy.deepcopy(profile)
                changed[key] = value
                path.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, "frozen cohort"):
                    protocol.load_profile(path)

    def test_seed_means_do_not_triple_the_sample_size_and_keep_partial_credit(self):
        ids = protocol.load_profile()["case_ids"][:2]
        rows = primary_rows(ids)
        for row in rows:
            if row["case_id"] == ids[0] and row["condition"] == "pure-random-64-s101":
                row["response"] = "partial"
                row["scoring"].update(correct=False, full_credit=False, consensus_score=.6, prediction="partial")
        result = protocol.summarize(rows, expected_ids=ids)["all100"]
        self.assertEqual(result["questions"], 2)
        self.assertEqual(result["conditions"]["pure-random-64-s101"]["mean_consensus_score"], .8)
        self.assertEqual(result["conditions"]["pure-random-64-s101"]["APR"], .5)
        pair = result["paired_comparisons"]["pure-random-64-seed-mean minus native-64-merge"]
        self.assertEqual(pair["questions"], 2)
        self.assertAlmostEqual(pair["mean_score_delta"], -.4 / 6)
        self.assertEqual(result["conditions"]["pure-random-64-s101"]["full_credit_losses"], [ids[0]])

    def test_summary_keeps_six_case_complement_and_rejects_missing_duplicate_or_extra_rows(self):
        rows = primary_rows()
        result = protocol.summarize(rows, expected_ids=protocol.load_profile()["case_ids"])
        self.assertEqual(result["all100"]["questions"], 100)
        self.assertEqual(result["outside_previous_six"]["questions"], 94)
        self.assertEqual(result["previous_six"]["questions"], 6)
        for bad in (rows[:-1], rows + [rows[0]], [{**rows[0], "case_id": "textvqa:unknown"}]):
            with self.assertRaises(ValueError):
                protocol.summarize(bad)
        with self.assertRaisesRegex(ValueError, "incomplete or filtered"):
            protocol.summarize(primary_rows(protocol.load_profile()["case_ids"][:1]),
                               expected_ids=protocol.load_profile()["case_ids"])

    def test_zero_baseline_full_credit_yields_no_APR_and_no_division_error(self):
        rows = primary_rows(protocol.load_profile()["case_ids"][:1])
        for row in rows:
            if row["condition"] == "baseline576":
                row["scoring"].update(correct=False, full_credit=False, consensus_score=0.)
        result = protocol.summarize(rows)["all100"]
        self.assertIsNone(result["conditions"]["native-64-merge"]["APR"])
        self.assertIsNone(result["conditions"]["native-64-merge"]["paired_mean_score_ratio"])
        self.assertIsNone(result["random_seed_summary"]["64"]["APR"]["mean"])

    def test_historical_baseline_drift_is_recorded_without_accuracy_gate(self):
        reference = protocol.load_reference()
        case = reference["cases"][0]
        arms = {name: {"response": "unrelated", "scoring": score_response(case, "unrelated")}
                for name in ("baseline576", "native-288-merge")}
        result = protocol.historical_comparison(case, arms, reference)
        self.assertFalse(result["baseline576"]["response_equal"])
        self.assertIn("never filter", result["baseline576"]["policy"])

    def test_paired_intervals_reproducible_and_random_membership_preserves_calibration(self):
        ids = ["one", "two"]
        with patch.object(protocol, "BOOTSTRAP_REPLICATES", 50):
            first = protocol.paired_effect(dict.fromkeys(ids, .5), dict.fromkeys(ids, .25), ids, intervals=True)
            again = protocol.paired_effect(dict.fromkeys(ids, .5), dict.fromkeys(ids, .25), ids, intervals=True)
        self.assertEqual(first, again)
        self.assertEqual(first["paired_bootstrap_95_ci"], [.25, .25])
        old = random_selection("a" * 64, 101, 64)
        self.assertEqual(protocol.random_selection("a" * 64, 101, 64), old)


if __name__ == "__main__":
    unittest.main()
