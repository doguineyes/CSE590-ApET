"""CPU policy checks for the matched staged calibration."""
import copy
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import rq2_staged_protocol as p


class StagedProtocolTests(unittest.TestCase):
    def test_profile_has_two_matched_schedules_and_clean_soccer_control(self):
        profile = p.load_profile()
        self.assertEqual(len(profile["conditions"]), 9)
        self.assertEqual(profile["schedules"], [[96, 32], [288, 96]])
        self.assertEqual(profile["layer"], 16)
        self.assertIn("mmstar:136", profile["case_ids"])
        self.assertNotIn("mmstar:1014", profile["case_ids"])
        self.assertTrue(all(not any(c["flags"].values()) for c in profile["cases"]))

    def test_two_stage_samples_are_unique_spatial_nested_and_answer_independent(self):
        state = random.getstate()
        for seed in p.RANDOM_SEEDS:
            small = p.random_plan("a" * 64, seed, 96, 32)
            large = p.random_plan("a" * 64, seed, 288, 96)
            self.assertTrue(set(small["input_original_positions"]) <= set(large["input_original_positions"]))
            for plan, first, last in ((small, 96, 32), (large, 288, 96)):
                self.assertEqual(len(set(plan["input_original_positions"])), first)
                self.assertEqual(len(set(plan["decoder_compact_positions"])), last)
                self.assertEqual(plan["decoder_original_positions"],
                                 [plan["input_original_positions"][i] for i in plan["decoder_compact_positions"]])
                self.assertNotEqual(plan["input_seed_sha256"], plan["decoder_seed_sha256"])
            self.assertEqual(small, p.random_plan("a" * 64, seed, 96, 32))
        self.assertEqual(state, random.getstate())
        identity = p.random_plan("a" * 64, 101, 576, 576)
        self.assertEqual(identity["decoder_original_positions"], list(range(576)))

    def test_invalid_random_inputs_and_profile_changes_fail(self):
        for arguments in (("bad", 101, 96, 32), ("a" * 64, 999, 96, 32), ("a" * 64, 101, 64, 64)):
            with self.assertRaises(ValueError):
                p.random_plan(*arguments)
        changed = copy.deepcopy(p.load_profile())
        changed["layer"] = 15
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "changed.json"
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "protocol/cohort"):
                p.load_profile(path)

    def test_summaries_use_questions_not_seeds_and_keep_text_partial_credit(self):
        rows = [{"case_id": cid, "condition": name,
                 "scoring": {"correct": True, "full_credit": True, "consensus_score": 1.}}
                for cid in p.CASE_IDS for name in p.CONDITIONS]
        altered = next(r for r in rows if r["case_id"] == "textvqa:37533" and r["condition"] == "pure-random-96-to-32-s101")
        altered["scoring"].update(correct=False, full_credit=False, consensus_score=.6)
        summary = p.summarize(rows)
        self.assertEqual(summary["textvqa"]["questions"], 3)
        arm = summary["textvqa"]["conditions"]["pure-random-96-to-32-s101"]
        self.assertAlmostEqual(arm["mean_score"], 2.6 / 3)
        self.assertAlmostEqual(arm["APR"], 2 / 3)
        with self.assertRaises(ValueError):
            p.summarize(rows + [rows[0]])
        with self.assertRaises(ValueError):
            p.summarize(rows[:-1])


if __name__ == "__main__":
    unittest.main()
