"""CPU checks for nested, E-free random retention and matched summaries."""
import copy
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from rq2_random_protocol import (PROFILE, CONDITIONS, CASE_IDS, RANDOM_SEEDS,
    load_profile, random_selection, specification, summarize)


class RandomProtocolTests(unittest.TestCase):
    def test_profile_is_separate_and_covers_both_budgets_all_datasets(self):
        p = load_profile()
        self.assertEqual(len(CONDITIONS), 11)
        self.assertEqual(p["budgets"], [288, 64])
        self.assertEqual(len(CASE_IDS), 6)
        self.assertEqual({c.split(":")[0] for c in CASE_IDS}, {"gqa", "mmstar", "textvqa"})
        self.assertEqual(specification("pure-random-576-s101")["budget"], 576)
        with self.assertRaises(ValueError):
            specification("random-64-with-FPS")

    def test_exact_counts_nested_budgets_spatial_order_and_isolated_rng(self):
        state = random.getstate()
        selections = []
        for seed in RANDOM_SEEDS:
            small, metadata = random_selection("a" * 64, seed, 64)
            large, _ = random_selection("a" * 64, seed, 288)
            full, _ = random_selection("a" * 64, seed, 576)
            self.assertEqual(len(set(small)), 64)
            self.assertEqual(len(set(large)), 288)
            self.assertTrue(set(small) <= set(large))
            self.assertEqual(small, sorted(small))
            self.assertEqual(full, list(range(576)))
            self.assertEqual(small, random_selection("a" * 64, seed, 64)[0])
            self.assertEqual(metadata["effective_basis_tokens"], 0)
            selections.append(small)
        self.assertEqual(state, random.getstate())
        self.assertNotEqual(selections[0], selections[1])
        self.assertNotEqual(selections[0], random_selection("b" * 64, 101, 64)[0])

    def test_invalid_inputs_and_changed_budget_profile_are_rejected(self):
        for image, seed, budget in (("bad", 101, 64), ("a" * 64, 999, 64),
                                    ("a" * 64, True, 64), ("a" * 64, 101, 32)):
            with self.assertRaises(ValueError):
                random_selection(image, seed, budget)
        data = json.loads(PROFILE.read_text())
        data["budgets"] = [288]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.json"
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                load_profile(path)

    def test_summaries_do_not_pool_seeds_or_repeat_controls_as_questions(self):
        rows = []
        for condition in CONDITIONS:
            for i, cid in enumerate(CASE_IDS[:2]):
                correct = condition == "baseline576" or i == 0
                rows.append({"case_id": cid, "condition": condition, "scoring": {"correct": correct}})
        result = summarize(rows)["gqa"]
        self.assertEqual(result["unique_questions"], 2)
        self.assertEqual(result["random_seed_summary"]["64"]["accuracy"]["seed_values"], [.5, .5, .5])
        with self.assertRaises(ValueError):
            summarize(rows + rows[:1])
        with self.assertRaises(ValueError):
            summarize(rows[:-1])


if __name__ == "__main__":
    unittest.main()
