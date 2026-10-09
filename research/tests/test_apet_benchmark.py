"""Check paired benchmark scheduling and summaries without ML libraries."""

import sys
import unittest
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import benchmark_apet as benchmark
from llava_runtime import generation_options


class BenchmarkTests(unittest.TestCase):
    def test_plan_is_repeatable_complete_paired_and_rotates_order(self):
        plan = benchmark.trial_plan([4, 8], [32, 128], [576, 288, 96], 3, 590)
        self.assertEqual(plan, benchmark.trial_plan([4, 8], [32, 128], [576, 288, 96], 3, 590))
        keys = {(t["row"], t["output_tokens"], t["repeat"], t["visual_tokens"]) for t in plan}
        self.assertEqual(len(plan), 36)
        self.assertEqual(len(keys), 36)
        first_positions = Counter()
        for start in range(0, len(plan), 3):
            block = plan[start:start + 3]
            self.assertEqual(len({(t["row"], t["output_tokens"], t["repeat"]) for t in block}), 1)
            self.assertEqual({t["visual_tokens"] for t in block}, {576, 288, 96})
            first_positions[block[0]["visual_tokens"]] += 1
        self.assertEqual(first_positions, {576: 4, 288: 4, 96: 4})

    def test_partial_run_ratio_uses_only_matching_trials(self):
        def trial(index, budget, seconds):
            return {"index": index, "output_tokens": 32, "generated_tokens": 32,
                    "visual_tokens": budget, "repeat": 0, "generation_seconds": seconds,
                    "peak_allocated_gib": {"0": 7}}

        trials = [trial(0, 576, 2), trial(0, 96, 1), trial(1, 96, 1)]
        group = benchmark.summarize_trials(trials)["32"]["96"]
        self.assertEqual(group["requests"], 2)
        self.assertEqual(group["paired_requests"], 1)
        self.assertEqual(group["paired_generation_time_ratio_baseline_over_current"], 2)
        self.assertEqual(group["generated_tokens_per_generation_second"], 32)
        trials.append(trial(1, 576, 4))
        group = benchmark.summarize_trials(trials)["32"]["96"]
        self.assertEqual(group["paired_requests"], 2)
        self.assertEqual(group["paired_generation_time_ratio_baseline_over_current"], 3)

    def test_unequal_output_lengths_and_duplicate_trials_are_rejected(self):
        trial = {"index": 0, "output_tokens": 32, "generated_tokens": 2,
                 "visual_tokens": 96, "repeat": 0, "generation_seconds": 1,
                 "peak_allocated_gib": {"0": 7}}
        with self.assertRaisesRegex(ValueError, "fixed output length"):
            benchmark.summarize_trials([trial])
        trial["generated_tokens"] = 32
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            benchmark.summarize_trials([trial, trial])

    def test_accuracy_generation_keeps_early_stopping_and_benchmark_sets_minimum(self):
        self.assertEqual(generation_options(16),
                         {"max_new_tokens": 16, "do_sample": False, "use_cache": True})
        self.assertEqual(generation_options(128, 128)["min_new_tokens"], 128)
        for maximum, minimum in [(0, None), (32, -1), (32, 33)]:
            with self.subTest(maximum=maximum, minimum=minimum), self.assertRaises(ValueError):
                generation_options(maximum, minimum)


if __name__ == "__main__":
    unittest.main()
