"""CPU protocol checks; no model packages or GPU required."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from analyze_rq1_input import quadrant_examples
from rq1_protocol import (GQA_IDS, REPO, SOURCES, consensus_score, load_manifest,
    pilot_cases, spearman, textvqa_evaluator, textvqa_summary, token_order)


class ProtocolTests(unittest.TestCase):
    def test_spearman_ties_and_constant(self):
        self.assertAlmostEqual(spearman([1, 1, 3, 4], [9, 9, 2, 0]), -1)
        self.assertIsNone(spearman([1, 1, 1], [1, 2, 3]))
        self.assertIsNone(spearman([1, 2], [1, 2]))
        with self.assertRaises(ValueError):
            spearman([1, 2, float("nan")], [1, 2, 3])

    def test_token_sampling_nested_and_image_shared(self):
        order = token_order("image-sha", 590)
        self.assertEqual(sorted(order), list(range(576)))
        self.assertEqual(order[:32], token_order("image-sha", 590)[:32])
        self.assertNotEqual(order, token_order("other-sha", 590))

    def test_actual_accepted_manifests_and_same_image_pairs(self):
        results = REPO / "research/results"
        if not all((results / rel).exists() for rel, _ in SOURCES.values()):
            self.skipTest("Private accepted input artifacts are not present; attach bundle for runtime hash checks")
        gqa = load_manifest(results / SOURCES["gqa"][0])
        cases = pilot_cases(gqa)
        self.assertEqual(len(cases), 12)
        self.assertEqual([c["question_id"] for c in cases], GQA_IDS)
        self.assertEqual(len({c["image_id"] for c in cases}), 6)
        for a, b in zip(cases[::2], cases[1::2]):
            self.assertEqual(a["image_id"], b["image_id"])
            self.assertEqual(a["compression_seed"], b["compression_seed"])
        self.assertNotIn("n497658", {c["image_id"] for c in cases})
        mm = load_manifest(results / SOURCES["mmstar"][0])
        self.assertTrue(all(c["predictions"]["baseline"]["correct"] for c in pilot_cases(mm)))
        text = load_manifest(results / SOURCES["textvqa"][0])
        self.assertEqual(len(text["cases"]), 100)
        with self.assertRaises(ValueError):
            pilot_cases(text)

    def test_changed_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps({"dataset": "gqa", "cases": []}))
            with self.assertRaises(ValueError):
                load_manifest(path)

    def test_native_textvqa_consensus_and_normalization(self):
        evaluator = textvqa_evaluator()
        for count, expected in [(1, .3), (2, .6), (3, .9), (4, 1.0)]:
            answers = ["two"] * count + ["other"] * (10-count)
            actual = consensus_score(evaluator, "The 2.", answers)
            self.assertAlmostEqual(actual["consensus_score"], expected)
            self.assertEqual(actual["full_credit"], count >= 4)
            self.assertEqual(consensus_score(evaluator, "", answers)["consensus_score"], 0)
        with self.assertRaises(ValueError):
            consensus_score(evaluator, "two", ["two"])

    def test_consensus_denominator_and_partial_pairing(self):
        rows = [dict(case_id="a", configuration="baseline", consensus_score=1, full_credit=True),
                dict(case_id="b", configuration="baseline", consensus_score=.6, full_credit=False),
                dict(case_id="a", configuration="input-only", consensus_score=.9, full_credit=False)]
        result = textvqa_summary(rows)["input-only"]
        self.assertEqual(result["paired_n"], 1)
        self.assertEqual(result["baseline_full_credit_denominator"], 1)
        self.assertEqual(result["full_credit_preservation"], 0)
        self.assertAlmostEqual(result["paired_mean_score_ratio"], .9)

    def test_negative_sensitivity_not_called_high_loss(self):
        rows = [dict(token_index=i, reconstruction_error=i, intervention_score=-i-1) for i in range(32)]
        exemplars = quadrant_examples(rows)["examples"]
        self.assertEqual(exemplars["low_E_high_I"], [])
        self.assertEqual(exemplars["high_E_high_I"], [])

    def test_partial_analysis_retains_unmeasured_nulls(self):
        from analyze_rq1_input import analyze
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "questions").mkdir()
            (folder / "features").mkdir()
            errors = list(range(576))
            case = {"case_id": "gqa:test", "question_id": "1", "image_id": "x", "image_sha256": "abc",
                "clip_sha256": "clip", "status": "passed", "question": "Q?", "canonical_target": "yes",
                "free_generation": {"response": "Yes"}, "controls": {"passed": True, "reference": {"mean_logprob": -.4}},
                "tokens": [{"token_index": i, "reconstruction_error": i, "intervention_score": i/1000} for i in [1, 7, 8]]}
            (folder / "questions/1.json").write_text(json.dumps(case))
            (folder / "features/abc.json").write_text(json.dumps({"diagnostics": {"input": {"errors": errors}}}))
            with patch("analyze_rq1_input.plot_case"):
                analyze(folder)
            rows = [json.loads(line) for line in (folder / "analysis/tokens.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 576)
            self.assertIsNone(rows[0]["intervention_score"])
            self.assertFalse(rows[0]["measured"])
            self.assertAlmostEqual(rows[7]["intervention_score"], .007)
            summary = json.loads((folder / "analysis/summary.json").read_text())
            self.assertAlmostEqual(summary["questions"][0]["spearman"], 1)


if __name__ == "__main__":
    unittest.main()
