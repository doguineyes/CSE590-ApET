"""Checks for scientific grouping/sign and same-image safeguards; CPU only."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from analyze_rq1_saved import analyze, contrast, error_groups, pair_summary


def token(index, error, effect):
    return {"token_index": index, "reconstruction_error": error, "intervention_score": effect}


class SavedAnalysisTests(unittest.TestCase):
    def test_basis_does_not_create_low_error_group_and_signed_effects_stay_signed(self):
        tokens = [token(i, i, effect) for i, effect in enumerate([4, 2, 0, 0, -2, -4], 1)]
        tokens += [token(0, 0, 100)]
        groups, basis, mean_delta, median_delta = contrast(tokens, {0})
        self.assertEqual([g["n"] for g in groups], [2, 2, 2])
        self.assertEqual(basis["I_mean"], 100)
        self.assertEqual(mean_delta, -6)
        self.assertEqual(median_delta, -6)
        self.assertEqual(groups[2]["abs_I_mean_auxiliary"], 3)
        self.assertEqual(groups[2]["positive_fraction"], 0)

    def test_equal_error_values_are_not_split_by_patch_order(self):
        # Four identical residuals straddle both nominal boundaries. They form
        # one middle group, rather than fabricating a low-vs-high comparison.
        items = [token(i, error, i) for i, error in enumerate([0, 1, 1, 1, 1, 2])]
        groups, _ = error_groups(items, set())
        self.assertEqual([[t["token_index"] for t in g] for g in groups], [[0], [1, 2, 3, 4], [5]])
        constant = [token(i, 1, i) for i in range(6)]
        _, _, mean_delta, median_delta = contrast(constant, set())
        self.assertIsNone(mean_delta)
        self.assertIsNone(median_delta)

    def test_same_image_opposite_ranking_and_feature_mismatch(self):
        case = {"image_id": "same", "image_sha256": "sha", "seed": 590,
                "clip_sha256": "clip", "projected_sha256": "projected",
                "case_id": "gqa:a", "measurement_target": "A", "target_token_ids": [1],
                "controls": {"reference": {"mean_logprob": -.5}},
                "tokens": [token(i, i, i / 100) for i in range(30)]}
        a = {"case": case, "native": {"errors": list(range(30))}, "basis": {0}, "retained": {1}}
        b = copy.deepcopy(a);b["case"]["case_id"] = "gqa:b"
        for t in b["case"]["tokens"]:
            t["intervention_score"] *= -1
        row, tokens = pair_summary(a, b)
        self.assertAlmostEqual(row["rho_nonbasis"], -1)
        self.assertEqual(row["top_overlap"], 0)
        self.assertEqual(row["opposite_sign_beyond_1e_minus4"], 29)
        self.assertEqual(len(tokens), 30)
        b["case"]["projected_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "projected_sha256"):
            pair_summary(a, b)

    def test_completed_analysis_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "new output directory"):
                analyze(REPO / "research/results", Path(temp))

    def test_real_saved_cake_pair_has_identical_errors_but_disjoint_top_patches(self):
        source = REPO / "research/results/rq1-gqa-input-pilot-v2-20261016T143422920388Z"
        if not source.exists():
            self.skipTest("Private saved result is not available")
        records = []
        for qid in ["201636988", "201637316"]:
            case = json.loads((source / "questions" / (qid + ".json")).read_text())
            native = json.loads((source / "features" / (case["image_sha256"] + ".json")).read_text())["diagnostics"]
            records.append({"case": case, "native": native, "basis": set(native["input"]["basis_indices"]),
                            "retained": set(native["input_merge"]["target_indices"])})
        row, _ = pair_summary(*records)
        self.assertEqual(row["nonbasis_tokens"], 566)
        self.assertEqual(row["top_set_size"], 57)
        self.assertEqual(row["top_overlap"], 0)
        self.assertAlmostEqual(row["rho_nonbasis"], -.8337815776137931)


if __name__ == "__main__":
    unittest.main()
