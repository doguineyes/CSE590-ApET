"""Numerical-diagnostic controls; actual CUDA/pretrained checks remain Kaggle-only."""

import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))
from rq2_parity_diagnostic import (PASS_PLAN, _preserve_tensors, capture_pass, compare_snapshots,
    score_comparison, selection_comparison, tensor_delta)


class DiagnosticPolicyTests(unittest.TestCase):
    def test_completed_evidence_prefix_is_saved_if_later_capture_fails(self):
        """CPU-only serialization wiring; this stub does not pretend to be Torch."""
        def stub_save(value, destination):
            Path(destination).write_text(json.dumps(value))

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "features").mkdir()
            path = output / "features/parity-diagnostic.pt"
            report = {"feature_sha256": {}}
            diagnostic = {"status": "running", "scientific_pilot_passed": False}
            tensors = {"passes": {"completed-pass": {"synthetic_cpu_test": True}}}
            with patch.dict(sys.modules, {"torch": SimpleNamespace(save=stub_save)}):
                with self.assertRaisesRegex(RuntimeError, "later capture failed"):
                    with _preserve_tensors(path, tensors, report, diagnostic,
                                           output / "parity-diagnostic.json"):
                        raise RuntimeError("later capture failed")
            self.assertEqual(json.loads(path.read_text()), tensors)
            self.assertEqual(diagnostic["status"], "failed")
            self.assertFalse(diagnostic["scientific_pilot_passed"])
            self.assertEqual(report["feature_sha256"]["features/parity-diagnostic.pt"],
                             hashlib.sha256(path.read_bytes()).hexdigest())
            saved = json.loads((output / "parity-diagnostic.json").read_text())
            self.assertIn("later capture failed", saved["error"])
            self.assertEqual(saved["tensor_file"], "features/parity-diagnostic.pt")

    def test_bounded_repeats_and_budget_differences_are_reported(self):
        self.assertEqual(len(PASS_PLAN), 9)
        self.assertEqual([v for _, v in PASS_PLAN].count(None), 3)
        self.assertEqual([v for _, v in PASS_PLAN].count("native"), 3)
        self.assertEqual([v for _, v in PASS_PLAN].count("no-merge"), 2)
        self.assertEqual([v for _, v in PASS_PLAN].count("generation-native"), 1)
        comparison = selection_comparison([1, 3, 5], [1, 3, 7, 8])
        self.assertFalse(comparison["exact"])
        self.assertFalse(comparison["reference_expected_budget"])
        self.assertFalse(comparison["observed_expected_budget"])
        self.assertEqual(comparison["reference_only"], [5])
        self.assertEqual(comparison["observed_only"], [7, 8])
        with self.assertRaisesRegex(ValueError, "Invalid diagnostic selected"):
            selection_comparison([1, 1], [1])
        with self.assertRaises(ValueError):
            selection_comparison([576], [1])

    def test_reference_score_delta_is_signed_and_token_aligned(self):
        left = {"token_logprobs": [-2., -4.]}
        right = {"token_logprobs": [-1., -6.]}
        delta = score_comparison(left, right)
        self.assertEqual(delta["token_logprob_deltas_observed_minus_reference"], [1., -2.])
        self.assertEqual(delta["max_abs_token_logprob_delta"], 2.)
        self.assertEqual(delta["mean_logprob_delta"], -.5)
        self.assertAlmostEqual(delta["first_token_probability_delta"], math.exp(-1)-math.exp(-2))
        self.assertFalse(delta["exact"])
        self.assertTrue(score_comparison(left, left)["exact"])
        for invalid in ({"token_logprobs": []}, {"token_logprobs": [1.]},
                        {"token_logprobs": [float("nan"), 0.]}):
            with self.assertRaises(ValueError):
                score_comparison(left, invalid)

class ArchivedFailureTests(unittest.TestCase):
    """Optional local archive audit; excluded from required Kaggle prerequisites."""

    def test_archived_failure_contains_no_numeric_delta_or_feature_tensors(self):
        """Regression: the supplied evidence cannot justify a tolerance policy."""
        path = REPO / "research/results/rq2-input-merge-pilot-v1-20261019T185452294158Z.zip"
        if not path.is_file():
            self.skipTest("Optional local raw failure archive is not tracked by Git")
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         "26f8becc7a885cb50c7a320fb090b4ed54b046ef359e28b805bd130d863811ab")
        with zipfile.ZipFile(path) as z:
            self.assertIsNone(z.testzip())
            report = json.loads(z.read("report.json"))
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["progress"]["completed_questions"], 0)
            self.assertEqual(report["error"],
                             "RuntimeError: Instrumented native parity failed: features_exact")
            self.assertFalse(any(name.endswith(".pt") for name in z.namelist()))
            case = json.loads(z.read("questions/gqa-201636988.json"))
            raw = case["native_control"]["encoded"]
            heavy = case["arms"]["native-input288"]["encoding"]["encoded"]
            self.assertNotEqual(raw["features"], heavy["features"])
            self.assertEqual(raw["mask"], heavy["mask"])
            self.assertEqual(raw["selected_indices_sorted"], heavy["selected_indices_sorted"])
            self.assertEqual(case["native_control"]["generation_tokens"]["generated_token_ids"],
                             case["arms"]["native-input288"]["generation_tokens"]["generated_token_ids"])


class DiagnosticTensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("Numeric Torch controls run in the original Kaggle environment")
        cls.torch = torch

    def test_float64_deltas_changed_tokens_and_zero_reference(self):
        torch = self.torch
        a = torch.tensor([[[1., 2.], [3., 4.]]], dtype=torch.float16)
        b = a.clone()
        b[0, 0, 0] += .5
        b[0, 1, 1] -= 1.
        delta = tensor_delta(a, b)
        self.assertFalse(delta["exact"])
        self.assertTrue(delta["both_finite"])
        self.assertEqual(delta["max_abs"], 1.)
        self.assertEqual(delta["mean_abs"], .375)
        self.assertAlmostEqual(delta["rms"], math.sqrt(1.25/4))
        self.assertAlmostEqual(delta["relative_l2"], math.sqrt(1.25/30))
        self.assertEqual(delta["changed_elements"], 2)
        self.assertEqual(delta["changed_tokens"], 2)
        self.assertEqual(tensor_delta(a, a)["relative_l2"], 0.)
        zero = torch.zeros_like(a)
        self.assertIsNone(tensor_delta(zero, a)["relative_l2"])
        self.assertFalse(tensor_delta(zero, a)["relative_l2_defined"])
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            tensor_delta(a, torch.full_like(a, float("nan")))

    def test_selected_comparison_preserves_coordinates_under_mask_changes(self):
        torch = self.torch
        a = torch.zeros(1, 576, 2)
        b = a.clone()
        b[0, 17] = 1
        left, right = torch.zeros(1, 576, dtype=torch.bool), torch.zeros(1, 576, dtype=torch.bool)
        left[:, :288], right[:, 288:] = True, True
        comparison = compare_snapshots({"encoded": (a, left)}, {"encoded": (b, right)})
        self.assertFalse(comparison["selection"]["same_coordinates"])
        metrics = comparison["stages"]["projected"]
        self.assertEqual(metrics["full576"]["changed_tokens"], 1)
        self.assertEqual(metrics["reference_selected_coordinates"]["changed_tokens"], 1)
        self.assertEqual(metrics["observed_selected_coordinates"]["changed_tokens"], 0)

    def fixture(self):
        from llava.model.llava_arch import LlavaMetaForCausalLM
        torch = self.torch

        class Vision(torch.nn.Module):
            def __init__(self, source):
                super().__init__()
                self.source = source

            def forward(self, pixels):
                return self.source.clone(), None

        class Harness:
            encode_images = LlavaMetaForCausalLM.encode_images
            token_merging = LlavaMetaForCausalLM.token_merging

            def __init__(self, source):
                self.tower = Vision(source)
                self.model = SimpleNamespace(basis_token_num=10, layer_list=[],
                    get_vision_tower=lambda: self.tower, mm_projector=lambda x: x)

            def get_model(self):
                return self.model

            def get_vision_tower(self):
                return self.tower

            def get_visual_token_num(self):
                return self.visual_token_num

        torch.manual_seed(590)
        return Harness(torch.randn(1, 577, 32).half())

    def test_raw_and_heavy_capture_restore_native_and_preserve_exact_reference(self):
        torch = self.torch
        model = self.fixture()
        before = set(model.__dict__)
        raw, _ = capture_pass(model, None, 590, None)
        heavy, observation = capture_pass(model, None, 590, "native")
        self.assertEqual(set(model.__dict__) - before, {"visual_token_num"})
        delta = compare_snapshots(raw, heavy)
        self.assertTrue(delta["selection"]["exact"])
        self.assertTrue(delta["stages"]["projected"]["full576"]["exact"])
        self.assertEqual(len(observation["input"]["errors"]), 576)
        self.assertEqual(len(observation["basis_slot_to_source"]), 10)
        no_merge, _ = capture_pass(model, None, 590, "no-merge")
        self.assertTrue(torch.equal(no_merge["pre_merge"], no_merge["post_merge"]))
        self.assertFalse(torch.equal(heavy["post_merge"], no_merge["post_merge"]))

    def test_numeric_controls_detect_injected_postmerge_variation(self):
        """Controlled drift demonstrates localization; it does not model CUDA cause."""
        torch = self.torch
        model = self.fixture()
        original = torch.Tensor.scatter_add_
        count = [0]

        def injected(tensor, dim, index, source):
            output = original(tensor, dim, index, source)
            if tensor.is_floating_point():
                count[0] += 1
                output[0, 0, 0] += .25*count[0]
            return output

        with patch.object(torch.Tensor, "scatter_add_", injected):
            first, first_call = capture_pass(model, None, 590, "native")
            second, second_call = capture_pass(model, None, 590, "native")
        comparison = compare_snapshots(first, second)
        self.assertTrue(comparison["stages"]["pre_merge"]["full576"]["exact"])
        self.assertFalse(comparison["stages"]["post_merge"]["full576"]["exact"])
        self.assertGreater(comparison["stages"]["projected"]["full576"]["max_abs"], 0)
        self.assertEqual(first_call["input"], second_call["input"])
        self.assertEqual(first_call["source_clip"], second_call["source_clip"])


if __name__ == "__main__":
    unittest.main()
