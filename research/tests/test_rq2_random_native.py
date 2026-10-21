"""Actual tiny native64 and pure-random checks; original Kaggle runtime required."""
import copy
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from rq2_random_native import (capture_condition, runtime_scope, validate_snapshot,
                               replay_snapshot, generate_condition)
from rq2_native import assert_selection_equal


class RandomNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("Run these native tests in the original Kaggle Torch runtime")
        cls.torch = torch
        os.environ["MPLBACKEND"] = "Agg"

    def fixture(self):
        from research.tests import test_rq2_frozen_native as fixture_module
        return fixture_module.FrozenNativeTests.fixture(self)

    def test_pure_random_skips_FPS_and_solve_and_preserves_all_original_features(self):
        torch = self.torch
        model, _, _, pixels = self.fixture()
        method = model.encode_images.__func__
        state = torch.random.get_rng_state().clone()
        full, full_snapshot = capture_condition(model, pixels, 590, "baseline576", "a" * 64)
        with patch("llava.model.llava_arch.fps", side_effect=AssertionError("FPS ran")), \
             patch.object(torch.linalg, "solve", side_effect=AssertionError("reconstruction ran")):
            for budget in (64, 288, 576):
                condition = f"pure-random-{budget}-s101"
                obs, snapshot = capture_condition(model, pixels, 590, condition, "a" * 64)
                self.assertTrue(torch.equal(snapshot["encoded"][0], full_snapshot["encoded"][0]))
                self.assertEqual(obs["encoded"]["kept_tokens"], budget)
                self.assertTrue(obs["controls"]["no_fps_E_basis_or_merge"])
                if budget == 576:
                    self.assertEqual(obs["encoded"], full["encoded"])
        self.assertIs(model.encode_images.__func__, method)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))

    def test_actual_native64_keeps_54_ordinary_targets_and_ten_basis_slots(self):
        model, _, _, pixels = self.fixture()
        merge, snapshot = capture_condition(model, pixels, 590, "native-64-merge", "a" * 64)
        drop, _ = capture_condition(model, pixels, 590, "native-64-no-merge", "a" * 64)
        self.assertEqual(merge["encoded"]["kept_tokens"], 64)
        self.assertEqual(len(merge["basis_slot_to_source"]), 10)
        self.assertEqual(len(merge["input_merge"]["target_indices"]), 54)
        self.assertEqual(assert_selection_equal(merge, drop)["status"], "passed")
        self.assertEqual(validate_snapshot(snapshot, pixels)["encoded"], merge["encoded"])

    def test_runtime_settings_rng_and_native_methods_restore_on_exception(self):
        import smoke_original_apet as smoke
        torch = self.torch
        model, _, _, pixels = self.fixture()
        settings = copy.deepcopy(smoke.CONFIGURATIONS)
        old = model.visual_token_num
        state = torch.random.get_rng_state().clone()
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            with runtime_scope(model, 64, 590):
                self.assertEqual(model.visual_token_num, 64)
                self.assertEqual(model.model.layer_list, [])
                raise RuntimeError("deliberate")
        self.assertEqual(smoke.CONFIGURATIONS, settings)
        self.assertEqual(model.visual_token_num, old)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))

    def test_replay_returns_clones_and_rejects_mutated_masks(self):
        torch = self.torch
        model, _, _, pixels = self.fixture()
        _, snapshot = capture_condition(model, pixels, 590, "pure-random-64-s101", "a" * 64)
        with replay_snapshot(model, pixels, snapshot) as audit:
            values = model.encode_images(pixels)
            self.assertNotEqual(values[0].data_ptr(), snapshot["encoded"][0].data_ptr())
        self.assertEqual(audit["image_prefill_calls"], 1)
        self.assertEqual(audit["status"], "passed")
        with torch.inference_mode():
            snapshot["encoded"][1][0, 0] = ~snapshot["encoded"][1][0, 0]
        with self.assertRaisesRegex(ValueError, "mutated"):
            validate_snapshot(snapshot, pixels)

    def test_tiny_generation_and_fixed_scoring_use_actual_64_token_geometry(self):
        from PIL import Image
        model, tokenizer, case, pixels = self.fixture()
        for condition in ("native-64-merge", "pure-random-64-s101"):
            obs, snapshot = capture_condition(model, pixels, 590, condition, "a" * 64)
            with patch("llava.mm_utils.process_images", return_value=(pixels, None)):
                row = generate_condition(tokenizer, model, None, Image.new("RGB", (2, 2)),
                    case, pixels, condition, snapshot, 590, "Red")
            self.assertEqual(row["generation"]["trace"]["input"], {"before": 576, "after": 64})
            self.assertEqual(row["generation"]["trace"].get("decoder", []), [])
            self.assertEqual(row["encoding"], obs)
            self.assertTrue(row["generation_tokens"]["generated_token_ids"])
            for control in row["frozen_replay"].values():
                if isinstance(control, dict) and "image_prefill_calls" in control:
                    self.assertEqual(control["status"], "passed")
                    self.assertEqual(control["image_prefill_calls"], 1)

    def test_full576_pure_random_matches_actual_tiny_generation_and_reference_scores(self):
        from PIL import Image
        from rq1_native import max_score_delta
        model, tokenizer, case, pixels = self.fixture()
        rows = []
        for condition in ("baseline576", "pure-random-576-s101"):
            _, snapshot = capture_condition(model, pixels, 590, condition, "a" * 64)
            with patch("llava.mm_utils.process_images", return_value=(pixels, None)):
                rows.append(generate_condition(tokenizer, model, None, Image.new("RGB", (2, 2)),
                    case, pixels, condition, snapshot, 590, "Red"))
        self.assertEqual(rows[0]["encoding"]["encoded"], rows[1]["encoding"]["encoded"])
        self.assertEqual(rows[0]["generation_tokens"], rows[1]["generation_tokens"])
        self.assertEqual(max_score_delta(rows[0]["reference_score"], rows[1]["reference_score"]), 0)


if __name__ == "__main__":
    unittest.main()
