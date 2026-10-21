"""Tiny original-model generation tests; run on Kaggle, without checkpoint downloads."""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))


class TextVQANativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("Use the original Kaggle Torch runtime")
        cls.torch = torch
        os.environ["MPLBACKEND"] = "Agg"

    def fixture(self):
        from research.tests.test_rq2_frozen_native import FrozenNativeTests
        return FrozenNativeTests.fixture(self)

    def test_broad_generation_uses_actual_64_geometry_and_never_teacher_forces(self):
        from PIL import Image
        from rq2_random_native import capture_condition
        from run_rq2_textvqa100 import generate_answer
        model, tokenizer, case, pixels = self.fixture()
        case["image_sha256"] = "a" * 64
        for condition in ("native-64-merge", "native-64-no-merge", "pure-random-64-s101"):
            _, snapshot = capture_condition(model, pixels, 590, condition, "a" * 64)
            with patch("llava.mm_utils.process_images", return_value=(pixels, None)), \
                 patch("rq2_random_native.score_reference", side_effect=AssertionError("Teacher forcing ran")):
                row = generate_answer(tokenizer, model, None, Image.new("RGB", (2, 2)), case,
                                      pixels, condition, snapshot, 590)
            self.assertEqual(row["generation"]["trace"]["input"], {"before": 576, "after": 64})
            self.assertEqual(row["generation"]["trace"].get("decoder", []), [])
            self.assertEqual(row["frozen_replay"]["image_prefill_calls"], 1)
            self.assertEqual(row["frozen_replay"]["status"], "passed")

    def test_broad_full576_parity_includes_generated_ids_and_processed_scores(self):
        from PIL import Image
        from rq2_random_native import capture_condition
        from run_rq2_textvqa100 import generate_answer, compare_replay
        model, tokenizer, case, pixels = self.fixture()
        case["image_sha256"] = "a" * 64
        rows = []
        for condition in ("baseline576", "pure-random-576-s101"):
            _, snapshot = capture_condition(model, pixels, 590, condition, "a" * 64)
            with patch("llava.mm_utils.process_images", return_value=(pixels, None)):
                rows.append(generate_answer(tokenizer, model, None, Image.new("RGB", (2, 2)), case,
                                            pixels, condition, snapshot, 590))
        self.assertEqual(compare_replay(*rows)["status"], "passed")


if __name__ == "__main__":
    unittest.main()
