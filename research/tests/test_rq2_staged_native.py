"""Tiny actual native staged generation. Run in original Kaggle runtime, no weights."""
import copy
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))


class StagedNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("Use original Kaggle Torch runtime for actual staged tests")
        cls.torch = torch
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        os.environ["MPLBACKEND"] = "Agg"

    @classmethod
    def tearDownClass(cls):
        cls.torch.set_num_threads(cls.old_threads)

    def fixture(self):
        from research.tests.test_rq2_frozen_native import FrozenNativeTests
        model, tokenizer, case, pixels = FrozenNativeTests.fixture(self)
        layers = list(model.model.layers)
        layers.extend(copy.deepcopy(layers[-1]) for _ in range(18 - len(layers)))
        for index, layer in enumerate(layers):
            layer.self_attn.layer_idx = index
        model.model.layers = self.torch.nn.ModuleList(layers)
        model.config.num_hidden_layers = 18
        case["image_sha256"] = "a" * 64
        return model, tokenizer, case, pixels

    def test_native_observation_matches_direct_native_with_RNG_continuation(self):
        from PIL import Image
        from rq2_staged_native import capture_input, generate_condition, generate_raw_native
        from run_rq2_staged_calibration import compare_replay
        model, tokenizer, case, pixels = self.fixture()
        for condition in ("native-96-to-32", "native-288-to-96"):
            _, snapshot = capture_input(model, pixels, 590, condition, "a" * 64)
            with patch("llava.mm_utils.process_images", return_value=(pixels, None)):
                row, layer = generate_condition(tokenizer, model, None, Image.new("RGB", (2, 2)),
                    case, pixels, condition, snapshot, 590)
                raw = generate_raw_native(tokenizer, model, None, Image.new("RGB", (2, 2)), case, condition, 590)
            # The CPU fixture is deterministic; separate CUDA fresh runs are
            # recorded numerically, never forced through this byte-exact gate.
            self.assertEqual(compare_replay(row, {**raw, "image_sha256": case["image_sha256"]})["status"], "passed")
            self.assertEqual(row["decoder"]["E"]["source_tokens"], row["input_tokens"])
            self.assertEqual(len(row["decoder"]["E"]["selected_indices"]), row["decoder_tokens"])
            self.assertEqual(layer["rng_after"], row["decoder"]["rng_after"])

    def test_pure_random_skips_native_selection_solve_and_merging_at_both_stages(self):
        from PIL import Image
        from rq2_staged_native import capture_input, generate_condition
        model, tokenizer, case, pixels = self.fixture()
        def forbidden(*args, **kwargs):
            raise AssertionError("Pure random executed FPS, solve or merging")
        for condition in ("pure-random-96-to-32-s101", "pure-random-288-to-96-s101"):
            with patch("llava.model.llava_arch.fps", new=forbidden), \
                 patch("llava.model.modeling_llama_x.fps", new=forbidden), \
                 patch("llava.model.modeling_llama_x.token_merging", new=forbidden), \
                 patch.object(self.torch.linalg, "solve", new=forbidden), \
                 patch("llava.mm_utils.process_images", return_value=(pixels, None)):
                _, snapshot = capture_input(model, pixels, 590, condition, "a" * 64)
                row, _ = generate_condition(tokenizer, model, None, Image.new("RGB", (2, 2)), case,
                    pixels, condition, snapshot, 590, probe=True)
            self.assertEqual(row["cache"]["cached_decode_steps"], 3)
            self.assertTrue(row["decoder"]["original_contextual_vectors_exact"])
            self.assertFalse(row["decoder"]["E"])

    def test_full576_identity_noop_preserves_generation_and_cached_positions(self):
        from PIL import Image
        from rq2_staged_native import capture_input, generate_condition
        from rq2_staged_protocol import IDENTITY_CONDITION
        from run_rq2_staged_calibration import compare_replay
        model, tokenizer, case, pixels = self.fixture()
        rows = []
        for condition in ("baseline576", IDENTITY_CONDITION):
            _, snapshot = capture_input(model, pixels, 590, condition, "a" * 64)
            with patch("llava.mm_utils.process_images", return_value=(pixels, None)):
                row, _ = generate_condition(tokenizer, model, None, Image.new("RGB", (2, 2)), case,
                    pixels, condition, snapshot, 590, probe=True)
                rows.append(row)
        self.assertEqual(compare_replay(*rows)["status"], "passed")
        self.assertEqual(rows[0]["cache"]["events"], rows[1]["cache"]["events"])

    def test_frozen_decoder_replay_rejects_mutated_output_and_restores_scope(self):
        from PIL import Image
        import smoke_original_apet as smoke
        from rq2_staged_native import capture_input, generate_condition
        from run_rq2_staged_calibration import compare_replay
        model, tokenizer, case, pixels = self.fixture()
        condition = "native-96-to-32"
        native = model.model.layer_prune.__func__
        settings = copy.deepcopy(smoke.CONFIGURATIONS)
        _, snapshot = capture_input(model, pixels, 590, condition, "a" * 64)
        with patch("llava.mm_utils.process_images", return_value=(pixels, None)):
            row, layer = generate_condition(tokenizer, model, None, Image.new("RGB", (2, 2)), case,
                pixels, condition, snapshot, 590)
            again, _ = generate_condition(tokenizer, model, None, Image.new("RGB", (2, 2)), case,
                pixels, condition, snapshot, 590, decoder_replay=layer)
            self.assertEqual(compare_replay(row, again)["status"], "passed")
            with self.torch.inference_mode():
                layer["output"][2][0, 0, 0] += 1
            with self.assertRaisesRegex(ValueError, "snapshot mutated"):
                generate_condition(tokenizer, model, None, Image.new("RGB", (2, 2)), case,
                    pixels, condition, snapshot, 590, decoder_replay=layer)
        self.assertIs(model.model.layer_prune.__func__, native)
        self.assertEqual(smoke.CONFIGURATIONS, settings)

    def test_random_prune_preserves_text_labels_mask_and_compact_positions(self):
        from rq2_staged_native import random_prune
        torch = self.torch
        inner = SimpleNamespace(config=SimpleNamespace(tokenizer_padding_side="right"), image_token_posi=[2],
                                image_tokens=[96], image_token_list=[96, 32])
        features = torch.arange(100 * 4).reshape(1, 100, 4).float()
        labels = torch.arange(100).unsqueeze(0)
        supplied = {"features": features, "position_ids": labels.clone(),
                    "attention_mask": torch.ones(1, 100, dtype=torch.bool), "labels": labels}
        selected = list(range(0, 96, 3))
        result = random_prune(inner, supplied, selected)
        expected = [0, 1] + [2 + i for i in selected] + [98, 99]
        self.assertTrue(torch.equal(result[2], features[:, expected]))
        self.assertTrue(torch.equal(result[3], labels[:, expected]))
        self.assertTrue(torch.equal(result[0], torch.arange(36).unsqueeze(0)))
        self.assertTrue(result[1].all())


if __name__ == "__main__":
    unittest.main()
