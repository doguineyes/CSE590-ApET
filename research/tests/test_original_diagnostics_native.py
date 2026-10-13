"""Verify logging on actual native functions with tiny synthetic tensors, no checkpoints."""

import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from original_diagnostics import observe_native_values


class NativeLoggingTests(unittest.TestCase):
    def setUp(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Run in the original Kaggle environment with Torch")
        os.environ["MPLBACKEND"] = "Agg"
        self.torch = torch

    def test_native_input_masks_and_features_preserved_with_logging(self):
        from llava.model.llava_arch import LlavaMetaForCausalLM
        torch = self.torch

        class Harness:
            encode_images = LlavaMetaForCausalLM.encode_images
            token_merging = LlavaMetaForCausalLM.token_merging

            def __init__(self, features, count):
                self.count = count
                self.model = SimpleNamespace(basis_token_num=10,
                    get_vision_tower=lambda: lambda images: (features.clone(), None), mm_projector=lambda values: values)

            def get_model(self): return self.model
            def get_visual_token_num(self): return self.count

        torch.manual_seed(590)
        features = torch.randn(1, 577, 32).half()
        for count in [576, 288]:
            torch.manual_seed(590)
            expected, expected_mask = Harness(features, count).encode_images(None)
            torch.manual_seed(590)
            with observe_native_values() as data:
                actual, actual_mask = Harness(features, count).encode_images(None)
            torch.testing.assert_close(actual_mask, expected_mask, rtol=0, atol=0)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            if count < 576:
                self.assertEqual(len(data["input"]["errors"]), 576)
                self.assertEqual(sorted(data["input"]["selected_indices"]), actual_mask[0].nonzero().flatten().tolist())
                self.assertEqual(sum(data["input_merge"]["merge_counts"]), 576 - count)

    def test_native_decoder_features_preserved_with_logging(self):
        from llava.model.modeling_llama_x import LlamaModel
        torch = self.torch

        class Harness:
            layer_prune = LlamaModel.layer_prune
            config = SimpleNamespace(tokenizer_padding_side="right")
            image_token_list = [576, 96]
            image_tokens = [576]
            image_token_posi = [2]
            basis_token_num = 10

        torch.manual_seed(590)
        features = torch.randn(1, 583, 32).half()
        kwargs = dict(cur_num=0, rank_layer=16, position_ids=None, attention_mask=None, labels=None,
                      selected_indices=[], sum_visual_attention=None, indices_attention=None)
        torch.manual_seed(590)
        expected = Harness().layer_prune(features=features.clone(), **kwargs)
        torch.manual_seed(590)
        with observe_native_values() as data:
            actual = Harness().layer_prune(features=features.clone(), **kwargs)
        torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
        self.assertEqual(len(data["decoder"][0]["errors"]), 576)
        self.assertEqual(len(data["decoder"][0]["selected_indices"]), 96)
        self.assertEqual(sum(data["decoder_merge"][0]["merge_counts"]), 480)


if __name__ == "__main__":
    unittest.main()
