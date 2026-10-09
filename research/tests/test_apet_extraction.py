"""Compare extracted math to original source functions, then test the HF adapter.

Run on Kaggle with the project Python; no checkpoint downloads are needed.
"""

import importlib.util
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None


class ConfigTests(unittest.TestCase):
    def test_invalid_budgets_and_regularization_are_rejected(self):
        from apet_compression import CompressionConfig

        for kwargs in [{"keep_tokens": 10, "basis_tokens": 10}, {"basis_tokens": 0},
                       {"epsilon": 0}, {"epsilon": float("nan")}, {"keep_tokens": 96.5}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                CompressionConfig(**kwargs)


@unittest.skipUnless(HAS_TORCH, "Run with Kaggle project Python")
class ExtractionTests(unittest.TestCase):
    def check_parity(self, device, dtype, batch=1):
        import torch
        from apet_compression import CompressionConfig, compress_tokens, fps
        from apet_compression.legacy_reference import original_functions, original_input_compression

        values = torch.randn(batch, 576, 64, generator=torch.Generator().manual_seed(22)).to(device, dtype)
        unchanged = values.clone()
        config = CompressionConfig(96, 10)
        torch.manual_seed(590)
        expected_fps = original_functions()["fps"](values, 10)
        torch.manual_seed(590)
        torch.testing.assert_close(fps(values, 10), expected_fps, rtol=0, atol=0)
        torch.manual_seed(590)
        original, mask = original_input_compression(values, config)
        torch.manual_seed(590)
        extracted = compress_tokens(values, config)
        torch.testing.assert_close(extracted.full_features, original, rtol=5e-3, atol=5e-3)
        torch.testing.assert_close(extracted.kept_mask, mask, rtol=0, atol=0)
        torch.testing.assert_close(values, unchanged, rtol=0, atol=0)
        self.assertEqual(extracted.tokens.shape, (batch, 96, 64))
        for b in range(batch):
            torch.testing.assert_close(extracted.tokens[b], original[b][mask[b]], rtol=5e-3, atol=5e-3)
        self.assertTrue(torch.all(extracted.kept_indices[:, 1:] > extracted.kept_indices[:, :-1]).item())

    def test_cpu_fp32_and_fp16_batch_parity_with_original_source(self):
        import torch

        for dtype in [torch.float32, torch.float16]:
            with self.subTest(dtype=dtype):
                self.check_parity("cpu", dtype, batch=2)

    def test_cuda_fp16_parity_on_each_visible_gpu(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable; CPU parity still runs")
        for index in range(torch.cuda.device_count()):
            with self.subTest(gpu=index):
                self.check_parity(f"cuda:{index}", torch.float16)

    def test_all_tokens_is_exact_identity(self):
        import torch
        from apet_compression import CompressionConfig, compress_tokens
        from apet_compression.legacy_reference import original_input_compression

        values = torch.randn(1, 576, 32)
        config = CompressionConfig(576, 10)
        original, mask = original_input_compression(values, config)
        extracted = compress_tokens(values, config)
        torch.testing.assert_close(extracted.tokens, original, rtol=0, atol=0)
        torch.testing.assert_close(extracted.kept_mask, mask, rtol=0, atol=0)
        self.assertIs(extracted.full_features, values)

    def test_local_generator_is_repeatable_and_leaves_global_rng_unchanged(self):
        import torch
        from apet_compression import CompressionConfig, compress_tokens

        values = torch.randn(1, 64, 32)
        state = torch.random.get_rng_state().clone()
        before = compress_tokens(values, CompressionConfig(24, 4), torch.Generator().manual_seed(30))
        after = compress_tokens(values, CompressionConfig(24, 4), torch.Generator().manual_seed(30))
        torch.testing.assert_close(before.tokens, after.tokens, rtol=0, atol=0)
        torch.testing.assert_close(torch.random.get_rng_state(), state, rtol=0, atol=0)


@unittest.skipUnless(HAS_TORCH and HAS_TRANSFORMERS, "Run with Kaggle project Python")
class AdapterTests(unittest.TestCase):
    def test_tiny_hf_generation_identity_compression_and_cleanup(self):
        import torch
        from transformers import LlavaConfig, LlavaForConditionalGeneration
        from apet_compression import CompressionConfig
        from apet_compression.adapters.hf_llava import HFInputCompressionAdapter

        config = LlavaConfig(
            text_config={"model_type": "llama", "vocab_size": 128, "hidden_size": 64,
                         "intermediate_size": 128, "num_hidden_layers": 2, "num_attention_heads": 4,
                         "num_key_value_heads": 4, "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0},
            vision_config={"model_type": "clip_vision_model", "hidden_size": 32,
                           "intermediate_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
                           "image_size": 16, "patch_size": 4, "projection_dim": 32},
            image_token_index=127, image_seq_length=16, vision_feature_layer=-2,
        )
        torch.manual_seed(590)
        model = LlavaForConditionalGeneration(config).eval()
        ids = torch.tensor([[1] + [127] * 16 + [3, 4]])
        inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids),
                  "pixel_values": torch.zeros(1, 3, 16, 16)}
        hooks_before = (len(model.multi_modal_projector._forward_pre_hooks),
                        len(model.multi_modal_projector._forward_hooks))
        with torch.inference_mode():
            baseline = model(**inputs).logits
            with HFInputCompressionAdapter(model, CompressionConfig(16, 2), verify_legacy=False) as adapter:
                prepared = adapter.prepare_inputs(inputs, 590)
                torch.testing.assert_close(model(**prepared).logits, baseline, rtol=0, atol=0)
                self.assertEqual(adapter.statistics()["keep_tokens"], 16)
            with HFInputCompressionAdapter(model, CompressionConfig(8, 2)) as adapter:
                prepared = adapter.prepare_inputs(inputs, 590)
                self.assertEqual(prepared["input_ids"].shape, (1, 11))
                self.assertEqual(prepared["input_ids"][0, -2:].tolist(), [3, 4])
                generated = model.generate(**prepared, max_new_tokens=2, do_sample=False, use_cache=True)
                self.assertGreater(generated.shape[1], 11)
                self.assertEqual(adapter.calls, 1)  # Vision is only evaluated during prefill.
                self.assertEqual(adapter.legacy_check["status"], "passed")
        self.assertEqual((len(model.multi_modal_projector._forward_pre_hooks),
                          len(model.multi_modal_projector._forward_hooks)), hooks_before)
        with self.assertRaisesRegex(RuntimeError, "intentional cleanup check"):
            with HFInputCompressionAdapter(model, CompressionConfig(8, 2), verify_legacy=False):
                raise RuntimeError("intentional cleanup check")
        self.assertEqual((len(model.multi_modal_projector._forward_pre_hooks),
                          len(model.multi_modal_projector._forward_hooks)), hooks_before)
        with torch.inference_mode():
            torch.testing.assert_close(model(**inputs).logits, baseline, rtol=0, atol=0)


if __name__ == "__main__":
    if "--require-cuda" in sys.argv:
        sys.argv.remove("--require-cuda")
        if not HAS_TORCH or not HAS_TRANSFORMERS:
            raise SystemExit("Project Torch/Transformers dependencies are required")
        import torch

        if not torch.cuda.is_available():
            raise SystemExit("CUDA required; enable Kaggle GPU before running the comparison")
    unittest.main()
