"""Kaggle prerequisite: causal scoring, singleton zeroing and tiny native forward.

No pretrained weights/downloads. Local Mac skips Torch checks; Kaggle notebook
first imports Torch explicitly so these cannot silently skip there.
"""

import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from rq1_native import (encode_image, max_score_delta, prepare_question, reference_tokens,
                        observe_generation_tokens, replaced, score_embeddings, score_from_logits, zero_patch)


class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("Torch checks require the original Kaggle runtime")
        os.environ["MPLBACKEND"] = "Agg"
        cls.torch = torch

    def test_causal_shift_and_target_only(self):
        torch = self.torch
        logits = torch.tensor([[[80., 0., 0.], [0., 1., 5.], [0., 4., 2.], [90., 0., 0.]]])
        labels = torch.tensor([[-100, -100, 2, 1]])
        actual = score_from_logits(logits, labels)
        expected = [logits[0, 1].log_softmax(-1)[2].item(), logits[0, 2].log_softmax(-1)[1].item()]
        self.assertEqual(actual["token_logprobs"], expected)
        self.assertAlmostEqual(actual["mean_logprob"], sum(expected)/2)
        with self.assertRaises(ValueError):
            score_from_logits(logits, torch.full_like(labels, -100))
        if torch.cuda.is_available():
            # Actual sharded usage: labels can remain away from the lm_head GPU.
            device = f"cuda:{torch.cuda.device_count()-1}"
            gpu = score_from_logits(logits.to(device), labels)
            self.assertAlmostEqual(gpu["mean_logprob"], actual["mean_logprob"], places=5)

    def test_generation_observer_excludes_dummy_bos_and_preserves_native_output(self):
        torch = self.torch
        first, second = torch.zeros(1, 16), torch.zeros(1, 16)
        first[0, 5], second[0, 2] = 8, 7
        output = SimpleNamespace(sequences=torch.tensor([[1, 5, 2]]), scores=(first, second))
        class Model:
            def generate(self): return output
        class Tokenizer:
            def convert_ids_to_tokens(self, ids):
                return [str(v) for v in ids] if isinstance(ids, list) else str(ids)
        model, record = Model(), {}
        with observe_generation_tokens(model, Tokenizer(), record):
            self.assertIs(model.generate(), output)
        self.assertEqual(record["generated_token_ids"], [5, 2])
        self.assertEqual(record["first_step_top10"][0]["token_id"], 5)
        self.assertAlmostEqual(record["generated_token_logprobs"][0], first[0].log_softmax(-1)[5].item())
        self.assertNotIn("generate", model.__dict__)

    def test_zeroing_preserves_every_other_position_and_noop(self):
        torch = self.torch
        values = torch.arange(2 * 600 * 4).reshape(2, 600, 4).float() + 1
        no_op = zero_patch(values, 12)
        self.assertTrue(torch.equal(no_op, values))
        self.assertNotEqual(no_op.data_ptr(), values.data_ptr())
        changed = zero_patch(values, 12, 575)
        self.assertTrue(torch.equal(changed[:, :587], values[:, :587]))
        self.assertTrue(torch.equal(changed[:, 588:], values[:, 588:]))
        self.assertTrue((changed[:, 587] == 0).all())
        self.assertTrue((values[:, 587] != 0).all())

    def test_replacement_restores_on_failure(self):
        class Example:
            def method(self): return 1
        obj = Example()
        with self.assertRaises(RuntimeError):
            with replaced(obj, "method", lambda: 2):
                self.assertEqual(obj.method(), 2)
                raise RuntimeError("control")
        self.assertEqual(obj.method(), 1)
        self.assertNotIn("method", obj.__dict__)

    def test_boundary_rejects_prompt_retokenization(self):
        torch = self.torch
        case = {"generation_question": "Q?", "canonical_target": "yes"}
        fake = lambda *a, **kw: torch.tensor([1, -200, 7]) if a[0].endswith(":") else torch.tensor([1, -200, 8, 9])
        with patch("llava.mm_utils.tokenizer_image_token", fake), self.assertRaises(ValueError):
            reference_tokens(None, case)

    def test_error_pass_reuses_full_path_clip_without_mutation(self):
        torch = self.torch
        from llava.model.llava_arch import LlavaMetaForCausalLM

        class Vision(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.features = torch.randn(1, 577, 1024).half()
                self.calls = 0
            def forward(self, images):
                self.calls += 1
                return self.features.clone(), None

        class Harness(torch.nn.Module):
            encode_images = LlavaMetaForCausalLM.encode_images
            token_merging = LlavaMetaForCausalLM.token_merging
            config = SimpleNamespace()
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.vision = Vision()
                self.model.get_vision_tower = lambda: self.model.vision
                self.model.mm_projector = torch.nn.Identity()
                self.model.embed_tokens = torch.nn.Embedding(128, 1024)
            def get_model(self): return self.model
            def get_vision_tower(self): return self.model.vision
            def get_input_embeddings(self): return self.model.embed_tokens
            def get_visual_token_num(self): return self.visual_token_num

        torch.manual_seed(590)
        model = Harness()
        expected = model.model.vision.features.clone()
        processor = SimpleNamespace(image_mean=[.5]*3, image_std=[.5]*3)
        from PIL import Image
        with patch("llava.mm_utils.process_images", return_value=(torch.zeros(1, 3, 336, 336), None)):
            encoded = encode_image(model, processor, Image.new("RGB", (10, 20)), 590)
        self.assertTrue(torch.equal(encoded["clip"], expected))
        self.assertTrue(torch.equal(encoded["projected"], expected[:, 1:]))
        self.assertEqual(len(encoded["diagnostics"]["input"]["errors"]), 576)
        self.assertEqual(model.model.vision.calls, 1)
        self.assertEqual(model.visual_token_num, 576)
        self.assertNotIn("forward", model.model.vision.__dict__)

    def test_native_preparation_positions_target_and_tiny_forward(self):
        torch = self.torch
        from llava.model.language_model.llava_llama_x import LlavaConfig, LlavaLlamaForCausalLM_X
        config = LlavaConfig(vocab_size=128, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=1024, bos_token_id=1, eos_token_id=2, pad_token_id=0)
        config._attn_implementation = "sdpa"
        torch.manual_seed(590)
        model = LlavaLlamaForCausalLM_X(config, visual_token_num=576, selected_indices=[]).eval()
        model.model.image_token_posi = [1]
        model.model.image_tokens = [576]
        projected = torch.randn(1, 576, 32)
        encoded = {"projected": projected, "mask": torch.ones(1, 576, dtype=torch.bool),
                   "pixels": torch.zeros(1, 3, 336, 336)}
        # Actual native prep; only tokenizer and vision tower presence are synthetic.
        tokens = ("prompt", torch.tensor([[1, -200, 3, 4, 5]]),
                  torch.tensor([[-100, -100, -100, 4, 5]]), 3)
        with patch("rq1_native.reference_tokens", return_value=tokens), replaced(model, "get_vision_tower", lambda: object()):
            prepared = prepare_question(None, model, {}, encoded)
        self.assertEqual(prepared["target_start"], 578)
        self.assertEqual(prepared["image_start"], 1)
        self.assertEqual(prepared["embeddings"].shape, (1, 580, 32))
        self.assertEqual(prepared["labels"][0, -2:].tolist(), [4, 5])
        self.assertTrue((prepared["labels"][0, 1:577] == -100).all())
        first = score_embeddings(model, prepared)
        clone = score_embeddings(model, prepared, zero_patch(prepared["embeddings"], 1))
        self.assertEqual(max_score_delta(first, clone), 0)
        changed = score_embeddings(model, prepared, zero_patch(prepared["embeddings"], 1, 0))
        self.assertEqual(len(changed["token_logprobs"]), 2)


if __name__ == "__main__":
    unittest.main()
