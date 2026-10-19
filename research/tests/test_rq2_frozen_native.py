"""Self-contained tiny native capture/replay tests for the Kaggle prerequisites."""
import copy
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'scripts'))
from rq2_frozen_native import (arm_configuration, capture_encoding, validate_snapshot,
    generate_from_snapshot, compare_realizations, _generation_replay, _score_replayed)
from rq2_native import assert_selection_equal


class FrozenPolicyTests(unittest.TestCase):
    def test_unknown_arm_rejected_before_model_or_torch_execution(self):
        self.assertEqual(arm_configuration('baseline576'), 'baseline')
        self.assertEqual(arm_configuration('native-input288-no-merge'), 'input-only')
        with self.assertRaises(ValueError): arm_configuration('random-new-arm')


class FrozenNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try: import torch
        except ImportError: raise unittest.SkipTest('Frozen native tests require the original Kaggle Torch runtime')
        cls.torch = torch
        os.environ['MPLBACKEND'] = 'Agg'

    def fixture(self):
        torch = self.torch
        from llava.model.language_model.llava_llama_x import LlavaConfig, LlavaLlamaForCausalLM_X
        class Vision(torch.nn.Module):
            def __init__(self, source): super().__init__(); self.source = source
            def forward(self, images): return self.source.clone(), None
        class Projector(torch.nn.Module):
            def forward(self, source): return source.float()
        class Tokenizer:
            bos_token_id = 1
            def __call__(self, text):
                return SimpleNamespace(input_ids=[1] + [3 + ord(c) % 120 for c in text])
            def convert_ids_to_tokens(self, values):
                return [str(v) for v in values] if isinstance(values, list) else str(values)
            def batch_decode(self, values, **kwargs): return ['red']
        torch.manual_seed(590)
        config = LlavaConfig(vocab_size=128, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=2048, bos_token_id=1, eos_token_id=2, pad_token_id=0)
        config._attn_implementation = 'sdpa'; config.tokenizer_model_max_length = 2048
        model = LlavaLlamaForCausalLM_X(config, visual_token_num=576, selected_indices=[]).eval()
        model.model.vision_tower = Vision(torch.randn(1, 577, 32))
        model.model.mm_projector = Projector()
        model.model.basis_token_num = 10; model.model.layer_list = []
        case = {'case_id': 'gqa:synthetic', 'canonical_target': 'red', 'generation_question': 'What color?'}
        return model, Tokenizer(), case, torch.zeros(1, 3, 336, 336, dtype=torch.float16)

    def test_three_captures_preserve_native_selection_basis_and_original_method(self):
        model, tokenizer, case, pixels = self.fixture()
        torch = self.torch
        old = model.encode_images.__func__
        state = torch.random.get_rng_state().clone()
        obs, native = capture_encoding(model, pixels, 590, 'native-input288')
        var, no_merge = capture_encoding(model, pixels, 590, 'native-input288-no-merge')
        full, baseline = capture_encoding(model, pixels, 590, 'baseline576')
        self.assertEqual(assert_selection_equal(obs, var)['status'], 'passed')
        self.assertTrue(torch.equal(no_merge['pre_merge'], no_merge['post_merge']))
        self.assertEqual(full['encoded']['kept_tokens'], 576)
        self.assertEqual(obs['encoded']['kept_tokens'], 288)
        self.assertIs(model.encode_images.__func__, old)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertEqual(validate_snapshot(native, 'native-input288', pixels)['encoded'], obs['encoded'])

    def test_replay_returns_independent_exact_clones_and_rejects_mutation(self):
        model, _, _, pixels = self.fixture()
        torch = self.torch
        _, snapshot = capture_encoding(model, pixels, 590, 'native-input288')
        old = model.encode_images.__func__
        with _generation_replay(model, pixels, 'native-input288', snapshot) as audit:
            values = model.encode_images(pixels)
            self.assertNotEqual(values[0].data_ptr(), snapshot['encoded'][0].data_ptr())
            self.assertTrue(torch.equal(values[0], snapshot['encoded'][0]))
        self.assertEqual(audit.calls, 1)
        self.assertIs(model.encode_images.__func__, old)
        with self.assertRaisesRegex(ValueError, 'more than one image prefill'):
            with _generation_replay(model, pixels, 'native-input288', snapshot):
                model.encode_images(pixels); model.encode_images(pixels)
        self.assertIs(model.encode_images.__func__, old)
        # Capture intentionally creates inference tensors. Inject corruption in
        # the permitted context so our own fingerprint guard, rather than
        # PyTorch's outside-inference mutation guard, is what rejects it.
        with torch.inference_mode():
            snapshot['encoded'][0][0, 0, 0] += 1
        with self.assertRaisesRegex(ValueError, 'mutated'):
            validate_snapshot(snapshot, 'native-input288', pixels)

    def test_actual_fixed_scoring_uses_one_replay_and_matches_existing_native_scorer(self):
        from run_rq2_input_merge import score_fixed_reference
        from rq1_native import max_score_delta
        model, tokenizer, case, pixels = self.fixture()
        for arm in ('baseline576', 'native-input288', 'native-input288-no-merge'):
            _, snapshot = capture_encoding(model, pixels, 590, arm)
            score, control = _score_replayed(tokenizer, model, case, pixels, arm, snapshot, 'Red')
            expected = score_fixed_reference(tokenizer, model, case, pixels, snapshot['encoded'], arm, 'Red')
            self.assertEqual(control['image_prefill_calls'], 1)
            self.assertEqual(max_score_delta(score, expected), 0)
            self.assertTrue(control['returned_features_and_mask_exact'])

    def test_actual_tiny_decoder_generation_and_scoring_reuse_the_same_tuple(self):
        from PIL import Image
        model, tokenizer, case, pixels = self.fixture()
        _, snapshot = capture_encoding(model, pixels, 590, 'native-input288')
        original = model.encode_images.__func__
        with patch('llava.mm_utils.process_images', return_value=(pixels, None)):
            row = generate_from_snapshot(tokenizer, model, None, Image.new('RGB', (2, 2)),
                case, pixels, 'native-input288', snapshot, 590, 'Red')
        self.assertTrue(row['generation_tokens']['generated_token_ids'])
        self.assertEqual(row['frozen_replay']['generation']['image_prefill_calls'], 1)
        self.assertEqual(row['frozen_replay']['reference_scoring']['image_prefill_calls'], 1)
        self.assertEqual(row['encoding'], snapshot['capture']['encoding'])
        self.assertIs(model.encode_images.__func__, original)

    def test_independent_numeric_drift_is_recorded_but_structural_change_is_rejected(self):
        from rq2_native import tensor_fingerprint
        model, _, _, pixels = self.fixture()
        torch = self.torch
        obs, snap = capture_encoding(model, pixels, 590, 'native-input288')
        other = {k: tuple(v.clone() for v in value) if k == 'encoded'
                 else value.clone() if k in ('pre_merge', 'post_merge') else copy.deepcopy(value)
                 for k, value in snap.items()}
        index = int(other['encoded'][1][0].nonzero()[0])
        other['encoded'][0][0, index, 0] += .01
        other_obs = copy.deepcopy(obs)
        other_obs['encoded']['features'] = tensor_fingerprint(other['encoded'][0])
        result = compare_realizations(obs, snap, other_obs, other)
        self.assertEqual(result['identity_controls']['status'], 'passed')
        self.assertGreater(result['numeric_differences']['stages']['projected']['full576']['max_abs'], 0)
        other_obs['input_merge']['merge_counts'][0] += 1
        with self.assertRaisesRegex(ValueError, 'source/selection identity changed'):
            compare_realizations(obs, snap, other_obs, other)


if __name__ == '__main__': unittest.main()
