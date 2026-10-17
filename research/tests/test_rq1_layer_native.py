"""Kaggle prerequisites: real native block boundary, copy safety, and RNG parity.

Tiny random models only; no pretrained downloads. Local Mac can skip Torch;
the notebook explicitly imports Torch before executing these tests on Kaggle.
"""

import os
from pathlib import Path
import sys
import unittest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'scripts'))
from rq1_layer_native import boundary_hook, observe_layer_errors
from rq1_native import max_score_delta, score_embeddings, zero_patch
from smoke_original_apet import configure


class NativeLayerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try: import torch
        except ImportError: raise unittest.SkipTest('Torch checks require original Kaggle runtime')
        os.environ['MPLBACKEND'] = 'Agg'
        cls.torch = torch

    def model_and_sequence(self):
        torch = self.torch
        from llava.model.language_model.llava_llama_x import LlavaConfig, LlavaLlamaForCausalLM_X
        config = LlavaConfig(vocab_size=128, hidden_size=32, intermediate_size=64,
            num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=1024, bos_token_id=1, eos_token_id=2, pad_token_id=0)
        config._attn_implementation = 'sdpa'
        torch.manual_seed(590)
        model = LlavaLlamaForCausalLM_X(config, visual_token_num=576, selected_indices=[]).eval()
        configure(model, 'baseline')
        model.model.image_token_posi, model.model.image_tokens = [1], [576]
        labels = torch.full((1, 580), -100, dtype=torch.long); labels[0, -2:] = torch.tensor([10, 11])
        prepared = {'embeddings': torch.randn(1, 580, 32), 'position_ids': torch.arange(580).unsqueeze(0),
                    'attention_mask': torch.ones(1, 580, dtype=torch.bool), 'labels': labels, 'image_start': 1}
        return model, prepared

    def test_observation_noop_and_single_patch_change_real_native_forward(self):
        torch = self.torch
        model, prepared = self.model_and_sequence()
        original = prepared['embeddings'].clone()
        reference = score_embeddings(model, prepared)
        captured = {}
        with boundary_hook(model, prepared, layer=2, observer=lambda h: captured.update(hidden=h.clone())):
            observed = score_embeddings(model, prepared)
        with boundary_hook(model, prepared, layer=2, noop=True):
            noop = score_embeddings(model, prepared)
        self.assertEqual(max_score_delta(reference, observed), 0)
        self.assertEqual(max_score_delta(reference, noop), 0)
        expected_hidden = zero_patch(captured['hidden'], 1, 23)
        downstream = {}
        # A pre-hook on block 3 sees the modified output of block 2. This checks
        # actual boundary placement and that later layers consume the intervention.
        def before_next(module, args): downstream['hidden'] = args[0].clone()
        handle = model.model.layers[2].register_forward_pre_hook(before_next)
        try:
            with boundary_hook(model, prepared, layer=2, token_index=23):
                changed = score_embeddings(model, prepared)
        finally: handle.remove()
        self.assertTrue(torch.equal(downstream['hidden'], expected_hidden))
        self.assertTrue(torch.equal(prepared['embeddings'], original))
        self.assertGreater(max_score_delta(reference, changed), 0)
        self.assertEqual(len(model.model.layers[1]._forward_hooks), 0)

    def test_hook_removed_when_observer_raises_and_pruning_is_rejected(self):
        model, prepared = self.model_and_sequence()
        def fail(hidden): raise RuntimeError('observer failure')
        with self.assertRaisesRegex(RuntimeError, 'observer failure'):
            with boundary_hook(model, prepared, layer=2, observer=fail):
                score_embeddings(model, prepared)
        self.assertEqual(len(model.model.layers[1]._forward_hooks), 0)
        model.model.layer_list = [2]
        with self.assertRaises(ValueError):
            with boundary_hook(model, prepared, layer=2): pass

    def test_original_error_observation_preserves_states_metadata_and_rng(self):
        torch = self.torch
        model, prepared = self.model_and_sequence()
        captured = {}
        with boundary_hook(model, prepared, layer=2, observer=lambda h: captured.update(hidden=h.clone())):
            reference = score_embeddings(model, prepared)
        hidden = captured['hidden']; original = hidden.clone()
        positions = prepared['position_ids'].clone(); labels = prepared['labels'].clone()
        mask = prepared['attention_mask'].clone(); rng = torch.get_rng_state().clone()
        observation = observe_layer_errors(model, prepared, hidden, 590, layer=2)
        self.assertTrue(torch.equal(hidden, original))
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        for name, saved in [('position_ids', positions), ('labels', labels), ('attention_mask', mask)]:
            self.assertTrue(torch.equal(prepared[name], saved))
        self.assertEqual(model.model.layer_list, [])
        self.assertEqual(model.model.image_token_list, [576])
        self.assertEqual(model.model.image_token_posi, [1])
        self.assertEqual(len(observation['stage']['errors']), 576)
        self.assertEqual(observation['stage']['feature_dimension'], 32)
        self.assertEqual(observation['stage']['layer'], 2)
        self.assertEqual(len(observation['stage']['basis_indices']), 10)
        self.assertTrue(observation['compressed_output_discarded'])
        self.assertEqual(max_score_delta(reference, score_embeddings(model, prepared)), 0)
        repeated = observe_layer_errors(model, prepared, hidden, 590, layer=2)
        self.assertEqual(observation, repeated)


if __name__ == '__main__': unittest.main()
