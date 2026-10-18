"""Kaggle prerequisites: actual shortened native forwards and gapped RoPE IDs."""

from pathlib import Path
import sys
import unittest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'research/tests')); sys.path.insert(0, str(REPO/'scripts'))
import test_rq1_layer_native as native_layer_tests
from rq1_layer_ih_native import retained_visual_hook, score_retained
from rq1_layer_native import boundary_hook
from rq1_native import max_score_delta, replaced, score_embeddings, score_from_logits


class NativeIHTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        native_layer_tests.NativeLayerTests.setUpClass()
        cls.torch = native_layer_tests.NativeLayerTests.torch

    def fixture(self):
        fixture = native_layer_tests.NativeLayerTests()
        return fixture.model_and_sequence()

    def test_native_retention_matches_independent_manual_suffix_and_preserves_gapped_positions(self):
        torch = self.torch
        model, prepared = self.fixture()
        capture = {}
        with boundary_hook(model, prepared, layer=1, observer=lambda h: capture.update(hidden=h.clone())):
            reference = score_embeddings(model, prepared)
        original = prepared['embeddings'].clone()
        for retained in [[], [23], list(range(576))]:
            actual, trace = score_retained(model, prepared, retained, layer=1, expected_boundary=capture['hidden'])
            ids = torch.tensor(trace['kept_positions'])
            # Independent oracle: run the original suffix modules directly on
            # the manually selected boundary states and original position IDs.
            hidden = capture['hidden'].index_select(1, ids)
            with torch.inference_mode():
                for block in model.model.layers[1:]:
                    rotary = block.self_attn.rotary_emb
                    native_forward = rotary.forward
                    with replaced(rotary, 'forward', lambda x, seq_len=None, native=native_forward:
                                  native(x, seq_len=prepared['embeddings'].shape[1])):
                        hidden = block(hidden, attention_mask=None, position_ids=ids.unsqueeze(0),
                                       past_key_value=None, output_attentions=False, use_cache=False)[0]
                logits = model.lm_head(model.model.norm(hidden)).float()
                expected = score_from_logits(logits, prepared['labels'].index_select(1, ids))
            self.assertLess(max_score_delta(actual, expected), 1e-6)
            self.assertEqual(trace['shortened_length'], 4+len(retained))
            if not retained: self.assertEqual(trace['kept_positions'], [0, 577, 578, 579])
            if len(retained) == 576: self.assertEqual(max_score_delta(actual, reference), 0)
        self.assertTrue(torch.equal(original, prepared['embeddings']))
        self.assertTrue(torch.equal(prepared['position_ids'][0], torch.arange(580)))

    def test_downstream_consumes_original_selected_states_and_rotary_extent_is_sufficient(self):
        torch = self.torch
        model, prepared = self.fixture(); capture = {}; seen = {}
        with boundary_hook(model, prepared, layer=1, observer=lambda h: capture.update(hidden=h.clone())):
            score_embeddings(model, prepared)
        with retained_visual_hook(model, prepared, [23], layer=1, expected_boundary=capture['hidden']) as trace:
            def inspect(module, args, kwargs):
                seen['hidden'] = args[0].clone(); seen['positions'] = kwargs['position_ids'].clone()
            handle = model.model.layers[1].register_forward_pre_hook(inspect, with_kwargs=True)
            try:
                with torch.inference_mode():
                    model(inputs_embeds=prepared['embeddings'], position_ids=prepared['position_ids'],
                          attention_mask=prepared['attention_mask'], use_cache=False, return_dict=True)
            finally: handle.remove()
        self.assertEqual(seen['positions'].tolist(), [[0, 24, 577, 578, 579]])
        self.assertTrue(torch.equal(seen['hidden'], capture['hidden'][:, [0, 24, 577, 578, 579]]))
        self.assertEqual(trace['blocks'], [1, 2]); self.assertEqual(trace['rotary'], [1, 2])

    def test_failed_preboundary_check_removes_all_hooks_and_reference_recovers(self):
        torch = self.torch
        model, prepared = self.fixture()
        before = score_embeddings(model, prepared)
        with self.assertRaisesRegex(RuntimeError, 'Pre-retention states'):
            score_retained(model, prepared, [23], layer=1, expected_boundary=torch.zeros_like(prepared['embeddings']))
        for block in model.model.layers:
            self.assertEqual(len(block._forward_pre_hooks), 0)
            self.assertEqual(len(block.self_attn.rotary_emb._forward_pre_hooks), 0)
        self.assertEqual(max_score_delta(before, score_embeddings(model, prepared)), 0)


if __name__ == '__main__': unittest.main()
