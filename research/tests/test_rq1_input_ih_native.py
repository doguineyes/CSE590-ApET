"""Kaggle-only tiny native checks: input deletion, causal shift and original IDs."""

from pathlib import Path
import sys
import unittest

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'research/tests'));sys.path.insert(0,str(REPO/'scripts'))
import test_rq1_layer_native as native_tests
from rq1_input_ih_native import score_retained_input
from rq1_native import max_score_delta,replaced,score_embeddings,score_from_logits


class InputIHNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        native_tests.NativeLayerTests.setUpClass();cls.torch=native_tests.NativeLayerTests.torch

    def fixture(self):return native_tests.NativeLayerTests().model_and_sequence()

    def test_real_native_scores_match_independent_manual_input_oracle(self):
        torch=self.torch;model,prepared=self.fixture();original=prepared['embeddings'].clone()
        reference=score_embeddings(model,prepared)
        for retained in [[],[0],[23],[575],list(range(576))]:
            actual,trace=score_retained_input(model,prepared,retained)
            # Independent explicit geometry, not the retention helper's map.
            positions=[0]+[1+i for i in retained]+[577,578,579]
            ids=torch.tensor(positions);hidden=prepared['embeddings'].index_select(1,ids)
            with torch.inference_mode():
                for block in model.model.layers:
                    rotary=block.self_attn.rotary_emb;native=rotary.forward
                    with replaced(rotary,'forward',lambda x,seq_len=None,fn=native:fn(x,seq_len=580)):
                        hidden=block(hidden,attention_mask=None,position_ids=ids.unsqueeze(0),
                                     past_key_value=None,output_attentions=False,use_cache=False)[0]
                expected=score_from_logits(model.lm_head(model.model.norm(hidden)).float(),prepared['labels'].index_select(1,ids))
            self.assertLess(max_score_delta(actual,expected),1e-6)
            self.assertEqual(trace['kept_positions'],positions)
            self.assertEqual(trace['shortened_length'],4+len(retained))
            if len(retained)==576:self.assertEqual(max_score_delta(actual,reference),0)
        self.assertTrue(torch.equal(original,prepared['embeddings']))

    def test_first_block_sees_only_selected_raw_embeddings_with_gapped_positions(self):
        torch=self.torch;model,prepared=self.fixture();seen={}
        def observe(module,args,kwargs):
            seen.update(hidden=args[0].clone(),positions=kwargs['position_ids'].tolist())
        handle=model.model.layers[0].register_forward_pre_hook(observe,with_kwargs=True)
        try:score,trace=score_retained_input(model,prepared,[23])
        finally:handle.remove()
        self.assertEqual(seen['positions'],[[0,24,577,578,579]])
        self.assertTrue(torch.equal(seen['hidden'],prepared['embeddings'][:,[0,24,577,578,579]]))
        # Future gold embeddings cannot influence the first target prediction.
        changed={**prepared,'embeddings':prepared['embeddings'].clone()}
        changed['embeddings'][0,-2:]+=100
        other,_=score_retained_input(model,changed,[23])
        self.assertEqual(score['token_logprobs'][0],other['token_logprobs'][0])

    def test_exception_removes_hooks_and_native_reference_recovers(self):
        model,prepared=self.fixture();reference=score_embeddings(model,prepared)
        def fail(*a,**k):raise RuntimeError('test head failure')
        with replaced(model.lm_head,'forward',fail):
            with self.assertRaisesRegex(RuntimeError,'test head failure'):score_retained_input(model,prepared,[23])
        for block in model.model.layers:self.assertEqual(len(block.self_attn.rotary_emb._forward_pre_hooks),0)
        self.assertEqual(max_score_delta(reference,score_embeddings(model,prepared)),0)
        model.model.layer_list=[1]
        with self.assertRaises(ValueError):score_retained_input(model,prepared,[])


if __name__=='__main__':unittest.main()
