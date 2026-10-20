"""Actual native Torch prerequisites, executed in the original Kaggle runtime."""
from collections import Counter
import copy
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

REPO=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO));sys.path.insert(0,str(REPO/'scripts'))
from rq2_routing_native import capture_condition, generate_condition, routing_override
from rq2_routing_protocol import CONDITIONS, routing_plan
from rq2_frozen_native import validate_snapshot


class RoutingNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try: import torch
        except ImportError: raise unittest.SkipTest('Routing controls need the original Kaggle Torch runtime')
        cls.torch=torch;os.environ['MPLBACKEND']='Agg'

    def fixture(self):
        from research.tests.test_rq2_frozen_native import FrozenNativeTests
        helper=FrozenNativeTests('test_three_captures_preserve_native_selection_basis_and_original_method')
        helper.torch=self.torch
        return helper.fixture()

    def test_all_variants_use_native_selection_basis_arithmetic_and_restore_rng(self):
        torch=self.torch;model,_,_,pixels=self.fixture()
        state=torch.random.get_rng_state().clone();ref,_=capture_condition(model,pixels,590,'native-nearest')
        for condition in list(CONDITIONS)[2:]:
            obs,snapshot=capture_condition(model,pixels,590,condition,ref,'image')
            self.assertEqual(obs['pre_merge'],ref['pre_merge']);self.assertEqual(obs['input'],ref['input'])
            self.assertEqual(obs['basis_slot_to_source'],ref['basis_slot_to_source'])
            self.assertEqual(obs['encoded']['mask'],ref['encoded']['mask'])
            self.assertEqual(snapshot['experiment']['routing_identity']['status'],'passed')
            validate_snapshot(snapshot,CONDITIONS[condition]['replay_arm'],pixels)
        self.assertTrue(torch.equal(state,torch.random.get_rng_state()))

    def test_shuffled_destinations_preserve_counts_and_native_fp16_average(self):
        torch=self.torch;model,_,_,pixels=self.fixture();ref,_=capture_condition(model,pixels,590,'native-nearest')
        obs,snapshot=capture_condition(model,pixels,590,'shuffled-101',ref,'image')
        graph=obs['input_merge'];self.assertEqual(graph['merge_counts'],ref['input_merge']['merge_counts'])
        self.assertEqual(Counter(graph['assigned_target_indices']),Counter(ref['input_merge']['assigned_target_indices']))
        # The fixture runs on CPU, allowing an exact independently assembled
        # scatter/denominator check without assuming bitwise CUDA repeatability.
        targets=graph['target_indices'];src=graph['dropped_indices'];lookup={t:i for i,t in enumerate(targets)}
        indices=torch.tensor([[lookup[t] for t in graph['assigned_target_indices']]])
        expected=snapshot['pre_merge'][:,targets].clone();features=snapshot['pre_merge'][:,src]
        expected.scatter_add_(1,indices.unsqueeze(-1).expand_as(features),features)
        expected/=(1+torch.tensor([graph['merge_counts']]).unsqueeze(-1))
        self.assertTrue(torch.equal(expected,snapshot['post_merge'][:,targets]))

    def test_protection_keeps_exact_features_and_routes_to_nearest_eligible(self):
        torch=self.torch;model,_,_,pixels=self.fixture();ref,_=capture_condition(model,pixels,590,'native-nearest')
        for condition in ('protect-high-E','protect-random-101'):
            obs,snapshot=capture_condition(model,pixels,590,condition,ref,'image')
            plan=snapshot['experiment']['plan'];graph=obs['input_merge'];protected=plan['protected_targets']
            self.assertEqual(len(protected),28)
            self.assertTrue(torch.equal(snapshot['pre_merge'][:,protected],snapshot['post_merge'][:,protected]))
            self.assertFalse(set(protected)&set(graph['assigned_target_indices']))
            targets=graph['target_indices'];sources=graph['dropped_indices']
            similarity=torch.nn.functional.cosine_similarity(snapshot['pre_merge'][:,sources].unsqueeze(2),
                snapshot['pre_merge'][:,targets].unsqueeze(1),dim=3)
            similarity[:,:,[targets.index(t) for t in protected]]=-torch.inf
            self.assertEqual([targets[i] for i in similarity.argmax(dim=2)[0].tolist()],graph['assigned_target_indices'])

    def test_cosine_and_native_methods_restore_on_failure_and_unrelated_calls_delegate(self):
        torch=self.torch;model,_,_,pixels=self.fixture();ref,_=capture_condition(model,pixels,590,'native-nearest')
        import torch.nn.functional as functional
        original=functional.cosine_similarity;method=model.encode_images.__func__
        broken=copy.deepcopy(ref);d=broken['input_merge']['assigned_target_indices'];targets=broken['input_merge']['target_indices']
        d[0]=next(t for t in targets if t!=d[0]);broken['input_merge']['merge_counts']=[Counter(d)[t] for t in targets]
        with self.assertRaisesRegex(ValueError,'nearest graph changed'):
            capture_condition(model,pixels,590,'shuffled-101',broken,'image')
        self.assertIs(functional.cosine_similarity,original);self.assertIs(model.encode_images.__func__,method)
        plan=routing_plan(ref,'image',590,'shuffled-101')
        with self.assertRaisesRegex(ValueError,'exactly once'):
            with routing_override(ref,plan):
                x=torch.ones((1,2));self.assertTrue(torch.equal(functional.cosine_similarity(x,x),original(x,x)))
        self.assertIs(functional.cosine_similarity,original)

    def test_modified_tuple_runs_actual_tiny_generation_and_reference_replay(self):
        from PIL import Image
        model,tokenizer,case,pixels=self.fixture();ref,_=capture_condition(model,pixels,590,'native-nearest')
        obs,snapshot=capture_condition(model,pixels,590,'protect-high-E',ref,'image')
        with patch('llava.mm_utils.process_images',return_value=(pixels,None)):
            row=generate_condition(tokenizer,model,None,Image.new('RGB',(2,2)),case,pixels,
                                   'protect-high-E',snapshot,590,'Red')
        self.assertEqual(row['encoding'],obs);self.assertEqual(row['routing'],'protect-E')
        for kind in ('generation','reference_scoring'):
            self.assertEqual(row['frozen_replay'][kind]['image_prefill_calls'],1)


if __name__=='__main__':unittest.main()
