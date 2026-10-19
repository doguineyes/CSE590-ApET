"""Native random-selection and oracle controls; run in the locked Kaggle runtime."""
import copy
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

REPO=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO));sys.path.insert(0,str(REPO/'scripts'))
from rq2_crossed_native import capture_condition,generate_condition,oracle_snapshot,selection_override,matched_oracle_control
from rq2_crossed_protocol import RANDOM_SEEDS
from rq2_native import assert_selection_equal
from rq2_frozen_native import validate_snapshot


class CrossedOraclePolicyTests(unittest.TestCase):
    def test_control_choice_uses_only_merge_count_and_index(self):
        observation={'input_merge':{'target_indices':[317,50,20,10], 'merge_counts':[3,3,3,0]}}
        self.assertEqual(matched_oracle_control(observation),20)
        observation['input_merge']['merge_counts']=[3,5,2,0]
        self.assertEqual(matched_oracle_control(observation),20)


class CrossedNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:import torch
        except ImportError:raise unittest.SkipTest('Crossed native controls need the original Kaggle Torch runtime')
        cls.torch=torch;os.environ['MPLBACKEND']='Agg'

    def fixture(self):
        # Reuse the passed v2 tiny-model fixture without loading its TestCase into
        # this module's unittest discovery or duplicating decoder setup code.
        from research.tests.test_rq2_frozen_native import FrozenNativeTests
        helper=FrozenNativeTests('test_three_captures_preserve_native_selection_basis_and_original_method')
        helper.torch=self.torch
        return helper.fixture()

    def test_random_pairs_keep_native_E_basis_premerge_and_do_not_change_torch_rng(self):
        torch=self.torch;model,_,_,pixels=self.fixture();before=torch.random.get_rng_state().clone()
        ref,_=capture_condition(model,pixels,590,'E-merge');original=torch.topk;chosen=[]
        for seed in RANDOM_SEEDS:
            merge,snap=capture_condition(model,pixels,590,f'random-{seed}-merge',ref,'image')
            no_merge,other=capture_condition(model,pixels,590,f'random-{seed}-no-merge',ref,'image')
            self.assertEqual(assert_selection_equal(merge,no_merge)['status'],'passed')
            self.assertEqual(merge['pre_merge'],ref['pre_merge']);self.assertEqual(merge['input']['errors'],ref['input']['errors'])
            self.assertEqual(merge['input']['selected_indices'][-10:],ref['input']['selected_indices'][-10:])
            self.assertNotEqual(merge['encoded']['mask'],ref['encoded']['mask'])
            chosen.append(merge['input']['selected_indices']);self.assertIs(torch.topk,original)
            self.assertEqual(snap['experiment']['random_selection']['native_selector_calls'],1)
        self.assertNotEqual(chosen[0],chosen[1]);self.assertTrue(torch.equal(before,torch.random.get_rng_state()))

    def test_topk_and_native_methods_restore_when_selector_identity_fails(self):
        torch=self.torch;model,_,_,pixels=self.fixture();ref,_=capture_condition(model,pixels,590,'E-merge')
        damaged=copy.deepcopy(ref);damaged['input']['errors'][0]+=1;original=torch.topk;method=model.encode_images.__func__
        with self.assertRaisesRegex(ValueError,'Native E/top-k changed'):
            capture_condition(model,pixels,590,'random-101-merge',damaged,'image')
        self.assertIs(torch.topk,original);self.assertIs(model.encode_images.__func__,method)
        # Unrelated torch.topk calls inside the scope must remain original.
        with self.assertRaisesRegex(ValueError,'not executed exactly once'):
            with selection_override(ref,ref['input']['selected_indices']):
                self.assertTrue(torch.equal(torch.topk(torch.tensor([1.,3.,2.]),2)[0],torch.tensor([3.,2.])))
        self.assertIs(torch.topk,original)

    def test_random_tuple_runs_actual_tiny_generation_and_fixed_scoring(self):
        from PIL import Image
        model,tokenizer,case,pixels=self.fixture();ref,_=capture_condition(model,pixels,590,'E-merge')
        obs,snapshot=capture_condition(model,pixels,590,'random-101-merge',ref,'image')
        with patch('llava.mm_utils.process_images',return_value=(pixels,None)):
            row=generate_condition(tokenizer,model,None,Image.new('RGB',(2,2)),case,pixels,
                                   'random-101-merge',snapshot,590,'Red')
        self.assertEqual(row['condition'],'random-101-merge');self.assertEqual(row['selection_seed'],101)
        self.assertEqual(row['encoding'],obs)
        for k in ('generation','reference_scoring'):self.assertEqual(row['frozen_replay'][k]['image_prefill_calls'],1)

    def test_oracle_sham_is_exact_and_restoration_changes_only_one_retained_state(self):
        torch=self.torch;model,_,_,pixels=self.fixture();obs,snapshot=capture_condition(model,pixels,590,'E-merge')
        target=next(i for i,n in zip(obs['input_merge']['target_indices'],obs['input_merge']['merge_counts']) if n>0)
        sham=oracle_snapshot(model,pixels,snapshot,590,None)
        self.assertTrue(torch.equal(sham['encoded'][0],snapshot['encoded'][0]))
        changed=oracle_snapshot(model,pixels,snapshot,590,target)
        self.assertTrue(torch.equal(changed['encoded'][0][:,target],snapshot['pre_merge'][:,target].float()))
        others=[i for i in range(576) if i!=target]
        self.assertTrue(torch.equal(changed['encoded'][0][:,others],snapshot['encoded'][0][:,others]))
        self.assertEqual(changed['capture']['oracle']['changed_projected_positions'],[target])
        self.assertEqual(validate_snapshot(changed,'native-input288',pixels)['encoded'],changed['capture']['encoding']['encoded'])


if __name__=='__main__':unittest.main()
