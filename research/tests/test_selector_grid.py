"""CPU selector properties and actual tiny original-LLaVA execution on Kaggle."""
import os
os.environ['MPLBACKEND']='Agg'
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
from selector_grid_protocol import conditions,choose,specification
from textvqa_patch_properties import detail_without_padding,padding_fraction

class PolicyTests(unittest.TestCase):
    def test_counts_and_modes(self):
        self.assertEqual(len(conditions('textvqa')),31);self.assertEqual(len(conditions('gqa')),14)
        for name in conditions('textvqa'):specification(name)

    def test_random_matches_existing_and_square_nonpad(self):
        from rq_next_seed_protocol import random_plan
        for seed in (101,202,303,404,505,1010):
            plan=random_plan('a'*64,seed,96,32)
            for stage,n,k,key in [('input',576,96,'input_original_positions'),('decoder',96,32,'decoder_compact_positions')]:
                args=(k,np.zeros(n),np.ones(n),None,'a'*64,seed,stage)
                self.assertEqual(choose('random',*args),plan[key])
                self.assertEqual(choose('nonpad',*args),plan[key])

    def test_nonpad_fill_and_stable_score_ties(self):
        pad=np.array([0]*20+[.5]*20+[1]*56)
        ids=choose('nonpad',32,pad,np.zeros(96),None,'a'*64,101,'decoder')
        self.assertEqual(sum(i<20 for i in ids),20);self.assertEqual(sum(20<=i<40 for i in ids),12)
        self.assertEqual(choose('detail',32,pad,np.zeros(96),None,'a'*64,None,'decoder'),list(range(32)))

    def test_detail_removes_internal_pad_transition(self):
        # Image content starts at x=84; choose width170: boundary at83 falls inside a patch.
        pixels=np.zeros((1,3,336,336));pixels[:,:,:,83:253]=1
        self.assertTrue(np.all(detail_without_padding(pixels,170,336)==0))
        pixels[:,:,:,100:110]=3
        self.assertGreater(detail_without_padding(pixels,170,336).max(),0)

    def test_archived_offline_maps(self):
        ref=json.loads((ROOT/'research/protocols/selector_grid_textvqa_reference_v1.json').read_text())
        self.assertEqual(len(ref['cases']),100)
        for v in ref['cases'].values():
            for s,key in [('detail','expected_detail_top96'),('error','expected_E_top96')]:
                self.assertEqual(choose(s,96,v['padding'],v['detail'],v['errors'],v['image_sha256'],None,'input'),v[key])

class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:import torch
        except ImportError:raise unittest.SkipTest('Requires original Kaggle Torch runtime')
        cls.torch=torch;cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):cls.torch.set_num_threads(cls.threads)

    def test_merge_all_kept_targets_matches_manual_mean(self):
        from selector_grid_native import merge_all_targets
        x=self.torch.tensor([[2.,0.],[0.,2.],[1.,0.],[0.,1.]])
        out,graph=merge_all_targets(x,[0,1])
        self.torch.testing.assert_close(out[:2],self.torch.tensor([[1.5,0.],[0.,1.5]]))
        self.assertEqual(graph['assigned_target_indices'],[0,1]);self.assertEqual(graph['merge_counts'],[1,1])
        identity,g=merge_all_targets(x,list(range(4)))
        self.assertTrue(self.torch.equal(identity,x));self.assertEqual(g['dropped_indices'],[])

    def test_all_selector_modes_actual_generation_replay(self):
        from PIL import Image
        from research.tests.test_rq2_staged_native import StagedNativeTests
        from selector_grid_native import adapter,make_input
        from run_rq2_staged_calibration import compare_replay
        import rq2_staged_native as staged
        model,tokenizer,case,pixels=StagedNativeTests.fixture(self)
        raw=model.get_vision_tower().source[:,1:].clone()
        _,full=staged.capture_input(model,pixels,590,'baseline576',case['image_sha256'])
        native_obs,native=staged.capture_input(model,pixels,590,'native-96-to-32',case['image_sha256'])
        props=dict(padding=padding_fraction(224,336),detail=np.arange(576,dtype=float),errors=np.array(native_obs['input']['errors']))
        with patch('llava.mm_utils.process_images',return_value=(pixels,None)):
            for selector in ('random','nonpad','detail','error'):
                for compensation in ('drop','merge'):
                    seed=101 if selector in ('random','nonpad') else 0
                    name=f'grid-{selector}-{compensation}-s{seed}'
                    chosen=choose(selector,96,props['padding'],props['detail'],props['errors'],case['image_sha256'],seed,'input')
                    with adapter(props,chosen):
                        obs,snap=make_input(model,pixels,name,590,case['image_sha256'],full,native,raw,props)
                        row,layer=staged.generate_condition(tokenizer,model,None,Image.new('RGB',(2,2)),case,pixels,name,snap,590)
                        again,_=staged.generate_condition(tokenizer,model,None,Image.new('RGB',(2,2)),case,pixels,name,snap,590,decoder_replay=layer)
                        self.assertEqual(compare_replay(row,again)['status'],'passed')
                        self.assertEqual(len(row['decoder']['selected_compact_positions']),32)
                        self.assertFalse(row['decoder']['basis_inserted'])
                        self.assertEqual(obs['encoded']['kept_tokens'],96)
                        if compensation=='merge':self.assertEqual(sum(row['decoder']['merge']['merge_counts']),64)

    def test_random_drop_matches_existing_operator_scores(self):
        from PIL import Image
        from research.tests.test_rq2_staged_native import StagedNativeTests
        from selector_grid_native import adapter,make_input
        import rq2_staged_native as staged
        model,tokenizer,case,pixels=StagedNativeTests.fixture(self)
        key=case['image_sha256'];raw=model.get_vision_tower().source[:,1:].clone()
        _,full=staged.capture_input(model,pixels,590,'baseline576',key)
        no,native=staged.capture_input(model,pixels,590,'native-96-to-32',key)
        props=dict(padding=np.zeros(576),detail=np.zeros(576),errors=no['input']['errors'])
        original='pure-random-96-to-32-s101';name='grid-random-drop-s101'
        with patch('llava.mm_utils.process_images',return_value=(pixels,None)):
            _,snap=staged.capture_input(model,pixels,590,original,key)
            expected,_=staged.generate_condition(tokenizer,model,None,Image.new('RGB',(2,2)),case,pixels,original,snap,590)
            selected=choose('random',96,props['padding'],props['detail'],props['errors'],key,101,'input')
            with adapter(props,selected):
                _,snap=make_input(model,pixels,name,590,key,full,native,raw,props)
                actual,_=staged.generate_condition(tokenizer,model,None,Image.new('RGB',(2,2)),case,pixels,name,snap,590)
            self.assertEqual(actual['generation_tokens'],expected['generation_tokens'])
            self.assertEqual(actual['scoring'],expected['scoring'])

if __name__=='__main__':unittest.main()
