"""Policy checks locally; original tiny-model execution required in Kaggle."""
import os
os.environ['MPLBACKEND']='Agg'
import sys,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
from basis_localization_protocol import conditions,specification

class PolicyTests(unittest.TestCase):
    def test_fixed_grid(self):
        names=conditions();self.assertEqual(len(names),16);self.assertEqual(len(set(names)),16)
        for name in names:
            s=specification(name)
            if name!='baseline576':self.assertEqual((s['input_tokens'],s['decoder_tokens']),(96,32))
        self.assertEqual([specification(x)['random_seed'] for x in names if 'random-s' in x],[101,202,303,404,505])
    def test_reject_unknown(self):
        with self.assertRaises(Exception):specification('loc-error-random-s999')
        with self.assertRaises(Exception):conditions('gqa')
    def test_decoder_basis_uses_presort_ranked_slots(self):
        from basis_localization_native import decoder_basis_pairs
        ranked=[31,5,29,2,27,1,25,0,23,4,21,6,19,8,17,10,15,12,13,14,11,16,9,18,7,20,3,22,24,26,28,30]
        basis=list(range(40,50))
        pairs=decoder_basis_pairs(ranked,basis)
        self.assertEqual(pairs,list(zip([9,18,7,20,3,22,24,26,28,30],basis)))
        self.assertNotEqual(pairs,decoder_basis_pairs(sorted(ranked),basis))
    def test_lineage_composes_replacements(self):
        from basis_localization_native import compose_lineage
        self.assertEqual(compose_lineage([70,11,90,13],[2,0,3]),[90,70,13])
    def test_native_stage_factorial(self):
        for name,expected in [('loc-native-drop',('native','native')),('loc-error-native',('error','native')),('loc-native-error',('native','error'))]:
            s=specification(name);self.assertEqual((s['selector'],s['decoder_selector']),expected)
    def test_distinct_stages(self):
        s=specification('loc-detail-error');self.assertEqual(s['selector'],'detail');self.assertEqual(s['decoder_selector'],'error')

class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:import torch
        except ImportError:raise unittest.SkipTest('Requires original Torch runtime')
        cls.torch=torch;cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):cls.torch.set_num_threads(cls.threads)
    def test_fps_overflow_diagnostic_is_observational(self):
        from basis_localization_native import fps_numerical_audit
        from llava.model.utils import fps
        torch=self.torch
        for x,overflow in [(torch.tensor([[[0.,0.],[1.,0.],[0.,1.]]]),False),
                           (torch.tensor([[[0.,0.],[300.,0.],[0.,300.]]],dtype=torch.float16),True)]:
            ids=fps(x,3)[0].tolist();before=x.clone();rng=torch.get_rng_state().clone()
            audit=fps_numerical_audit(x[0],ids)
            self.assertEqual(audit['any_inf'],overflow)
            self.assertTrue(audit['selected_sequence_reproduced'])
            self.assertTrue(torch.equal(before,x));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
    def test_all_arms_replay_and_reference_parity(self):
        from PIL import Image
        from research.tests.test_rq2_staged_native import StagedNativeTests
        from basis_localization_native import adapter,make_input
        import selector_grid_native as grid
        import rq2_staged_native as staged
        from run_rq2_staged_calibration import compare_replay
        model,tok,case,pixels=StagedNativeTests.fixture(self);key=case['image_sha256']
        raw=model.get_vision_tower().source[:,1:].clone()
        _,full=staged.capture_input(model,pixels,590,'baseline576',key)
        no,native=staged.capture_input(model,pixels,590,'native-96-to-32',key)
        props=dict(padding=np.zeros(576),detail=np.arange(576,dtype=float),errors=np.asarray(no['input']['errors']))
        for name in conditions()[2:]:
            selected=[]
            with patch('llava.mm_utils.process_images',return_value=(pixels,None)),adapter(props,selected):
                obs,snap=make_input(model,pixels,name,590,key,full,native,raw,props);selected[:]=obs['selected_original_positions']
                row,layer=staged.generate_condition(tok,model,None,Image.new('RGB',(2,2)),case,pixels,name,snap,590)
                again,_=staged.generate_condition(tok,model,None,Image.new('RGB',(2,2)),case,pixels,name,snap,590,decoder_replay=layer)
                self.assertEqual(compare_replay(row,again)['status'],'passed')
                self.assertEqual(len(set(row['decoder']['selected_compact_positions'])),32)
                self.assertEqual(layer['source_visual'].shape[1],96)
                self.assertTrue(self.torch.isfinite(layer['source_visual']).all())
                probe,_=staged.generate_condition(tok,model,None,Image.new('RGB',(2,2)),case,pixels,name,snap,590,probe=True)
                self.assertEqual(probe['cache']['status'],'passed')
            if name.startswith('grid-'):
                with patch('llava.mm_utils.process_images',return_value=(pixels,None)),grid.adapter(props,selected):
                    _,old=grid.make_input(model,pixels,name,590,key,full,native,raw,props)
                    expected,_=staged.generate_condition(tok,model,None,Image.new('RGB',(2,2)),case,pixels,name,old,590)
                self.assertEqual(row['generation_tokens'],expected['generation_tokens'])
                self.assertEqual(row['decoder']['selected_compact_positions'],expected['decoder']['selected_compact_positions'])
            if name=='loc-native-drop':
                from rq2_native import native_input_variant
                with staged.runtime_scope(model,'native-96-to-32',590,capture_input=True),self.torch.inference_mode():
                    with native_input_variant(model,'no-merge',capture_tensors=True):expected=model.encode_images(pixels)
                self.assertTrue(all(self.torch.equal(a,b) for a,b in zip(expected,snap['encoded'])))
            self.assertEqual(staged.specification.__module__,'rq2_staged_protocol')

if __name__=='__main__':unittest.main()
