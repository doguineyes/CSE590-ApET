import os,sys,unittest
from pathlib import Path
from unittest.mock import patch
os.environ['MPLBACKEND']='Agg'
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
import numpy as np
from norm_intervention_protocol import conditions,specification,swap_plan
class PolicyTests(unittest.TestCase):
    def test_grid(self):
        self.assertEqual(len(conditions()),31)
        self.assertEqual(len(set(conditions())),31)
        for n in conditions()[1:]:self.assertEqual(specification(n)['decoder_tokens'],32)
    def test_replacements(self):
        norms=np.ones(96);norms[31]=100;norms[30]=50
        errors=np.arange(96,dtype=float)
        for n in conditions():
            if not n.startswith('loc-norm-') or n.endswith('fp32'):continue
            sources=list(range(32)) if '-c' not in n else list(range(32,64))
            a=swap_plan(n,norms,sources,list(range(32)),set(range(22,32)),errors,'image',lambda slots:95)
            self.assertEqual(a,swap_plan(n,norms,sources,list(range(32)),set(range(22,32)),errors,'image',lambda slots:95))
            for j,old,new in a:self.assertEqual(old,sources[j]);self.assertNotIn(new,sources)
            if '-a1c-' in n or '-a2c-' in n:
                self.assertTrue(all(j in range(22,30) for j,_,_ in a))
    def test_restoration_already_present(self):
        n=np.ones(96);n[0]=100
        self.assertEqual(swap_plan('loc-norm-c1',n,list(range(32)),list(range(32)),[],list(range(96)),'x'),[])
class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:import torch
        except ImportError:raise unittest.SkipTest('Requires original Torch runtime')
        cls.torch=torch;cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):cls.torch.set_num_threads(cls.threads)
    def test_all_arms_replay_and_cache(self):
        from PIL import Image
        from research.tests.test_rq2_staged_native import StagedNativeTests
        from norm_intervention_native import adapter,make_input
        import rq2_staged_native as staged
        from run_rq2_staged_calibration import compare_replay
        model,tok,case,pixels=StagedNativeTests.fixture(self);key=case['image_sha256']
        raw=model.get_vision_tower().source[:,1:].clone()
        _,full=staged.capture_input(model,pixels,590,'baseline576',key)
        no,native=staged.capture_input(model,pixels,590,'native-96-to-32',key)
        props=dict(padding=np.zeros(576),detail=np.arange(576,dtype=float),errors=np.asarray(no['input']['errors']))
        pools={}
        for name in conditions()[1:]:
            selected=[]
            with patch('llava.mm_utils.process_images',return_value=(pixels,None)),adapter(props,selected):
                obs,snap=make_input(model,pixels,name,590,key,full,native,raw,props);selected[:]=obs['selected_original_positions']
                row,layer=staged.generate_condition(tok,model,None,Image.new('RGB',(2,2)),case,pixels,name,snap,590)
                again,_=staged.generate_condition(tok,model,None,Image.new('RGB',(2,2)),case,pixels,name,snap,590,decoder_replay=layer)
                self.assertEqual(compare_replay(row,again)['status'],'passed')
                self.assertEqual(len(set(row['decoder']['selected_compact_positions'])),32)
                from norm_intervention_protocol import parent
                if name==parent(name):pools[name]=layer['source_visual'].clone()
                else:self.assertTrue(self.torch.equal(pools[parent(name)],layer['source_visual']))
                probe,_=staged.generate_condition(tok,model,None,Image.new('RGB',(2,2)),case,pixels,name,snap,590,probe=True)
                self.assertEqual(probe['cache']['status'],'passed')
if __name__=='__main__':unittest.main()
