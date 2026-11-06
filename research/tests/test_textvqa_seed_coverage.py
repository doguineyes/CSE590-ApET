"""CPU mask compatibility plus a real tiny-native test in the original runtime."""
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from rq_next_seed_protocol import SEEDS, random_plan, specification, adapter_policy

class SeedPolicyTests(unittest.TestCase):
    def test_all_archived_masks_unchanged(self):
        history=json.loads((ROOT/'research/protocols/textvqa_seed_history_v1.json').read_text())
        self.assertEqual(len(history['cases']),100)
        for c in history['cases'].values():
            for seed in SEEDS[:3]:
                self.assertEqual(random_plan(c['image_sha256'],seed,96,32),
                    c['arms'][f'pure-random-96-to-32-s{seed}']['random_plan'])

    def test_counts_mapping_and_independent_seed_streams(self):
        plans=[random_plan('a'*64,s,96,32) for s in SEEDS]
        self.assertEqual(len({tuple(p['input_original_positions']) for p in plans}),20)
        for p in plans:
            first,last=p['input_original_positions'],p['decoder_compact_positions']
            self.assertEqual(len(set(first)),96); self.assertEqual(len(set(last)),32)
            self.assertEqual(first,sorted(first)); self.assertEqual(last,sorted(last))
            self.assertTrue(set(p['decoder_original_positions']) <= set(first))
            self.assertEqual(p['decoder_original_positions'],[first[i] for i in last])

    def test_rejects_unregistered_policy(self):
        with self.assertRaises(ValueError): random_plan('a'*64,4,96,32)
        with self.assertRaises(ValueError): random_plan('a'*64,404,64,32)
        with self.assertRaises(ValueError): specification('pure-random-96-to-32-s4')

    def test_scoped_adapter_restores_on_exception(self):
        import rq2_staged_native as native
        original=(native.specification,native.random_plan)
        with self.assertRaises(RuntimeError):
            with adapter_policy():
                self.assertEqual(native.specification('pure-random-96-to-32-s2020')['random_seed'],2020)
                raise RuntimeError('test')
        self.assertEqual((native.specification,native.random_plan),original)

    def test_calibration_gate_rejects_partial_and_changed_sources(self):
        import tempfile
        from run_textvqa_seed_coverage import verify_calibration
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'report.json'
            r=dict(version='textvqa-seed-coverage-v1',mode='calibration',status='passed',source_sha256={'a':'b'},completed_questions=3,completed_predictions=18)
            p.write_text(json.dumps(r)); verify_calibration(p,{'a':'b'})
            with self.assertRaises(ValueError): verify_calibration(p,{'a':'changed'})
            r['completed_predictions']=17; p.write_text(json.dumps(r))
            with self.assertRaises(ValueError): verify_calibration(p,{'a':'b'})

class ExtendedSeedNativeTests(unittest.TestCase):
    def setUp(self):
        # Calling StagedNativeTests.fixture directly does not run its setUpClass,
        # which normally selects the headless backend before importing LLaVA.
        backend = patch.dict(os.environ, {'MPLBACKEND': 'Agg'})
        backend.start()
        self.addCleanup(backend.stop)

    def test_new_seed_actual_native_gather_and_cache(self):
        try:
            import torch
            from PIL import Image
        except ImportError:
            self.skipTest('Requires original Kaggle runtime')
        from research.tests.test_rq2_staged_native import StagedNativeTests
        from rq2_staged_native import capture_input,generate_condition
        from run_rq2_staged_calibration import compare_replay
        self.torch=torch
        previous=torch.get_num_threads(); torch.set_num_threads(1)
        try:
            model,tokenizer,case,pixels=StagedNativeTests.fixture(self)
            condition='pure-random-96-to-32-s404'
            with adapter_policy(), patch('llava.mm_utils.process_images',return_value=(pixels,None)):
                obs,snap=capture_input(model,pixels,590,condition,case['image_sha256'])
                row,layer=generate_condition(tokenizer,model,None,Image.new('RGB',(2,2)),case,pixels,condition,snap,590)
                repeat,_=generate_condition(tokenizer,model,None,Image.new('RGB',(2,2)),case,pixels,condition,snap,590,decoder_replay=layer)
                self.assertEqual(compare_replay(row,repeat)['status'],'passed')
                self.assertEqual(row['decoder']['random_plan'],obs['random_plan'])
                self.assertTrue(row['decoder']['original_contextual_vectors_exact'])
        finally: torch.set_num_threads(previous)

if __name__=='__main__': unittest.main()
