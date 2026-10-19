"""CPU orchestration/export gates; synthetic fixtures are never GPU evidence."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

REPO=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO/'scripts'))
import run_rq2_crossed as runner
from rq1_protocol import write_json,sha256
from rq2_crossed_protocol import load_profile,CONDITIONS,RANDOM_SEEDS,summarize,random_indices
from rq2_frozen_protocol import load_profile as parent_profile
from rq2_protocol import score_response
from research.tests.test_rq2_runner import fixture_inputs,synthetic_encoding


class CrossedRunnerTests(unittest.TestCase):
    def test_successful_mode_exports_keep_oracle_separate_from_calibration_and_full(self):
        profile=load_profile();fake_torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True,device_count=lambda:2))
        def synthetic(args,report,profile,manifests,records,loaded):
            ids=profile['calibration_case_ids'] if args.mode=='calibration' else [runner.ORACLE_CASE]
            report.update(synthetic_cpu_test=True,expected_case_ids=ids,
                questions=[{'case_id':cid,'status':'passed'} for cid in ids],
                progress={'completed_questions':len(ids),'expected_questions':len(ids)})
        for mode,flag in [('calibration','--calibration'),('oracle','--oracle-only')]:
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as folder:
                root=Path(folder);inputs=fixture_inputs(root);output=root/'out'
                with patch.dict(sys.modules,{'torch':fake_torch}),patch.object(runner,'provenance',return_value={}), \
                     patch.object(runner,'validate_runtime'),patch.object(runner,'verify_native_reference'), \
                     patch.object(runner,'validate_bundle',return_value=(profile,{}, {}, [])), \
                     patch.object(runner,'measurement_hashes',return_value={}),patch.object(runner,'identity',return_value={'synthetic_cpu_test':True}), \
                     patch.object(runner,'run_panel',side_effect=synthetic),patch.object(runner,'load_model',return_value=(None,None,None)), \
                     patch.object(runner,'validate_calibration') as gate,patch.object(runner.signal,'signal'),patch.dict(os.environ),contextlib.redirect_stdout(io.StringIO()):
                    code=runner.main(['--inputs',str(inputs),'--output',str(output),flag])
                self.assertEqual(code,0);gate.assert_not_called()
                with zipfile.ZipFile(output.with_suffix('.zip')) as z:
                    self.assertIsNone(z.testzip());r=json.loads(z.read('report.json'))
                    self.assertEqual(r['mode'],mode);self.assertFalse(r['scientific_pilot_passed'])
                    self.assertEqual(r['calibration_passed'],mode=='calibration');self.assertEqual(r['diagnostic_only'],mode=='oracle')

    def test_runtime_failure_archives_the_failed_mode_without_loading_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'failure'
            with patch.object(runner,'provenance',return_value={}),patch.object(runner,'validate_runtime',side_effect=RuntimeError('synthetic runtime failure')), \
                 patch.object(runner,'load_model') as load,patch.object(runner.signal,'signal'),patch.dict(os.environ),contextlib.redirect_stdout(io.StringIO()):
                code=runner.main(['--inputs',str(Path(folder)/'inputs'),'--output',str(output),'--oracle-only'])
            self.assertEqual(code,1);load.assert_not_called()
            with zipfile.ZipFile(output.with_suffix('.zip')) as z:
                self.assertIsNone(z.testzip());r=json.loads(z.read('report.json'))
                self.assertEqual(r['mode'],'oracle');self.assertTrue(r['diagnostic_only'])
                self.assertFalse(r['scientific_pilot_passed']);self.assertFalse(r['calibration_passed'])

    def test_mode_and_output_checks_stop_before_runtime(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);existing=root/'existing';existing.mkdir();marker=existing/'preserved';marker.write_text('keep')
            with patch.object(runner,'provenance') as get,contextlib.redirect_stderr(io.StringIO()):
                for args in ([],['--calibration','--oracle-only'],['--calibration']):
                    with self.assertRaises(SystemExit):runner.main(['--inputs',str(root/'inputs'),'--output',str(existing),*args])
            get.assert_not_called();self.assertEqual(marker.read_text(),'keep')

    def test_same_image_pair_reuses_exactly_one_nine_condition_bank(self):
        import numpy as np
        from PIL import Image
        profile=load_profile();records=[]
        for cid in profile['case_ids'][2:4]:
            case={'case_id':cid,'image_sha256':'same-image','question':'Synthetic?',
                  'generation_question':'Synthetic?','canonical_target':'no'}
            records.append({'case':case,'rq2':case,'reference':{'seed':590,'clip_sha256':'synthetic-clip',
                'projected_sha256':'synthetic-full','prompt':'same prompt','measurement_target':'No',
                'target_token_ids':[1],'controls':{'reference':{'token_logprobs':[-1.]}}},
                'feature':{'diagnostics':{'input':{'errors':[0.]*576,'basis_indices':list(range(10))}}}})
        base={'pixels':object(),'clip_sha256':'synthetic-clip','projected_sha256':'synthetic-full','seed':590,
              'canvas':np.zeros((2,2,3),dtype=np.uint8)}
        def capture(model,pixels,seed,condition,reference,key):
            spec=CONDITIONS[condition];obs=synthetic_encoding(spec['replay_arm'])
            if condition=='baseline576':obs['encoded']['features']['sha256']='synthetic-full'
            if spec['selection']=='random':
                selected,_=random_indices(reference,key,seed,spec['random_seed'])
                obs['input']['selected_indices']=selected;obs['input']['selected_indices_sorted']=sorted(selected)
                obs['encoded']['mask']={'sha256':repr(sorted(selected))}
            return obs,{'observation':obs}
        def generate(tokenizer,model,processor,image,case,pixels,condition,snapshot,seed,target):
            return {'case_id':case['case_id'],'condition':condition,'arm':condition,'response':'No',
                'scoring':score_response(case,'No'),'encoding':snapshot['observation'],
                'generation_tokens':{'generated_token_ids':[1,2]},
                'reference_score':{'prompt':'same prompt','target_token_ids':[1],'token_logprobs':[-1.],'mean_logprob':-1.}}
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder);(output/'images').mkdir();args=SimpleNamespace(output=output,inputs=output/'inputs',mode='calibration');report={}
            with patch.dict(sys.modules,{'torch':SimpleNamespace()}),patch.object(runner,'panel',return_value=records), \
                 patch.object(runner,'save_image',return_value=Image.new('RGB',(2,2))),patch.object(runner,'save_bank',return_value={'feature_file':'one bank'}) as save, \
                 patch('rq1_native.encode_image',return_value=base) as encode, \
                 patch('rq2_crossed_native.capture_condition',side_effect=capture) as captured, \
                 patch('rq2_crossed_native.generate_condition',side_effect=generate) as generated,contextlib.redirect_stdout(io.StringIO()):
                runner.run_panel(args,report,profile,{'gqa':{}},records,(None,None,None))
            self.assertEqual(captured.call_count,9);self.assertEqual(generated.call_count,18);self.assertEqual(encode.call_count,1)
            save.assert_called_once();self.assertEqual(report['summaries']['gqa']['unique_questions'],2)
            q=[json.loads((output/'questions'/(c['case']['case_id'].replace(':','-')+'.json')).read_text()) for c in records]
            self.assertEqual(q[0]['feature_bank'],q[1]['feature_bank']);self.assertEqual(len(q[1]['controls']['paired_selection']),4)

    def test_calibration_gate_requires_complete_hashed_crossed_rows_and_controls(self):
        p=load_profile();old=parent_profile();ids=p['calibration_case_ids']
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path=root/'report.json';(root/'evidence').mkdir();(root/'features').mkdir()
            (root/'evidence/source.bin').write_bytes(b'synthetic source');(root/'features/bank.bin').write_bytes(b'synthetic tensors')
            report={'version':runner.VERSION,'mode':'calibration','status':'passed','calibration_passed':True,
                'measurement_sources_sha256':runner.measurement_hashes(),'profile_sha256':sha256(runner.PROFILE),
                'arguments':{'model':'same','vision_model':'same'},'expected_case_ids':ids,
                'questions':[{'case_id':cid,'status':'passed'} for cid in ids],
                'progress':{'completed_questions':7,'expected_questions':7},
                'evidence_sha256':{'source.bin':sha256(root/'evidence/source.bin')},
                'feature_sha256':{'features/bank.bin':sha256(root/'features/bank.bin')},'question_sha256':{}}
            rows=[]
            for case in old['cases']:
                cid=case['case_id']
                if cid not in ids:continue
                arms={}
                for condition in CONDITIONS:
                    row={'case_id':cid,'condition':condition,'scoring':score_response(case,case['historical_predictions']['baseline']['response']),
                        'frozen_replay':{key:{'status':'passed','image_prefill_calls':1} for key in ('generation','reference_scoring')}}
                    arms[condition]=row;rows.append(row)
                controls={'archived_source_exact':True,'same_image_frozen_reuse_exact':True,'baseline_reference_max_delta':0,
                    'paired_selection':{key:{'status':'passed'} for key in ['E',*map(str,RANDOM_SEEDS)]}}
                if cid in runner.REPEAT_CASE_IDS:
                    controls['fixed_replay_repeats']={k:{'status':'passed'} for k in runner.FIXED_REPEAT_CONDITIONS}
                    controls['fresh_native_repeats']=[{'condition':condition,'comparison':{'identity_controls':{'status':'passed'}}}
                                                    for condition in runner.FRESH_CONDITIONS for _ in range(2)]
                name='questions/'+cid.replace(':','-')+'.json';write_json(root/name,{'case_id':cid,'status':'passed','arms':arms,'controls':controls})
                report['question_sha256'][name]=sha256(root/name)
            report['summaries']=summarize(rows);write_json(path,report)
            args=SimpleNamespace(model='same',vision_model='same')
            self.assertEqual(runner.validate_calibration(path,args)['expected_case_ids'],ids)
            (root/'features/bank.bin').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Missing/changed'):runner.validate_calibration(path,args)
            (root/'features/bank.bin').write_bytes(b'synthetic tensors')
            name='questions/'+ids[0].replace(':','-')+'.json';q=json.loads((root/name).read_text())
            del q['arms']['random-303-no-merge'];write_json(root/name,q);report['question_sha256'][name]=sha256(root/name);write_json(path,report)
            with self.assertRaisesRegex(ValueError,'source/baseline controls failed'):runner.validate_calibration(path,args)

    def test_old_calibration_or_oracle_cannot_gate_full_crossed_experiment(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'report.json';args=SimpleNamespace(model='same',vision_model='same')
            for version,mode in [('rq2-input-merge-v2','calibration'),(runner.VERSION,'oracle')]:
                path.write_text(json.dumps({'version':version,'mode':mode,'status':'passed','calibration_passed':True}))
                with self.assertRaises(ValueError):runner.validate_calibration(path,args)


if __name__=='__main__':unittest.main()
