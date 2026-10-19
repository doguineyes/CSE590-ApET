"""CPU orchestration/evidence controls; synthetic fixtures are not GPU results."""
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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts'))
import run_rq2_input_merge_v2 as runner
from rq1_protocol import write_json
from rq2_protocol import ARMS, paired_summary, score_response
from rq2_frozen_protocol import load_profile
from research.tests.test_rq2_runner import fixture_inputs
from rq1_protocol import sha256


class FrozenRunnerTests(unittest.TestCase):
    def test_fixed_replay_checks_remain_exact_for_scores_and_generated_ids(self):
        row = {'generation_tokens': {'generated_token_ids': [1, 2]}, 'encoding': {'sha': 'frozen'},
               'reference_score': {'token_logprobs': [-1.]}}
        self.assertEqual(runner.row_repeat_check(row, row)['status'], 'passed')
        for name in ('generation_tokens', 'encoding', 'reference_score'):
            altered = copy.deepcopy(row)
            altered[name] = {'generation_tokens': {'generated_token_ids': [3]},
                             'encoding': {'sha': 'different'},
                             'reference_score': {'token_logprobs': [-1.001]}}[name]
            with self.assertRaises(ValueError): runner.row_repeat_check(row, altered)

    def test_mode_and_overwrite_rejections_precede_runtime_and_preserve_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); output = root/'existing'; output.mkdir()
            marker = output/'preserve'; marker.write_text('unchanged')
            with patch.object(runner, 'provenance') as provenance, contextlib.redirect_stderr(io.StringIO()):
                for extra in ([], ['--calibration', '--calibration-report', str(root/'x')], ['--calibration']):
                    with self.assertRaises(SystemExit):
                        runner.main(['--inputs', str(root/'inputs'), '--output', str(output), *extra])
            provenance.assert_not_called(); self.assertEqual(marker.read_text(), 'unchanged')

    def test_runtime_failure_is_archived_and_never_loads_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/'failed'
            with patch.object(runner, 'provenance', return_value={}), \
                 patch.object(runner, 'validate_runtime', side_effect=RuntimeError('test runtime failure')), \
                 patch.object(runner, 'load_model') as load, patch.object(runner.signal, 'signal'), \
                 patch.dict(os.environ), contextlib.redirect_stdout(io.StringIO()):
                result = runner.main(['--inputs', str(Path(folder)/'inputs'), '--output', str(output), '--calibration'])
            self.assertEqual(result, 1); load.assert_not_called()
            with zipfile.ZipFile(output.with_suffix('.zip')) as z:
                self.assertIsNone(z.testzip())
                report = json.loads(z.read('report.json'))
                self.assertEqual(report['mode'], 'calibration')
                self.assertFalse(report['calibration_passed']); self.assertFalse(report['scientific_pilot_passed'])
                self.assertIn('test runtime failure', report['error'])
                self.assertIn('index.html', z.namelist())

    def test_synthetic_calibration_dispatch_cannot_count_as_full_scientific_panel(self):
        profile = load_profile()
        def synthetic(args, report, profile, manifests, records, loaded):
            report['synthetic_cpu_test'] = True
            report['expected_case_ids'] = profile['calibration']['case_ids']
            report['questions'] = [{'case_id': cid, 'status': 'passed'} for cid in report['expected_case_ids']]
            report['progress'] = {'completed_questions': 7, 'expected_questions': 7}
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 2))
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); inputs=fixture_inputs(root); output=root/'calibration'
            with patch.dict(sys.modules, {'torch': fake_torch}), \
                 patch.object(runner, 'provenance', return_value={}), \
                 patch.object(runner, 'validate_runtime'), patch.object(runner, 'verify_native_reference'), \
                 patch.object(runner, 'validate_bundle', return_value=(profile, {}, {}, [])), \
                 patch.object(runner, 'measurement_hashes', return_value={}), \
                 patch.object(runner, 'calibration_identity', return_value={'synthetic_cpu_test': True}), \
                 patch.object(runner, 'run_panel', side_effect=synthetic), \
                 patch.object(runner, 'load_model', return_value=(None,None,None)), \
                 patch.object(runner, 'validate_calibration') as gate, \
                 patch.object(runner.signal,'signal'),patch.dict(os.environ), \
                 contextlib.redirect_stdout(io.StringIO()):
                code=runner.main(['--inputs',str(inputs),'--output',str(output),'--calibration'])
            self.assertEqual(code,0);gate.assert_not_called()
            report=json.loads((output/'report.json').read_text())
            self.assertTrue(report['calibration_passed']);self.assertFalse(report['scientific_pilot_passed'])
            self.assertEqual(report['version'],'rq2-input-merge-v2')

    def test_full_gate_rejects_v1_diagnostic_incomplete_and_changed_source_reports(self):
        profile=load_profile()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'report.json'
            args=SimpleNamespace(model='same',vision_model='same')
            variants=[{}, {'status':'passed','version':'rq2-native-parity-diagnostic-v1'},
                      {'version':runner.VERSION,'mode':'full','status':'passed'},
                      {'version':runner.VERSION,'mode':'calibration','status':'passed',
                       'calibration_passed':True,'scientific_pilot_passed':False,
                       'expected_case_ids':profile['calibration']['case_ids'],'questions':[]}]
            for variant in variants:
                path.write_text(json.dumps(variant))
                with self.assertRaises(ValueError): runner.validate_calibration(path,args)

    def test_full_gate_requires_real_detail_hashes_and_complete_controls(self):
        """Synthetic bytes exercise actual gate/scorers, never claim model parity."""
        profile=load_profile();ids=profile['calibration']['case_ids']
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path=root/'report.json';args=SimpleNamespace(model='same',vision_model='same')
            evidence=root/'evidence';evidence.mkdir();(evidence/'source.bin').write_bytes(b'synthetic evidence')
            features=root/'features';features.mkdir();(features/'bank.bin').write_bytes(b'synthetic tensor bytes')
            report={'version':runner.VERSION,'mode':'calibration','status':'passed',
                'calibration_passed':True,'scientific_pilot_passed':False,'synthetic_cpu_test':True,
                'expected_case_ids':ids,'questions':[{'case_id':c,'status':'passed'} for c in ids],
                'progress':{'completed_questions':7,'expected_questions':7},
                'measurement_sources_sha256':runner.measurement_hashes(),'profile_sha256':sha256(runner.PROFILE),
                'arguments':{'model':'same','vision_model':'same'},
                'implementation':{'native_matches_reference':True},'lock_sha256':profile['lock_sha256'],
                'evidence_sha256':{'source.bin':sha256(evidence/'source.bin')},
                'feature_sha256':{'features/bank.bin':sha256(features/'bank.bin')},'question_sha256':{}}
            rows=[]
            for case in profile['cases']:
                cid=case['case_id']
                if cid not in ids:continue
                arms={}
                for arm in ARMS:
                    old='input-only' if arm=='native-input288' else 'baseline'
                    response=case['historical_predictions'][old]['response']
                    row={'case_id':cid,'arm':arm,'scoring':score_response(case,response),
                        'frozen_replay':{key:{'status':'passed','image_prefill_calls':1}
                                         for key in ('generation','reference_scoring')}}
                    arms[arm]=row;rows.append(row)
                controls={'archived_source_exact':True,'same_image_frozen_reuse_exact':True,
                    'selection_and_basis':{'status':'passed'},'baseline_reference_max_delta':0}
                if cid in profile['controls']['fixed_replay_repeats']:
                    controls['fixed_replay_repeats']={arm:{'status':'passed'} for arm in ARMS}
                    controls['fresh_native_repeats']={'status':'recorded','repeats':[
                        {'comparison':{'identity_controls':{'status':'passed'}}} for _ in range(2)]}
                relative='questions/'+cid.replace(':','-')+'.json'
                write_json(root/relative,{'case_id':cid,'status':'passed','controls':controls,'arms':arms})
                report['question_sha256'][relative]=sha256(root/relative)
            report['summaries']=paired_summary(rows);write_json(path,report)
            self.assertEqual(runner.validate_calibration(path,args)['expected_case_ids'],ids)
            (features/'bank.bin').write_bytes(b'changed tensor bytes')
            with self.assertRaisesRegex(ValueError,'Missing or changed'):
                runner.validate_calibration(path,args)
            (features/'bank.bin').write_bytes(b'synthetic tensor bytes')
            relative='questions/'+ids[0].replace(':','-')+'.json'
            question=json.loads((root/relative).read_text())
            question['controls']['fresh_native_repeats']['repeats']=[]
            write_json(root/relative,question);report['question_sha256'][relative]=sha256(root/relative)
            write_json(path,report)
            with self.assertRaisesRegex(ValueError,'fresh native controls missing'):
                runner.validate_calibration(path,args)

    def test_actual_panel_captures_each_arm_once_for_same_image_questions(self):
        """Real orchestration with synthetic encoders isolates feature-bank reuse."""
        import numpy as np
        from PIL import Image
        from research.tests.test_rq2_runner import synthetic_encoding
        profile=load_profile();profile['controls']['fixed_replay_repeats']=[]
        items=[]
        for cid in profile['case_ids'][:2]:
            case={'case_id':cid,'image_sha256':'same-image','generation_question':'Synthetic?',
                  'canonical_target':'no'}
            items.append({'case':case,'rq2':{'historical_predictions':{
                'baseline':{'response':'No'},'input-only':{'response':'No'}}},
                'reference':{'seed':590,'clip_sha256':'synthetic-clip','projected_sha256':'synthetic-full',
                             'measurement_target':'No','prompt':'same prompt','target_token_ids':[1],
                             'controls':{'reference':{'token_logprobs':[-1.]}}},
                'feature':{'diagnostics':{'input':{'errors':[0.]*576,'basis_indices':list(range(10))}}}})
        base={'pixels':object(),'clip_sha256':'synthetic-clip','projected_sha256':'synthetic-full',
              'seed':590,'canvas':np.zeros((2,2,3),dtype=np.uint8)}
        def capture(model,pixels,seed,arm):
            obs=synthetic_encoding(arm)
            if arm=='baseline576':obs['encoded']['features']['sha256']='synthetic-full'
            return obs,{'arm':arm,'observation':obs}
        def generate(tokenizer,model,processor,image,case,pixels,arm,snapshot,seed,target):
            return {'case_id':case['case_id'],'arm':arm,'response':'No','scoring':score_response(case,'No'),
                'generation_tokens':{'generated_token_ids':[1,2]},'encoding':snapshot['observation'],
                'reference_score':{'prompt':'same prompt','target_token_ids':[1],'token_logprobs':[-1.],
                                   'mean_logprob':-1.}}
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'images').mkdir()
            args=SimpleNamespace(output=root,inputs=root/'inputs',calibration=True)
            report={}
            with patch.dict(sys.modules,{'torch':SimpleNamespace()}), \
                 patch.object(runner,'choose_panel',return_value=items), \
                 patch.object(runner,'save_image',return_value=Image.new('RGB',(2,2))), \
                 patch.object(runner,'save_bank',return_value={'feature_file':'same-bank'}) as save, \
                 patch('rq1_native.encode_image',return_value=base) as encode, \
                 patch('rq2_frozen_native.capture_encoding',side_effect=capture) as captured, \
                 patch('rq2_frozen_native.generate_from_snapshot',side_effect=generate), \
                 contextlib.redirect_stdout(io.StringIO()):
                runner.run_panel(args,report,profile,{'gqa':{}},items,(None,None,None))
            self.assertEqual(encode.call_count,1);self.assertEqual(captured.call_count,3);save.assert_called_once()
            self.assertEqual(report['progress'],{'completed_questions':2,'expected_questions':2})
            first=json.loads((root/'questions/gqa-201636988.json').read_text())
            second=json.loads((root/'questions/gqa-201637316.json').read_text())
            self.assertEqual(first['feature_bank'],second['feature_bank'])
            self.assertTrue(second['controls']['same_image_frozen_reuse_exact'])


if __name__ == '__main__': unittest.main()
