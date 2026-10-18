"""CPU checks for frozen reference bundle, pairing, corruption and exact reuse."""

import copy
from datetime import datetime,timezone
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'scripts'))
from rq1_input_ih import (PROFILE,comparison,exact_score,file_at,locate_inputs,
                          source_hashes,validate_bundle,validate_calibration,verify_record,verify_reuse_identity)
from rq1_protocol import sha256,write_json

BUNDLE=REPO/'research/results/rq1-ih-inputs-v1.zip'


def synthetic_question(record,count=32):
    """Synthetic probabilities validate archival gates, never model behavior."""
    from rq1_layer_ih import kept_positions,probability
    ref=record['reference'];length=ref['expanded_sequence_length'];start=ref['image_expanded_start']
    score=ref['controls']['reference'];n=len(ref['target_token_ids'])
    null={'token_logprobs':[-2.]*n,'sum_logprob':-2.*n,'mean_logprob':-2.}
    controls={k:copy.deepcopy(score) for k in ['archived_reference','reference','reference_repeat','native_multimodal',
        'clone_noop','reference_after','retain_all_noop','first_retained_repeat']}
    controls.update({k:copy.deepcopy(null) for k in ['remove_all_before','remove_all_repeat','remove_all_after']})
    controls.update(passed=True,archived_reuse_exact=True,final_input_exact=True,tolerance_per_answer_token=.001,
                    max_reference_delta=0.,remove_all_repeat_delta=0.,remove_all_after_delta=0.,first_retained_repeat_delta=0.,
                    zero_audit_repeats=[{'token_index':t['token_index'],'score':t['perturbed_score']} for t in ref['tokens'][:3]])
    order=ref['token_execution_order'][:count]
    ih={'status':'passed','layer':0,'boundary':'before_block_1','window_size':1,
        'operator':'physical-retain-one-original-positions','target_token_ids':ref['target_token_ids'],
        'first_reference_token_string':'synthetic','expanded_sequence_length':length,'image_expanded_start':start,
        'target_expanded_start':ref['target_expanded_start'],'token_execution_order':order,'controls':controls,
        'remove_all_probability':probability(null),'remove_all_kept_positions':kept_positions(length,start,[]),
        'remove_all_shortened_length':length-576,'tokens':[]}
    for t in ref['tokens'][:count]:
        i=t['token_index'];ih['tokens'].append({'token_index':i,'patch_row':i//24,'patch_col':i%24,
            'reconstruction_error':t['reconstruction_error'],'retained_score':copy.deepcopy(score),
            'retained_probability':probability(score),'ih_first_probability':probability(score)-probability(null),
            'kept_positions':kept_positions(length,start,[i]),'shortened_length':length-575})
    q={'case_id':ref['case_id'],'status':'passed','free_generation':ref['free_generation'],
       'generation_tokens':ref['generation_tokens'],'ih':ih}
    _,q['comparison']=comparison(ref,ih,record['feature']['diagnostics']['input']['basis_indices'])
    return q


class InputIHTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        supplied=os.environ.get('RQ1_IH_TEST_INPUTS')
        if not BUNDLE.exists() and not supplied:raise unittest.SkipTest('Private portable input bundle unavailable')
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name)/'inputs'
        if supplied:shutil.copytree(supplied,cls.root)
        else:
            cls.root.mkdir()
            with zipfile.ZipFile(BUNDLE) as z:z.extractall(cls.root)
        cls.profile,cls.index,cls.manifests,cls.records=validate_bundle(cls.root)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_frozen_panel_and_existing_native_data_are_unchanged(self):
        self.assertEqual(len(self.records),30)
        self.assertEqual([sum(r['entry']['dataset']==d for r in self.records) for d in ['gqa','mmstar','textvqa']],[12,12,6])
        self.assertEqual(len(self.profile['calibration_case_ids']),6)
        self.assertEqual(sum(len(r['reference']['tokens']) for r in self.records),17280)

    def test_kaggle_auto_unpacked_inputs_and_unsafe_paths(self):
        self.assertEqual(locate_inputs([self.root.parent],self.root.parent/'unused'),self.root.resolve())
        with self.assertRaises(ValueError):file_at(self.root,'../escape')
        with self.assertRaises(ValueError):locate_inputs([],self.root.parent/'unused')

    def test_metadata_or_tiny_numerical_drift_blocks_archived_reuse(self):
        r=self.records[0];a=r['reference'];f=r['feature']
        encoded={k:a[k] for k in ['clip_sha256','projected_sha256','seed']};encoded['diagnostics']=copy.deepcopy(f['diagnostics'])
        prepared={k:a[k] for k in ['prompt','target_token_ids']}
        prepared.update(image_start=a['image_expanded_start'],target_start=a['target_expanded_start'],
                        embeddings=SimpleNamespace(shape=(1,a['expanded_sequence_length'],4096)))
        verify_reuse_identity(encoded,prepared,a,f)
        bad=copy.deepcopy(encoded);bad['diagnostics']['input']['errors'][0]+=1e-12
        with self.assertRaises(ValueError):verify_reuse_identity(bad,prepared,a,f)
        score=copy.deepcopy(a['controls']['reference']);score['token_logprobs'][0]-=1e-14
        score['sum_logprob']=sum(score['token_logprobs']);score['mean_logprob']=score['sum_logprob']/len(score['token_logprobs'])
        with self.assertRaises(ValueError):exact_score(score,a['controls']['reference'],'tiny drift')

    def test_saved_score_units_and_retention_geometry_corruption(self):
        r=self.records[0];q=synthetic_question(r)
        self.assertEqual(verify_record(q,r['reference'],r['feature']),0)
        rows,s=comparison(r['reference'],q['ih'],r['feature']['diagnostics']['input']['basis_indices'])
        self.assertEqual(len(rows),32)
        self.assertEqual(rows[0]['zero_whole_log'],r['reference']['tokens'][0]['intervention_score'])
        for field,value in [('ih_first_probability',.1),('kept_positions',[])]:
            bad=copy.deepcopy(q);bad['ih']['tokens'][0][field]=value
            with self.assertRaises(ValueError):verify_record(bad,r['reference'],r['feature'])
        bad=copy.deepcopy(q);bad['ih']['tokens'].pop()
        with self.assertRaises(ValueError):verify_record(bad,r['reference'],r['feature'])

    def test_failed_or_wrong_boundary_calibrations_cannot_authorize_full(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'report.json'
            write_json(p,{'version':'rq1-layer16-ih-v1','status':'passed','arguments':{'tokens':32}})
            with self.assertRaises(ValueError):validate_calibration(p,'model','vision')
            write_json(p,{'version':'rq1-input-ih-v1','status':'failed'})
            with self.assertRaises(ValueError):validate_calibration(p,'model','vision')

    def test_matching_complete_calibration_gate_checks_details_and_current_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);shutil.copytree(self.root/'references',root/'evidence/references')
            lookup={r['case']['case_id']:r for r in self.records};rows=[];digests={}
            for cid in self.profile['calibration_case_ids']:
                q=synthetic_question(lookup[cid]);relative='questions/'+cid.replace(':','-')+'.json'
                write_json(root/relative,q);digests[relative]=sha256(root/relative)
                rows.append({'case_id':cid,'status':'passed','positions':32,'archived_reuse_exact':True})
            report={'version':'rq1-input-ih-v1','status':'passed','arguments':{'tokens':32,'model':'model','vision_model':'vision'},
                'profile_sha256':sha256(PROFILE),'reference_index_sha256':self.profile['reference_index_sha256'],
                'measurement_sources_sha256':source_hashes(),'lock_sha256':sha256(REPO/'environments/original/uv.lock'),
                'implementation':{'native_matches_reference':True},'questions':rows,'question_sha256':digests,
                'progress':{'completed_retention_scores':192}}
            p=root/'report.json';write_json(p,report)
            gate,details=validate_calibration(p,'model','vision')
            self.assertTrue(gate['passed']);self.assertEqual(len(details),6)
            detail=root/next(iter(digests));raw=detail.read_bytes();detail.write_bytes(raw+b' ')
            with self.assertRaisesRegex(ValueError,'detail hash'):validate_calibration(p,'model','vision')
            detail.write_bytes(raw);report['measurement_sources_sha256']['scripts/rq1_input_ih_native.py']='changed'
            write_json(p,report)
            with self.assertRaises(ValueError):validate_calibration(p,'model','vision')

    def test_reference_file_corruption_is_rejected_before_inference(self):
        path=file_at(self.root,self.index['cases'][0]['question_file']);raw=path.read_bytes()
        try:
            path.write_bytes(raw+b' ')
            with self.assertRaisesRegex(ValueError,'evidence hash'):validate_bundle(self.root)
        finally:path.write_bytes(raw)

    def test_notebook_helper_uses_isolated_utf8_child_and_requires_full_prerequisite(self):
        notebook=json.loads((REPO/'research/notebooks/kaggle_rq1_input_pilot.ipynb').read_text())
        helper=next(''.join(c['source']) for c in notebook['cells'] if ''.join(c['source']).startswith('def run_input_ih('))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'scripts').mkdir()
            (root/'scripts/run_rq1_input_ih.py').write_text(
                'import json,pathlib,sys\nassert sys.flags.utf8_mode == 1 and sys.flags.isolated == 1\n'
                'out=pathlib.Path(sys.argv[sys.argv.index("--output")+1]);out.mkdir()\n'
                'out.joinpath("report.json").write_text(json.dumps({"status":"passed","wall_seconds":0,"questions":[]}))\n'
                'print("Input IH → ✓")\n',encoding='utf-8')
            def local_path(value):
                p=Path(value);return root/p.name if str(p).startswith('/kaggle/working/') else p
            namespace={'datetime':datetime,'timezone':timezone,'Path':local_path,'original_python':sys.executable,
                'repo':root,'inputs':root,'CHECKPOINT_ARGS':[],'subprocess':subprocess,'json':json}
            exec(compile(helper,'<tracked input IH notebook>','exec'),namespace)
            with patch('sys.stdout',new_callable=io.StringIO):
                out=namespace['run_input_ih']()
                with self.assertRaises(ValueError):namespace['run_input_ih'](tokens=576)
            self.assertIn('Input IH → ✓',out.with_suffix('.log').read_text(encoding='utf-8'))

    def test_invalid_reference_preflight_saves_failure_zip_without_loading_model(self):
        import run_rq1_input_ih as runner
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'failure'
            argv=['run_rq1_input_ih.py','--inputs',str(self.root),'--output',str(out)]
            with patch.object(sys,'argv',argv),patch.object(runner,'provenance',return_value={}),\
                 patch.object(runner,'validate_runtime'),patch.object(runner,'verify_native_reference'),\
                 patch.object(runner,'validate_bundle',side_effect=ValueError('frozen reference mismatch')),\
                 patch.object(runner,'load_model') as load,patch.object(runner.signal,'signal'),\
                 patch('sys.stdout',new_callable=io.StringIO):
                self.assertEqual(runner.main(),1)
            load.assert_not_called()
            with zipfile.ZipFile(out.with_suffix('.zip')) as z:
                report=json.loads(z.read('report.json'))
                self.assertEqual(report['status'],'failed')
                self.assertIn('frozen reference mismatch',report['error'])


if __name__=='__main__':unittest.main()
