"""CPU checks for frozen selection, semantic option mapping and failed-run export."""

from collections import Counter
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'scripts'))
from rq1_protocol import SOURCES
from rq1_validation import (PROFILE,rotate_options,score_rotated,neighbors,
                            perturbation_summary,shortcut_summary,validate_profile)
import run_rq1_validation as runner


class ValidationTests(unittest.TestCase):
    def test_cyclic_choices_keep_identity_prompt_and_semantic_gold(self):
        q='Question: Find an angle.\nChoices:\n(A) 97\n(B) 102\n(C) 107\n(D) 122'
        displays=[]
        for shift in range(4):
            variant=rotate_options(q,'A',shift)
            displays.append(variant['gold_display'])
            scored=score_rotated(variant['gold_display'],variant)
            self.assertTrue(scored['correct'])
            self.assertEqual(scored['selected_content'],'97')
            self.assertEqual(scored['selected_original_label'],'A')
            if shift==0:self.assertEqual(variant['question'],q)
        self.assertEqual(set(displays),set('ABCD'))

    def test_punctuation_is_preserved_and_fixed_A_is_not_always_correct(self):
        q='Which?\nOptions: A: Red carpet, B: Black shoes, C: White sign, D: White banner'
        variants=[rotate_options(q,'A',i) for i in range(4)]
        self.assertEqual(sum(score_rotated('A',v)['correct'] for v in variants),1)
        self.assertEqual([score_rotated('A',v)['selected_content'] for v in variants],
                         ['Red carpet','Black shoes','White sign','White banner'])
        self.assertEqual(variants[0]['question'],q)
        with self.assertRaises(ValueError):rotate_options(q.replace('White sign','nan'),'A',1)

    def test_neighbors_do_not_wrap_across_rows(self):
        self.assertEqual(neighbors(0),[24,1])
        self.assertEqual(neighbors(23),[47,22])
        self.assertEqual(set(neighbors(25)),{1,49,24,26})
        self.assertEqual(neighbors(575),[551,574])
        with self.assertRaises(ValueError):neighbors(576)

    def test_reference_repeat_does_not_inflate_shortcut_accuracy(self):
        variant=rotate_options('A: red, B: blue, C: green, D: white','B',0)
        row={'case_id':'mmstar:x','condition':'real','shift':0,'kind':'trial','response':'B','scoring':score_rotated('B',variant)}
        summary=shortcut_summary([row,{**row,'kind':'reference_repeat'}])
        self.assertEqual(summary['conditions']['real']['trials'],1)
        self.assertEqual(summary['conditions']['real']['correct'],1)
        self.assertTrue(summary['cases'][0]['original_repeat_same_semantic_choice'])

    def test_intervention_statistics_preserve_negative_signs_and_strata(self):
        rows=[{'stratum':'random','current_E':v+1,'effects':{m:{'I':v*scale} for m,scale in [('zero',1),('half',.5),('neighbor_mean',-1)]}}
              for v in [-.3,-.2,.1,.4]]
        s=perturbation_summary(rows)['random']
        self.assertAlmostEqual(s['half']['rho_vs_fresh_zero'],1)
        self.assertAlmostEqual(s['neighbor_mean']['rho_vs_fresh_zero'],-1)
        self.assertEqual(s['neighbor_mean']['sign_agreement_beyond_1e_minus4'],0)
        self.assertEqual(s['zero']['positive_count'],2)

    def test_profile_matches_saved_sources_and_rebuilds_exact_patch_selection(self):
        profile=json.loads(PROFILE.read_text())
        self.assertEqual(len(profile['shortcuts']['case_ids'])*3*4,72)
        self.assertEqual(len(profile['perturbations']),6)
        for entry in profile['perturbations']:
            self.assertEqual(Counter(p['stratum'] for p in entry['patches']),{'random':16,'high_I':8,'low_E_high_I':8})
            self.assertEqual(len({p['token_index'] for p in entry['patches']}),32)
        inputs=REPO/'research/results/rq1-inputs-v1'
        if not inputs.exists():self.skipTest('Private accepted inputs unavailable')
        _,_,lookup=validate_profile(inputs)
        self.assertTrue(all(cid in lookup for cid in profile['shortcuts']['case_ids']))
        self.assertEqual(lookup['textvqa:35962']['outcomes']['input-only'],'loss')
        self.assertEqual(lookup['textvqa:35670']['outcomes']['input-only'],'preserved_correct')
        self.assertTrue(lookup['textvqa:35962']['predictions']['baseline']['full_credit'])
        from prepare_rq1_validation import prepare
        with tempfile.TemporaryDirectory() as temp,patch('builtins.print'):
            dest=Path(temp)/'profile.json';prepare(REPO/'research/results',dest)
            self.assertEqual(json.loads(dest.read_text()),profile)

    def test_rejected_inputs_fail_before_model_load_and_still_export_zip(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'run'
            with patch.object(sys,'argv',['run_rq1_validation.py','--inputs',temp,'--output',str(output)]), \
                 patch.object(runner,'provenance',return_value={}),patch.object(runner,'validate_runtime'), \
                 patch.object(runner,'verify_native_reference'),patch.object(runner.signal,'signal'), \
                 patch.object(runner,'validate_profile',side_effect=ValueError('Frozen input mismatch')), \
                 patch.object(runner,'load_model') as load,patch('builtins.print'):
                # Output is deliberately outside the configured input directory.
                with tempfile.TemporaryDirectory() as inputs:
                    sys.argv[2]=inputs
                    self.assertEqual(runner.main(),1)
            load.assert_not_called()
            with zipfile.ZipFile(output.with_suffix('.zip')) as z:
                report=json.loads(z.read('report.json'))
                self.assertEqual(report['status'],'failed')
                self.assertIn('Frozen input mismatch',report['error'])

    def test_new_notebook_defaults_block_inference_and_code_cells_compile(self):
        path=REPO/'research/notebooks/kaggle_rq1_validity_checks.ipynb'
        notebook=json.loads(path.read_text());code=[]
        for cell in notebook['cells']:
            if cell['cell_type']=='code':
                source=''.join(cell['source']);compile(source,str(path),'exec');code.append(source)
                self.assertEqual(cell['outputs'],[])
                self.assertIsNone(cell['execution_count'])
        self.assertIn('RUN_VALIDATION = False',code[0])
        self.assertIn('if not RUN_VALIDATION:',code[0])
        self.assertIn('run_rq1_validation.py',''.join(code))

    def test_shortcut_runner_executes_72_trials_and_six_excluded_repeats(self):
        profile=json.loads(PROFILE.read_text())
        lookup={cid:{'case_id':cid,'image_sha256':cid,'compression_seed':590,
            'question':'A: red, B: blue, C: green, D: white','canonical_target':'B',
            'generation_question':'A: red, B: blue, C: green, D: white\nAnswer with only the option letter (A, B, C, or D).'}
            for cid in profile['shortcuts']['case_ids']}
        class FakeImage:
            def save(self,path):Path(path).write_bytes(b'blank')
            def getpixel(self,where):return (127,127,127)
        fake_pil=SimpleNamespace(Image=SimpleNamespace(new=lambda *a:FakeImage()))
        def generate(*args,**kwargs):kwargs['record'].update(response='A',status='passed')
        with tempfile.TemporaryDirectory() as temp,patch.dict(sys.modules,{'PIL':fake_pil}), \
             patch.object(runner,'image_for',return_value=FakeImage()),patch.object(runner,'generate_native_answer',side_effect=generate), \
             patch('rq1_validation_native.generate_text_only',return_value={'response':'A','status':'passed'}),patch('builtins.print'):
            report={};runner.run_shortcuts(SimpleNamespace(output=Path(temp)),report,profile,{},lookup,
                (None,None,SimpleNamespace(image_mean=[.5,.5,.5])))
            records=[json.loads(line) for line in (Path(temp)/'shortcuts.jsonl').read_text().splitlines()]
            self.assertEqual(len(records),78)
            self.assertEqual(report['shortcut_progress']['completed_trials'],72)
            self.assertEqual(report['shortcut_progress']['completed_repeats'],6)
            for condition in report['shortcut_summary']['conditions'].values():
                self.assertEqual(condition['trials'],24)
                self.assertEqual(condition['correct'],6)

    def test_all_mode_loads_model_once_for_both_diagnostics(self):
        fake_torch=SimpleNamespace(cuda=SimpleNamespace(device_count=lambda:0))
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'run';inputs=Path(temp)/'inputs';inputs.mkdir()
            with patch.object(sys,'argv',['run_rq1_validation.py','--inputs',str(inputs),'--output',str(output)]), \
                 patch.object(runner,'provenance',return_value={}),patch.object(runner,'validate_runtime'), \
                 patch.object(runner,'verify_native_reference'),patch.object(runner.signal,'signal'), \
                 patch.object(runner,'validate_profile',return_value=({}, {}, {})),patch.object(runner,'copy_evidence'), \
                 patch.object(runner,'load_model',return_value=('tokenizer','model','processor')) as load, \
                 patch.object(runner,'run_shortcuts') as shortcuts,patch.object(runner,'run_perturbations') as perturbations, \
                 patch.dict(sys.modules,{'torch':fake_torch}),patch('builtins.print'):
                self.assertEqual(runner.main(),0)
            load.assert_called_once();shortcuts.assert_called_once();perturbations.assert_called_once()


if __name__=='__main__':unittest.main()
