"""CPU checks: source-pinned IH scope, matched units, gating and saved exports."""

import copy
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'scripts'))
from rq1_layer import CASE_IDS, PROFILE as PARENT_PROFILE
from rq1_layer_ih import (PROFILE, SOURCE_FILES, VERSION, comparison_rows, comparison_summary,
                          kept_positions, probability, validate_calibration, validate_profile)
from rq1_protocol import sha256
import run_rq1_layer_ih as runner


def score(p):
    return {'token_logprobs': [math.log(p)], 'sum_logprob': math.log(p), 'mean_logprob': math.log(p)}


class IHProtocolTests(unittest.TestCase):
    def test_physical_retention_preserves_text_order_and_original_positions(self):
        self.assertEqual(kept_positions(580, 1, []), [0, 577, 578, 579])
        self.assertEqual(kept_positions(580, 1, [23]), [0, 24, 577, 578, 579])
        self.assertEqual(kept_positions(580, 1, [575, 0]), [0, 1, 576, 577, 578, 579])
        self.assertEqual(kept_positions(580, 1, range(576)), list(range(580)))
        for bad in [[-1], [576], [2, 2], [True]]:
            with self.assertRaises(ValueError): kept_positions(580, 1, bad)

    def test_profile_pins_original_panel_and_documents_released_operator(self):
        p = json.loads(PROFILE.read_text())
        self.assertEqual(p['parent_profile_sha256'], sha256(PARENT_PROFILE))
        self.assertEqual(p['official_commit'], '75909b2936a13d7214e83514f0bc0cb9cc91139e')
        self.assertEqual((p['layer'], p['next_module_index'], p['window_size']), (16, 16, 1))
        self.assertEqual(len(p['official_source_sha256']), 4)
        self.assertIn('SIZE=2', p['adaptations'][0])
        inputs = REPO/'research/results/rq1-inputs-v1'
        if inputs.exists():
            ih, parent, data, lookup = validate_profile(inputs)
            self.assertEqual([c['case_id'] for c in parent['cases']], CASE_IDS)

    def test_probability_differences_have_comparable_units_and_signs(self):
        layer = {'target_token_ids': [10], 'token_execution_order': [0, 1, 2],
                 'controls': {'reference': score(.8)}, 'tokens': [
                    {'token_index': i, 'reconstruction_error': i+1,
                     'intervention_score': math.log(.8)-math.log(p), 'perturbed_score': score(p)}
                    for i, p in enumerate([.7, .9, .8])]}
        ih = {'target_token_ids': [10], 'controls': {'remove_all_before': score(.2)},
              'tokens': [{'token_index': i, 'retained_score': score(p)} for i, p in enumerate([.3, .1, .2])]}
        rows = comparison_rows(layer, ih)
        self.assertAlmostEqual(rows[0]['zero_first_probability'], .1)
        self.assertAlmostEqual(rows[0]['ih_first_probability'], .1)
        self.assertLess(rows[1]['zero_first_probability'], 0)
        self.assertLess(rows[1]['ih_first_probability'], 0)
        self.assertAlmostEqual(rows[2]['ih_first_probability'], 0)
        summary = comparison_summary(layer, ih, [0])
        self.assertEqual(summary['all']['positions'], 3)
        self.assertEqual(summary['nonbasis']['positions'], 2)
        bad = copy.deepcopy(ih); bad['tokens'].reverse()
        with self.assertRaises(ValueError): comparison_rows(layer, bad)
        with self.assertRaises(ValueError): probability(score(float('nan')))

    def test_old_calibration_cannot_unlock_new_full_sweep_and_sources_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'report.json'
            with self.assertRaises(ValueError): validate_calibration(None, 'm', 'v')
            report = {'version': 'rq1-layer16-v1', 'status': 'passed'}
            path.write_text(json.dumps(report))
            with self.assertRaises(ValueError): validate_calibration(path, 'm', 'v')
            report = {'version': VERSION, 'status': 'passed', 'arguments': {'tokens':32,'model':'m','vision_model':'v'},
                'ih_profile_sha256': sha256(PROFILE), 'profile_sha256': sha256(PARENT_PROFILE),
                'questions': [{'case_id': cid, 'status':'passed'} for cid in CASE_IDS],
                'ih_questions': [{'case_id': cid, 'status':'passed','positions':32,'controls_passed':True} for cid in CASE_IDS],
                'progress': {'completed_patch_stage_scores':384},
                'ih_progress': {'completed_questions':6,'completed_retention_scores':192},
                'total_patch_method_scores':576}
            native = json.loads((REPO/'research/provenance/original_llava_reference.json').read_text())['native_source_sha256']
            report['implementation'] = {'native_matches_reference':True,'native_source_sha256':native}
            report['evidence_sha256'] = {'sources/'+f:sha256(REPO/f) for f in SOURCE_FILES+list(native)}
            path.write_text(json.dumps(report))
            self.assertTrue(validate_calibration(path, 'm', 'v')['passed'])
            report['evidence_sha256']['sources/scripts/rq1_layer_ih_native.py'] = 'changed'
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'source/environment'): validate_calibration(path, 'm', 'v')

    def test_partial_review_is_utf8_even_under_an_ascii_default(self):
        original = io.open
        def ascii_default(file, mode='r', buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None):
            if 'b' not in mode and encoding in (None, 'locale'): encoding = 'ascii'
            return original(file, mode, buffering, encoding, errors, newline, closefd, opener)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory); (out/'questions').mkdir(); (out/'evidence').mkdir()
            (out/'evidence/x-manifest.json').write_text('{"images":{},"note":"窗 → ✓"}', encoding='utf-8')
            (out/'questions/x.json').write_text('{"case_id":"窗","status":"running"}', encoding='utf-8')
            with patch('io.open', side_effect=ascii_default): runner.write_review(out, {'status':'failed → ✓'})
            self.assertIn('failed → ✓', (out/'index.html').read_text(encoding='utf-8'))

    def test_bad_protocol_fails_before_model_load_and_preserves_zip(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/'run'; inputs = Path(directory)/'inputs'; inputs.mkdir()
            with patch.object(sys, 'argv', ['run_rq1_layer_ih.py','--inputs',str(inputs),'--output',str(out)]), \
                 patch.object(runner, 'provenance', return_value={}), patch.object(runner, 'validate_runtime'), \
                 patch.object(runner, 'verify_native_reference'), patch.object(runner.signal, 'signal'), \
                 patch.object(runner, 'validate_profile', side_effect=ValueError('IH input mismatch')), \
                 patch.object(runner, 'load_model') as load, patch('builtins.print'):
                self.assertEqual(runner.main(), 1)
            load.assert_not_called()
            with zipfile.ZipFile(out.with_suffix('.zip')) as archive:
                report = json.loads(archive.read('report.json'))
                self.assertEqual(report['status'], 'failed')
                self.assertIn('IH input mismatch', report['error'])

    def test_actual_notebook_helper_selects_ih_entrypoint_and_utf8_log(self):
        notebook = json.loads((REPO/'research/notebooks/kaggle_rq1_layer16_pilot.ipynb').read_text())
        helper = next(''.join(c['source']) for c in notebook['cells'] if ''.join(c['source']).startswith('def run_layer('))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root/'scripts').mkdir()
            (root/'scripts/run_rq1_layer_ih.py').write_text(
                'import json, pathlib, sys\n'
                'assert sys.flags.utf8_mode == 1 and sys.flags.isolated == 1\n'
                'out = pathlib.Path(sys.argv[sys.argv.index("--output")+1]); out.mkdir(parents=True)\n'
                'out.joinpath("report.json").write_text(json.dumps({"status":"passed","wall_seconds":0,"questions":[],"ih_questions":[]}))\n'
                'print("IH → ✓")\n', encoding='utf-8')
            def local_path(value):
                path = Path(value)
                return root/path.name if str(path).startswith('/kaggle/working/') else path
            namespace = {'datetime':datetime,'timezone':timezone,'Path':local_path,
                'original_python':sys.executable,'repo':root,'inputs':root,'CHECKPOINT_ARGS':[],
                'subprocess':subprocess,'json':json}
            exec(compile(helper,'<tracked notebook helper>','exec'), namespace)
            with patch('sys.stdout', new_callable=io.StringIO): out = namespace['run_layer'](include_ih=True)
            self.assertIn('-ih-calibration-v1-', out.name)
            self.assertIn('IH → ✓', out.with_suffix('.log').read_text(encoding='utf-8'))


if __name__ == '__main__': unittest.main()
