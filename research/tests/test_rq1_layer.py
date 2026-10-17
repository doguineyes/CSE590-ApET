"""CPU-only checks of frozen identities, unbiased patch sampling and paired ranks."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts'))
from rq1_layer import (CASE_IDS, PROFILE, VERSION, paired_summary, selected_tokens,
                       validate_calibration, validate_layer_profile)
from rq1_protocol import sha256, token_order
from rq1_validation import PROFILE as VALIDITY_PROFILE
from run_rq1_layer import write_review
import run_rq1_layer as runner


def stage(errors, effects, target=None):
    return {'target_token_ids': target or [10], 'tokens': [
        {'token_index': i, 'patch_row': i//24, 'patch_col': i%24,
         'reconstruction_error': e, 'intervention_score': v,
         'perturbed_score': {'mean_logprob': -1-v}}
        for i, (e, v) in enumerate(zip(errors, effects))]}


class LayerProtocolTests(unittest.TestCase):
    def test_profile_is_fixed_before_validity_outcomes(self):
        profile = json.loads(PROFILE.read_text())
        parent = json.loads(VALIDITY_PROFILE.read_text())
        self.assertEqual(profile['validity_profile_sha256'], sha256(VALIDITY_PROFILE))
        self.assertEqual(profile['version'], VERSION)
        self.assertEqual([e['case_id'] for e in profile['cases']], CASE_IDS)
        self.assertEqual(len({e['image_sha256'] for e in profile['cases']}), 5)
        self.assertEqual((profile['layer'], profile['module_index']), (16, 15))
        for new, old in zip(profile['cases'], parent['perturbations']):
            self.assertEqual(new['target_token_ids'], old['target_token_ids'])
            self.assertEqual(new['archived_role'], old['archived_role'])

    def test_sampling_is_nested_and_same_for_same_image(self):
        entries = json.loads(PROFILE.read_text())['cases']
        for entry in entries:
            self.assertEqual(selected_tokens(entry, 576), token_order(entry['image_sha256'], 590))
            self.assertEqual(selected_tokens(entry, 32), selected_tokens(entry, 576)[:32])
        self.assertEqual(selected_tokens(entries[0], 32), selected_tokens(entries[1], 32))
        with self.assertRaises(ValueError):
            selected_tokens(entries[0], 64)
        bad = copy.deepcopy(entries[0]); bad['token_order'][-1] = bad['token_order'][0]
        with self.assertRaises(ValueError):
            selected_tokens(bad, 32)

    def test_layer_profile_accepts_original_bundle_and_preserves_screening_roles(self):
        inputs = REPO / 'research/results/rq1-inputs-v1'
        if not inputs.exists(): self.skipTest('Private accepted input bundle unavailable')
        profile, data, lookup = validate_layer_profile(inputs)
        self.assertEqual([e['case_id'] for e in profile['cases']], CASE_IDS)
        self.assertEqual(lookup['textvqa:35962']['outcomes']['input-only'], 'loss')
        self.assertEqual(lookup['textvqa:35670']['outcomes']['input-only'], 'preserved_correct')
        self.assertEqual(set(data), {'gqa', 'mmstar', 'textvqa'})

    def test_paired_statistics_use_same_positions_and_common_nonbasis(self):
        a = stage([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5])
        b = stage([5, 4, 3, 2, 1, 0], [0, 1, 2, 3, 4, 5])
        result = paired_summary(a, b, [0], [1])
        self.assertEqual(result['measured_positions'], 6)
        self.assertEqual(result['common_nonbasis_positions'], 4)
        self.assertAlmostEqual(result['input']['rho_E_I_all'], 1)
        self.assertAlmostEqual(result['layer16']['rho_E_I_all'], -1)
        self.assertAlmostEqual(result['rho_I_between_stages'], 1)
        self.assertAlmostEqual(result['rho_E_between_stages'], -1)
        b['tokens'].reverse()
        self.assertEqual(paired_summary(a, b, [0], [1]), result)

    def test_misaligned_or_duplicate_positions_and_targets_rejected(self):
        a = stage([1, 2, 3, 4], [-1, 2, -3, 4])
        for b in [stage([1, 2, 3], [-1, 2, -3]), stage([1, 2, 3, 4], [-1, 2, -3, 4], [11])]:
            with self.assertRaises(ValueError): paired_summary(a, b, [], [])
        b = copy.deepcopy(a); b['tokens'].append(b['tokens'][0])
        with self.assertRaises(ValueError): paired_summary(a, b, [], [])

    def test_insufficient_or_constant_rank_is_not_fabricated(self):
        a = stage([1, 1, 1, 1], [1, -1, 1, -1])
        b = stage([1, 2, 3, 4], [1, -1, 1, -1])
        result = paired_summary(a, b, [0, 1, 2], [])
        self.assertIsNone(result['input']['rho_E_I_all'])
        self.assertIsNone(result['layer16']['rho_E_I_common_nonbasis'])

    def test_full_sweep_requires_compatible_calibration_and_evidence(self):
        profile = json.loads(PROFILE.read_text())
        with self.assertRaises(ValueError): validate_calibration(None, profile, 'm', 'v')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.json'
            report = {'version': VERSION, 'status': 'passed', 'profile_sha256': sha256(PROFILE),
                      'arguments': {'tokens': 32, 'model': 'm', 'vision_model': 'v'},
                      'questions': [{'case_id': cid, 'status': 'passed'} for cid in CASE_IDS]}
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'source/environment'):
                validate_calibration(path, profile, 'm', 'v')
            # Supply a real source inventory, as the exported calibration does.
            native = json.loads((REPO / 'research/provenance/original_llava_reference.json').read_text())
            report['implementation'] = native
            relatives = [str(p.relative_to(REPO)) for p in (REPO / 'scripts').glob('*.py')]
            relatives += ['environments/original/uv.lock', 'research/provenance/original_llava_reference.json']
            relatives += list(native['native_source_sha256'])
            report['evidence_sha256'] = {'sources/' + name: sha256(REPO / name) for name in relatives}
            path.write_text(json.dumps(report))
            self.assertTrue(validate_calibration(path, profile, 'm', 'v')['passed'])
            with self.assertRaises(ValueError): validate_calibration(path, profile, 'other-model', 'v')
            report['evidence_sha256']['sources/scripts/rq1_layer_native.py'] = 'wrong'
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'source/environment'):
                validate_calibration(path, profile, 'm', 'v')
            report['status'] = 'failed'; path.write_text(json.dumps(report))
            with self.assertRaises(ValueError): validate_calibration(path, profile, 'm', 'v')

    def test_partial_results_can_export_review_without_claiming_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / 'questions').mkdir()
            (output / 'evidence').mkdir()
            (output / 'questions' / 'gqa-test.json').write_text(json.dumps({
                'case_id': 'gqa:test', 'status': 'running', 'input': stage([1, 2, 3], [1, 0, -1])}))
            write_review(output, {'status': 'failed'})
            self.assertIn('Status: failed', (output / 'index.html').read_text())
            self.assertIn('gqa:test', (output / 'tokens.csv').read_text())

    def test_rejected_layer_protocol_stops_before_model_load_and_exports_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'run'
            inputs = Path(directory) / 'inputs'; inputs.mkdir()
            with patch.object(sys, 'argv', ['run_rq1_layer.py', '--inputs', str(inputs), '--output', str(output)]), \
                 patch.object(runner, 'provenance', return_value={}), patch.object(runner, 'validate_runtime'), \
                 patch.object(runner, 'verify_native_reference'), patch.object(runner.signal, 'signal'), \
                 patch.object(runner, 'validate_layer_profile', side_effect=ValueError('Layer input mismatch')), \
                 patch.object(runner, 'load_model') as load, patch('builtins.print'):
                self.assertEqual(runner.main(), 1)
            load.assert_not_called()
            with zipfile.ZipFile(output.with_suffix('.zip')) as archive:
                report = json.loads(archive.read('report.json'))
                self.assertEqual(report['status'], 'failed')
                self.assertIn('Layer input mismatch', report['error'])


if __name__ == '__main__': unittest.main()
