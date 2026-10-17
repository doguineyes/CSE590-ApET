"""Saved layer-evidence checks and real notebook launch encoding; no Torch/GPU."""

import copy
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'scripts'))
from analyze_rq1_layer import analyze, verify_question
from rq1_protocol import sha256

ARCHIVE = REPO/'research/results/rq1-layer16-calibration-v1-20261017T230335470061Z.zip'
DIGEST = '8758b31819ef9f73992dcd46776247b7015107e583ca2615f691b64dd8249fd3'


class LayerReviewTests(unittest.TestCase):
    def source_case(self):
        if not ARCHIVE.exists(): self.skipTest('Private layer calibration not available')
        with zipfile.ZipFile(ARCHIVE) as z:
            q = json.loads(z.read('questions/gqa-202218829.json'))
            entry = json.loads(z.read('evidence/layer-profile.json'))['cases'][0]
            case = next(c for c in json.loads(z.read('evidence/gqa-manifest.json'))['cases']
                        if c['case_id'] == entry['case_id'])
        return q, entry, case

    def test_missing_position_and_changed_stage_targets_rejected(self):
        q, entry, case = self.source_case()
        for mutation in ('missing', 'target'):
            changed = copy.deepcopy(q)
            if mutation == 'missing': changed['decoder']['tokens'].pop()
            else: changed['decoder']['target_token_ids'] = [1]
            with self.assertRaises(ValueError): verify_question(changed, entry, case, 32)

    def test_corrupt_effect_and_nonfinite_scores_rejected(self):
        q, entry, case = self.source_case()
        for mutation in ('effect', 'nan'):
            changed = copy.deepcopy(q)
            if mutation == 'effect': changed['decoder']['tokens'][0]['intervention_score'] += .1
            else: changed['input']['tokens'][0]['perturbed_score']['token_logprobs'][0] = float('nan')
            with self.assertRaises(ValueError): verify_question(changed, entry, case, 32)

    def test_boundary_control_failure_and_drift_rejected(self):
        q, entry, case = self.source_case()
        changed = copy.deepcopy(q); changed['decoder']['controls']['final_boundary_exact'] = False
        with self.assertRaises(ValueError): verify_question(changed, entry, case, 32)
        changed = copy.deepcopy(q)
        score = changed['decoder']['controls']['noop_at_boundary']
        score['token_logprobs'][0] -= .1
        score['sum_logprob'] = sum(score['token_logprobs'])
        score['mean_logprob'] = score['sum_logprob']/len(score['token_logprobs'])
        with self.assertRaisesRegex(ValueError, 'drift'): verify_question(changed, entry, case, 32)

    def test_overwrite_and_wrong_archive_hash_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'new derived'): analyze('missing.zip', directory)
            if ARCHIVE.exists():
                with self.assertRaisesRegex(ValueError, 'SHA256'):
                    analyze(ARCHIVE, Path(directory)/'new', expected_sha256='wrong')

    def test_real_archive_rebuild_preserves_originals(self):
        self.source_case()
        log = ARCHIVE.with_suffix('.log')
        before = sha256(ARCHIVE), sha256(log)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/'review'
            result = analyze(ARCHIVE, out, log, DIGEST)
            self.assertEqual(result['patch_stage_scores'], 384)
            self.assertEqual(result['verified_feature_file_bytes'], 11)
            self.assertEqual(result['verified_evidence_files'], 39)
            self.assertEqual(result['max_full_forward_control_drift'], 0)
            self.assertEqual(result['historical_input_scores_repeated_exactly'], 192)
            self.assertTrue(result['full_run_source_eligibility']['passed'])
            self.assertFalse(result['inference_performed'])
            self.assertFalse(result['tensor_contents_loaded'])
            self.assertFalse(result['same_image']['layer_visual_equal_hash'])
            self.assertGreater(result['same_image']['rho_E_all_576'], .99999)
            self.assertIn('Verified local review', (out/'index.html').read_text(encoding='utf-8'))
        self.assertEqual(before, (sha256(ARCHIVE), sha256(log)))

    def test_notebook_helper_launches_utf8_isolated_child_and_captures_unicode(self):
        notebook = json.loads((REPO/'research/notebooks/kaggle_rq1_layer16_pilot.ipynb').read_text(encoding='utf-8'))
        helper = next(''.join(c['source']) for c in notebook['cells'] if 'def run_layer(' in ''.join(c['source']))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root/'scripts').mkdir()
            (root/'scripts/run_rq1_layer.py').write_text(
                'import json, pathlib, sys\n'
                'assert sys.flags.utf8_mode == 1 and sys.flags.isolated == 1\n'
                'out = pathlib.Path(sys.argv[sys.argv.index("--output")+1])\n'
                'out.mkdir(parents=True)\n'
                'out.joinpath("report.json").write_text(json.dumps({"status":"passed","wall_seconds":0,"questions":[]}))\n'
                'print("Window → 窗 ✓")\n', encoding='utf-8')
            # Redirect the helper's Kaggle working path into the temporary tree.
            def local_path(value):
                path = Path(value)
                return root/path.name if str(path).startswith('/kaggle/working/') else path
            namespace = {'datetime': datetime, 'timezone': timezone, 'Path': local_path,
                'original_python': sys.executable, 'repo': root, 'inputs': root,
                'CHECKPOINT_ARGS': [], 'subprocess': subprocess, 'json': json}
            exec(compile(helper, '<actual notebook helper>', 'exec'), namespace)
            with patch.dict('os.environ', {'LC_ALL':'C', 'PYTHONUTF8':'0', 'PYTHONCOERCECLOCALE':'0'}), \
                 patch('sys.stdout', new_callable=io.StringIO):
                out = namespace['run_layer']()
            self.assertIn('Window → 窗 ✓', out.with_suffix('.log').read_text(encoding='utf-8'))


if __name__ == '__main__': unittest.main()
