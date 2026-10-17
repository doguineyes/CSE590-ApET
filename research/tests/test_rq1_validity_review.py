"""Saved validity evidence and Unicode export regression; no GPU/inference."""

import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'scripts'))
from analyze_rq1_validity import analyze, assert_close, verify_question
from rq1_protocol import sha256
from run_rq1_validation import write_review as validity_review
from run_rq1_layer import write_review as layer_review
from summarize_rq1_pilot import write_csv

ARCHIVE = REPO/'research/results/rq1-validity-all-20261017T204446432120Z.zip'


class ValidityReviewTests(unittest.TestCase):
    def test_exports_are_utf8_even_when_default_encoding_is_ascii(self):
        original = io.open
        def ascii_default(file, mode='r', buffering=-1, encoding=None, errors=None,
                          newline=None, closefd=True, opener=None):
            if 'b' not in mode and encoding in (None, 'locale'): encoding = 'ascii'
            return original(file, mode, buffering, encoding, errors, newline, closefd, opener)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a,b = root/'validity',root/'layer'; a.mkdir(); b.mkdir()
            with patch('io.open', side_effect=ascii_default):
                validity_review(a, {'status':'passed–✓'})
                layer_review(b, {'status':'passed–✓'})
                write_csv(root/'unicode.csv', [{'answer':'Window → 窗'}])
            for folder in [a,b]:
                self.assertIn('passed–✓', (folder/'index.html').read_text(encoding='utf-8'))
            self.assertIn('Window → 窗', (root/'unicode.csv').read_text(encoding='utf-8'))

    def test_numerical_mismatch_and_nonfinite_values_rejected(self):
        with self.assertRaises(ValueError): assert_close({'I':.1}, {'I':-.1}, 'effect')
        with self.assertRaises(ValueError): assert_close(float('nan'), .1, 'effect')

    def test_existing_derived_folder_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError,'new derived'):
                analyze('nonexistent.zip', Path(directory))

    def require_source(self):
        if not ARCHIVE.exists(): self.skipTest('Private source result unavailable')
        try: import tqdm
        except ImportError: self.skipTest('Use environments/editor/.venv/bin/python for native TextVQA rescoring')

    def test_real_saved_run_is_reverified_without_changing_source(self):
        self.require_source()
        source_hash = sha256(ARCHIVE)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)/'review'
            result = analyze(ARCHIVE,folder,ARCHIVE.with_suffix('.log'))
            self.assertEqual(result['semantic_trials'],72)
            self.assertEqual(result['patch_method_scores'],576)
            self.assertEqual(result['verified_evidence_files'],32)
            self.assertEqual(result['max_full_forward_control_drift'],0)
            self.assertGreater((folder/'index.html').stat().st_size,0)
            self.assertTrue((folder/'method-summary.csv').exists())
        self.assertEqual(sha256(ARCHIVE),source_hash)

    def test_tampered_sensitivity_is_rejected(self):
        self.require_source()
        with zipfile.ZipFile(ARCHIVE) as z:
            profile = json.loads(z.read('evidence/validation-profile.json'))
            entry = profile['perturbations'][0]
            q = json.loads(z.read('questions/'+entry['case_id'].replace(':','-')+'.json'))
            cases = json.loads(z.read('evidence/gqa-manifest.json'))['cases']
            case = next(c for c in cases if c['case_id']==entry['case_id'])
        bad = copy.deepcopy(q)
        bad['alternatives'][0]['effects']['half']['I'] += 1
        with self.assertRaisesRegex(ValueError,'Reference-minus-perturbed'):
            verify_question(bad,entry,case)


if __name__ == '__main__': unittest.main()
